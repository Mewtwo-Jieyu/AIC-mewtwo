#!/usr/bin/env python3
"""Closed-loop fixed-shape benchmark client for vLLM OpenAI serving."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import aiohttp
from transformers import AutoTokenizer

_ONE_TOKEN_CANDIDATES = [
    " a",
    " b",
    " c",
    " the",
    " hello",
    " world",
    ".",
    " .",
    " test",
    "\n",
    "\n\n",
]


@dataclass(frozen=True)
class RequestRecord:
    request_index: int
    ok: int
    status_code: int
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    error: str


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a fixed-shape closed-loop benchmark against /v1/completions."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="kimi-k2.5")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--num-prompts", type=int, default=256)
    parser.add_argument("--max-concurrency", type=int, default=128)
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--output-len", type=int, required=True)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=7200.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--ignore-eos", action="store_true", default=True)
    parser.add_argument("--disable-ignore-eos", dest="ignore_eos", action="store_false")
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--records-jsonl", type=Path)
    return parser


def _get_base_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def _roundtrip_prompt_from_token_id(
    tokenizer: Any,
    token_id: int,
    target_len: int,
) -> str | None:
    prompt = tokenizer.decode(
        [token_id] * target_len,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    actual_len = len(tokenizer.encode(prompt, add_special_tokens=False))
    if actual_len == target_len:
        return prompt
    return None


def build_fixed_prompt(tokenizer_path: str, target_len: int) -> tuple[str, list[int]]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    for candidate in _ONE_TOKEN_CANDIDATES:
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) != 1:
            continue
        prompt = _roundtrip_prompt_from_token_id(tokenizer, ids[0], target_len)
        if prompt is None:
            continue
        return prompt, [ids[0]] * target_len
    raise RuntimeError(
        "failed to build an exact-length prompt from one-token candidates; "
        "please add tokenizer-specific candidates"
    )


async def _post_completion(
    session: aiohttp.ClientSession,
    endpoint: str,
    payload: dict[str, Any],
    request_index: int,
) -> RequestRecord:
    start = time.perf_counter()
    try:
        async with session.post(endpoint, json=payload) as response:
            latency_ms = (time.perf_counter() - start) * 1000.0
            body = await response.text()
            if response.status != 200:
                return RequestRecord(
                    request_index=request_index,
                    ok=0,
                    status_code=response.status,
                    latency_ms=latency_ms,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    error=body[:500],
                )
            data = json.loads(body)
            usage = data.get("usage", {}) or {}
            return RequestRecord(
                request_index=request_index,
                ok=1,
                status_code=response.status,
                latency_ms=latency_ms,
                prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                total_tokens=int(usage.get("total_tokens", 0) or 0),
                error="",
            )
    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.perf_counter() - start) * 1000.0
        return RequestRecord(
            request_index=request_index,
            ok=0,
            status_code=0,
            latency_ms=latency_ms,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            error=str(exc),
        )


async def _supports_prompt_token_ids(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    sample_prompt_token_ids: list[int],
) -> bool:
    payload = {
        "model": model,
        "prompt_token_ids": sample_prompt_token_ids,
        "max_tokens": 1,
        "temperature": 0.0,
        "ignore_eos": True,
    }
    async with session.post(endpoint, json=payload) as response:
        return response.status == 200


def _make_payload(
    *,
    model: str,
    output_len: int,
    temperature: float,
    ignore_eos: bool,
    prompt: str,
    prompt_token_ids: list[int] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": output_len,
        "min_tokens": output_len,
        "temperature": temperature,
        "ignore_eos": ignore_eos,
    }
    if prompt_token_ids is not None:
        payload["prompt_token_ids"] = prompt_token_ids
    else:
        payload["prompt"] = prompt
    return payload


async def _run_warmup(
    session: aiohttp.ClientSession,
    endpoint: str,
    payload: dict[str, Any],
    warmup_requests: int,
) -> None:
    for idx in range(warmup_requests):
        record = await _post_completion(session, endpoint, payload, request_index=-(idx + 1))
        if record.ok != 1:
            raise RuntimeError(f"warmup failed: status={record.status_code} error={record.error}")


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    prompt, prompt_token_ids = build_fixed_prompt(args.tokenizer, args.input_len)
    endpoint = f"{_get_base_url(args.host, args.port)}/v1/completions"
    timeout = aiohttp.ClientTimeout(total=args.timeout_s)

    connector = aiohttp.TCPConnector(limit=args.max_concurrency)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        use_prompt_token_ids = await _supports_prompt_token_ids(
            session=session,
            endpoint=endpoint,
            model=args.model,
            sample_prompt_token_ids=prompt_token_ids[: min(8, len(prompt_token_ids))],
        )
        payload = _make_payload(
            model=args.model,
            output_len=args.output_len,
            temperature=args.temperature,
            ignore_eos=args.ignore_eos,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids if use_prompt_token_ids else None,
        )

        if args.warmup_requests > 0:
            await _run_warmup(
                session=session,
                endpoint=endpoint,
                payload=payload,
                warmup_requests=args.warmup_requests,
            )

        records: list[RequestRecord] = []
        next_request_index = 0
        lock = asyncio.Lock()

        async def worker() -> None:
            nonlocal next_request_index
            while True:
                async with lock:
                    if next_request_index >= args.num_prompts:
                        return
                    request_index = next_request_index
                    next_request_index += 1
                record = await _post_completion(
                    session=session,
                    endpoint=endpoint,
                    payload=payload,
                    request_index=request_index,
                )
                records.append(record)
                done = len(records)
                if done % 8 == 0 or done == args.num_prompts:
                    ok = sum(r.ok for r in records)
                    print(f"progress: {done}/{args.num_prompts}, ok={ok}, fail={done - ok}", flush=True)

        started_at = time.perf_counter()
        await asyncio.gather(*[worker() for _ in range(args.max_concurrency)])
        wall_s = time.perf_counter() - started_at

    records.sort(key=lambda item: item.request_index)
    ok_records = [r for r in records if r.ok == 1]
    fail_records = [r for r in records if r.ok != 1]
    total_prompt_tokens = sum(r.prompt_tokens for r in ok_records)
    total_completion_tokens = sum(r.completion_tokens for r in ok_records)
    total_tokens = sum(r.total_tokens for r in ok_records)
    summary = {
        "endpoint": endpoint,
        "model": args.model,
        "num_prompts": args.num_prompts,
        "max_concurrency": args.max_concurrency,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "warmup_requests": args.warmup_requests,
        "use_prompt_token_ids": use_prompt_token_ids,
        "wall_s": wall_s,
        "ok_requests": len(ok_records),
        "failed_requests": len(fail_records),
        "mean_latency_ms": (
            sum(r.latency_ms for r in ok_records) / len(ok_records) if ok_records else 0.0
        ),
        "p50_latency_ms": _percentile([r.latency_ms for r in ok_records], 50.0),
        "p99_latency_ms": _percentile([r.latency_ms for r in ok_records], 99.0),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "request_rate_rps": len(ok_records) / wall_s if wall_s > 0 else 0.0,
        "output_tok_s": total_completion_tokens / wall_s if wall_s > 0 else 0.0,
        "total_tok_s": total_tokens / wall_s if wall_s > 0 else 0.0,
        "records": [asdict(r) for r in records],
    }
    return summary


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _write_records_jsonl(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in summary["records"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = _build_parser().parse_args()
    summary = asyncio.run(run_benchmark(args))
    _write_json(args.result_json, summary)
    if args.records_jsonl is not None:
        _write_records_jsonl(args.records_jsonl, summary)
    print(
        json.dumps(
            {
                "ok_requests": summary["ok_requests"],
                "failed_requests": summary["failed_requests"],
                "use_prompt_token_ids": summary["use_prompt_token_ids"],
                "output_tok_s": round(summary["output_tok_s"], 3),
                "total_tok_s": round(summary["total_tok_s"], 3),
                "mean_latency_ms": round(summary["mean_latency_ms"], 3),
                "p50_latency_ms": round(summary["p50_latency_ms"], 3),
                "p99_latency_ms": round(summary["p99_latency_ms"], 3),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
