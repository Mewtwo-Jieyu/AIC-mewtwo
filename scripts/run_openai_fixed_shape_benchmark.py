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

STREAM_DONE = object()


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
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    stream_chunk_count: int = 0


@dataclass
class StreamResponseState:
    started_at: float
    first_token_at: float | None = None
    finished_at: float | None = None
    usage: dict[str, Any] | None = None
    stream_chunk_count: int = 0

    def observe(self, payload: dict[str, Any], *, observed_at: float) -> None:
        usage = payload.get("usage")
        if usage:
            self.usage = usage
        choices = payload.get("choices") or []
        has_text = any(bool(choice.get("text")) for choice in choices)
        if not has_text:
            return
        if self.first_token_at is None:
            self.first_token_at = observed_at
        self.stream_chunk_count += 1

    def finish(self, *, observed_at: float) -> None:
        self.finished_at = observed_at

    def to_record(
        self,
        *,
        request_index: int,
        status_code: int,
        expected_prompt_tokens: int,
        expected_completion_tokens: int,
    ) -> RequestRecord:
        if self.usage is None:
            raise ValueError("stream_usage_missing")
        if self.first_token_at is None:
            raise ValueError("stream_first_token_missing")
        if self.finished_at is None:
            raise ValueError("stream_done_missing")
        prompt_tokens = int(self.usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(self.usage.get("completion_tokens", 0) or 0)
        total_tokens = int(self.usage.get("total_tokens", 0) or 0)
        if prompt_tokens != expected_prompt_tokens:
            raise ValueError(
                f"prompt_tokens_mismatch:{prompt_tokens}!={expected_prompt_tokens}"
            )
        if completion_tokens != expected_completion_tokens:
            raise ValueError(
                "completion_tokens_mismatch:"
                f"{completion_tokens}!={expected_completion_tokens}"
            )
        if total_tokens != prompt_tokens + completion_tokens:
            raise ValueError("total_tokens_mismatch")
        if not self.started_at <= self.first_token_at <= self.finished_at:
            raise ValueError("stream_time_not_monotonic")
        ttft_ms = (self.first_token_at - self.started_at) * 1000.0
        latency_ms = (self.finished_at - self.started_at) * 1000.0
        tpot_ms = (latency_ms - ttft_ms) / max(completion_tokens - 1, 1)
        return RequestRecord(
            request_index=request_index,
            ok=1,
            status_code=status_code,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            error="",
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            stream_chunk_count=self.stream_chunk_count,
        )


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
    parser.add_argument(
        "--prompt-variant-mode",
        choices=("fixed", "rotating"),
        default="fixed",
        help="fixed keeps the historical single prompt; rotating changes the prompt prefix per request.",
    )
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--records-jsonl", type=Path)
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use SSE streaming and record per-request TTFT and TPOT.",
    )
    parser.add_argument(
        "--skip-prompt-token-id-probe",
        action="store_true",
        help="Send text prompts without a capability probe so measured metrics exclude probe traffic.",
    )
    parser.add_argument(
        "--request-id-prefix",
        default="",
        help="Optional diagnostic X-Request-Id prefix; empty preserves existing behavior.",
    )
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
    safe_token_ids = _collect_safe_single_token_ids(tokenizer, target_len)
    token_id = safe_token_ids[0]
    prompt = _roundtrip_prompt_from_token_id(tokenizer, token_id, target_len)
    if prompt is not None:
        return prompt, [token_id] * target_len
    raise RuntimeError(
        "failed to build an exact-length prompt from one-token candidates; "
        "please add tokenizer-specific candidates"
    )


def _collect_safe_single_token_ids(tokenizer: Any, target_len: int) -> list[int]:
    safe_token_ids: list[int] = []
    for candidate in _ONE_TOKEN_CANDIDATES:
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) != 1:
            continue
        if _roundtrip_prompt_from_token_id(tokenizer, ids[0], target_len) is not None:
            safe_token_ids.append(ids[0])
    if not safe_token_ids:
        raise RuntimeError(
            "failed to find any exact-length one-token candidates; "
            "please add tokenizer-specific candidates"
        )
    return safe_token_ids


def _decode_exact_prompt(tokenizer: Any, prompt_token_ids: list[int]) -> str | None:
    prompt = tokenizer.decode(
        prompt_token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    actual_len = len(tokenizer.encode(prompt, add_special_tokens=False))
    if actual_len == len(prompt_token_ids):
        return prompt
    return None


def _build_rotating_prompt_variants(
    *,
    tokenizer: Any,
    safe_token_ids: list[int],
    target_len: int,
    variant_count: int,
) -> list[tuple[str, list[int]]]:
    if len(safe_token_ids) < 2:
        raise RuntimeError("rotating prompt variants require at least two safe one-token candidates")
    variants: list[tuple[str, list[int]]] = []
    for variant_index in range(variant_count):
        prompt_token_ids = [
            safe_token_ids[(pos + variant_index) % len(safe_token_ids)]
            for pos in range(target_len)
        ]
        prompt = _decode_exact_prompt(tokenizer, prompt_token_ids)
        if prompt is None:
            raise RuntimeError(
                "failed to build an exact-length rotating prompt variant; "
                "please add tokenizer-specific candidates"
            )
        variants.append((prompt, prompt_token_ids))
    return variants


def build_prompt_variants(
    tokenizer_path: str,
    target_len: int,
    variant_count: int,
    mode: str,
) -> list[tuple[str, list[int]]]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    safe_token_ids = _collect_safe_single_token_ids(tokenizer, target_len)
    if mode == "fixed":
        token_id = safe_token_ids[0]
        prompt = _roundtrip_prompt_from_token_id(tokenizer, token_id, target_len)
        if prompt is None:
            raise RuntimeError("failed to build fixed prompt from a previously validated token")
        return [(prompt, [token_id] * target_len)]
    if mode == "rotating":
        return _build_rotating_prompt_variants(
            tokenizer=tokenizer,
            safe_token_ids=safe_token_ids,
            target_len=target_len,
            variant_count=variant_count,
        )
    raise ValueError(f"unknown prompt variant mode: {mode}")


async def _post_completion(
    session: aiohttp.ClientSession,
    endpoint: str,
    payload: dict[str, Any],
    request_index: int,
    headers: dict[str, str] | None = None,
) -> RequestRecord:
    start = time.perf_counter()
    try:
        async with session.post(endpoint, json=payload, headers=headers) as response:
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


def _parse_sse_line(line: bytes) -> dict[str, Any] | object | None:
    stripped = line.strip()
    if not stripped or stripped.startswith(b":") or not stripped.startswith(b"data:"):
        return None
    data = stripped.removeprefix(b"data:").strip()
    if data == b"[DONE]":
        return STREAM_DONE
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise ValueError("stream_payload_not_object")
    return payload


async def _post_streaming_completion(
    session: aiohttp.ClientSession,
    endpoint: str,
    payload: dict[str, Any],
    request_index: int,
    expected_prompt_tokens: int,
    expected_completion_tokens: int,
    headers: dict[str, str] | None = None,
) -> RequestRecord:
    start = time.perf_counter()
    state = StreamResponseState(started_at=start)
    try:
        async with session.post(endpoint, json=payload, headers=headers) as response:
            if response.status != 200:
                body = await response.text()
                return RequestRecord(
                    request_index=request_index,
                    ok=0,
                    status_code=response.status,
                    latency_ms=(time.perf_counter() - start) * 1000.0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    error=body[:500],
                )
            while True:
                line = await response.content.readline()
                if not line:
                    break
                parsed = _parse_sse_line(line)
                if parsed is None:
                    continue
                observed_at = time.perf_counter()
                if parsed is STREAM_DONE:
                    state.finish(observed_at=observed_at)
                    break
                state.observe(parsed, observed_at=observed_at)
            return state.to_record(
                request_index=request_index,
                status_code=response.status,
                expected_prompt_tokens=expected_prompt_tokens,
                expected_completion_tokens=expected_completion_tokens,
            )
    except Exception as exc:  # noqa: BLE001
        return RequestRecord(
            request_index=request_index,
            ok=0,
            status_code=0,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            error=str(exc),
        )


def _request_headers(
    request_id_prefix: str, prompt_tokens: int, request_index: int
) -> dict[str, str] | None:
    if not request_id_prefix:
        return None
    return {
        "X-AIC-Prompt-Tokens": str(prompt_tokens),
        "X-Request-Id": f"{request_id_prefix}-{request_index:06d}",
    }


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
    stream: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": output_len,
        "min_tokens": output_len,
        "temperature": temperature,
        "ignore_eos": ignore_eos,
    }
    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
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
    stream: bool,
    expected_prompt_tokens: int,
    expected_completion_tokens: int,
) -> None:
    for idx in range(warmup_requests):
        if stream:
            record = await _post_streaming_completion(
                session,
                endpoint,
                payload,
                request_index=-(idx + 1),
                expected_prompt_tokens=expected_prompt_tokens,
                expected_completion_tokens=expected_completion_tokens,
            )
        else:
            record = await _post_completion(session, endpoint, payload, request_index=-(idx + 1))
        if record.ok != 1:
            raise RuntimeError(f"warmup failed: status={record.status_code} error={record.error}")


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    prompt_variants = build_prompt_variants(
        args.tokenizer,
        args.input_len,
        max(1, args.num_prompts),
        args.prompt_variant_mode,
    )
    endpoint = f"{_get_base_url(args.host, args.port)}/v1/completions"
    timeout = aiohttp.ClientTimeout(total=args.timeout_s)

    connector = aiohttp.TCPConnector(limit=args.max_concurrency)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        use_prompt_token_ids = False
        if not args.skip_prompt_token_id_probe:
            use_prompt_token_ids = await _supports_prompt_token_ids(
                session=session,
                endpoint=endpoint,
                model=args.model,
                sample_prompt_token_ids=prompt_variants[0][1][: min(8, len(prompt_variants[0][1]))],
            )
        payloads = [
            _make_payload(
                model=args.model,
                output_len=args.output_len,
                temperature=args.temperature,
                ignore_eos=args.ignore_eos,
                prompt=prompt,
                prompt_token_ids=prompt_token_ids if use_prompt_token_ids else None,
                stream=args.stream,
            )
            for prompt, prompt_token_ids in prompt_variants
        ]

        if args.warmup_requests > 0:
            await _run_warmup(
                session=session,
                endpoint=endpoint,
                payload=payloads[0],
                warmup_requests=args.warmup_requests,
                stream=args.stream,
                expected_prompt_tokens=args.input_len,
                expected_completion_tokens=args.output_len,
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
                headers = _request_headers(
                    args.request_id_prefix,
                    args.input_len,
                    request_index,
                )
                if args.stream:
                    record = await _post_streaming_completion(
                        session=session,
                        endpoint=endpoint,
                        payload=payloads[request_index % len(payloads)],
                        request_index=request_index,
                        expected_prompt_tokens=args.input_len,
                        expected_completion_tokens=args.output_len,
                        headers=headers,
                    )
                else:
                    record = await _post_completion(
                        session=session,
                        endpoint=endpoint,
                        payload=payloads[request_index % len(payloads)],
                        request_index=request_index,
                        headers=headers,
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
        "stream": args.stream,
        "skip_prompt_token_id_probe": args.skip_prompt_token_id_probe,
        "use_prompt_token_ids": use_prompt_token_ids,
        "prompt_variant_mode": args.prompt_variant_mode,
        "prompt_variant_count": len(prompt_variants),
        "unique_prompt_prefixes": len({tuple(ids[: min(64, len(ids))]) for _, ids in prompt_variants}),
        "wall_s": wall_s,
        "ok_requests": len(ok_records),
        "failed_requests": len(fail_records),
        "mean_latency_ms": (
            sum(r.latency_ms for r in ok_records) / len(ok_records) if ok_records else 0.0
        ),
        "p50_latency_ms": _percentile([r.latency_ms for r in ok_records], 50.0),
        "p90_latency_ms": _percentile([r.latency_ms for r in ok_records], 90.0),
        "p99_latency_ms": _percentile([r.latency_ms for r in ok_records], 99.0),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "request_rate_rps": len(ok_records) / wall_s if wall_s > 0 else 0.0,
        "output_tok_s": total_completion_tokens / wall_s if wall_s > 0 else 0.0,
        "total_tok_s": total_tokens / wall_s if wall_s > 0 else 0.0,
        "records": [asdict(r) for r in records],
    }
    if args.stream:
        summary.update(_stream_metric_summary(ok_records))
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


def _stream_metric_summary(records: list[RequestRecord]) -> dict[str, float]:
    ttft_values = [r.ttft_ms for r in records if r.ok == 1 and r.ttft_ms is not None]
    tpot_values = [r.tpot_ms for r in records if r.ok == 1 and r.tpot_ms is not None]
    return {
        "mean_ttft_ms": sum(ttft_values) / len(ttft_values) if ttft_values else 0.0,
        "p50_ttft_ms": _percentile(ttft_values, 50.0),
        "p90_ttft_ms": _percentile(ttft_values, 90.0),
        "p99_ttft_ms": _percentile(ttft_values, 99.0),
        "mean_tpot_ms": sum(tpot_values) / len(tpot_values) if tpot_values else 0.0,
        "p50_tpot_ms": _percentile(tpot_values, 50.0),
        "p90_tpot_ms": _percentile(tpot_values, 90.0),
        "p99_tpot_ms": _percentile(tpot_values, 99.0),
    }


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
                "stream": summary["stream"],
                "prompt_variant_mode": summary["prompt_variant_mode"],
                "unique_prompt_prefixes": summary["unique_prompt_prefixes"],
                "output_tok_s": round(summary["output_tok_s"], 3),
                "total_tok_s": round(summary["total_tok_s"], 3),
                "mean_latency_ms": round(summary["mean_latency_ms"], 3),
                "p50_latency_ms": round(summary["p50_latency_ms"], 3),
                "p90_latency_ms": round(summary["p90_latency_ms"], 3),
                "p99_latency_ms": round(summary["p99_latency_ms"], 3),
                **(
                    {
                        key: round(summary[key], 3)
                        for key in (
                            "mean_ttft_ms",
                            "p50_ttft_ms",
                            "p90_ttft_ms",
                            "p99_ttft_ms",
                            "mean_tpot_ms",
                            "p50_tpot_ms",
                            "p90_tpot_ms",
                            "p99_tpot_ms",
                        )
                    }
                    if summary["stream"]
                    else {}
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if summary["failed_requests"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
