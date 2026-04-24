#!/usr/bin/env python3
"""Run a 3k-3k b=128 chat-completions benchmark against the live vLLM service."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = "http://s-20260420181527-6g8xk.ailab-pj.pjh-service.org.cn"
_PROM_METRIC_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)
_PROM_METRIC_FIELDS = {
    "vllm:iteration_tokens_total_count": "iteration_count",
    "vllm:iteration_tokens_total_sum": "iteration_tokens",
    "vllm:prompt_tokens_total": "prompt_tokens",
    "vllm:generation_tokens_total": "generation_tokens",
    "vllm:request_success_total": "request_success",
    "vllm:num_preemptions_total": "preemptions",
}


@dataclass(frozen=True)
class RequestRecord:
    request_index: int
    ok: int
    status_code: int
    latency_ms: float
    response_bytes: int
    error: str


@dataclass(frozen=True)
class MetricsSample:
    sample_index: int
    elapsed_s: float
    iteration_count: float
    iteration_tokens: float
    prompt_tokens: float
    generation_tokens: float
    request_success: float
    preemptions: float


@dataclass(frozen=True)
class MetricsWindow:
    start_index: int
    end_index: int
    duration_s: float
    iteration_count_delta: float
    iteration_tokens_delta: float
    avg_tokens_per_iter: float
    avg_iter_latency_ms: float
    prompt_tokens_delta: float
    generation_tokens_delta: float
    generation_tok_s: float
    request_success_delta: float
    preemptions_delta: float


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send fixed-shape chat requests to /v1/chat/completions."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default="kimi-k2.5")
    parser.add_argument("--num-requests", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--input-words", type=int, default=3000)
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--timeout-s", type=float, default=7200.0)
    parser.add_argument("--prompt-word", default="hi")
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--capture-metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--metrics-interval-s", type=float, default=0.0)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/chat_3k3k_b128"))
    return parser


def _url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + path


def _http_get(url: str, timeout_s: float) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return response.read()


def _http_post_json(url: str, payload: dict[str, Any], timeout_s: float) -> tuple[int, bytes]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _make_prompt(request_index: int, input_words: int, prompt_word: str) -> str:
    if input_words <= 0:
        raise ValueError("input_words must be positive")
    words = [f"req{request_index}"]
    words.extend([prompt_word] * max(input_words - 1, 0))
    return " ".join(words)


def _make_payload(args: argparse.Namespace, request_index: int) -> dict[str, Any]:
    return {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": _make_prompt(
                    request_index=request_index,
                    input_words=args.input_words,
                    prompt_word=args.prompt_word,
                ),
            }
        ],
        "max_tokens": args.max_tokens,
        "min_tokens": args.max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": args.stream,
    }


def _send_one(args: argparse.Namespace, endpoint: str, request_index: int) -> RequestRecord:
    payload = _make_payload(args, request_index)
    start = time.perf_counter()
    try:
        status_code, response_body = _http_post_json(
            endpoint,
            payload,
            timeout_s=args.timeout_s,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        if status_code != 200:
            return RequestRecord(
                request_index=request_index,
                ok=0,
                status_code=status_code,
                latency_ms=latency_ms,
                response_bytes=len(response_body),
                error=response_body[:500].decode("utf-8", errors="replace"),
            )
        return RequestRecord(
            request_index=request_index,
            ok=1,
            status_code=status_code,
            latency_ms=latency_ms,
            response_bytes=len(response_body),
            error="",
        )
    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.perf_counter() - start) * 1000.0
        return RequestRecord(
            request_index=request_index,
            ok=0,
            status_code=0,
            latency_ms=latency_ms,
            response_bytes=0,
            error=str(exc),
        )


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


def _write_jsonl(path: Path, rows: list[RequestRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def _parse_metrics_sample(
    *,
    sample_index: int,
    started_at: float,
    raw_metrics: bytes,
) -> MetricsSample:
    values = {field: 0.0 for field in _PROM_METRIC_FIELDS.values()}
    for raw_line in raw_metrics.decode("utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PROM_METRIC_RE.match(line)
        if match is None:
            continue
        field = _PROM_METRIC_FIELDS.get(match.group("name"))
        if field is None:
            continue
        values[field] += float(match.group("value"))
    return MetricsSample(
        sample_index=sample_index,
        elapsed_s=time.perf_counter() - started_at,
        **values,
    )


def _capture_metrics_sample(
    *,
    sample_index: int,
    started_at: float,
    metrics_url: str,
    samples_dir: Path,
) -> MetricsSample:
    raw_metrics = _http_get(metrics_url, timeout_s=30.0)
    samples_dir.mkdir(parents=True, exist_ok=True)
    (samples_dir / f"metrics_{sample_index:04d}.txt").write_bytes(raw_metrics)
    return _parse_metrics_sample(
        sample_index=sample_index,
        started_at=started_at,
        raw_metrics=raw_metrics,
    )


def _metrics_sampler(
    *,
    stop_event: threading.Event,
    metrics_url: str,
    samples_dir: Path,
    interval_s: float,
    started_at: float,
    samples: list[MetricsSample],
    lock: threading.Lock,
) -> None:
    sample_index = 0
    while not stop_event.is_set():
        try:
            sample = _capture_metrics_sample(
                sample_index=sample_index,
                started_at=started_at,
                metrics_url=metrics_url,
                samples_dir=samples_dir,
            )
            with lock:
                samples.append(sample)
        except Exception as exc:  # noqa: BLE001
            print(f"metrics sample failed: {exc}", flush=True)
        sample_index += 1
        stop_event.wait(interval_s)


def _metrics_windows(samples: list[MetricsSample]) -> list[MetricsWindow]:
    windows: list[MetricsWindow] = []
    ordered = sorted(samples, key=lambda row: row.sample_index)
    for before, after in zip(ordered, ordered[1:], strict=False):
        duration_s = after.elapsed_s - before.elapsed_s
        iter_delta = after.iteration_count - before.iteration_count
        token_delta = after.iteration_tokens - before.iteration_tokens
        gen_delta = after.generation_tokens - before.generation_tokens
        windows.append(
            MetricsWindow(
                start_index=before.sample_index,
                end_index=after.sample_index,
                duration_s=duration_s,
                iteration_count_delta=iter_delta,
                iteration_tokens_delta=token_delta,
                avg_tokens_per_iter=token_delta / iter_delta if iter_delta > 0 else 0.0,
                avg_iter_latency_ms=duration_s * 1000.0 / iter_delta if iter_delta > 0 else 0.0,
                prompt_tokens_delta=after.prompt_tokens - before.prompt_tokens,
                generation_tokens_delta=gen_delta,
                generation_tok_s=gen_delta / duration_s if duration_s > 0 else 0.0,
                request_success_delta=after.request_success - before.request_success,
                preemptions_delta=after.preemptions - before.preemptions,
            )
        )
    return windows


def _write_metrics_windows(path: Path, windows: list[MetricsWindow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(MetricsWindow.__dataclass_fields__)
    with path.open("w") as f:
        f.write("\t".join(fields) + "\n")
        for window in windows:
            row = asdict(window)
            f.write("\t".join(str(row[field]) for field in fields) + "\n")


def _summarize_tail_windows(windows: list[MetricsWindow]) -> dict[str, float]:
    pure_decode = [
        row for row in windows
        if row.prompt_tokens_delta == 0
        and row.generation_tokens_delta > 0
        and row.iteration_count_delta > 0
    ]
    if not pure_decode:
        return {}
    half_index = len(windows) // 2
    tail = [row for row in pure_decode if row.start_index >= half_index] or pure_decode
    duration_s = sum(row.duration_s for row in tail)
    iter_delta = sum(row.iteration_count_delta for row in tail)
    token_delta = sum(row.iteration_tokens_delta for row in tail)
    gen_delta = sum(row.generation_tokens_delta for row in tail)
    return {
        "tail_window_count": float(len(tail)),
        "tail_duration_s": duration_s,
        "tail_iteration_count_delta": iter_delta,
        "tail_iteration_tokens_delta": token_delta,
        "tail_avg_tokens_per_iter": token_delta / iter_delta if iter_delta > 0 else 0.0,
        "tail_avg_iter_latency_ms": duration_s * 1000.0 / iter_delta if iter_delta > 0 else 0.0,
        "tail_generation_tok_s": gen_delta / duration_s if duration_s > 0 else 0.0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    endpoint = _url(args.base_url, "/v1/chat/completions")
    metrics_url = _url(args.base_url, "/metrics")

    if args.capture_metrics:
        (args.out_dir / "metrics_before.txt").write_bytes(
            _http_get(metrics_url, timeout_s=30.0)
        )

    started_at = time.perf_counter()
    records: list[RequestRecord] = []
    metrics_samples: list[MetricsSample] = []
    metrics_lock = threading.Lock()
    metrics_stop = threading.Event()
    sampler_thread: threading.Thread | None = None

    if args.metrics_interval_s > 0:
        sampler_thread = threading.Thread(
            target=_metrics_sampler,
            kwargs={
                "stop_event": metrics_stop,
                "metrics_url": metrics_url,
                "samples_dir": args.out_dir / "metrics_samples",
                "interval_s": args.metrics_interval_s,
                "started_at": started_at,
                "samples": metrics_samples,
                "lock": metrics_lock,
            },
            daemon=True,
        )
        sampler_thread.start()

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(_send_one, args, endpoint, request_index)
            for request_index in range(args.num_requests)
        ]
        for done, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            if done % 8 == 0 or done == args.num_requests:
                ok = sum(row.ok for row in records)
                print(
                    f"progress: {done}/{args.num_requests}, ok={ok}, fail={done - ok}",
                    flush=True,
                )
    wall_s = time.perf_counter() - started_at

    if sampler_thread is not None:
        metrics_stop.set()
        sampler_thread.join(timeout=max(args.metrics_interval_s, 1.0) + 5.0)
        with metrics_lock:
            next_index = max((row.sample_index for row in metrics_samples), default=-1) + 1
        try:
            final_sample = _capture_metrics_sample(
                sample_index=next_index,
                started_at=started_at,
                metrics_url=metrics_url,
                samples_dir=args.out_dir / "metrics_samples",
            )
            with metrics_lock:
                metrics_samples.append(final_sample)
        except Exception as exc:  # noqa: BLE001
            print(f"final metrics sample failed: {exc}", flush=True)

    if args.capture_metrics:
        (args.out_dir / "metrics_after.txt").write_bytes(
            _http_get(metrics_url, timeout_s=30.0)
        )

    with metrics_lock:
        metrics_samples = list(metrics_samples)
    windows = _metrics_windows(metrics_samples)
    if metrics_samples:
        _write_json(
            args.out_dir / "metrics_samples.json",
            {"samples": [asdict(row) for row in sorted(metrics_samples, key=lambda item: item.sample_index)]},
        )
    if windows:
        _write_metrics_windows(args.out_dir / "metrics_windows.tsv", windows)
    tail_summary = _summarize_tail_windows(windows)

    records.sort(key=lambda row: row.request_index)
    ok_latencies = [row.latency_ms for row in records if row.ok == 1]
    failed = [row for row in records if row.ok != 1]
    summary = {
        "base_url": args.base_url,
        "endpoint": endpoint,
        "model": args.model,
        "num_requests": args.num_requests,
        "concurrency": args.concurrency,
        "input_words": args.input_words,
        "max_tokens": args.max_tokens,
        "stream": args.stream,
        "wall_s": wall_s,
        "ok_requests": len(ok_latencies),
        "failed_requests": len(failed),
        "mean_latency_ms": statistics.mean(ok_latencies) if ok_latencies else 0.0,
        "p50_latency_ms": _percentile(ok_latencies, 50.0),
        "p99_latency_ms": _percentile(ok_latencies, 99.0),
        "first_error": failed[0].error if failed else "",
        "metrics_interval_s": args.metrics_interval_s,
        "metrics_tail_summary": tail_summary,
    }
    _write_json(args.out_dir / "summary.json", summary)
    _write_jsonl(args.out_dir / "records.jsonl", records)
    return summary


def main() -> None:
    args = _build_parser().parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["failed_requests"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
