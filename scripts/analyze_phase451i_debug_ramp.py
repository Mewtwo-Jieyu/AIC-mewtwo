#!/usr/bin/env python3
"""Phase451-I: parse the DEBUG ramp observation run."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

SCENARIO = "K2.5-tp4ep8dp2-8k2k"
RUN_DIR = REPO_ROOT / "docs/iter_gap_investigation/phase451i_debug_ramp"
SCENARIO_DIR = RUN_DIR / SCENARIO
DEFAULT_SERVE_LOG = SCENARIO_DIR / "serve.log.gz"
DEFAULT_METRICS = SCENARIO_DIR / "metrics.jsonl"
DEFAULT_BENCH_RESULT = SCENARIO_DIR / "bench_result.json"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase451i_debug_ramp.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase451i_debug_ramp.md"

CSV_FIELDS = ["section", "scenario", "metric", "value", "target", "status", "note"]

COUNTS_RE = re.compile(
    r"Received counts: \[\[(?P<w0>\d+), (?P<r0>\d+)\], "
    r"\[(?P<w1>\d+), (?P<r1>\d+)\]\]"
)
PROM_RE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[-+0-9.eE]+)$"
)
LABEL_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="([^"]*)"')
ITER_RE = re.compile(
    r"\(EngineCore_DP(?P<dp_rank>\d+) pid=.*?\).*?Iteration\((?P<iteration>\d+)\): "
    r"(?P<ctx_requests>\d+) context requests, (?P<ctx_tokens>\d+) context tokens, "
    r"(?P<generation_requests>\d+) generation requests, (?P<generation_tokens>\d+) generation tokens, "
    r"iteration elapsed time: (?P<elapsed_ms>[-+0-9.eE]+) ms"
)
KEYWORD_RES = {
    "added": re.compile(r"\b(added|add(?:ing)?|new)\s+request\b|\brequest\b.*\badded\b", re.IGNORECASE),
    "preempt": re.compile(r"preempt", re.IGNORECASE),
    "victim": re.compile(r"victim", re.IGNORECASE),
}


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def _fmt(value: object) -> str:
    if value == "":
        return ""
    if isinstance(value, bool):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "nan"
    if abs(number) >= 1000:
        return f"{number:.0f}"
    if number.is_integer():
        return f"{number:.0f}"
    return f"{number:.6f}".rstrip("0").rstrip(".")


def _row(
    section: str,
    metric: str,
    value: object,
    *,
    target: object = "",
    status: str = "",
    note: str = "",
    scenario: str = SCENARIO,
) -> dict[str, object]:
    return {
        "section": section,
        "scenario": scenario,
        "metric": metric,
        "value": value,
        "target": target,
        "status": status,
        "note": note,
    }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def parse_received_counts_line(line: str, *, index: int) -> dict[str, int] | None:
    match = COUNTS_RE.search(line)
    if not match:
        return None
    return {
        "index": index,
        "w0": int(match.group("w0")),
        "r0": int(match.group("r0")),
        "w1": int(match.group("w1")),
        "r1": int(match.group("r1")),
    }


def parse_iteration_line(line: str, *, line_no: int) -> dict[str, int | float] | None:
    match = ITER_RE.search(line)
    if not match:
        return None
    return {
        "line_no": line_no,
        "dp_rank": int(match.group("dp_rank")),
        "iteration": int(match.group("iteration")),
        "ctx_requests": int(match.group("ctx_requests")),
        "ctx_tokens": int(match.group("ctx_tokens")),
        "generation_requests": int(match.group("generation_requests")),
        "generation_tokens": int(match.group("generation_tokens")),
        "elapsed_ms": float(match.group("elapsed_ms")),
    }


def parse_debug_log(path: Path) -> tuple[list[dict[str, int]], list[dict[str, int | float]], dict[str, int]]:
    counts: list[dict[str, int]] = []
    iterations: list[dict[str, int | float]] = []
    keywords: Counter[str] = Counter()
    count_index = 0
    with _open_text(path) as f:
        for line_no, line in enumerate(f, start=1):
            parsed_counts = parse_received_counts_line(line, index=count_index)
            if parsed_counts is not None:
                counts.append(parsed_counts)
                count_index += 1
            parsed_iter = parse_iteration_line(line, line_no=line_no)
            if parsed_iter is not None:
                iterations.append(parsed_iter)
            for key, pattern in KEYWORD_RES.items():
                if pattern.search(line):
                    keywords[key] += 1
    return counts, iterations, dict(keywords)


def parse_prometheus_line(line: str) -> tuple[str, dict[str, str], float] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    match = PROM_RE.match(stripped)
    if not match:
        return None
    labels = {
        label.group(1): label.group(2)
        for label in LABEL_RE.finditer(match.group("labels") or "")
    }
    return match.group("name"), labels, float(match.group("value"))


def extract_final_metric_values(path: Path, metric_names: set[str]) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = {name: {} for name in metric_names}
    if not path.exists():
        return values
    with _open_text(path) as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            for metric_line in str(payload.get("body", "")).splitlines():
                parsed = parse_prometheus_line(metric_line)
                if parsed is None:
                    continue
                name, labels, value = parsed
                if name not in metric_names:
                    continue
                engine = labels.get("engine", "global")
                extra = labels.get("finished_reason")
                key = f"{engine}:{extra}" if extra else engine
                values[name][key] = value
    return values


def _counts_to_text(row: dict[str, int] | None) -> str:
    if row is None:
        return ""
    return f"[[{row['w0']},{row['r0']}],[{row['w1']},{row['r1']}]]"


def _find_first(
    rows: Iterable[dict[str, int]],
    predicate,
) -> dict[str, int] | None:
    for row in rows:
        if predicate(row):
            return row
    return None


def summarize_debug_run(
    counts: list[dict[str, int]],
    iterations: list[dict[str, int | float]],
    keywords: dict[str, int],
) -> dict[str, object]:
    max_waiting_per_engine = 0
    max_waiting_global = 0
    max_running_per_engine = 0
    max_running_global = 0
    max_waiting_skew = 0
    for row in counts:
        max_waiting_per_engine = max(max_waiting_per_engine, row["w0"], row["w1"])
        max_waiting_global = max(max_waiting_global, row["w0"] + row["w1"])
        max_running_per_engine = max(max_running_per_engine, row["r0"], row["r1"])
        max_running_global = max(max_running_global, row["r0"] + row["r1"])
        max_waiting_skew = max(max_waiting_skew, abs(row["w0"] - row["w1"]))

    first_nonzero = _find_first(counts, lambda row: row["w0"] + row["w1"] + row["r0"] + row["r1"] > 0)
    first_asym_burst = _find_first(
        counts,
        lambda row: (
            (row["w0"] >= 50 and row["w1"] <= 5)
            or (row["w1"] >= 50 and row["w0"] <= 5)
        ),
    )
    first_symmetric_burst = _find_first(counts, lambda row: row["w0"] >= 50 and row["w1"] >= 50)

    by_rank: dict[int, list[dict[str, int | float]]] = defaultdict(list)
    for row in iterations:
        by_rank[int(row["dp_rank"])].append(row)

    ctx_rows = [row for row in iterations if int(row["ctx_tokens"]) > 0]
    mixed_rows = [
        row for row in iterations
        if int(row["ctx_tokens"]) > 0 and int(row["generation_requests"]) > 0
    ]
    decode_rows = [
        row for row in iterations
        if int(row["ctx_tokens"]) == 0 and int(row["generation_requests"]) > 0
    ]
    mixed_generation_requests = [float(row["generation_requests"]) for row in mixed_rows]
    decode_generation_requests = [float(row["generation_requests"]) for row in decode_rows]

    visibility = (
        "engine_burst_confirmed"
        if max_waiting_per_engine >= 50 or max_waiting_global >= 100
        else "engine_burst_not_observed"
    )
    victim_observability = "available" if keywords.get("victim", 0) > 0 else "not_available"
    return {
        "counts_samples": len(counts),
        "iteration_rows": len(iterations),
        "iteration_rows_dp0": len(by_rank.get(0, [])),
        "iteration_rows_dp1": len(by_rank.get(1, [])),
        "max_waiting_per_engine": max_waiting_per_engine,
        "max_waiting_global": max_waiting_global,
        "max_running_per_engine": max_running_per_engine,
        "max_running_global": max_running_global,
        "max_waiting_skew": max_waiting_skew,
        "first_nonzero_counts": _counts_to_text(first_nonzero),
        "first_asymmetric_burst_counts": _counts_to_text(first_asym_burst),
        "first_symmetric_burst_counts": _counts_to_text(first_symmetric_burst),
        "visibility": visibility,
        "ctx_step_count": len(ctx_rows),
        "mixed_step_count": len(mixed_rows),
        "decode_step_count": len(decode_rows),
        "mixed_decode_batch_p10": _percentile(mixed_generation_requests, 10),
        "mixed_decode_batch_p50": _percentile(mixed_generation_requests, 50),
        "mixed_decode_batch_p90": _percentile(mixed_generation_requests, 90),
        "mixed_decode_batch_max": max(mixed_generation_requests) if mixed_generation_requests else math.nan,
        "decode_batch_p50": _percentile(decode_generation_requests, 50),
        "decode_batch_max": max(decode_generation_requests) if decode_generation_requests else math.nan,
        "added_line_count": keywords.get("added", 0),
        "preempt_line_count": keywords.get("preempt", 0),
        "victim_line_count": keywords.get("victim", 0),
        "victim_observability": victim_observability,
    }


def read_bench_result(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _empty_file(path: Path) -> bool:
    return path.exists() and path.stat().st_size == 0


def build_report_rows(
    *,
    serve_log: Path = DEFAULT_SERVE_LOG,
    metrics_path: Path = DEFAULT_METRICS,
    bench_result: Path = DEFAULT_BENCH_RESULT,
    run_dir: Path = RUN_DIR,
) -> list[dict[str, object]]:
    counts, iterations, keywords = parse_debug_log(serve_log)
    summary = summarize_debug_run(counts, iterations, keywords)
    bench = read_bench_result(bench_result)
    final_metrics = extract_final_metric_values(
        metrics_path,
        {
            "vllm:num_preemptions_total",
            "vllm:prompt_tokens_recomputed_total",
            "vllm:request_success_total",
        },
    )
    preemptions = final_metrics["vllm:num_preemptions_total"]
    recomputed = final_metrics["vllm:prompt_tokens_recomputed_total"]
    successes = final_metrics["vllm:request_success_total"]

    rows: list[dict[str, object]] = [
        _row("i3_gpu_run", "bench_ok_requests", bench.get("ok_requests", ""), target="128", status="pass" if bench.get("ok_requests") == 128 else "blocked"),
        _row("i3_gpu_run", "bench_failed_requests", bench.get("failed_requests", ""), target="0", status="pass" if bench.get("failed_requests") == 0 else "blocked"),
        _row("i3_gpu_run", "output_tok_s", bench.get("output_tok_s", "")),
        _row("i3_gpu_run", "gpu_residual_after_empty", _empty_file(run_dir / "gpu_compute_apps_after.txt"), status="pass" if _empty_file(run_dir / "gpu_compute_apps_after.txt") else "blocked"),
        _row("i3_gpu_run", "process_residual_after_empty", _empty_file(run_dir / "process_residual_after.txt"), status="pass" if _empty_file(run_dir / "process_residual_after.txt") else "blocked"),
        _row("i3_received_counts", "counts_samples", summary["counts_samples"]),
        _row("i3_received_counts", "max_waiting_per_engine", summary["max_waiting_per_engine"], target=">=50 confirms burst", status="pass" if summary["visibility"] == "engine_burst_confirmed" else "blocked"),
        _row("i3_received_counts", "max_waiting_global", summary["max_waiting_global"], target=">=100 confirms burst"),
        _row("i3_received_counts", "max_running_per_engine", summary["max_running_per_engine"]),
        _row("i3_received_counts", "max_running_global", summary["max_running_global"]),
        _row("i3_received_counts", "max_waiting_skew", summary["max_waiting_skew"], note="early [[0,1],[63,1]] API view before both engines sync"),
        _row("i3_received_counts", "first_nonzero_counts", summary["first_nonzero_counts"]),
        _row("i3_received_counts", "first_asymmetric_burst_counts", summary["first_asymmetric_burst_counts"]),
        _row("i3_received_counts", "first_symmetric_burst_counts", summary["first_symmetric_burst_counts"]),
        _row("i3_received_counts", "engine_visibility_debug", summary["visibility"], target="engine_burst_confirmed", status="pass" if summary["visibility"] == "engine_burst_confirmed" else "blocked"),
        _row("i3_iterations", "iteration_rows", summary["iteration_rows"]),
        _row("i3_iterations", "iteration_rows_dp0", summary["iteration_rows_dp0"]),
        _row("i3_iterations", "iteration_rows_dp1", summary["iteration_rows_dp1"]),
        _row("i3_iterations", "ctx_step_count", summary["ctx_step_count"]),
        _row("i3_iterations", "mixed_step_count", summary["mixed_step_count"]),
        _row("i3_iterations", "decode_step_count", summary["decode_step_count"]),
        _row("i3_iterations", "mixed_decode_batch_p10", summary["mixed_decode_batch_p10"]),
        _row("i3_iterations", "mixed_decode_batch_p50", summary["mixed_decode_batch_p50"]),
        _row("i3_iterations", "mixed_decode_batch_p90", summary["mixed_decode_batch_p90"]),
        _row("i3_iterations", "mixed_decode_batch_max", summary["mixed_decode_batch_max"]),
        _row("i3_iterations", "decode_batch_p50", summary["decode_batch_p50"]),
        _row("i3_iterations", "decode_batch_max", summary["decode_batch_max"]),
        _row("i3_observability", "added_line_count", summary["added_line_count"], target=">0 would expose request arrival", status="blocked" if summary["added_line_count"] == 0 else "pass"),
        _row("i3_observability", "preempt_line_count", summary["preempt_line_count"], target=">0 would expose preemption", status="blocked" if summary["preempt_line_count"] == 0 else "pass"),
        _row("i3_observability", "victim_line_count", summary["victim_line_count"], target=">0 would expose victim identity", status="blocked" if summary["victim_line_count"] == 0 else "pass"),
        _row("i3_observability", "victim_observability", summary["victim_observability"], target="available", status="blocked" if summary["victim_observability"] == "not_available" else "pass"),
        _row("i3_metrics", "num_preemptions_engine0", preemptions.get("0", ""), target="metric counter available"),
        _row("i3_metrics", "num_preemptions_engine1", preemptions.get("1", ""), target="metric counter available"),
        _row("i3_metrics", "prompt_tokens_recomputed_engine0", recomputed.get("0", ""), target="0 means no recompute tokens observed"),
        _row("i3_metrics", "prompt_tokens_recomputed_engine1", recomputed.get("1", ""), target="0 means no recompute tokens observed"),
        _row("i3_metrics", "request_success_length_engine0", successes.get("0:length", "")),
        _row("i3_metrics", "request_success_length_engine1", successes.get("1:length", "")),
        _row("i4_accept", "runtime_patch", "not_applied", status="blocked", note="DEBUG confirms burst but does not expose victim-level decision state"),
        _row("i4_accept", "default_aic", "No-Go", status="blocked"),
    ]
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def write_md(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_metric = {str(row["metric"]): row for row in rows}
    visibility = by_metric.get("engine_visibility_debug", {}).get("value", "unknown")
    victim = by_metric.get("victim_observability", {}).get("value", "unknown")
    lines = [
        "# Phase451-I DEBUG ramp",
        "",
        (
            f"结论: DEBUG 短跑确认 `{visibility}`;"
            f" victim 级观测 `{victim}`。"
            "因此本轮只收报告,不做调度语义修复。"
        ),
        "",
        "## Summary",
        "",
        "| section | metric | value | target | status | note |",
        "|---|---|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['section']} | {row['metric']} | {_fmt(row['value'])} | "
            f"{row.get('target', '')} | {row.get('status', '')} | {row.get('note', '')} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- `Received counts` proves EngineCore-visible arrival is burst-scale, not client-side drizzle hidden by API/tokenization.",
            "- DEBUG level exposes queue counts and per-step composition, but not victim identity or per-request preemption decision state in this run.",
            "- No runtime, PerfDB, scheduler, or validate gate change is justified by this evidence alone; Default AIC remains No-Go.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve-log", type=Path, default=DEFAULT_SERVE_LOG)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--bench-result", type=Path, default=DEFAULT_BENCH_RESULT)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows = build_report_rows(
        serve_log=args.serve_log,
        metrics_path=args.metrics,
        bench_result=args.bench_result,
        run_dir=args.run_dir,
    )
    write_csv(args.csv_out, rows)
    write_md(args.md_out, rows)
    print(f"wrote {args.csv_out}")
    print(f"wrote {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
