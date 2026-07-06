#!/usr/bin/env python3
"""Phase423: summarize the clean DP2 bt65536 recollect attempt."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase423_bt65536_recollect"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase423_bt65536_recollect.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase423_bt65536_recollect.md"

SOURCE = "phase423_bt65536_recollect"
SCENARIO = "K2.5-tp4ep8dp2-8k2k-bt65536"
DEFAULT_READINESS = "No-Go"

OLD_DIRTY_KV_CACHE_TOKENS = 25_744
OLD_DIRTY_NUM_GPU_BLOCKS = 1_609
EXPECTED_FROM_TP8_HALF_TOKENS = 171_776

KV_ERROR_RE = re.compile(
    r"max seq len \((?P<max_len>\d+)\), "
    r"\((?P<needed>[0-9.]+) GiB KV cache is needed, which is larger than the "
    r"available KV cache memory \((?P<available>[0-9.]+) GiB\).*?"
    r"estimated maximum model length is (?P<estimated>\d+)",
    re.DOTALL,
)

OUTPUT_FIELDS = [
    "source",
    "row_type",
    "scenario",
    "collection_status",
    "tp",
    "dp",
    "ep",
    "isl",
    "osl",
    "max_bt",
    "batch_size",
    "world_size",
    "max_model_len",
    "max_num_seqs",
    "gpu_memory_utilization",
    "prefix_caching",
    "prompt_variant_mode",
    "enable_logging_iteration_details",
    "cudagraph_metrics",
    "bench_result_present",
    "metrics_samples",
    "needed_kv_cache_gib",
    "available_kv_cache_gib",
    "estimated_max_model_len",
    "estimated_capacity_vs_old_dirty",
    "estimated_capacity_vs_expected",
    "old_dirty_kv_cache_tokens",
    "old_dirty_num_gpu_blocks",
    "expected_from_tp8_half_tokens",
    "gpu_after_empty",
    "process_after_empty",
    "serve_log",
    "serve_log_error_lines",
    "outer_log",
    "reference_action",
    "ab_action",
    "large_step_trace_action",
    "capacity_arbitration",
    "mechanism_verdict",
    "next_phase_target",
    "phase405_penalty_read",
    "gpu_allowed",
    "ssh_allowed",
    "runtime_modified",
    "perf_database",
    "valid_for_default",
    "diagnostic_only",
    "default_readiness",
]


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6f}"
    if isinstance(value, (list, tuple)):
        return ";".join(str(v) for v in value)
    return str(value)


def _guard_row(row: dict[str, object]) -> dict[str, str]:
    out = {field: _fmt(row.get(field, "")) for field in OUTPUT_FIELDS}
    out.update(
        {
            "source": SOURCE,
            "phase405_penalty_read": "false",
            "gpu_allowed": "true",
            "ssh_allowed": "true",
            "runtime_modified": "false",
            "perf_database": "false",
            "valid_for_default": "false",
            "diagnostic_only": "true",
            "default_readiness": DEFAULT_READINESS,
        }
    )
    return out


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _line_numbers(text: str, needle: str) -> list[int]:
    return [idx for idx, line in enumerate(text.splitlines(), start=1) if needle in line]


def _parse_kv_error(serve_log: str) -> dict[str, object]:
    match = KV_ERROR_RE.search(serve_log)
    if match is None:
        return {}
    needed = float(match.group("needed"))
    available = float(match.group("available"))
    estimated = int(match.group("estimated"))
    return {
        "max_model_len": int(match.group("max_len")),
        "needed_kv_cache_gib": needed,
        "available_kv_cache_gib": available,
        "estimated_max_model_len": estimated,
        "estimated_capacity_vs_old_dirty": estimated / OLD_DIRTY_KV_CACHE_TOKENS,
        "estimated_capacity_vs_expected": estimated / EXPECTED_FROM_TP8_HALF_TOKENS,
    }


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text:
        return 0
    return len(text.splitlines())


def _file_empty(path: Path) -> bool:
    return path.exists() and path.stat().st_size == 0


def _base_fields(meta: dict[str, object]) -> dict[str, object]:
    return {
        "scenario": meta.get("name", SCENARIO),
        "tp": meta.get("tp", 4),
        "dp": meta.get("dp", 2),
        "ep": meta.get("ep", 8),
        "isl": meta.get("isl", 8000),
        "osl": meta.get("osl", 2000),
        "max_bt": meta.get("max_num_batched_tokens", 65536),
        "batch_size": meta.get("batch_size", 128),
        "world_size": meta.get("world_size", 8),
        "max_model_len": meta.get("max_model_len", 262144),
        "max_num_seqs": meta.get("max_num_seqs", 256),
        "gpu_memory_utilization": meta.get("gpu_memory_utilization", 0.8),
        "prefix_caching": meta.get("prefix_caching", False),
        "prompt_variant_mode": meta.get("prompt_variant_mode", "rotating"),
        "enable_logging_iteration_details": meta.get("enable_logging_iteration_details", True),
        "cudagraph_metrics": meta.get("cudagraph_metrics", True),
    }


def build_phase423_rows(raw_root: Path = RAW_ROOT) -> list[dict[str, str]]:
    scenario_dir = raw_root / SCENARIO
    meta = _read_json(scenario_dir / "meta.json")
    serve_log = _read_text(scenario_dir / "serve.log")
    kv_error = _parse_kv_error(serve_log)
    base = _base_fields(meta)

    bench_present = (scenario_dir / "bench_result.json").exists()
    metrics_samples = _count_lines(scenario_dir / "metrics.jsonl")
    gpu_after_empty = _file_empty(raw_root / "gpu_compute_apps_after.txt")
    process_after_empty = _file_empty(raw_root / "process_residual_after.txt")
    error_lines = _line_numbers(serve_log, "ValueError: To serve at least one request")

    collection_status = "service_init_failed" if kv_error and not bench_present else "unexpected_collection_state"
    capacity_arbitration = (
        "clean_protocol_cannot_start_at_max_model_len_262144"
        if collection_status == "service_init_failed"
        else "inconclusive"
    )
    mechanism_verdict = (
        "clean_recollect_blocked_by_kv_init_capacity"
        if collection_status == "service_init_failed"
        else "phase423_inconclusive"
    )

    rows = [
        _guard_row(
            {
                **base,
                "row_type": "config",
                "collection_status": collection_status,
                "bench_result_present": bench_present,
                "metrics_samples": metrics_samples,
                "old_dirty_kv_cache_tokens": OLD_DIRTY_KV_CACHE_TOKENS,
                "old_dirty_num_gpu_blocks": OLD_DIRTY_NUM_GPU_BLOCKS,
                "expected_from_tp8_half_tokens": EXPECTED_FROM_TP8_HALF_TOKENS,
                "serve_log": scenario_dir / "serve.log",
                "outer_log": raw_root / "outer.log",
                "capacity_arbitration": capacity_arbitration,
                "mechanism_verdict": mechanism_verdict,
            }
        ),
        _guard_row(
            {
                **base,
                **kv_error,
                "row_type": "failure",
                "collection_status": collection_status,
                "bench_result_present": bench_present,
                "metrics_samples": metrics_samples,
                "old_dirty_kv_cache_tokens": OLD_DIRTY_KV_CACHE_TOKENS,
                "old_dirty_num_gpu_blocks": OLD_DIRTY_NUM_GPU_BLOCKS,
                "expected_from_tp8_half_tokens": EXPECTED_FROM_TP8_HALF_TOKENS,
                "serve_log": scenario_dir / "serve.log",
                "serve_log_error_lines": error_lines,
                "outer_log": raw_root / "outer.log",
                "reference_action": "no_reference_replacement",
                "ab_action": "not_run_no_bench_result",
                "large_step_trace_action": "not_run_service_not_ready",
                "capacity_arbitration": capacity_arbitration,
                "mechanism_verdict": mechanism_verdict,
                "next_phase_target": "phase424_reference_protocol_decision",
            }
        ),
        _guard_row(
            {
                **base,
                "row_type": "cleanup",
                "collection_status": collection_status,
                "gpu_after_empty": gpu_after_empty,
                "process_after_empty": process_after_empty,
                "capacity_arbitration": capacity_arbitration,
                "mechanism_verdict": mechanism_verdict,
            }
        ),
        _guard_row(
            {
                **base,
                **kv_error,
                "row_type": "summary",
                "collection_status": collection_status,
                "bench_result_present": bench_present,
                "metrics_samples": metrics_samples,
                "old_dirty_kv_cache_tokens": OLD_DIRTY_KV_CACHE_TOKENS,
                "old_dirty_num_gpu_blocks": OLD_DIRTY_NUM_GPU_BLOCKS,
                "expected_from_tp8_half_tokens": EXPECTED_FROM_TP8_HALF_TOKENS,
                "gpu_after_empty": gpu_after_empty,
                "process_after_empty": process_after_empty,
                "reference_action": "no_reference_replacement",
                "ab_action": "not_run_no_bench_result",
                "large_step_trace_action": "not_run_service_not_ready",
                "capacity_arbitration": capacity_arbitration,
                "mechanism_verdict": mechanism_verdict,
                "next_phase_target": "phase424_reference_protocol_decision",
            }
        ),
    ]
    return rows


def _validate_rows(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("no rows")
    for idx, row in enumerate(rows):
        for field in ("valid_for_default", "perf_database", "runtime_modified"):
            if row.get(field) != "false":
                raise ValueError(f"{field} must be false on row {idx}")
        if row.get("diagnostic_only") != "true":
            raise ValueError(f"diagnostic_only must be true on row {idx}")
        if row.get("default_readiness") != DEFAULT_READINESS:
            raise ValueError(f"default_readiness must be {DEFAULT_READINESS} on row {idx}")
    summary = next((row for row in rows if row["row_type"] == "summary"), None)
    if summary is None:
        raise ValueError("missing summary row")
    if summary["collection_status"] == "service_init_failed":
        if summary["reference_action"] != "no_reference_replacement":
            raise ValueError("service-init failure cannot replace reference")
        if summary["ab_action"] != "not_run_no_bench_result":
            raise ValueError("service-init failure cannot run A/B without bench_result")


def write_phase423_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_phase423_md(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    summary = next(row for row in rows if row["row_type"] == "summary")
    failure = next(row for row in rows if row["row_type"] == "failure")
    cleanup = next(row for row in rows if row["row_type"] == "cleanup")

    text = f"""# Phase423 DP2 bt65536 Clean Recollect

## Verdict

Phase423 没有产出可替换的 benchmark reference。清洁协议在 vLLM KV cache 初始化阶段失败，API server 没有进入 ready，benchmark 没有开始。

| Item | Value |
|---|---:|
| scenario | {summary['scenario']} |
| max_model_len | {summary['max_model_len']} |
| max_num_batched_tokens | {summary['max_bt']} |
| max_num_seqs | {summary['max_num_seqs']} |
| prefix_caching | {summary['prefix_caching']} |
| prompt_variant_mode | {summary['prompt_variant_mode']} |
| needed_kv_cache_gib | {failure['needed_kv_cache_gib']} |
| available_kv_cache_gib | {failure['available_kv_cache_gib']} |
| estimated_max_model_len | {failure['estimated_max_model_len']} |
| collection_status | {summary['collection_status']} |

## Decision

| Decision | Value |
|---|---|
| capacity_arbitration | {summary['capacity_arbitration']} |
| mechanism_verdict | {summary['mechanism_verdict']} |
| reference_action | {summary['reference_action']} |
| ab_action | {summary['ab_action']} |
| large_step_trace_action | {summary['large_step_trace_action']} |
| next_phase_target | {summary['next_phase_target']} |

这次结果不能把旧 `retry5e` reference 直接替换掉，也不能执行 A/B 全表。它只证明：在 `max_model_len=262144`、`max_num_batched_tokens=65536`、`gpu_memory_utilization=0.8`、prefix off 的清洁协议下，DP2 bt65536 自然 profile 出来的可用 KV 内存不足以启动服务。

## Cleanup

| Check | Value |
|---|---|
| gpu_after_empty | {cleanup['gpu_after_empty']} |
| process_after_empty | {cleanup['process_after_empty']} |

## Boundary

本阶段只记录 diagnostic evidence：不写 PerfDatabase，不改 runtime 计费，不替换 validate reference，不打开 Default AIC。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows = build_phase423_rows(args.raw_root)
    write_phase423_csv(args.csv, rows)
    write_phase423_md(args.md, rows)
    summary = next(row for row in rows if row["row_type"] == "summary")
    print(
        "phase423 "
        f"status={summary['collection_status']} "
        f"verdict={summary['mechanism_verdict']} "
        f"default={summary['default_readiness']}"
    )


if __name__ == "__main__":
    main()
