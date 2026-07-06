#!/usr/bin/env python3
"""Phase424: package the max_model_len=131072 DP2 bt65536 recollect."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase424_bt65536_recollect"
DEFAULT_PHASE422_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase422_dp2_capacity_fix.csv"
DEFAULT_AFTER_CSV = Path("/private/tmp/phase424_validation_after.csv")
DEFAULT_TRACE_CSV = Path("/private/tmp/phase424_bt65536_trace_after.csv")
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase424_bt65536_recollect.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase424_bt65536_recollect.md"

SOURCE = "phase424_bt65536_recollect"
SCENARIO = "K2.5-tp4ep8dp2-8k2k-bt65536"
DEFAULT_READINESS = "No-Go"

OLD_DIRTY_KV_CACHE_TOKENS = 25_744
OLD_DIRTY_NUM_GPU_BLOCKS = 1_609
PHASE423_ESTIMATED_MAX_MODEL_LEN = 131_648
KV_BLOCK_SIZE = 16

# Phase418 corrected lower bounds for the 32k mixed-prefill step.
PHASE418_32K_MOE_BF16_BOUND_MS = 170.822563
PHASE418_32K_EP_BOUND_MS = 489.335467

KV_CACHE_RE = re.compile(r"GPU KV cache size: (?P<tokens>[0-9,]+) tokens")
ITER_RE = re.compile(
    r"EngineCore_DP(?P<engine>\d+).*?"
    r"Iteration\((?P<iteration>\d+)\): "
    r"(?P<context_requests>\d+) context requests, "
    r"(?P<context_tokens>\d+) context tokens, "
    r"(?P<generation_requests>\d+) generation requests, "
    r"(?P<generation_tokens>\d+) generation tokens, "
    r"iteration elapsed time: (?P<elapsed_ms>[0-9.]+) ms"
)

OUTPUT_FIELDS = [
    "source",
    "row_type",
    "scenario",
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
    "bench_result_present",
    "bench_ok_requests",
    "bench_failed_requests",
    "bench_output_tok_s",
    "bench_total_tok_s",
    "real_output_tok_s_gpu",
    "real_total_tok_s_gpu",
    "phase422_sim_tok_s_gpu",
    "phase424_sim_tok_s_gpu",
    "phase422_error_ratio",
    "phase424_error_ratio",
    "error_ratio_delta",
    "classification",
    "kv_cache_tokens",
    "num_gpu_blocks",
    "old_dirty_kv_cache_tokens",
    "old_dirty_num_gpu_blocks",
    "phase423_estimated_max_model_len",
    "capacity_vs_old_dirty",
    "capacity_vs_phase423_estimate",
    "metrics_samples",
    "gpu_after_empty",
    "process_after_empty",
    "max_context_tokens",
    "max_context_requests",
    "max_context_generation_requests",
    "max_context_elapsed_ms",
    "trace_initial_prefill_tokens",
    "trace_initial_prefill_reqs",
    "trace_representative_step_ms",
    "trace_context_non_attention_ms",
    "moe_query_path",
    "ep_query_path",
    "combined_64k_lower_bound_ms",
    "large_step_status",
    "large_step_charge_status",
    "reference_action",
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
    if isinstance(value, (tuple, list)):
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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _count_lines(path: Path) -> int:
    text = _read_text(path)
    return len(text.splitlines()) if text else 0


def _file_empty(path: Path) -> bool:
    return path.exists() and path.stat().st_size == 0


def _config_base(meta: dict[str, object]) -> dict[str, object]:
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
        "max_model_len": meta.get("max_model_len", 131072),
        "max_num_seqs": meta.get("max_num_seqs", 256),
        "gpu_memory_utilization": meta.get("gpu_memory_utilization", 0.8),
        "prefix_caching": meta.get("prefix_caching", False),
        "prompt_variant_mode": meta.get("prompt_variant_mode", "rotating"),
    }


def _parse_kv_tokens(serve_log: str) -> list[int]:
    return [int(match.group("tokens").replace(",", "")) for match in KV_CACHE_RE.finditer(serve_log)]


def _parse_max_context_step(serve_log: str) -> dict[str, object]:
    best: dict[str, object] = {}
    best_tokens = -1
    for match in ITER_RE.finditer(serve_log):
        tokens = int(match.group("context_tokens"))
        if tokens <= best_tokens:
            continue
        best_tokens = tokens
        best = {
            "max_context_tokens": tokens,
            "max_context_requests": int(match.group("context_requests")),
            "max_context_generation_requests": int(match.group("generation_requests")),
            "max_context_elapsed_ms": float(match.group("elapsed_ms")),
        }
    return best


def _phase422_config_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["scenario"]: row for row in rows if row.get("row_type") == "config"}


def _after_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["scenario"]: row for row in rows}


def _trace_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["scenario"]: row for row in rows if row.get("row_type") == "trace"}


def _classify(before_error: float, after_error: float, scenario: str) -> str:
    if scenario == SCENARIO:
        return "reference_replaced_improved" if after_error < before_error else "reference_replaced_not_improved"
    if abs(after_error - before_error) < 1e-9:
        return "unchanged"
    return "unexpected_changed"


def _combined_64k_lower_bound_ms() -> float:
    return (PHASE418_32K_MOE_BF16_BOUND_MS + PHASE418_32K_EP_BOUND_MS) * 2.0


def build_phase424_rows(
    *,
    raw_root: Path = RAW_ROOT,
    phase422_csv: Path = DEFAULT_PHASE422_CSV,
    after_csv: Path = DEFAULT_AFTER_CSV,
    trace_csv: Path = DEFAULT_TRACE_CSV,
) -> list[dict[str, str]]:
    scenario_dir = raw_root / SCENARIO
    meta = _read_json(scenario_dir / "meta.json")
    bench = _read_json(scenario_dir / "bench_result.json")
    serve_log = _read_text(scenario_dir / "serve.log")
    phase422 = _phase422_config_rows(_read_csv(phase422_csv))
    after = _after_rows(_read_csv(after_csv))
    traces = _trace_rows(_read_csv(trace_csv))
    if set(phase422) != set(after):
        raise ValueError("Phase422 and Phase424 validation scenarios do not match")

    base = _config_base(meta)
    kv_tokens = _parse_kv_tokens(serve_log)
    if not kv_tokens:
        raise ValueError("missing GPU KV cache size in Phase424 serve.log")
    if len(set(kv_tokens)) != 1:
        raise ValueError(f"inconsistent Phase424 KV cache tokens: {kv_tokens}")
    kv_cache_tokens = kv_tokens[0]
    num_gpu_blocks = kv_cache_tokens // KV_BLOCK_SIZE
    if kv_cache_tokens != num_gpu_blocks * KV_BLOCK_SIZE:
        raise ValueError("KV cache tokens must divide by block size")

    rows: list[dict[str, str]] = []
    for scenario in sorted(after):
        before_row = phase422[scenario]
        after_row = after[scenario]
        before_error = float(before_row["phase422_error_ratio"])
        after_error = float(after_row["error_ratio"])
        rows.append(
            _guard_row(
                {
                    "row_type": "config",
                    "scenario": scenario,
                    "tp": after_row["tp"],
                    "dp": after_row["dp"],
                    "ep": after_row["ep"],
                    "max_bt": after_row["max_bt"],
                    "real_output_tok_s_gpu": after_row["real_output_tok_s_gpu"],
                    "phase422_sim_tok_s_gpu": before_row["phase422_sim_tok_s_gpu"],
                    "phase424_sim_tok_s_gpu": after_row["sim_output_tok_s_gpu"],
                    "phase422_error_ratio": before_row["phase422_error_ratio"],
                    "phase424_error_ratio": after_row["error_ratio"],
                    "error_ratio_delta": after_error - before_error,
                    "classification": _classify(before_error, after_error, scenario),
                }
            )
        )

    bench_output_gpu = float(bench.get("output_tok_s", 0.0)) / float(base["world_size"])
    bench_total_gpu = float(bench.get("total_tok_s", 0.0)) / float(base["world_size"])
    capacity_fields = {
        **base,
        "bench_result_present": bool(bench),
        "bench_ok_requests": bench.get("ok_requests", ""),
        "bench_failed_requests": bench.get("failed_requests", ""),
        "bench_output_tok_s": bench.get("output_tok_s", ""),
        "bench_total_tok_s": bench.get("total_tok_s", ""),
        "real_output_tok_s_gpu": bench_output_gpu,
        "real_total_tok_s_gpu": bench_total_gpu,
        "kv_cache_tokens": kv_cache_tokens,
        "num_gpu_blocks": num_gpu_blocks,
        "old_dirty_kv_cache_tokens": OLD_DIRTY_KV_CACHE_TOKENS,
        "old_dirty_num_gpu_blocks": OLD_DIRTY_NUM_GPU_BLOCKS,
        "phase423_estimated_max_model_len": PHASE423_ESTIMATED_MAX_MODEL_LEN,
        "capacity_vs_old_dirty": kv_cache_tokens / OLD_DIRTY_KV_CACHE_TOKENS,
        "capacity_vs_phase423_estimate": kv_cache_tokens / PHASE423_ESTIMATED_MAX_MODEL_LEN,
        "metrics_samples": _count_lines(scenario_dir / "metrics.jsonl"),
        "gpu_after_empty": _file_empty(raw_root / "gpu_compute_apps_after.txt"),
        "process_after_empty": _file_empty(raw_root / "process_residual_after.txt"),
        "reference_action": "replace_dirty_dp2_bt65536_reference_with_phase424_clean_run",
        "capacity_arbitration": "phase424_capacity_healthy_at_max_model_len_131072",
    }
    rows.append(_guard_row({"row_type": "capacity", **capacity_fields}))

    max_step = _parse_max_context_step(serve_log)
    trace = traces.get(SCENARIO, {})
    combined_bound = _combined_64k_lower_bound_ms()
    trace_non_attn = float(trace.get("context_non_attention_ms", 0.0) or 0.0)
    large_step_tokens = int(trace.get("initial_prefill_tokens", 0) or 0)
    large_step_status = "true_64k_step_observed" if int(max_step.get("max_context_tokens", 0)) >= 64_000 else "missing_64k_step"
    charge_status = (
        "combined_non_attention_above_64k_lower_bound"
        if trace_non_attn >= combined_bound and large_step_tokens >= 65_536
        else "large_step_charge_below_combined_lower_bound"
    )
    rows.append(
        _guard_row(
            {
                "row_type": "trace",
                **base,
                **max_step,
                "kv_cache_tokens": kv_cache_tokens,
                "num_gpu_blocks": num_gpu_blocks,
                "trace_initial_prefill_tokens": trace.get("initial_prefill_tokens", ""),
                "trace_initial_prefill_reqs": trace.get("initial_prefill_reqs", ""),
                "trace_representative_step_ms": trace.get("representative_step_ms", ""),
                "trace_context_non_attention_ms": trace.get("context_non_attention_ms", ""),
                "moe_query_path": trace.get("moe_query_path", ""),
                "ep_query_path": trace.get("ep_query_path", ""),
                "combined_64k_lower_bound_ms": combined_bound,
                "large_step_status": large_step_status,
                "large_step_charge_status": charge_status,
            }
        )
    )

    after_errors = [float(row["error_ratio"]) for row in after.values()]
    dp2_after = after[SCENARIO]
    dp2_before = phase422[SCENARIO]
    verdict = (
        "dp2_bt65536_reference_replaced_converged"
        if float(dp2_after["error_ratio"]) <= 1.5 and charge_status == "combined_non_attention_above_64k_lower_bound"
        else "phase424_recollect_inconclusive"
    )
    rows.append(
        _guard_row(
            {
                "row_type": "summary",
                "scenario": "all_multi_config",
                "phase422_error_ratio": dp2_before["phase422_error_ratio"],
                "phase424_error_ratio": max(after_errors),
                "classification": "max_returns_to_dp2_8k2k_residual"
                if max(after_errors) > float(dp2_after["error_ratio"])
                else "dp2_bt65536_still_max",
                "kv_cache_tokens": kv_cache_tokens,
                "num_gpu_blocks": num_gpu_blocks,
                "bench_ok_requests": bench.get("ok_requests", ""),
                "bench_failed_requests": bench.get("failed_requests", ""),
                "reference_action": "replace_dirty_dp2_bt65536_reference_with_phase424_clean_run",
                "capacity_arbitration": "old_25744_token_reference_was_dirty",
                "large_step_status": large_step_status,
                "large_step_charge_status": charge_status,
                "mechanism_verdict": verdict,
                "next_phase_target": "phase425_return_to_main_dp2_residual"
                if verdict == "dp2_bt65536_reference_replaced_converged"
                else "phase425_recheck_phase424_large_step_charge",
            }
        )
    )
    return rows


def write_phase424_csv(path: Path, rows: list[dict[str, str]]) -> None:
    for row in rows:
        if row.get("valid_for_default") != "false":
            raise ValueError(f"valid_for_default must stay false for {row.get('row_type')}")
        if row.get("default_readiness") != DEFAULT_READINESS:
            raise ValueError(f"default_readiness must stay {DEFAULT_READINESS}")
        if row.get("perf_database") != "false":
            raise ValueError(f"perf_database must stay false for {row.get('row_type')}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_phase424_md(path: Path, rows: list[dict[str, str]]) -> None:
    summary = next(row for row in rows if row["row_type"] == "summary")
    capacity = next(row for row in rows if row["row_type"] == "capacity")
    trace = next(row for row in rows if row["row_type"] == "trace")
    configs = [row for row in rows if row["row_type"] == "config"]
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase424 bt65536 recollect",
        "",
        f"- verdict: `{summary['mechanism_verdict']}`",
        f"- default_readiness: `{summary['default_readiness']}`",
        "- boundary: measurement/reference replacement only; no PerfDB or runtime charge change.",
        "",
        "## Capacity",
        "",
        "| field | value |",
        "|---|---:|",
        f"| max_model_len | {capacity['max_model_len']} |",
        f"| kv_cache_tokens | {capacity['kv_cache_tokens']} |",
        f"| num_gpu_blocks | {capacity['num_gpu_blocks']} |",
        f"| old_dirty_kv_cache_tokens | {capacity['old_dirty_kv_cache_tokens']} |",
        f"| capacity_vs_old_dirty | {capacity['capacity_vs_old_dirty']} |",
        f"| output_tok_s_gpu | {capacity['real_output_tok_s_gpu']} |",
        "",
        "## Validation Delta",
        "",
        "| scenario | Phase422 error | Phase424 error | class |",
        "|---|---:|---:|---|",
    ]
    for row in configs:
        lines.append(
            f"| {row['scenario']} | {row['phase422_error_ratio']} | "
            f"{row['phase424_error_ratio']} | {row['classification']} |"
        )
    lines.extend(
        [
            "",
            "## Large Step",
            "",
            "| field | value |",
            "|---|---:|",
            f"| max_context_tokens_real_trace | {trace['max_context_tokens']} |",
            f"| trace_initial_prefill_tokens_sim | {trace['trace_initial_prefill_tokens']} |",
            f"| trace_context_non_attention_ms | {trace['trace_context_non_attention_ms']} |",
            f"| combined_64k_lower_bound_ms | {trace['combined_64k_lower_bound_ms']} |",
            f"| charge_status | {trace['large_step_charge_status']} |",
            "",
            "## Next",
            "",
            f"`{summary['next_phase_target']}`. Default AIC stays No-Go.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--phase422-csv", type=Path, default=DEFAULT_PHASE422_CSV)
    parser.add_argument("--after-csv", type=Path, default=DEFAULT_AFTER_CSV)
    parser.add_argument("--trace-csv", type=Path, default=DEFAULT_TRACE_CSV)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows = build_phase424_rows(
        raw_root=args.raw_root,
        phase422_csv=args.phase422_csv,
        after_csv=args.after_csv,
        trace_csv=args.trace_csv,
    )
    write_phase424_csv(args.csv, rows)
    write_phase424_md(args.md, rows)
    print(f"wrote {args.csv}")
    print(f"wrote {args.md}")


if __name__ == "__main__":
    main()
