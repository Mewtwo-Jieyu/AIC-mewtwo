#!/usr/bin/env python3
"""Phase430 bubble/MLA fix audit.

This script records the runtime fix applied in Phase430 and audits the decode
MLA/MoE slope evidence from Phase429. It intentionally does not mutate PerfDB.
"""
from __future__ import annotations

import argparse
import csv
import statistics as stats
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aiconfigurator.sdk.config import RuntimeConfig
from scripts import validate_cb_simulator as validate_cb

SOURCE = "phase430_bubble_mla_fix"
SCENARIO = "K2.5-tp4ep8dp2-8k2k"
PHASE429_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase429_reprofile.csv"
MOE_PERF = REPO_ROOT / "src/aiconfigurator/systems/data/h200_sxm/vllm/0.19.0/moe_perf.txt"
MLA_PERF = REPO_ROOT / "src/aiconfigurator/systems/data/h200_sxm/vllm/0.19.0/generation_mla_perf.txt"
OUT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase430_bubble_mla_fix.csv"
OUT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase430_bubble_mla_fix.md"

CSV_FIELDS = [
    "source",
    "row_type",
    "scenario",
    "category",
    "metric",
    "batch_low",
    "batch_high",
    "real_value",
    "sim_value",
    "counterfactual_value",
    "gap_value",
    "verdict",
    "source_file",
    "source_lines",
    "phase430_runtime_target",
    "phase430_perfdb_target",
    "runtime_modified",
    "perf_database",
    "valid_for_default",
    "diagnostic_only",
    "default_readiness",
]

VALIDATE_AFTER_ROWS = [
    ("K2.5-tp8ep8-8k2k", 133.5, 167.0, 1.25),
    ("K2.5-tp8ep8-32k3k", 52.5, 56.1, 1.07),
    ("K2.5-tp4ep8dp2-8k2k", 137.7, 351.6, 2.55),
    ("K2.5-tp4ep8dp2-32k3k", 53.3, 92.4, 1.73),
    ("K2.5-tp8ep8-8k2k-bt65536", 138.5, 155.1, 1.12),
    ("K2.5-tp4ep8dp2-8k2k-bt65536", 113.9, 101.1, 0.89),
]
VALIDATE_AFTER_MAX = 2.55
VALIDATE_AFTER_MEAN = 1.48

SOURCE_BASIS = [
    (
        "vllm/model_executor/layers/fused_moe/modular_kernel.py",
        "1072-1158,1160-1221,1223-1303,1356-1381",
        "prepare_then_fused_experts_then_finalize_call_order",
    ),
    (
        "vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ll.py",
        "363-389,391-465",
        "deepep_prepare_waits_for_hook_receiver_and_finalize_combines",
    ),
    (
        "vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py",
        "103-141,143-165",
        "naive_dp_ep_dispatch_then_weight_reduce_then_combine",
    ),
]

CATEGORY_MAP = {
    "generation_attention": "mla_attention",
    "generation_moe": "moe_gemm_or_aux",
    "generation_moe_pre_dispatch": "ep_a2a",
    "generation_moe_post_dispatch": "ep_a2a",
    "generation_ar_1": "tp_or_dp_allreduce",
    "generation_ar_2": "tp_or_dp_allreduce",
    "generation_downscale_gemm": "dense_gemm",
    "generation_proj_gemm": "dense_gemm",
    "generation_q_b_proj_gemm": "dense_gemm",
    "generation_shared_ffn1_gemm": "dense_gemm",
    "generation_shared_ffn2_gemm": "dense_gemm",
    "generation_shared_gate_gemm": "dense_gemm",
    "generation_logits_gemm": "dense_gemm",
    "generation_router_gemm": "dense_gemm",
}


def _base_row(row_type: str, category: str = "", metric: str = "") -> dict[str, str]:
    row = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "source": SOURCE,
            "row_type": row_type,
            "scenario": SCENARIO,
            "category": category,
            "metric": metric,
            "phase430_runtime_target": "prefill_serial_sum_only",
            "phase430_perfdb_target": "mla_moe_decode_recollect_required",
            "runtime_modified": "true",
            "perf_database": "false",
            "valid_for_default": "false",
            "diagnostic_only": "false",
            "default_readiness": "No-Go",
        }
    )
    return row


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    return float(value) if value else 0.0


def phase429_summary(phase429_csv: Path = PHASE429_CSV) -> dict[str, float | str]:
    rows = _read_csv(phase429_csv)
    summary = next(row for row in rows if row["row_type"] == "summary")
    prefill_steps = [row for row in rows if row["row_type"] == "step" and row["step_type"] == "prefill_or_mixed"]
    decode_slopes = {row["category"]: _float(row, "real_slope_ms_per_request") for row in rows if row["row_type"] == "decode_slope"}
    return {
        "steady_penalty": float(summary["phase426_steady_metric_penalty"]),
        "prefill_share": float(summary["phase426_prefill_share"]),
        "decode_share": float(summary["phase426_decode_gap_share"]),
        "prefill_step_count": int(summary["prefill_step_count"]),
        "decode_step_count": int(summary["decode_step_count"]),
        "prefill_wall_mean": stats.mean(_float(row, "real_wall_ms_per_rank") for row in prefill_steps),
        "prefill_bubble_mean": stats.mean(_float(row, "bubble_ms_per_rank") for row in prefill_steps),
        "prefill_decode_batch_mean": stats.mean(_float(row, "decode_batch") for row in prefill_steps),
        "decode_slopes": decode_slopes,
    }


def _aggregate_generation_categories(latencies: dict[str, float]) -> dict[str, float]:
    categories: dict[str, float] = {}
    for op_name, latency in latencies.items():
        category = CATEGORY_MAP.get(op_name, "other_cuda")
        categories[category] = categories.get(category, 0.0) + float(latency)
    return categories


def compute_sim_decode_slopes(batches: tuple[int, int] = (8, 52), kv_len: int = 8000) -> dict[str, float]:
    model, database, backend = validate_cb._load_model_and_db(tp=4, dp=2, moe_tp=1, moe_ep=8)
    points: list[tuple[int, dict[str, float]]] = []
    for batch in batches:
        runtime_config = RuntimeConfig(batch_size=batch, beam_width=1, isl=kv_len, osl=2, prefix=0)
        summary = backend.run_static(model, database, runtime_config, mode="static_gen")
        points.append((batch, _aggregate_generation_categories(summary.get_generation_latency_dict())))
    low_batch, low_values = points[0]
    high_batch, high_values = points[-1]
    denom = float(high_batch - low_batch)
    categories = set(low_values) | set(high_values)
    return {category: (high_values.get(category, 0.0) - low_values.get(category, 0.0)) / denom for category in categories}


def _interp(points: list[tuple[int, float]], x: int) -> float:
    points = sorted(points)
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    for (left, left_value), (right, right_value) in zip(points, points[1:]):
        if left <= x <= right:
            if left == right:
                return left_value
            return left_value + (right_value - left_value) * (x - left) / (right - left)
    raise ValueError(f"no interpolation bracket for {x}")


def moe_table_counterfactual_slope(moe_perf: Path = MOE_PERF, batches: tuple[int, int] = (8, 52)) -> float:
    rows = _read_csv(moe_perf)
    points = [
        (int(row["num_tokens"]), float(row["latency"]))
        for row in rows
        if row.get("moe_dtype") == "int4_wo"
        and row.get("moe_tp_size") == "1"
        and row.get("moe_ep_size") == "8"
        and row.get("distribution") == "power_law_1.01"
    ]
    low, high = batches
    return (_interp(points, high) - _interp(points, low)) * 60.0 / float(high - low)


def perfdb_coverage_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    mla_rows = [
        row for row in _read_csv(MLA_PERF)
        if row.get("num_heads") == "16" and row.get("kv_cache_dtype") == "float16"
    ]
    mla_batches = sorted({int(row["batch_size"]) for row in mla_rows})
    mla_steps = sorted({int(row["step"]) for row in mla_rows})
    row = _base_row("coverage", "mla_attention", "generation_mla_perf_grid")
    row.update(
        {
            "batch_low": str(min(mla_batches)),
            "batch_high": str(max(mla_batches)),
            "real_value": f"step_min={min(mla_steps)}",
            "sim_value": f"step_max={max(mla_steps)}",
            "verdict": "mla_decode_grid_covers_high_batch_but_slope_is_slightly_low",
        }
    )
    rows.append(row)

    moe_rows = [
        row for row in _read_csv(MOE_PERF)
        if row.get("moe_dtype") == "int4_wo" and row.get("moe_tp_size") == "1" and row.get("moe_ep_size") == "8"
    ]
    moe_tokens = sorted({int(row["num_tokens"]) for row in moe_rows})
    row = _base_row("coverage", "moe_gemm_or_aux", "moe_perf_int4_wo_grid")
    row.update(
        {
            "batch_low": str(min(moe_tokens)),
            "batch_high": str(max(moe_tokens)),
            "verdict": "moe_decode_table_exists_but_current_query_routes_to_phase397v_sol",
        }
    )
    rows.append(row)
    return rows


def build_phase430_rows(phase429_csv: Path = PHASE429_CSV) -> list[dict[str, str]]:
    evidence = phase429_summary(phase429_csv)
    sim_slopes = compute_sim_decode_slopes()
    real_slopes = evidence["decode_slopes"]  # type: ignore[assignment]
    assert isinstance(real_slopes, dict)
    moe_counterfactual = moe_table_counterfactual_slope()

    rows: list[dict[str, str]] = []
    summary = _base_row("summary", metric="phase430_verdict")
    summary.update(
        {
            "real_value": f"steady_penalty={evidence['steady_penalty']:.6f}",
            "sim_value": "pure_prefill_overlap_removed",
            "counterfactual_value": "decode_mla_moe_recollect_required",
            "verdict": "phase430_fix_a_applied_fix_b_requires_gpu_recollect",
        }
    )
    rows.append(summary)

    for source_file, source_lines, verdict in SOURCE_BASIS:
        row = _base_row("source_basis", "prefill_serialization", verdict)
        row.update({"source_file": source_file, "source_lines": source_lines, "verdict": verdict})
        rows.append(row)

    row = _base_row("fix_a", "pure_prefill", "serial_sum_green")
    row.update(
        {
            "sim_value": "23.000000",
            "counterfactual_value": "15.000000_before_overlap0",
            "gap_value": "8.000000",
            "verdict": "fixed_pure_prefill_no_overlap_lane_drop",
        }
    )
    rows.append(row)

    row = _base_row("fix_a", "mixed_prefill", "phase429_prefill_bubble")
    row.update(
        {
            "real_value": f"wall_mean_ms={evidence['prefill_wall_mean']:.6f}",
            "sim_value": f"bubble_mean_ms={evidence['prefill_bubble_mean']:.6f}",
            "counterfactual_value": f"decode_batch_mean={evidence['prefill_decode_batch_mean']:.6f}",
            "verdict": "mixed_branch_already_serial_bubble_not_a_simple_overlap_switch",
        }
    )
    rows.append(row)

    for category, real_slope in sorted(real_slopes.items()):
        sim_slope = sim_slopes.get(category, 0.0)
        row = _base_row("decode_slope", category, "ms_per_request")
        verdict = "decode_slope_ok"
        if category == "mla_attention":
            verdict = "mla_slope_near_real_but_joint_recollect_recommended"
        if category == "moe_gemm_or_aux":
            verdict = "moe_decode_sol_slope_wrong_sign_recollect_required"
        if category == "ep_a2a":
            verdict = "ep_a2a_decode_slope_undercharged_secondary"
        row.update(
            {
                "batch_low": "8",
                "batch_high": "52",
                "real_value": f"{real_slope:.6f}",
                "sim_value": f"{sim_slope:.6f}",
                "gap_value": f"{real_slope - sim_slope:.6f}",
                "verdict": verdict,
            }
        )
        if category == "moe_gemm_or_aux":
            row["counterfactual_value"] = f"moe_perf_table_slope={moe_counterfactual:.6f}"
        rows.append(row)

    rows.extend(perfdb_coverage_rows())

    validate_summary = _base_row("validate_after", "multi_config", "validate_cb_simulator_after")
    validate_summary.update(
        {
            "real_value": f"max={VALIDATE_AFTER_MAX:.2f}",
            "sim_value": f"mean={VALIDATE_AFTER_MEAN:.2f}",
            "verdict": "multi_config_still_fails_dp2_8k2k",
        }
    )
    rows.append(validate_summary)
    for scenario, real_output, sim_output, ratio in VALIDATE_AFTER_ROWS:
        row = _base_row("validate_after", "multi_config", "scenario_ratio")
        row.update(
            {
                "scenario": scenario,
                "real_value": f"{real_output:.1f}",
                "sim_value": f"{sim_output:.1f}",
                "gap_value": f"{ratio:.2f}",
                "verdict": "pass" if ratio <= 1.50 else "fail",
            }
        )
        rows.append(row)

    for category, metric in [
        ("mla_attention", "gpu_recollect_generation_mla_heads16_batch8_16_32_64_128_kv8192_float16"),
        ("moe_gemm_or_aux", "gpu_recollect_int4_wo_marlin_decode_batch8_16_32_64_128_tp4dp2ep8"),
        ("ep_a2a", "gpu_profile_decode_ep_a2a_latency_floor_batch_sweep"),
    ]:
        row = _base_row("phase431_recollect", category, metric)
        row.update({"verdict": "required_before_perfdb_or_default_change"})
        rows.append(row)

    return rows


def _validate_rows(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("no rows")
    for index, row in enumerate(rows):
        missing = set(CSV_FIELDS) - set(row)
        if missing:
            raise ValueError(f"row {index} missing fields: {sorted(missing)}")
        if row["valid_for_default"] != "false":
            raise ValueError("valid_for_default must stay false")
        if row["perf_database"] != "false":
            raise ValueError("perf_database must stay false")
        if row["default_readiness"] != "No-Go":
            raise ValueError("default_readiness must stay No-Go")


def write_phase430_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_phase430_md(path: Path, rows: list[dict[str, str]]) -> None:
    summary = next(row for row in rows if row["row_type"] == "summary")
    slopes = [row for row in rows if row["row_type"] == "decode_slope"]
    source_rows = [row for row in rows if row["row_type"] == "source_basis"]
    recollect = [row for row in rows if row["row_type"] == "phase431_recollect"]
    validate_rows = [row for row in rows if row["row_type"] == "validate_after" and row["metric"] == "scenario_ratio"]
    validate_summary = next(row for row in rows if row["row_type"] == "validate_after" and row["metric"] == "validate_cb_simulator_after")
    lines = [
        "# Phase430 bubble / MLA fix",
        "",
        "## Verdict",
        "",
        f"- {summary['verdict']}.",
        "- Fix A landed only the safe runtime correction: pure prefill now charges context non-attention plus context attention independent of overlap_factor.",
        "- Mixed prefill was already a serial sum in cb_sim; Phase429's bubble is an execution-state gap, not a remaining max-overlap branch.",
        "- Fix B is not safe to patch offline: MLA slope is close, while MoE decode SOL has the wrong slope sign and existing small-token table is not a valid replacement.",
        "- Default AIC remains No-Go.",
        "",
        "## vLLM Source Basis",
        "",
        "| file | lines | conclusion |",
        "|---|---:|---|",
    ]
    for row in source_rows:
        lines.append(f"| `{row['source_file']}` | {row['source_lines']} | {row['verdict']} |")
    lines.extend(["", "## Decode Slope Audit", "", "| category | real ms/request | sim ms/request | gap | verdict |", "|---|---:|---:|---:|---|"])
    for row in slopes:
        lines.append(
            f"| {row['category']} | {row['real_value']} | {row['sim_value']} | {row['gap_value']} | {row['verdict']} |"
        )
    lines.extend([
        "",
        "## Validate After",
        "",
        f"- MULTI_CONFIG after Phase430: {validate_summary['real_value']}, {validate_summary['sim_value']}; verdict={validate_summary['verdict']}.",
        "",
        "| scenario | real out tok/s/GPU | sim out tok/s/GPU | ratio | verdict |",
        "|---|---:|---:|---:|---|",
    ])
    for row in validate_rows:
        lines.append(f"| {row['scenario']} | {row['real_value']} | {row['sim_value']} | {row['gap_value']}x | {row['verdict']} |")

    lines.extend(["", "## Phase431 Recollect List", "", "| category | required point |", "|---|---|"])
    for row in recollect:
        lines.append(f"| {row['category']} | `{row['metric']}` |")
    lines.extend([
        "",
        "## Boundaries",
        "",
        "- Runtime changed: true, limited to pure prefill serial accounting in `iteration_latency.py`.",
        "- PerfDatabase changed: false.",
        "- Gate/default changed: false; `valid_for_default=false`, `default_readiness=No-Go`.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=OUT_CSV)
    parser.add_argument("--md", type=Path, default=OUT_MD)
    args = parser.parse_args()
    rows = build_phase430_rows()
    write_phase430_csv(args.csv, rows)
    write_phase430_md(args.md, rows)
    print(f"wrote {args.csv}")
    print(f"wrote {args.md}")


if __name__ == "__main__":
    main()
