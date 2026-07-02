#!/usr/bin/env python3
"""Phase397x: offline attribution for 0.19 MULTI_CONFIG over-prediction."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aiconfigurator.sdk import common  # noqa: E402
from aiconfigurator.sdk.backends.vllm_backend import VLLMBackend  # noqa: E402
from aiconfigurator.sdk.config import RuntimeConfig  # noqa: E402
from scripts import validate_cb_simulator as validate_cb  # noqa: E402


SOURCE = "phase397x_multi_config_attribution"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase397x_multi_config_attribution.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase397x_multi_config_attribution.md"
REAL_BREAKDOWN_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase397l_decode_profile"
DEFAULT_READINESS = "No-Go"
TRUE = "true"
FALSE = "false"

TIER_A_REAL_BREAKDOWNS = {
    "K2.5-tp8ep8-8k2k": REAL_BREAKDOWN_ROOT / "K2.5-tp8ep8-8k2k/op_breakdown.csv",
    "K2.5-tp4ep8dp2-8k2k": REAL_BREAKDOWN_ROOT / "K2.5-tp4ep8dp2-8k2k/op_breakdown.csv",
}

CSV_FIELDS = [
    "source",
    "row_type",
    "scenario",
    "tier",
    "category",
    "tp",
    "dp",
    "ep",
    "isl",
    "osl",
    "batch_size",
    "max_num_batched_tokens",
    "real_output_tok_s_gpu",
    "sim_output_tok_s_gpu",
    "throughput_ratio_sim_over_real",
    "error_ratio",
    "direction",
    "throughput_source",
    "sim_modeled_ms_per_iter",
    "sim_generation_ms_per_iter",
    "sim_mix_modeled_ms_per_step",
    "sim_category_ms_per_iter",
    "sim_category_pct",
    "sim_generation_category_ms",
    "real_decode_category_ms",
    "latency_ratio_sim_over_real",
    "latency_direction",
    "context_attention_excluded_ms_per_mix_step",
    "context_attention_excluded_ms_per_iter",
    "num_mix_steps",
    "num_genonly_steps",
    "real_per_op_available",
    "verdict",
    "runtime_modified",
    "write_real_data_file",
    "gate_modified",
    "gpu_allowed",
    "ssh_allowed",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
    "default_readiness",
]


@dataclass(frozen=True)
class Schedule:
    num_mix_steps: float
    num_mix_ctx_tokens: int
    num_mix_gen_tokens: int
    num_genonly_steps: float
    num_genonly_tokens: int


def _fmt(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if math.isnan(value):
        return ""
    return f"{value:.6f}"


def _slash(values: Iterable[str]) -> str:
    return "/".join(values)


def _base_row(row_type: str) -> dict[str, str]:
    return {
        "source": SOURCE,
        "row_type": row_type,
        "scenario": "",
        "tier": "",
        "category": "",
        "tp": "",
        "dp": "",
        "ep": "",
        "isl": "",
        "osl": "",
        "batch_size": "",
        "max_num_batched_tokens": "",
        "real_output_tok_s_gpu": "",
        "sim_output_tok_s_gpu": "",
        "throughput_ratio_sim_over_real": "",
        "error_ratio": "",
        "direction": "",
        "throughput_source": "",
        "sim_modeled_ms_per_iter": "",
        "sim_generation_ms_per_iter": "",
        "sim_mix_modeled_ms_per_step": "",
        "sim_category_ms_per_iter": "",
        "sim_category_pct": "",
        "sim_generation_category_ms": "",
        "real_decode_category_ms": "",
        "latency_ratio_sim_over_real": "",
        "latency_direction": "",
        "context_attention_excluded_ms_per_mix_step": "",
        "context_attention_excluded_ms_per_iter": "",
        "num_mix_steps": "",
        "num_genonly_steps": "",
        "real_per_op_available": "",
        "verdict": "",
        "runtime_modified": FALSE,
        "write_real_data_file": FALSE,
        "gate_modified": FALSE,
        "gpu_allowed": FALSE,
        "ssh_allowed": FALSE,
        "diagnostic_only": TRUE,
        "valid_for_default": FALSE,
        "perf_database": FALSE,
        "default_readiness": DEFAULT_READINESS,
    }


def symmetric_error_ratio(sim_output: float, real_output: float) -> float:
    if sim_output <= 0.0 or real_output <= 0.0:
        return float("inf")
    ratio = sim_output / real_output
    return max(ratio, 1.0 / ratio)


def classify_direction(sim_output: float, real_output: float) -> str:
    if sim_output > real_output:
        return "sim_overpredicts_throughput"
    if sim_output < real_output:
        return "sim_underpredicts_throughput"
    return "matched"


def classify_latency_direction(sim_latency: float, real_latency: float) -> str:
    if sim_latency > real_latency:
        return "sim_overcharges_latency"
    if sim_latency < real_latency:
        return "sim_undercharges_latency"
    return "matched"


def categorize_latency_dict(latency: dict[str, float]) -> dict[str, float]:
    categories: dict[str, float] = defaultdict(float)
    for name, value in latency.items():
        lower = name.lower()
        if "attention" in lower:
            category = "attention"
        elif (
            "pre_dispatch" in lower
            or "post_dispatch" in lower
            or "_ar_" in lower
            or "allreduce" in lower
            or "allgather" in lower
            or "p2p" in lower
        ):
            category = "comm"
        elif "moe" in lower or "router" in lower:
            category = "moe"
        elif "gemm" in lower or "embedding" in lower:
            category = "gemm"
        elif "norm" in lower or "act" in lower:
            category = "other"
        else:
            category = "other"
        categories[category] += float(value)
    for category in ("attention", "moe", "gemm", "comm", "overlap", "overhead", "other"):
        categories.setdefault(category, 0.0)
    return dict(categories)


def _schedule(point) -> Schedule:
    steps_to_finish_ctx = float(np.ceil(point.isl * point.batch_size / point.max_num_batched_tokens))
    if point.batch_size > 1 and steps_to_finish_ctx >= point.osl:
        return Schedule(
            num_mix_steps=steps_to_finish_ctx,
            num_mix_ctx_tokens=point.max_num_batched_tokens,
            num_mix_gen_tokens=max(1, int(point.batch_size // (steps_to_finish_ctx / point.osl))),
            num_genonly_steps=0.0,
            num_genonly_tokens=0,
        )
    if point.batch_size > 1:
        mix_gen_tokens = int(point.batch_size - np.ceil(point.max_num_batched_tokens / point.isl))
        if mix_gen_tokens < 1:
            raise ValueError(f"invalid mixed generation tokens for {point.name}: {mix_gen_tokens}")
        return Schedule(
            num_mix_steps=steps_to_finish_ctx,
            num_mix_ctx_tokens=point.max_num_batched_tokens,
            num_mix_gen_tokens=mix_gen_tokens,
            num_genonly_steps=float(point.osl - steps_to_finish_ctx),
            num_genonly_tokens=point.batch_size,
        )
    return Schedule(
        num_mix_steps=1.0,
        num_mix_ctx_tokens=point.max_num_batched_tokens,
        num_mix_gen_tokens=0,
        num_genonly_steps=float(point.osl - 1),
        num_genonly_tokens=1,
    )


def _sim_step_breakdown(point) -> tuple[Schedule, dict[str, float], dict[str, float], float]:
    model, database, _ = validate_cb._load_model_and_db(
        tp=point.tp,
        dp=point.dp,
        moe_tp=point.moe_tp,
        moe_ep=point.moe_ep,
    )
    backend = VLLMBackend()
    schedule = _schedule(point)

    mix_summary = backend.run_static(
        model,
        database,
        RuntimeConfig(
            batch_size=1,
            beam_width=1,
            isl=schedule.num_mix_ctx_tokens + schedule.num_mix_gen_tokens,
            osl=1,
            prefix=0,
        ),
        mode="static_ctx",
    )
    raw_mix_latency = dict(mix_summary.get_context_latency_dict())
    excluded_context_attention = float(raw_mix_latency.pop("context_attention", 0.0))
    mix_categories = categorize_latency_dict(raw_mix_latency)

    if schedule.num_genonly_tokens > 0:
        gen_summary = backend.run_static(
            model,
            database,
            RuntimeConfig(
                batch_size=schedule.num_genonly_tokens,
                beam_width=1,
                isl=point.isl + point.osl // 2,
                osl=2,
            ),
            mode="static_gen",
        )
        gen_categories = categorize_latency_dict(dict(gen_summary.get_generation_latency_dict()))
    else:
        gen_categories = {category: 0.0 for category in ("attention", "moe", "gemm", "comm", "overlap", "overhead", "other")}

    return schedule, mix_categories, gen_categories, excluded_context_attention


def _load_real_decode_categories(path: Path) -> dict[str, float]:
    mapped: dict[str, float] = defaultdict(float)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            category = row["category"]
            if category in {"TOTAL_gpu_busy", "decode_wall_ms_per_iter", "comm_ms_per_iter", "n_profiled_steps", "decode_bs"}:
                continue
            value = float(row["ms_per_iter"])
            if category == "attention":
                mapped["attention"] += value
            elif category in {"moe_expert_gemm", "moe_aux"}:
                mapped["moe"] += value
            elif category in {"allreduce", "allgather"}:
                mapped["comm"] += value
            elif "gemm" in category:
                mapped["gemm"] += value
            else:
                mapped["other"] += value
    for category in ("attention", "moe", "gemm", "comm", "overlap", "overhead", "other"):
        mapped.setdefault(category, 0.0)
    return dict(mapped)


def _scenario_common(point, row_type: str) -> dict[str, str]:
    row = _base_row(row_type)
    row.update(
        {
            "scenario": point.name,
            "tier": "A" if point.name in TIER_A_REAL_BREAKDOWNS else "B",
            "tp": str(point.tp),
            "dp": str(point.dp),
            "ep": str(point.moe_ep),
            "isl": str(point.isl),
            "osl": str(point.osl),
            "batch_size": str(point.batch_size),
            "max_num_batched_tokens": str(point.max_num_batched_tokens),
            "real_per_op_available": str(point.name in TIER_A_REAL_BREAKDOWNS).lower(),
        }
    )
    return row


def analyze_phase397x_multi_config_attribution(repo_root: Path = REPO_ROOT) -> list[dict[str, str]]:
    raw_rows = validate_cb._run_multi_config_diagnostic_raw(
        overlap_factor=0.0,
        ep8_per_iteration_overhead_ms=validate_cb.DEFAULT_EP8_PER_ITERATION_OVERHEAD_MS,
    )
    real_decode = {
        scenario: _load_real_decode_categories(repo_root / path.relative_to(REPO_ROOT))
        for scenario, path in TIER_A_REAL_BREAKDOWNS.items()
    }

    rows: list[dict[str, str]] = []
    for point, real_output, sim_output, per_ops in raw_rows:
        schedule, mix_categories, gen_categories, excluded_context_attention = _sim_step_breakdown(point)
        total_steps = schedule.num_mix_steps + schedule.num_genonly_steps
        weighted_categories = {
            category: (
                mix_categories.get(category, 0.0) * schedule.num_mix_steps
                + gen_categories.get(category, 0.0) * schedule.num_genonly_steps
            )
            / total_steps
            for category in ("attention", "moe", "gemm", "comm", "overlap", "overhead", "other")
        }
        modeled_ms_per_iter = sum(weighted_categories.values())
        mix_ms = sum(mix_categories.values())
        gen_ms = sum(gen_categories.values())
        excluded_ms_per_iter = excluded_context_attention * schedule.num_mix_steps / total_steps
        throughput_ratio = sim_output / real_output
        error_ratio = symmetric_error_ratio(sim_output, real_output)
        boundary = per_ops.get("cb_sim_boundary", {})

        summary = _scenario_common(point, "scenario_summary")
        summary.update(
            {
                "real_output_tok_s_gpu": _fmt(real_output),
                "sim_output_tok_s_gpu": _fmt(sim_output),
                "throughput_ratio_sim_over_real": _fmt(throughput_ratio),
                "error_ratio": _fmt(error_ratio),
                "direction": classify_direction(sim_output, real_output),
                "throughput_source": str(boundary.get("throughput_source", "unknown")),
                "sim_modeled_ms_per_iter": _fmt(modeled_ms_per_iter),
                "sim_generation_ms_per_iter": _fmt(gen_ms),
                "sim_mix_modeled_ms_per_step": _fmt(mix_ms),
                "context_attention_excluded_ms_per_mix_step": _fmt(excluded_context_attention),
                "context_attention_excluded_ms_per_iter": _fmt(excluded_ms_per_iter),
                "num_mix_steps": _fmt(schedule.num_mix_steps),
                "num_genonly_steps": _fmt(schedule.num_genonly_steps),
                "verdict": (
                    "active_gate_failed"
                    if error_ratio > validate_cb.MULTI_CONFIG_MAX_ACCEPTANCE
                    else "within_active_gate"
                ),
            }
        )
        rows.append(summary)

        for category, value in sorted(weighted_categories.items()):
            category_row = _scenario_common(point, "sim_category")
            category_row.update(
                {
                    "category": category,
                    "sim_modeled_ms_per_iter": _fmt(modeled_ms_per_iter),
                    "sim_generation_ms_per_iter": _fmt(gen_ms),
                    "sim_mix_modeled_ms_per_step": _fmt(mix_ms),
                    "sim_category_ms_per_iter": _fmt(value),
                    "sim_category_pct": _fmt(value / modeled_ms_per_iter * 100.0 if modeled_ms_per_iter else 0.0),
                    "verdict": "sim_internal_cost_structure",
                }
            )
            rows.append(category_row)

        if point.name in real_decode:
            for category, real_ms in sorted(real_decode[point.name].items()):
                sim_gen_ms = gen_categories.get(category, 0.0)
                compare = _scenario_common(point, "tier_a_real_compare")
                compare.update(
                    {
                        "category": category,
                        "sim_generation_category_ms": _fmt(sim_gen_ms),
                        "real_decode_category_ms": _fmt(real_ms),
                        "latency_ratio_sim_over_real": _fmt(sim_gen_ms / real_ms if real_ms > 0.0 else float("nan")),
                        "latency_direction": classify_latency_direction(sim_gen_ms, real_ms) if real_ms > 0.0 else "missing_real",
                        "verdict": "decode_real_per_op_compare_only_not_full_request",
                    }
                )
                rows.append(compare)
        else:
            missing = _scenario_common(point, "tier_b_missing_real_per_op")
            missing.update(
                {
                    "verdict": "real_per_op_absent_aggregate_only",
                }
            )
            rows.append(missing)

    return rows


def _validate_rows(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("no Phase397x rows")
    for row in rows:
        for flag in ("runtime_modified", "write_real_data_file", "gate_modified", "gpu_allowed", "ssh_allowed", "valid_for_default", "perf_database"):
            if row.get(flag) != FALSE:
                raise ValueError(f"{flag} must be false")
        if row.get("diagnostic_only") != TRUE:
            raise ValueError("diagnostic_only must be true")
        if row.get("default_readiness") != DEFAULT_READINESS:
            raise ValueError("default_readiness must be No-Go")


def write_phase397x_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def render_phase397x_md(rows: list[dict[str, str]]) -> str:
    _validate_rows(rows)
    summaries = [row for row in rows if row["row_type"] == "scenario_summary"]
    worst = max(summaries, key=lambda row: float(row["error_ratio"]))
    mean_error = sum(float(row["error_ratio"]) for row in summaries) / len(summaries)
    all_over = all(row["direction"] == "sim_overpredicts_throughput" for row in summaries)

    tier_a_compare = [row for row in rows if row["row_type"] == "tier_a_real_compare"]
    undercharged = [
        row
        for row in tier_a_compare
        if row["latency_direction"] == "sim_undercharges_latency"
        and row["category"] in {"attention", "moe", "gemm", "comm"}
    ]
    strongest_undercharge = min(
        undercharged,
        key=lambda row: float(row["latency_ratio_sim_over_real"]),
        default=None,
    )

    lines = [
        "# Phase397x Multi-Config Attribution",
        "",
        "Phase397x is offline and report-only. It does not change runtime, DB tables, gates, or thresholds.",
        "",
        "## Summary",
        "",
        "| Item | Result |",
        "|---|---|",
        f"| Active gate | 0.19-real 8-card MULTI_CONFIG x6 |",
        f"| Direction | {'all six scenarios overpredict throughput' if all_over else 'mixed'} |",
        f"| Error | max={float(worst['error_ratio']):.2f}x mean={mean_error:.2f}x |",
        f"| Worst scenario | `{worst['scenario']}` |",
        "| Default AIC | Default AIC remains No-Go |",
        "",
        "## Scenario Results",
        "",
        "| Scenario | Tier | Real tok/s/GPU | Sim tok/s/GPU | Sim/real | Error | Direction |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(summaries, key=lambda item: float(item["error_ratio"]), reverse=True):
        lines.append(
            "| {scenario} | {tier} | {real} | {sim} | {ratio}x | {error}x | {direction} |".format(
                scenario=row["scenario"],
                tier=row["tier"],
                real=f"{float(row['real_output_tok_s_gpu']):.1f}",
                sim=f"{float(row['sim_output_tok_s_gpu']):.1f}",
                ratio=f"{float(row['throughput_ratio_sim_over_real']):.2f}",
                error=f"{float(row['error_ratio']):.2f}",
                direction=row["direction"],
            )
        )

    lines.extend(
        [
            "",
            "## Attribution",
            "",
            "Tier A has decode per-op profiles only. It can explain decode composition, not full prefill/mixed request cost.",
            "",
            "| Scenario | Category | Sim decode ms | Real decode ms | Sim/real latency | Direction |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in tier_a_compare:
        if row["category"] in {"overlap", "overhead"}:
            continue
        lines.append(
            "| {scenario} | {category} | {sim} | {real} | {ratio}x | {direction} |".format(
                scenario=row["scenario"],
                category=row["category"],
                sim=f"{float(row['sim_generation_category_ms']):.2f}",
                real=f"{float(row['real_decode_category_ms']):.2f}",
                ratio=f"{float(row['latency_ratio_sim_over_real']):.2f}",
                direction=row["latency_direction"],
            )
        )
    lines.append("")
    if strongest_undercharge is None:
        lines.append(
            "Tier A decode comparison does not show a dominant decode undercharge. The aggregate over-prediction points outside decode per-op anchors, toward mixed/prefill composition and scheduling cost."
        )
    else:
        lines.append(
            "Strongest actionable Tier A decode undercharge: `{scenario}` `{category}` sim/real latency = {ratio:.2f}x.".format(
                scenario=strongest_undercharge["scenario"],
                category=strongest_undercharge["category"],
                ratio=float(strongest_undercharge["latency_ratio_sim_over_real"]),
            )
        )
        lines.append(
            "Decode attention is not the aggregate undercharge: Tier A attention is overcharged in both available profiles."
        )

    excluded = [
        row
        for row in summaries
        if float(row["context_attention_excluded_ms_per_iter"]) > 0.0
    ]
    if excluded:
        top_excluded = max(excluded, key=lambda row: float(row["context_attention_excluded_ms_per_iter"]))
        lines.append(
            "The largest modeled mixed-step exclusion is context attention in `{scenario}`: {value:.2f} ms/iter equivalent.".format(
                scenario=top_excluded["scenario"],
                value=float(top_excluded["context_attention_excluded_ms_per_iter"]),
            )
        )

    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- Tier B rows have no real per-op profile and stay aggregate-only.",
            "- No GPU or SSH was used.",
            "- No DB row, runtime path, threshold, or Default AIC gate was changed.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_phase397x_md(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_phase397x_md(rows), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args(argv)

    rows = analyze_phase397x_multi_config_attribution(args.repo_root)
    write_phase397x_csv(args.out_csv, rows)
    write_phase397x_md(args.out_md, rows)


if __name__ == "__main__":
    main()
