#!/usr/bin/env python3
"""Phase397z: attribute MULTI_CONFIG over-prediction to effective decode batch."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import validate_cb_simulator as validate_cb  # noqa: E402


SOURCE = "phase397z_effective_batch_attribution"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase397z_effective_batch_attribution.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase397z_effective_batch_attribution.md"
PHASE397X_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase397x_multi_config_attribution.csv"
DEFAULT_READINESS = "No-Go"

CSV_FIELDS = [
    "source",
    "row_type",
    "scenario",
    "tier",
    "tp",
    "dp",
    "ep",
    "max_num_batched_tokens",
    "real_output_tok_s_gpu",
    "sim_output_tok_s_gpu",
    "throughput_ratio_sim_over_real",
    "sim_effective_decode_batch_per_dp",
    "real_effective_decode_batch_per_dp",
    "effective_batch_ratio",
    "sim_decode_occupancy",
    "real_decode_occupancy",
    "sim_to_real_iteration_count_factor",
    "sim_to_real_wall_factor",
    "sim_iter_ms",
    "real_decode_iter_ms",
    "iter_latency_factor",
    "reconstructed_throughput_ratio",
    "avg_prefill_reqs_per_iter",
    "avg_decode_reqs_per_iter",
    "avg_tokens_per_iter",
    "peak_decode_reqs_per_iter",
    "steady_state_iterations",
    "steady_state_time_ms",
    "anchor_source_scenario",
    "anchor_provenance",
    "dominant_driver",
    "scheduler_localization",
    "tier_boundary",
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
class RealDecodeAnchor:
    source_scenario: str
    tier: str
    real_decode_iter_ms: float
    provenance: str


def _fmt(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isinf(value) or math.isnan(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def _base_row(row_type: str) -> dict[str, object]:
    return {
        "source": SOURCE,
        "row_type": row_type,
        "scenario": "",
        "tier": "",
        "tp": "",
        "dp": "",
        "ep": "",
        "max_num_batched_tokens": "",
        "real_output_tok_s_gpu": "",
        "sim_output_tok_s_gpu": "",
        "throughput_ratio_sim_over_real": "",
        "sim_effective_decode_batch_per_dp": "",
        "real_effective_decode_batch_per_dp": "",
        "effective_batch_ratio": "",
        "sim_decode_occupancy": "",
        "real_decode_occupancy": "",
        "sim_to_real_iteration_count_factor": "",
        "sim_to_real_wall_factor": "",
        "sim_iter_ms": "",
        "real_decode_iter_ms": "",
        "iter_latency_factor": "",
        "reconstructed_throughput_ratio": "",
        "avg_prefill_reqs_per_iter": "",
        "avg_decode_reqs_per_iter": "",
        "avg_tokens_per_iter": "",
        "peak_decode_reqs_per_iter": "",
        "steady_state_iterations": "",
        "steady_state_time_ms": "",
        "anchor_source_scenario": "",
        "anchor_provenance": "",
        "dominant_driver": "",
        "scheduler_localization": "",
        "tier_boundary": "",
        "runtime_modified": False,
        "write_real_data_file": False,
        "gate_modified": False,
        "gpu_allowed": False,
        "ssh_allowed": False,
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
        "default_readiness": DEFAULT_READINESS,
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("inf")
    return numerator / denominator


def _topology_key(tp: int) -> str:
    return f"tp{tp}"


def _point_by_name() -> dict[str, object]:
    return {point.name: point for point in validate_cb.MULTI_CONFIG_DATA}


def load_real_decode_anchors(path: Path = PHASE397X_CSV) -> dict[str, RealDecodeAnchor]:
    totals: dict[str, float] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["row_type"] != "tier_a_real_compare":
                continue
            if row["category"] in {"overhead", "overlap"}:
                continue
            scenario = row["scenario"]
            totals[scenario] = totals.get(scenario, 0.0) + float(row["real_decode_category_ms"])

    required = {
        "tp8": "K2.5-tp8ep8-8k2k",
        "tp4": "K2.5-tp4ep8dp2-8k2k",
    }
    anchors: dict[str, RealDecodeAnchor] = {}
    for key, scenario in required.items():
        if scenario not in totals:
            raise ValueError(f"missing Phase397x real decode anchor for {scenario}")
        anchors[key] = RealDecodeAnchor(
            source_scenario=scenario,
            tier="A",
            real_decode_iter_ms=totals[scenario],
            provenance="phase397l_op_breakdown_exact",
        )
    return anchors


def decompose_throughput_ratio(
    *,
    sim_effective_decode_batch: float,
    real_effective_decode_batch: float,
    sim_iter_ms: float,
    real_iter_ms: float,
) -> dict[str, float]:
    effective_batch_ratio = _safe_ratio(
        sim_effective_decode_batch,
        real_effective_decode_batch,
    )
    iter_latency_factor = _safe_ratio(real_iter_ms, sim_iter_ms)
    return {
        "effective_batch_ratio": effective_batch_ratio,
        "iter_latency_factor": iter_latency_factor,
        "reconstructed_throughput_ratio": effective_batch_ratio * iter_latency_factor,
    }


def _dominant_driver(effective_batch_ratio: float, iter_latency_factor: float) -> str:
    if effective_batch_ratio >= 1.5 and effective_batch_ratio > iter_latency_factor:
        return "effective_batch_overestimate"
    if iter_latency_factor >= 1.5:
        return "iteration_latency_underestimate"
    return "mixed_or_perop_secondary"


def build_phase397z_rows(
    *,
    budget_rows: Iterable[object] | None = None,
    real_decode_anchors: dict[str, RealDecodeAnchor] | None = None,
) -> list[dict[str, object]]:
    if budget_rows is None:
        budget_rows = validate_cb.run_diagnostic_multi_config_budget_breakdown(
            overlap_factor=0.0,
            ep8_per_iteration_overhead_ms=0.0,
            verbose=False,
        )
    if real_decode_anchors is None:
        real_decode_anchors = load_real_decode_anchors()

    points = _point_by_name()
    rows: list[dict[str, object]] = []
    for budget in budget_rows:
        point = points.get(budget.name)
        if point is None:
            raise ValueError(f"missing MULTI_CONFIG point for {budget.name}")
        anchor = real_decode_anchors[_topology_key(int(budget.tp))]
        batch_per_dp = point.batch_size / max(int(budget.dp), 1)
        sim_effective = float(budget.avg_decode_reqs_per_iter)
        real_iter_ms = anchor.real_decode_iter_ms
        real_effective = float(budget.real_output_tok_s_gpu) * int(budget.tp) * real_iter_ms / 1000.0
        sim_iter_ms = _safe_ratio(
            sim_effective * 1000.0,
            float(budget.sim_output_tok_s_gpu) * int(budget.tp),
        )
        factors = decompose_throughput_ratio(
            sim_effective_decode_batch=sim_effective,
            real_effective_decode_batch=real_effective,
            sim_iter_ms=sim_iter_ms,
            real_iter_ms=real_iter_ms,
        )
        throughput_ratio = _safe_ratio(
            float(budget.sim_output_tok_s_gpu),
            float(budget.real_output_tok_s_gpu),
        )
        tier = "A" if budget.name == anchor.source_scenario else "B"
        tier_boundary = (
            "exact_phase397l_decode_anchor"
            if tier == "A"
            else f"estimated_from_{anchor.source_scenario}_decode_anchor"
        )

        row = _base_row("scenario")
        row.update(
            {
                "scenario": budget.name,
                "tier": tier,
                "tp": int(budget.tp),
                "dp": int(budget.dp),
                "ep": int(budget.ep),
                "max_num_batched_tokens": int(budget.max_bt),
                "real_output_tok_s_gpu": float(budget.real_output_tok_s_gpu),
                "sim_output_tok_s_gpu": float(budget.sim_output_tok_s_gpu),
                "throughput_ratio_sim_over_real": throughput_ratio,
                "sim_effective_decode_batch_per_dp": sim_effective,
                "real_effective_decode_batch_per_dp": real_effective,
                "effective_batch_ratio": factors["effective_batch_ratio"],
                "sim_decode_occupancy": _safe_ratio(sim_effective, batch_per_dp),
                "real_decode_occupancy": _safe_ratio(real_effective, batch_per_dp),
                "sim_to_real_iteration_count_factor": _safe_ratio(real_effective, sim_effective),
                "sim_to_real_wall_factor": _safe_ratio(1.0, throughput_ratio),
                "sim_iter_ms": sim_iter_ms,
                "real_decode_iter_ms": real_iter_ms,
                "iter_latency_factor": factors["iter_latency_factor"],
                "reconstructed_throughput_ratio": factors["reconstructed_throughput_ratio"],
                "avg_prefill_reqs_per_iter": float(budget.avg_prefill_reqs_per_iter),
                "avg_decode_reqs_per_iter": float(budget.avg_decode_reqs_per_iter),
                "avg_tokens_per_iter": float(budget.avg_tokens_per_iter),
                "peak_decode_reqs_per_iter": float(budget.peak_decode_reqs_per_iter),
                "steady_state_iterations": int(budget.steady_state_iterations),
                "steady_state_time_ms": float(budget.steady_state_time_ms),
                "anchor_source_scenario": anchor.source_scenario,
                "anchor_provenance": anchor.provenance,
                "dominant_driver": _dominant_driver(
                    factors["effective_batch_ratio"],
                    factors["iter_latency_factor"],
                ),
                "scheduler_localization": "decode_reserved_before_prefill_and_closed_loop_replacement",
                "tier_boundary": tier_boundary,
            }
        )
        rows.append(row)

    rows.sort(key=lambda row: float(row["throughput_ratio_sim_over_real"]), reverse=True)
    return rows


def _validate_rows(rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("no Phase397z rows")
    for row in rows:
        if row.get("valid_for_default") is True or row.get("valid_for_default") == "true":
            raise ValueError("valid_for_default must stay false")
        if row.get("perf_database") is True or row.get("perf_database") == "true":
            raise ValueError("perf_database must stay false")
        if row.get("gate_modified") is True or row.get("gate_modified") == "true":
            raise ValueError("gate_modified must stay false")
        if row.get("runtime_modified") is True or row.get("runtime_modified") == "true":
            raise ValueError("runtime_modified must stay false")
        if row.get("default_readiness") != DEFAULT_READINESS:
            raise ValueError("default_readiness must stay No-Go")


def write_phase397z_csv(path: Path, rows: list[dict[str, object]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field, "")) for field in CSV_FIELDS})


def render_phase397z_md(rows: list[dict[str, object]]) -> str:
    _validate_rows(rows)
    worst = max(rows, key=lambda row: float(row["throughput_ratio_sim_over_real"]))
    mean_error = sum(float(row["throughput_ratio_sim_over_real"]) for row in rows) / len(rows)
    mean_batch_ratio = sum(float(row["effective_batch_ratio"]) for row in rows) / len(rows)
    driver_count = sum(1 for row in rows if row["dominant_driver"] == "effective_batch_overestimate")

    lines = [
        "# Phase397z Effective Batch Attribution",
        "",
        "Phase397z is offline and report-only. It does not change scheduler code, runtime, DB tables, thresholds, or Default AIC gates.",
        "",
        "## Summary",
        "",
        "| Item | Result |",
        "|---|---|",
        "| Active surface | 0.19-real 8-card MULTI_CONFIG x6 |",
        f"| Worst error | {worst['scenario']} = {float(worst['throughput_ratio_sim_over_real']):.2f}x |",
        f"| Mean error | {mean_error:.2f}x |",
        f"| Mean effective batch ratio | {mean_batch_ratio:.2f}x |",
        f"| Dominant driver rows | {driver_count}/{len(rows)} effective batch overestimate |",
        "| Scheduler localization | decode reserved before prefill + closed-loop replacement keeps sim decode occupancy high |",
        "| Default AIC | Default AIC remains No-Go |",
        "",
        "## Scenario Decomposition",
        "",
        "| Scenario | Tier | Sim/real throughput | Effective batch ratio | Iter latency factor | Sim eff batch | Real eff batch | Driver |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {scenario} | {tier} | {thr:.2f}x | {batch:.2f}x | {lat:.2f}x | {sim:.1f} | {real:.1f} | {driver} |".format(
                scenario=row["scenario"],
                tier=row["tier"],
                thr=float(row["throughput_ratio_sim_over_real"]),
                batch=float(row["effective_batch_ratio"]),
                lat=float(row["iter_latency_factor"]),
                sim=float(row["sim_effective_decode_batch_per_dp"]),
                real=float(row["real_effective_decode_batch_per_dp"]),
                driver=row["dominant_driver"],
            )
        )
    lines.extend(
        [
            "",
            "## Iteration Structure",
            "",
            "| Scenario | Sim iter count / real | Sim wall / real | Sim occupancy | Real occupancy | Steady sim iters | Steady sim ms |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {scenario} | {iter_factor:.2f}x | {wall_factor:.2f}x | {sim_occ:.2f} | {real_occ:.2f} | {iters} | {ms:.1f} |".format(
                scenario=row["scenario"],
                iter_factor=float(row["sim_to_real_iteration_count_factor"]),
                wall_factor=float(row["sim_to_real_wall_factor"]),
                sim_occ=float(row["sim_decode_occupancy"]),
                real_occ=float(row["real_decode_occupancy"]),
                iters=int(row["steady_state_iterations"]),
                ms=float(row["steady_state_time_ms"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The throughput gap is dominated by effective batch / decode occupancy, not by mixed-step latency. In the Tier A `tp8ep8-8k2k` row, sim holds about 125 decode requests per iteration while the real aggregate throughput implies about 42 effective decode requests. The sim decode iteration is slower than the real pure-decode anchor, so latency is not the source of the over-predicted throughput.",
            "",
            "Tier A rows use the Phase397l real decode op-sum directly. Tier B rows reuse the matching topology anchor and are aggregate estimates only.",
            "",
            "## Scheduler Localization",
            "",
            "- `CBScheduler.schedule()` reserves one token for every decoding request before admitting prefill.",
            "- `CBSimulator.run()` replaces completed requests immediately in a closed loop.",
            "- The simulator therefore keeps decode occupancy near the configured per-replica batch, while the real aggregate surface includes prefill blocking, KV/prefix capacity effects, and lower sustained decode occupancy.",
            "",
            "## Boundaries",
            "",
            "- No GPU or SSH was used.",
            "- No runtime, scheduler, DB row, threshold, or gate was changed.",
            "- Default AIC remains No-Go.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_phase397z_md(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_phase397z_md(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows = build_phase397z_rows()
    write_phase397z_csv(args.csv, rows)
    write_phase397z_md(args.md, rows)


if __name__ == "__main__":
    main()
