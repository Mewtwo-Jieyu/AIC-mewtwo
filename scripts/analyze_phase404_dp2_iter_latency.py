#!/usr/bin/env python3
"""Phase404: offline DP2 decode-iteration latency attribution.

This phase is report-only. It does not change scheduler, runtime, PerfDatabase,
or any gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aiconfigurator.sdk import common  # noqa: E402
from aiconfigurator.sdk.config import RuntimeConfig  # noqa: E402
from aiconfigurator.sdk.backends.cb_simulator.iteration_latency import (  # noqa: E402
    IterationLatencyCalculator,
    _bucket,
)
from scripts import validate_cb_simulator as validate_cb  # noqa: E402


SOURCE = "phase404_dp2_iter_latency"
DEFAULT_RAW_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase403_dp2_stats"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase404_dp2_iter_latency.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase404_dp2_iter_latency.md"
PHASE403_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase403_dp2_running_batch.csv"
PHASE400_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase400_clean_recollect.csv"
DEFAULT_READINESS = "No-Go"

DP2_SCENARIOS = (
    "K2.5-tp4ep8dp2-8k2k",
    "K2.5-tp4ep8dp2-32k3k",
)
TP8_CROSS_SCENARIO = "K2.5-tp8ep8-32k3k"
ALL_SCENARIOS = DP2_SCENARIOS + (TP8_CROSS_SCENARIO,)

CSV_FIELDS = [
    "source",
    "row_type",
    "scenario",
    "tp",
    "dp",
    "ep",
    "isl",
    "osl",
    "max_num_batched_tokens",
    "per_replica_batch",
    "sim_running_global",
    "real_running_global_max",
    "real_running_global_mean",
    "real_running_global_p50",
    "mid_kv_len",
    "bucketed_kv_len",
    "real_active_decode_ms_per_iter",
    "real_active_decode_ms_p10",
    "real_active_decode_ms_p90",
    "active_decode_intervals",
    "real_aggregate_implied_ms_per_iter",
    "sim_decode_ms_per_iter",
    "gen_attention_ms",
    "gen_moe_compute_ms",
    "gen_dispatch_ms",
    "gen_gemm_ms",
    "gen_other_non_attention_ms",
    "overhead_ms",
    "active_decode_residual_ms",
    "aggregate_implied_residual_ms",
    "dispatch_ms_vs_active_residual_ratio",
    "attention_ms_vs_active_residual_ratio",
    "aggregation_gap_minus_active_gap_ms",
    "phase403_or_phase400_output_tok_s_global",
    "phase403_or_phase400_output_tok_s_gpu",
    "attribution_verdict",
    "primary_driver",
    "phase405_target",
    "gpu_allowed",
    "ssh_allowed",
    "runtime_modified",
    "perf_database",
    "valid_for_default",
    "diagnostic_only",
    "default_readiness",
]


@dataclass(frozen=True)
class ActiveDecodeStats:
    intervals: int
    ms_per_iter_median: float
    ms_per_iter_p10: float
    ms_per_iter_p90: float


@dataclass(frozen=True)
class SimBreakdown:
    scenario: str
    sim_running_global: int
    per_replica_batch: int
    mid_kv_len: int
    bucketed_kv_len: int
    sim_decode_ms_per_iter: float
    gen_attention_ms: float
    gen_moe_compute_ms: float
    gen_dispatch_ms: float
    gen_gemm_ms: float
    gen_other_non_attention_ms: float
    overhead_ms: float


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
    return str(value)


def _read_csv_by_scenario(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as f:
        return {row["scenario"]: row for row in csv.DictReader(f)}


def _point_by_name() -> dict[str, validate_cb.MultiConfigPoint]:
    return {point.name: point for point in validate_cb.MULTI_CONFIG_DATA}


def _median(values: list[float]) -> float:
    if not values:
        return math.nan
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * pct
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return sorted_values[int(pos)]
    weight = pos - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


_PROM_LINE = re.compile(
    r"^(?P<metric>[A-Za-z_:][A-Za-z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)


def _parse_counter_body(body: str) -> tuple[float, float]:
    running_total = 0.0
    generation_total = 0.0
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PROM_LINE.match(line)
        if not match:
            continue
        metric = match.group("metric")
        value = float(match.group("value"))
        if metric == "vllm:num_requests_running":
            running_total += value
        elif metric == "vllm:generation_tokens_total":
            generation_total += value
    return running_total, generation_total


def _parse_ts(text: str) -> float:
    return datetime.fromisoformat(text).timestamp()


def parse_active_decode_stats(path: Path) -> ActiveDecodeStats:
    samples: list[tuple[float, float, float]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") != 200:
                continue
            running, generation_total = _parse_counter_body(record.get("body", ""))
            samples.append((_parse_ts(record["ts"]), running, generation_total))

    intervals: list[float] = []
    for prev, cur in zip(samples, samples[1:]):
        prev_ts, prev_running, prev_generation = prev
        cur_ts, cur_running, cur_generation = cur
        dt_s = cur_ts - prev_ts
        generation_delta = cur_generation - prev_generation
        avg_running = (prev_running + cur_running) / 2.0
        if dt_s <= 0 or generation_delta <= 0 or avg_running <= 0:
            continue
        intervals.append(dt_s * 1000.0 * avg_running / generation_delta)

    if not intervals:
        raise ValueError(f"missing active decode intervals in {path}")
    return ActiveDecodeStats(
        intervals=len(intervals),
        ms_per_iter_median=_median(intervals),
        ms_per_iter_p10=_percentile(intervals, 0.10),
        ms_per_iter_p90=_percentile(intervals, 0.90),
    )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _mid_decode_kv_len(isl: int, osl: int) -> int:
    return isl + max(osl // 2, 1)


def _categorize_generation_latency(gen_dict: dict[str, float]) -> tuple[float, float, float, float, float]:
    attention = 0.0
    moe = 0.0
    dispatch = 0.0
    gemm = 0.0
    other = 0.0
    for name, value in gen_dict.items():
        latency = float(value)
        if name == "generation_attention":
            attention += latency
        elif "dispatch" in name:
            dispatch += latency
        elif "moe" in name:
            moe += latency
        elif "gemm" in name:
            gemm += latency
        else:
            other += latency
    return attention, moe, dispatch, gemm, other


def collect_sim_breakdowns(
    scenarios: Iterable[str] = ALL_SCENARIOS,
) -> list[SimBreakdown]:
    points = _point_by_name()
    backend = validate_cb.VLLMBackend()
    loaded: dict[tuple[int, int, int, int], tuple] = {}
    rows: list[SimBreakdown] = []

    for scenario in scenarios:
        point = points[scenario]
        key = (point.tp, point.dp, point.moe_tp, point.moe_ep)
        if key not in loaded:
            loaded[key] = validate_cb._load_model_and_db(
                tp=point.tp,
                dp=point.dp,
                moe_tp=point.moe_tp,
                moe_ep=point.moe_ep,
            )[:2]
        model, database = loaded[key]
        if scenario == TP8_CROSS_SCENARIO:
            phase400 = _read_csv_by_scenario(PHASE400_CSV)
            real_running = float(phase400[scenario]["serve_running_reqs_max"])
        else:
            phase403 = _read_csv_by_scenario(PHASE403_CSV)
            real_running = float(phase403[scenario]["real_running_global_max"])
        per_replica_batch = max(1, int(round(real_running / point.dp)))
        sim_running_global = per_replica_batch * point.dp
        mid_kv_len = _mid_decode_kv_len(point.isl, point.osl)
        bucketed_kv_len = _bucket(mid_kv_len, 512)
        calculator = IterationLatencyCalculator(
            backend,
            model,
            database,
            overlap_factor=0.0,
            per_iteration_overhead_ms=0.0,
        )
        sim_total = calculator.compute(
            prefill_tokens=0,
            prefill_batch_size=0,
            prefill_seq_len=point.isl,
            decode_batch_size=per_replica_batch,
            decode_avg_kv_len=mid_kv_len,
        )
        breakdown = calculator.get_last_breakdown()
        summary = backend.run_static(
            model,
            database,
            RuntimeConfig(
                batch_size=per_replica_batch,
                beam_width=1,
                isl=bucketed_kv_len,
                osl=2,
            ),
            mode="static_gen",
        )
        gen_dict = summary.get_generation_latency_dict()
        attention, moe, dispatch, gemm, other = _categorize_generation_latency(gen_dict)
        rows.append(
            SimBreakdown(
                scenario=scenario,
                sim_running_global=sim_running_global,
                per_replica_batch=per_replica_batch,
                mid_kv_len=mid_kv_len,
                bucketed_kv_len=bucketed_kv_len,
                sim_decode_ms_per_iter=float(sim_total),
                gen_attention_ms=attention,
                gen_moe_compute_ms=moe,
                gen_dispatch_ms=dispatch,
                gen_gemm_ms=gemm,
                gen_other_non_attention_ms=other,
                overhead_ms=breakdown.iteration_overhead_ms if breakdown else 0.0,
            )
        )
    return rows


def _ratio_to_residual(value: float, residual: float) -> float | None:
    if residual <= 0:
        return None
    return value / residual


def _verdict(
    *,
    row_type: str,
    active_residual: float | None,
    aggregate_residual: float,
    sim_decode_ms: float,
) -> tuple[str, str, str]:
    tolerance = max(5.0, 0.20 * sim_decode_ms)
    if row_type == "tp8_crosscheck":
        return (
            "tp8_control_no_dp_only_iter_gap",
            "control",
            "phase405_dp_specific_only_if_dp2_active_gap_exists",
        )
    if (
        active_residual is not None
        and active_residual > 0.0
        and aggregate_residual > 0.0
        and active_residual / aggregate_residual <= 0.35
    ):
        return (
            "aggregate_gap_dominates_minor_active_decode_gap",
            "decode_duty_or_queueing_with_minor_active_dp_gap",
            "dp_decode_duty_cycle_or_queueing_first",
        )
    if active_residual is not None and abs(active_residual) <= tolerance:
        if aggregate_residual > tolerance:
            return (
                "reject_gen_op_latency_gap",
                "aggregate_decode_duty_or_queueing_not_active_gen_op",
                "dp_decode_duty_cycle_or_queueing",
            )
        return (
            "decode_iter_matched",
            "no_active_decode_iter_gap",
            "phase405_no_decode_iter_fix_required",
        )
    if active_residual is not None and active_residual > tolerance:
        return (
            "active_decode_iter_gap_present",
            "dp_sync_or_comm_bubble_candidate",
            "dp_sync_or_comm_bubble_modeling",
        )
    return (
        "inconclusive_missing_active_decode_counter",
        "missing_active_decode_truth",
        "repeat_metrics_capture",
    )


def _sim_by_scenario(sim_rows: Iterable[SimBreakdown] | Iterable[object]) -> dict[str, object]:
    return {row.scenario: row for row in sim_rows}


def _active_stats_by_scenario(
    raw_root: Path,
    active_stats: dict[str, ActiveDecodeStats] | None,
) -> dict[str, ActiveDecodeStats]:
    if active_stats is not None:
        return active_stats
    return {
        scenario: parse_active_decode_stats(raw_root / scenario / "metrics.jsonl")
        for scenario in DP2_SCENARIOS
    }


def _dp2_row(
    *,
    source_row: dict[str, str],
    sim: object,
    active: ActiveDecodeStats,
) -> dict[str, str]:
    scenario = source_row["scenario"]
    real_running_max = float(source_row["real_running_global_max"])
    output_global = float(source_row["phase403_output_tok_s_global"])
    output_gpu = float(source_row["phase403_output_tok_s_gpu"])
    sim_total = float(sim.sim_decode_ms_per_iter)
    active_ms = active.ms_per_iter_median
    aggregate_ms = real_running_max * 1000.0 / output_global
    active_residual = active_ms - sim_total
    aggregate_residual = aggregate_ms - sim_total
    verdict, primary, target = _verdict(
        row_type="dp2_attribution",
        active_residual=active_residual,
        aggregate_residual=aggregate_residual,
        sim_decode_ms=sim_total,
    )
    row = {
        "source": SOURCE,
        "row_type": "dp2_attribution",
        "scenario": scenario,
        "tp": int(source_row["tp"]),
        "dp": int(source_row["dp"]),
        "ep": int(source_row["ep"]),
        "isl": int(source_row["isl"]),
        "osl": int(source_row["osl"]),
        "max_num_batched_tokens": int(source_row["max_num_batched_tokens"]),
        "per_replica_batch": int(sim.per_replica_batch),
        "sim_running_global": int(sim.sim_running_global),
        "real_running_global_max": real_running_max,
        "real_running_global_mean": float(source_row["real_running_global_mean"]),
        "real_running_global_p50": float(source_row["real_running_global_p50"]),
        "mid_kv_len": int(sim.mid_kv_len),
        "bucketed_kv_len": int(sim.bucketed_kv_len),
        "real_active_decode_ms_per_iter": active_ms,
        "real_active_decode_ms_p10": active.ms_per_iter_p10,
        "real_active_decode_ms_p90": active.ms_per_iter_p90,
        "active_decode_intervals": active.intervals,
        "real_aggregate_implied_ms_per_iter": aggregate_ms,
        "sim_decode_ms_per_iter": sim_total,
        "gen_attention_ms": float(sim.gen_attention_ms),
        "gen_moe_compute_ms": float(sim.gen_moe_compute_ms),
        "gen_dispatch_ms": float(sim.gen_dispatch_ms),
        "gen_gemm_ms": float(sim.gen_gemm_ms),
        "gen_other_non_attention_ms": float(sim.gen_other_non_attention_ms),
        "overhead_ms": float(sim.overhead_ms),
        "active_decode_residual_ms": active_residual,
        "aggregate_implied_residual_ms": aggregate_residual,
        "dispatch_ms_vs_active_residual_ratio": _ratio_to_residual(
            float(sim.gen_dispatch_ms),
            active_residual,
        ),
        "attention_ms_vs_active_residual_ratio": _ratio_to_residual(
            float(sim.gen_attention_ms),
            active_residual,
        ),
        "aggregation_gap_minus_active_gap_ms": aggregate_residual - active_residual,
        "phase403_or_phase400_output_tok_s_global": output_global,
        "phase403_or_phase400_output_tok_s_gpu": output_gpu,
        "attribution_verdict": verdict,
        "primary_driver": primary,
        "phase405_target": target,
        "gpu_allowed": False,
        "ssh_allowed": False,
        "runtime_modified": False,
        "perf_database": False,
        "valid_for_default": False,
        "diagnostic_only": True,
        "default_readiness": DEFAULT_READINESS,
    }
    return {field: _fmt(row.get(field)) for field in CSV_FIELDS}


def _tp8_row(*, source_row: dict[str, str], sim: object) -> dict[str, str]:
    scenario = source_row["scenario"]
    real_running_max = float(source_row["serve_running_reqs_max"])
    output_global = float(source_row["output_tok_s_global"])
    output_gpu = float(source_row["output_tok_s_gpu"])
    sim_total = float(sim.sim_decode_ms_per_iter)
    aggregate_ms = real_running_max * 1000.0 / output_global
    aggregate_residual = aggregate_ms - sim_total
    verdict, primary, target = _verdict(
        row_type="tp8_crosscheck",
        active_residual=None,
        aggregate_residual=aggregate_residual,
        sim_decode_ms=sim_total,
    )
    row = {
        "source": SOURCE,
        "row_type": "tp8_crosscheck",
        "scenario": scenario,
        "tp": int(source_row["tp"]),
        "dp": int(source_row["dp"]),
        "ep": int(source_row["ep"]),
        "isl": int(source_row["isl"]),
        "osl": int(source_row["osl"]),
        "max_num_batched_tokens": int(source_row["max_num_batched_tokens"]),
        "per_replica_batch": int(sim.per_replica_batch),
        "sim_running_global": int(sim.sim_running_global),
        "real_running_global_max": real_running_max,
        "real_running_global_mean": float(source_row["serve_running_reqs_mean"]),
        "real_running_global_p50": None,
        "mid_kv_len": int(sim.mid_kv_len),
        "bucketed_kv_len": int(sim.bucketed_kv_len),
        "real_active_decode_ms_per_iter": None,
        "real_active_decode_ms_p10": None,
        "real_active_decode_ms_p90": None,
        "active_decode_intervals": None,
        "real_aggregate_implied_ms_per_iter": aggregate_ms,
        "sim_decode_ms_per_iter": sim_total,
        "gen_attention_ms": float(sim.gen_attention_ms),
        "gen_moe_compute_ms": float(sim.gen_moe_compute_ms),
        "gen_dispatch_ms": float(sim.gen_dispatch_ms),
        "gen_gemm_ms": float(sim.gen_gemm_ms),
        "gen_other_non_attention_ms": float(sim.gen_other_non_attention_ms),
        "overhead_ms": float(sim.overhead_ms),
        "active_decode_residual_ms": None,
        "aggregate_implied_residual_ms": aggregate_residual,
        "dispatch_ms_vs_active_residual_ratio": None,
        "attention_ms_vs_active_residual_ratio": None,
        "aggregation_gap_minus_active_gap_ms": None,
        "phase403_or_phase400_output_tok_s_global": output_global,
        "phase403_or_phase400_output_tok_s_gpu": output_gpu,
        "attribution_verdict": verdict,
        "primary_driver": primary,
        "phase405_target": target,
        "gpu_allowed": False,
        "ssh_allowed": False,
        "runtime_modified": False,
        "perf_database": False,
        "valid_for_default": False,
        "diagnostic_only": True,
        "default_readiness": DEFAULT_READINESS,
    }
    return {field: _fmt(row.get(field)) for field in CSV_FIELDS}


def build_phase404_rows(
    *,
    sim_rows: Iterable[SimBreakdown] | Iterable[object] | None = None,
    raw_root: Path = DEFAULT_RAW_ROOT,
    phase403_csv: Path = PHASE403_CSV,
    phase400_csv: Path = PHASE400_CSV,
    active_stats: dict[str, ActiveDecodeStats] | None = None,
) -> list[dict[str, str]]:
    sim_by_scenario = _sim_by_scenario(
        collect_sim_breakdowns() if sim_rows is None else sim_rows
    )
    phase403 = _read_csv_by_scenario(phase403_csv)
    phase400 = _read_csv_by_scenario(phase400_csv)
    active_by_scenario = _active_stats_by_scenario(raw_root, active_stats)
    rows = [
        _dp2_row(
            source_row=phase403[scenario],
            sim=sim_by_scenario[scenario],
            active=active_by_scenario[scenario],
        )
        for scenario in DP2_SCENARIOS
    ]
    rows.append(_tp8_row(source_row=phase400[TP8_CROSS_SCENARIO], sim=sim_by_scenario[TP8_CROSS_SCENARIO]))
    _validate_rows(rows)
    return rows


def _validate_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != 3:
        raise ValueError("Phase404 must contain two DP2 rows plus one TP8 crosscheck")
    if {row["scenario"] for row in rows} != set(ALL_SCENARIOS):
        raise ValueError("Phase404 scenario set mismatch")
    for row in rows:
        if row["source"] != SOURCE:
            raise ValueError("source")
        if row["gpu_allowed"] != "false":
            raise ValueError("gpu_allowed")
        if row["ssh_allowed"] != "false":
            raise ValueError("ssh_allowed")
        for field in ("runtime_modified", "perf_database", "valid_for_default"):
            if row[field] != "false":
                raise ValueError(field)
        if row["diagnostic_only"] != "true":
            raise ValueError("diagnostic_only")
        if row["default_readiness"] != DEFAULT_READINESS:
            raise ValueError("default_readiness")
        if row["row_type"] == "dp2_attribution" and not row["real_active_decode_ms_per_iter"]:
            raise ValueError("real_active_decode_ms_per_iter")


def write_phase404_csv(path: Path, rows: list[dict[str, str]] | None = None) -> None:
    rows = build_phase404_rows() if rows is None else rows
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_phase404_md(rows: list[dict[str, str]]) -> str:
    _validate_rows(rows)
    dp_rows = [row for row in rows if row["row_type"] == "dp2_attribution"]
    verdicts = {row["attribution_verdict"] for row in dp_rows}
    if verdicts == {"reject_gen_op_latency_gap"}:
        summary = (
            "Phase404 rejects the per-iteration generation-op gap hypothesis: "
            "active decode counters are close to sim, while aggregate-implied "
            "latency is larger because it folds in non-active decode time."
        )
    elif verdicts == {"aggregate_gap_dominates_minor_active_decode_gap"}:
        summary = (
            "Phase404 finds only a minor active decode iteration gap. The larger "
            "residual is aggregate-implied latency, so Phase405 should target "
            "decode duty cycle or queueing first, not a single generation op."
        )
    else:
        summary = "Phase404 found an active decode iteration gap; inspect the table before Phase405."

    lines = [
        "# Phase404 DP2 decode-iteration attribution",
        "",
        summary,
        "",
        "## Rows",
        "",
        "| scenario | row_type | real active ms/iter | aggregate implied ms/iter | sim ms/iter | residual active | residual aggregate | dispatch ms | verdict |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {scenario} | {row_type} | {real_active_decode_ms_per_iter} | {real_aggregate_implied_ms_per_iter} | {sim_decode_ms_per_iter} | {active_decode_residual_ms} | {aggregate_implied_residual_ms} | {gen_dispatch_ms} | {attribution_verdict} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- real_active_decode_ms_per_iter comes from Phase403 /metrics generation-token counters and running batch.",
            "- real_aggregate_implied_ms_per_iter comes from total output throughput and running max; it includes queueing, prefill, and idle/duty-cycle effects.",
            "- gen_dispatch_ms is already charged serially in the pure decode path, and its magnitude is too small to explain a large aggregate-implied residual when active decode matches.",
            "- TP8-32k is retained only as a control row; Phase403 did not recollect active decode counters for TP8.",
            "",
            "## Boundary",
            "",
            "- gpu_allowed=false, ssh_allowed=false.",
            "- runtime_modified=false, perf_database=false, valid_for_default=false, diagnostic_only=true, default_readiness=No-Go.",
            "- Phase405 should target the aggregate decode duty/queueing gap unless a later trace contradicts the active decode counter result.",
            "",
        ]
    )
    return "\n".join(lines)


def write_phase404_md(path: Path, rows: list[dict[str, str]] | None = None) -> None:
    rows = build_phase404_rows() if rows is None else rows
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_phase404_md(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--phase403-csv", type=Path, default=PHASE403_CSV)
    parser.add_argument("--phase400-csv", type=Path, default=PHASE400_CSV)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows = build_phase404_rows(
        raw_root=args.raw_root,
        phase403_csv=args.phase403_csv,
        phase400_csv=args.phase400_csv,
    )
    write_phase404_csv(args.csv, rows)
    write_phase404_md(args.md, rows)
    print(f"wrote {args.csv}")
    print(f"wrote {args.md}")


if __name__ == "__main__":
    main()
