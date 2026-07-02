#!/usr/bin/env python3
"""Phase405: decompose the DP2 gap into duty-cycle and active-iter terms."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = "phase405_dp2_duty_cycle"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase405_dp2_duty_cycle.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase405_dp2_duty_cycle.md"
PHASE401_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase401_preempt_admission.csv"
PHASE403_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase403_dp2_running_batch.csv"
PHASE404_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase404_dp2_iter_latency.csv"
PHASE400_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase400_clean_recollect.csv"
PHASE403_RAW_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase403_dp2_stats"
DEFAULT_READINESS = "No-Go"

DP2_SCENARIOS = (
    "K2.5-tp4ep8dp2-8k2k",
    "K2.5-tp4ep8dp2-32k3k",
)
TP8_CONTROL = "K2.5-tp8ep8-32k3k"
ALL_SCENARIOS = DP2_SCENARIOS + (TP8_CONTROL,)

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
    "phase401_sim_output_tok_s_gpu",
    "real_output_tok_s_gpu",
    "observed_tput_ratio",
    "observed_direction",
    "sim_avg_prefill_reqs_global",
    "sim_avg_decode_reqs_global",
    "sim_peak_decode_reqs_global",
    "sim_decode_peak_minus_avg_global",
    "real_running_global_mean",
    "real_running_global_p50",
    "real_running_global_p90",
    "real_running_global_max",
    "real_running_peak_minus_mean",
    "raw_batch_occupancy_ratio",
    "peak_batch_ratio",
    "real_active_decode_ms_per_iter",
    "sim_decode_ms_per_iter",
    "active_iter_ratio",
    "real_mean_implied_ms_per_iter",
    "duty_cycle_ratio",
    "reconstructed_tput_ratio",
    "reconstruction_error_pct",
    "phase404_active_residual_ms",
    "phase404_aggregate_gap_minus_active_gap_ms",
    "attribution_verdict",
    "primary_driver",
    "phase406_target",
    "gpu_allowed",
    "ssh_allowed",
    "runtime_modified",
    "perf_database",
    "valid_for_default",
    "diagnostic_only",
    "default_readiness",
]


@dataclass(frozen=True)
class RunningStats:
    mean: float
    p50: float
    p90: float
    max: float


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


def _safe_ratio(a: float, b: float) -> float:
    if b <= 0:
        return math.inf
    return a / b


def _direction(predicted: float, real: float) -> str:
    if predicted > real:
        return "sim_over_predicts_throughput"
    if predicted < real:
        return "sim_under_predicts_throughput"
    return "matched"


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


def _running_total(body: str) -> float | None:
    total = 0.0
    seen = False
    for raw_line in body.splitlines():
        match = _PROM_LINE.match(raw_line.strip())
        if not match:
            continue
        if match.group("metric") == "vllm:num_requests_running":
            total += float(match.group("value"))
            seen = True
    return total if seen else None


def parse_running_stats(path: Path) -> RunningStats:
    values: list[float] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") != 200:
                continue
            running = _running_total(record.get("body", ""))
            if running is not None:
                values.append(running)
    if not values:
        raise ValueError(f"missing running metrics in {path}")
    return RunningStats(
        mean=sum(values) / len(values),
        p50=_percentile(values, 0.50),
        p90=_percentile(values, 0.90),
        max=max(values),
    )


def _running_stats_by_scenario(
    *,
    raw_root: Path,
    phase403: dict[str, dict[str, str]],
    phase400: dict[str, dict[str, str]],
    running_stats: dict[str, RunningStats] | None,
) -> dict[str, RunningStats]:
    if running_stats is not None:
        stats = dict(running_stats)
        if TP8_CONTROL not in stats:
            tp8 = phase400[TP8_CONTROL]
            stats[TP8_CONTROL] = RunningStats(
                mean=float(tp8["serve_running_reqs_mean"]),
                p50=math.nan,
                p90=math.nan,
                max=float(tp8["serve_running_reqs_max"]),
            )
        return stats
    stats = {
        scenario: parse_running_stats(raw_root / scenario / "metrics.jsonl")
        for scenario in DP2_SCENARIOS
    }
    tp8 = phase400[TP8_CONTROL]
    stats[TP8_CONTROL] = RunningStats(
        mean=float(tp8["serve_running_reqs_mean"]),
        p50=math.nan,
        p90=math.nan,
        max=float(tp8["serve_running_reqs_max"]),
    )
    # Keep Phase403 CSV as the stable mean/p50/max source if parser and prior
    # packaging disagree only because of future metrics label churn.
    for scenario in DP2_SCENARIOS:
        packaged = phase403[scenario]
        stats[scenario] = RunningStats(
            mean=float(packaged["real_running_global_mean"]),
            p50=float(packaged["real_running_global_p50"]),
            p90=stats[scenario].p90,
            max=float(packaged["real_running_global_max"]),
        )
    return stats


def _observed_ratio(sim_output: float, real_output: float) -> float:
    if sim_output <= 0 or real_output <= 0:
        return math.inf
    return sim_output / real_output


def _reconstruction_error_pct(reconstructed: float, observed: float) -> float:
    if observed <= 0:
        return math.inf
    return abs(reconstructed - observed) / observed * 100.0


def _verdict(
    *,
    row_type: str,
    reconstruction_error_pct: float,
    duty_cycle_ratio: float,
    active_iter_ratio: float | None,
    raw_batch_occupancy_ratio: float,
) -> tuple[str, str, str]:
    if row_type == "tp8_control":
        if 0.9 <= raw_batch_occupancy_ratio <= 1.15:
            return (
                "tp8_control_no_dp_duty_gap",
                "control",
                "phase406_dp_only_if_dp2_duty_gap_persists",
            )
        return (
            "tp8_control_has_occupancy_drift",
            "control_drift",
            "recheck_phase400_tp8_control",
        )
    if reconstruction_error_pct <= 10.0:
        if active_iter_ratio is not None and duty_cycle_ratio >= active_iter_ratio:
            primary = "duty_cycle_gap_primary_active_iter_secondary"
        else:
            primary = "active_iter_gap_primary_duty_secondary"
        return (
            "duty_cycle_plus_active_iter_reconstructs_dp2_gap",
            primary,
            "dp_prefill_occupancy_or_lockstep_duty_model",
        )
    return (
        "duty_cycle_decomposition_incomplete",
        "missing_unmodeled_factor",
        "recheck_metrics_or_additional_dp_sync_factor",
    )


def _build_row(
    *,
    scenario: str,
    row_type: str,
    phase401: dict[str, dict[str, str]],
    phase403: dict[str, dict[str, str]],
    phase404: dict[str, dict[str, str]],
    phase400: dict[str, dict[str, str]],
    stats: RunningStats,
) -> dict[str, str]:
    sim = phase401[scenario]
    p404 = phase404[scenario]
    dp = int(sim["dp"])
    sim_output = float(sim["phase401_sim_output_tok_s_gpu"])
    if row_type == "tp8_control":
        real_output_gpu = float(phase400[scenario]["output_tok_s_gpu"])
        real_output_global = float(phase400[scenario]["output_tok_s_global"])
    else:
        real_output_gpu = float(phase403[scenario]["phase403_output_tok_s_gpu"])
        real_output_global = float(phase403[scenario]["phase403_output_tok_s_global"])

    sim_avg_decode_global = float(sim["sim_avg_decode_reqs_per_iter"]) * dp
    sim_peak_decode_global = float(sim["sim_peak_decode_reqs_per_iter"]) * dp
    sim_avg_prefill_global = float(sim["sim_avg_prefill_reqs_per_iter"]) * dp
    observed_ratio = _observed_ratio(sim_output, real_output_gpu)
    raw_batch_ratio = _safe_ratio(sim_avg_decode_global, stats.mean)
    peak_batch_ratio = _safe_ratio(sim_peak_decode_global, stats.max)
    real_active = float(p404["real_active_decode_ms_per_iter"]) if p404["real_active_decode_ms_per_iter"] else None
    sim_iter = float(p404["sim_decode_ms_per_iter"])
    active_iter_ratio = _safe_ratio(real_active, sim_iter) if real_active is not None else None
    real_mean_implied_ms = stats.mean * 1000.0 / real_output_global
    duty_cycle_ratio = _safe_ratio(real_mean_implied_ms, real_active) if real_active is not None else math.nan
    reconstructed = (
        duty_cycle_ratio * active_iter_ratio
        if active_iter_ratio is not None
        else _safe_ratio(real_mean_implied_ms, sim_iter)
    )
    reconstruction_error = _reconstruction_error_pct(reconstructed, observed_ratio)
    verdict, primary, target = _verdict(
        row_type=row_type,
        reconstruction_error_pct=reconstruction_error,
        duty_cycle_ratio=duty_cycle_ratio,
        active_iter_ratio=active_iter_ratio,
        raw_batch_occupancy_ratio=raw_batch_ratio,
    )
    row = {
        "source": SOURCE,
        "row_type": row_type,
        "scenario": scenario,
        "tp": int(sim["tp"]),
        "dp": dp,
        "ep": int(sim["ep"]),
        "isl": int(sim["isl"]),
        "osl": int(sim["osl"]),
        "max_num_batched_tokens": int(sim["max_num_batched_tokens"]),
        "phase401_sim_output_tok_s_gpu": sim_output,
        "real_output_tok_s_gpu": real_output_gpu,
        "observed_tput_ratio": observed_ratio,
        "observed_direction": _direction(sim_output, real_output_gpu),
        "sim_avg_prefill_reqs_global": sim_avg_prefill_global,
        "sim_avg_decode_reqs_global": sim_avg_decode_global,
        "sim_peak_decode_reqs_global": sim_peak_decode_global,
        "sim_decode_peak_minus_avg_global": sim_peak_decode_global - sim_avg_decode_global,
        "real_running_global_mean": stats.mean,
        "real_running_global_p50": stats.p50,
        "real_running_global_p90": stats.p90,
        "real_running_global_max": stats.max,
        "real_running_peak_minus_mean": stats.max - stats.mean,
        "raw_batch_occupancy_ratio": raw_batch_ratio,
        "peak_batch_ratio": peak_batch_ratio,
        "real_active_decode_ms_per_iter": real_active,
        "sim_decode_ms_per_iter": sim_iter,
        "active_iter_ratio": active_iter_ratio,
        "real_mean_implied_ms_per_iter": real_mean_implied_ms,
        "duty_cycle_ratio": duty_cycle_ratio,
        "reconstructed_tput_ratio": reconstructed,
        "reconstruction_error_pct": reconstruction_error,
        "phase404_active_residual_ms": float(p404["active_decode_residual_ms"]) if p404["active_decode_residual_ms"] else None,
        "phase404_aggregate_gap_minus_active_gap_ms": float(p404.get("aggregation_gap_minus_active_gap_ms", "")) if p404.get("aggregation_gap_minus_active_gap_ms", "") else None,
        "attribution_verdict": verdict,
        "primary_driver": primary,
        "phase406_target": target,
        "gpu_allowed": False,
        "ssh_allowed": False,
        "runtime_modified": False,
        "perf_database": False,
        "valid_for_default": False,
        "diagnostic_only": True,
        "default_readiness": DEFAULT_READINESS,
    }
    return {field: _fmt(row.get(field)) for field in CSV_FIELDS}


def build_phase405_rows(
    *,
    phase401_csv: Path = PHASE401_CSV,
    phase403_csv: Path = PHASE403_CSV,
    phase404_csv: Path = PHASE404_CSV,
    phase400_csv: Path = PHASE400_CSV,
    raw_root: Path = PHASE403_RAW_ROOT,
    running_stats: dict[str, RunningStats] | None = None,
) -> list[dict[str, str]]:
    phase401 = _read_csv_by_scenario(phase401_csv)
    phase403 = _read_csv_by_scenario(phase403_csv)
    phase404 = _read_csv_by_scenario(phase404_csv)
    phase400 = _read_csv_by_scenario(phase400_csv)
    stats = _running_stats_by_scenario(
        raw_root=raw_root,
        phase403=phase403,
        phase400=phase400,
        running_stats=running_stats,
    )
    rows = [
        _build_row(
            scenario=scenario,
            row_type="dp2_duty_attribution",
            phase401=phase401,
            phase403=phase403,
            phase404=phase404,
            phase400=phase400,
            stats=stats[scenario],
        )
        for scenario in DP2_SCENARIOS
    ]
    rows.append(
        _build_row(
            scenario=TP8_CONTROL,
            row_type="tp8_control",
            phase401=phase401,
            phase403=phase403,
            phase404=phase404,
            phase400=phase400,
            stats=stats[TP8_CONTROL],
        )
    )
    _validate_rows(rows)
    return rows


def _validate_rows(rows: list[dict[str, str]]) -> None:
    for row in rows:
        if row.get("source") != SOURCE:
            raise ValueError("source")
        if row.get("gpu_allowed") != "false":
            raise ValueError("gpu_allowed")
        if row.get("ssh_allowed") != "false":
            raise ValueError("ssh_allowed")
        for field in ("runtime_modified", "perf_database", "valid_for_default"):
            if row.get(field) != "false":
                raise ValueError(field)
        if row.get("diagnostic_only") != "true":
            raise ValueError("diagnostic_only")
        if row.get("default_readiness") != DEFAULT_READINESS:
            raise ValueError("default_readiness")
    if len(rows) != 3:
        raise ValueError("Phase405 must contain two DP2 rows plus one TP8 control")
    if {row["scenario"] for row in rows} != set(ALL_SCENARIOS):
        raise ValueError("Phase405 scenario set mismatch")


def write_phase405_csv(path: Path, rows: list[dict[str, str]] | None = None) -> None:
    rows = build_phase405_rows() if rows is None else rows
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_phase405_md(rows: list[dict[str, str]]) -> str:
    _validate_rows(rows)
    dp_rows = [row for row in rows if row["row_type"] == "dp2_duty_attribution"]
    verdicts = {row["attribution_verdict"] for row in dp_rows}
    if verdicts == {"duty_cycle_plus_active_iter_reconstructs_dp2_gap"}:
        summary = (
            "Phase405 explains the DP2 throughput gap as duty-cycle loss plus "
            "a smaller active-iteration term. TP8 is retained as the control and "
            "does not show the same DP duty gap."
        )
    else:
        summary = "Phase405 decomposition is incomplete; inspect the residual columns before Phase406."

    lines = [
        "# Phase405 DP2 duty-cycle attribution",
        "",
        summary,
        "",
        "## Decomposition",
        "",
        "| scenario | row_type | observed ratio | duty ratio | active iter ratio | reconstructed | error pct | raw batch ratio | verdict |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {scenario} | {row_type} | {observed_tput_ratio} | {duty_cycle_ratio} | {active_iter_ratio} | {reconstructed_tput_ratio} | {reconstruction_error_pct} | {raw_batch_occupancy_ratio} | {attribution_verdict} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- raw_batch_occupancy_ratio is the direct sim avg decode batch over real running mean check; it is diagnostic but not sufficient alone because running includes non-output active time.",
            "- duty_cycle_ratio uses real running mean, real output throughput, and Phase404 active decode iter latency to capture the wall-clock duty loss.",
            "- reconstructed_tput_ratio = duty_cycle_ratio * active_iter_ratio for DP2; residual under 10% means the two-factor model explains the observed gap.",
            "- TP8-32k is the control: peak batch matches and no DP-specific duty verdict is emitted.",
            "",
            "## Boundary",
            "",
            "- gpu_allowed=false, ssh_allowed=false.",
            "- runtime_modified=false, perf_database=false, valid_for_default=false, diagnostic_only=true, default_readiness=No-Go.",
            "- Phase406 should target DP prefill occupancy / lockstep duty modeling, not PerfDatabase values.",
            "",
        ]
    )
    return "\n".join(lines)


def write_phase405_md(path: Path, rows: list[dict[str, str]] | None = None) -> None:
    rows = build_phase405_rows() if rows is None else rows
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_phase405_md(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase401-csv", type=Path, default=PHASE401_CSV)
    parser.add_argument("--phase403-csv", type=Path, default=PHASE403_CSV)
    parser.add_argument("--phase404-csv", type=Path, default=PHASE404_CSV)
    parser.add_argument("--phase400-csv", type=Path, default=PHASE400_CSV)
    parser.add_argument("--raw-root", type=Path, default=PHASE403_RAW_ROOT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows = build_phase405_rows(
        phase401_csv=args.phase401_csv,
        phase403_csv=args.phase403_csv,
        phase404_csv=args.phase404_csv,
        phase400_csv=args.phase400_csv,
        raw_root=args.raw_root,
    )
    write_phase405_csv(args.csv, rows)
    write_phase405_md(args.md, rows)
    print(f"wrote {args.csv}")
    print(f"wrote {args.md}")


if __name__ == "__main__":
    main()
