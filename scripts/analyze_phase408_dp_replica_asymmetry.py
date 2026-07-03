#!/usr/bin/env python3
"""Phase408: attribute the DP2 lockstep gap to replica asymmetry."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = "phase408_dp_replica_asymmetry"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase408_dp_replica_asymmetry.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase408_dp_replica_asymmetry.md"
PHASE407_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase407_dp_lockstep_joint_sim.csv"
PHASE403_RAW_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase403_dp2_stats"
DEFAULT_READINESS = "No-Go"

DP2_SCENARIOS = (
    "K2.5-tp4ep8dp2-8k2k",
    "K2.5-tp4ep8dp2-32k3k",
)

CSV_FIELDS = [
    "source",
    "row_type",
    "scenario",
    "metric_samples",
    "rate_samples",
    "active_prefill_samples",
    "prompt_rate_corr_all",
    "prompt_rate_corr_active",
    "engine0_prefill_rate_mean",
    "engine1_prefill_rate_mean",
    "engine0_prefill_rate_sd",
    "engine1_prefill_rate_sd",
    "engine0_prefill_active_frac",
    "engine1_prefill_active_frac",
    "engine0_prefill_time_rate_mean",
    "engine1_prefill_time_rate_mean",
    "waiting_engine0_mean",
    "waiting_engine1_mean",
    "kv_cache_engine0_mean",
    "kv_cache_engine1_mean",
    "mean_rate_imbalance_ratio",
    "active_frac_imbalance_ratio",
    "phase407_uncoupled_output_tok_s_gpu",
    "real_output_tok_s_gpu",
    "needed_penalty_source",
    "needed_penalty",
    "phi0_penalty",
    "best_phi_steps",
    "best_phi_penalty",
    "target_phi_steps",
    "target_phi_penalty",
    "target_penalty_error_pct",
    "phase_adjusted_output_tok_s_gpu",
    "phase_adjusted_ratio",
    "diagnostic_knob",
    "asymmetry_sufficient",
    "load_imbalance_secondary",
    "mechanism_verdict",
    "phase409_target",
    "phase405_penalty_read",
    "gpu_allowed",
    "ssh_allowed",
    "runtime_modified",
    "perf_database",
    "valid_for_default",
    "diagnostic_only",
    "default_readiness",
]

_PROM_LINE = re.compile(
    r"^(?P<metric>[A-Za-z_:][A-Za-z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)


@dataclass(frozen=True)
class MetricSample:
    ts: datetime
    prompt_tokens: dict[str, float]
    prefill_time_seconds: dict[str, float]
    waiting: dict[str, float]
    kv_cache_usage: dict[str, float]


@dataclass(frozen=True)
class EngineStats:
    mean: float
    sd: float
    active_frac: float
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


def _parse_labels(text: str | None) -> dict[str, str]:
    if not text:
        return {}
    return {
        key: value
        for key, value in re.findall(r'([A-Za-z_][A-Za-z0-9_]*)="([^"]*)"', text)
    }


def _engine(labels: dict[str, str]) -> str | None:
    for key in ("engine", "data_parallel_rank", "rank"):
        if key in labels:
            value = labels[key]
            return value.split("_")[-1] if value.startswith("EngineCore_") else value
    return None


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _parse_metric_body(body: str) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
]:
    prompt_tokens: dict[str, float] = {}
    prefill_time_seconds: dict[str, float] = {}
    waiting: dict[str, float] = {}
    kv_cache_usage: dict[str, float] = {}
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PROM_LINE.match(line)
        if not match:
            continue
        labels = _parse_labels(match.group("labels"))
        engine = _engine(labels)
        if engine not in {"0", "1"}:
            continue
        metric = match.group("metric")
        value = float(match.group("value"))
        if metric == "vllm:prompt_tokens_total":
            prompt_tokens[engine] = value
        elif metric == "vllm:request_prefill_time_seconds_sum":
            prefill_time_seconds[engine] = value
        elif metric == "vllm:num_requests_waiting":
            waiting[engine] = value
        elif metric == "vllm:kv_cache_usage_perc":
            kv_cache_usage[engine] = value
    return prompt_tokens, prefill_time_seconds, waiting, kv_cache_usage


def read_metric_samples(path: Path) -> list[MetricSample]:
    samples: list[MetricSample] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") != 200:
                continue
            prompt, prefill_time, waiting, kv_cache = _parse_metric_body(
                record.get("body", "")
            )
            if not prompt:
                continue
            samples.append(
                MetricSample(
                    ts=_timestamp(record["ts"]),
                    prompt_tokens=prompt,
                    prefill_time_seconds=prefill_time,
                    waiting=waiting,
                    kv_cache_usage=kv_cache,
                )
            )
    if len(samples) < 2:
        raise ValueError(f"need at least two prompt-token samples in {path}")
    return samples


def _counter_rates(samples: list[MetricSample], attr: str) -> tuple[list[float], list[float]]:
    engine0: list[float] = []
    engine1: list[float] = []
    for prev, cur in zip(samples, samples[1:]):
        dt = (cur.ts - prev.ts).total_seconds()
        if dt <= 0:
            continue
        prev_counter = getattr(prev, attr)
        cur_counter = getattr(cur, attr)
        for engine, out in (("0", engine0), ("1", engine1)):
            if engine not in cur_counter or engine not in prev_counter:
                out.append(0.0)
                continue
            delta = cur_counter[engine] - prev_counter[engine]
            out.append(max(0.0, delta) / dt)
    return engine0, engine1


def _gauge_mean(samples: list[MetricSample], attr: str, engine: str) -> float:
    values = [
        getattr(sample, attr)[engine]
        for sample in samples
        if engine in getattr(sample, attr)
    ]
    return sum(values) / len(values) if values else math.nan


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def _sd(values: list[float]) -> float:
    if not values:
        return math.nan
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return math.nan
    mean_x = _mean(xs)
    mean_y = _mean(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0 or var_y <= 0:
        return math.nan
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return cov / math.sqrt(var_x * var_y)


def _engine_stats(values: list[float]) -> EngineStats:
    if not values:
        return EngineStats(mean=math.nan, sd=math.nan, active_frac=math.nan, max=math.nan)
    return EngineStats(
        mean=_mean(values),
        sd=_sd(values),
        active_frac=sum(1 for value in values if value > 0) / len(values),
        max=max(values),
    )


def _imbalance_ratio(a: float, b: float) -> float:
    low = min(a, b)
    high = max(a, b)
    if low <= 0:
        return math.inf if high > 0 else 1.0
    return high / low


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return math.inf
    return numerator / denominator


def phase_offset_penalty(engine0_rates: list[float], engine1_rates: list[float], phi_steps: int) -> float:
    if len(engine0_rates) != len(engine1_rates):
        raise ValueError("rate series must have equal length")
    if not engine0_rates:
        raise ValueError("rate series is empty")
    n = len(engine0_rates)
    shifted = [engine1_rates[(idx + phi_steps) % n] for idx in range(n)]
    baseline = (sum(engine0_rates) + sum(shifted)) / 2.0
    if baseline <= 0:
        return 1.0
    coupled = sum(max(a, b) for a, b in zip(engine0_rates, shifted))
    return coupled / baseline


def sweep_phase_offsets(
    engine0_rates: list[float],
    engine1_rates: list[float],
    *,
    target_penalty: float,
) -> dict[str, float | int]:
    penalties = [
        (phi, phase_offset_penalty(engine0_rates, engine1_rates, phi))
        for phi in range(len(engine0_rates))
    ]
    best_phi, best_penalty = max(penalties, key=lambda item: item[1])
    target_phi, target_phi_penalty = min(
        penalties,
        key=lambda item: abs(item[1] - target_penalty),
    )
    return {
        "phi0_penalty": penalties[0][1],
        "best_phi_steps": best_phi,
        "best_phi_penalty": best_penalty,
        "target_phi_steps": target_phi,
        "target_phi_penalty": target_phi_penalty,
    }


def _active_pairs(xs: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
    active_x: list[float] = []
    active_y: list[float] = []
    for x, y in zip(xs, ys):
        if x > 0 or y > 0:
            active_x.append(x)
            active_y.append(y)
    return active_x, active_y


def _row_from_scenario(
    *,
    scenario: str,
    phase407_row: dict[str, str],
    raw_root: Path,
) -> dict[str, str]:
    samples = read_metric_samples(raw_root / scenario / "metrics.jsonl")
    prompt0, prompt1 = _counter_rates(samples, "prompt_tokens")
    prefill_time0, prefill_time1 = _counter_rates(samples, "prefill_time_seconds")
    active0, active1 = _active_pairs(prompt0, prompt1)
    stats0 = _engine_stats(prompt0)
    stats1 = _engine_stats(prompt1)

    uncoupled = float(phase407_row["uncoupled_joint_output_tok_s_gpu"])
    real = float(phase407_row["real_output_tok_s_gpu"])
    needed_penalty = _safe_ratio(uncoupled, real)
    sweep = sweep_phase_offsets(prompt0, prompt1, target_penalty=needed_penalty)
    target_phi_penalty = float(sweep["target_phi_penalty"])
    target_penalty_error = abs(_safe_ratio(target_phi_penalty, needed_penalty) - 1.0) * 100.0
    adjusted_output = _safe_ratio(uncoupled, target_phi_penalty)
    adjusted_ratio = _safe_ratio(adjusted_output, real)
    adjusted_error = abs(adjusted_ratio - 1.0) * 100.0
    sufficient = target_penalty_error <= 10.0 and adjusted_error <= 10.0
    load_imbalance_secondary = (
        _imbalance_ratio(stats0.mean, stats1.mean) >= 1.20
        or _imbalance_ratio(stats0.active_frac, stats1.active_frac) >= 1.20
    )
    verdict = (
        "replica_asymmetry_sufficient_missing_variable"
        if sufficient
        else "replica_asymmetry_insufficient"
    )
    phase409_target = (
        "runtime_replica_asymmetry_or_prefill_phase_model"
        if sufficient and not load_imbalance_secondary
        else "runtime_replica_asymmetry_plus_load_imbalance_model"
        if sufficient
        else "recheck_prefill_occupancy_or_other_dp_gap"
    )

    row = {
        "source": SOURCE,
        "row_type": "dp2_asymmetry_attribution",
        "scenario": scenario,
        "metric_samples": len(samples),
        "rate_samples": len(prompt0),
        "active_prefill_samples": len(active0),
        "prompt_rate_corr_all": _pearson(prompt0, prompt1),
        "prompt_rate_corr_active": _pearson(active0, active1),
        "engine0_prefill_rate_mean": stats0.mean,
        "engine1_prefill_rate_mean": stats1.mean,
        "engine0_prefill_rate_sd": stats0.sd,
        "engine1_prefill_rate_sd": stats1.sd,
        "engine0_prefill_active_frac": stats0.active_frac,
        "engine1_prefill_active_frac": stats1.active_frac,
        "engine0_prefill_time_rate_mean": _mean(prefill_time0),
        "engine1_prefill_time_rate_mean": _mean(prefill_time1),
        "waiting_engine0_mean": _gauge_mean(samples, "waiting", "0"),
        "waiting_engine1_mean": _gauge_mean(samples, "waiting", "1"),
        "kv_cache_engine0_mean": _gauge_mean(samples, "kv_cache_usage", "0"),
        "kv_cache_engine1_mean": _gauge_mean(samples, "kv_cache_usage", "1"),
        "mean_rate_imbalance_ratio": _imbalance_ratio(stats0.mean, stats1.mean),
        "active_frac_imbalance_ratio": _imbalance_ratio(stats0.active_frac, stats1.active_frac),
        "phase407_uncoupled_output_tok_s_gpu": uncoupled,
        "real_output_tok_s_gpu": real,
        "needed_penalty_source": "phase407_uncoupled_div_real",
        "needed_penalty": needed_penalty,
        "phi0_penalty": sweep["phi0_penalty"],
        "best_phi_steps": sweep["best_phi_steps"],
        "best_phi_penalty": sweep["best_phi_penalty"],
        "target_phi_steps": sweep["target_phi_steps"],
        "target_phi_penalty": target_phi_penalty,
        "target_penalty_error_pct": target_penalty_error,
        "phase_adjusted_output_tok_s_gpu": adjusted_output,
        "phase_adjusted_ratio": adjusted_ratio,
        "diagnostic_knob": "observed_independent_completion_plus_circular_phase_offset",
        "asymmetry_sufficient": sufficient,
        "load_imbalance_secondary": load_imbalance_secondary,
        "mechanism_verdict": verdict,
        "phase409_target": phase409_target,
        "phase405_penalty_read": False,
        "gpu_allowed": False,
        "ssh_allowed": False,
        "runtime_modified": False,
        "perf_database": False,
        "valid_for_default": False,
        "diagnostic_only": True,
        "default_readiness": DEFAULT_READINESS,
    }
    return {key: _fmt(row.get(key, "")) for key in CSV_FIELDS}


def build_phase408_rows(
    *,
    phase407_csv: Path = PHASE407_CSV,
    raw_root: Path = PHASE403_RAW_ROOT,
) -> list[dict[str, str]]:
    phase407 = _read_csv_by_scenario(phase407_csv)
    rows = [
        _row_from_scenario(
            scenario=scenario,
            phase407_row=phase407[scenario],
            raw_root=raw_root,
        )
        for scenario in DP2_SCENARIOS
    ]
    return rows


def _validate_rows(rows: list[dict[str, str]]) -> None:
    for row in rows:
        if row.get("source") != SOURCE:
            raise ValueError("unexpected source")
        if row.get("phase405_penalty_read") != "false":
            raise ValueError("phase405_penalty_read must stay false")
        for field in ("gpu_allowed", "ssh_allowed", "runtime_modified", "perf_database"):
            if row.get(field) != "false":
                raise ValueError(f"{field} must stay false")
        if row.get("valid_for_default") != "false":
            raise ValueError("valid_for_default must stay false")
        if row.get("diagnostic_only") != "true":
            raise ValueError("diagnostic_only must stay true")
        if row.get("default_readiness") != DEFAULT_READINESS:
            raise ValueError("default_readiness must stay No-Go")
    scenarios = {row.get("scenario") for row in rows}
    if scenarios != set(DP2_SCENARIOS):
        raise ValueError(f"Phase408 rows must cover DP2 scenarios only: {scenarios}")


def write_phase408_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_phase408_md(rows: list[dict[str, str]]) -> str:
    _validate_rows(rows)
    lines = [
        "# Phase408 DP Replica Asymmetry Attribution",
        "",
        "Phase408 is diagnostic-only. It reads Phase403 per-engine metrics and Phase407 uncoupled/real outputs; it does not read Phase405 penalty rows and does not modify runtime, PerfDatabase, or gates.",
        "",
        "## Verdict",
        "",
    ]
    if all(row["asymmetry_sufficient"] == "true" for row in rows):
        lines.append(
            "Replica asymmetry has enough magnitude to explain the missing DP2 penalty: a phase offset over the observed per-engine prefill-token streams can move the Phase407 uncoupled prediction back to the Phase403 real throughput for both DP2 scenarios."
        )
    else:
        lines.append(
            "Replica asymmetry alone is insufficient for at least one DP2 scenario; Phase409 must keep other DP occupancy mechanisms in scope."
        )
    lines.extend(
        [
            "",
            "## DP2 Results",
            "",
            "| scenario | corr_all | corr_active | needed_penalty | phi0_penalty | target_phi | target_penalty | adjusted_ratio | verdict |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| {scenario} | {corr_all} | {corr_active} | {needed} | {phi0} | {phi} | {target} | {ratio} | {verdict} |".format(
                scenario=row["scenario"],
                corr_all=row["prompt_rate_corr_all"],
                corr_active=row["prompt_rate_corr_active"],
                needed=row["needed_penalty"],
                phi0=row["phi0_penalty"],
                phi=row["target_phi_steps"],
                target=row["target_phi_penalty"],
                ratio=row["phase_adjusted_ratio"],
                verdict=row["mechanism_verdict"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The target penalty is `phase407_uncoupled_joint_output_tok_s_gpu / real_output_tok_s_gpu`, not a Phase405 reconstructed penalty.",
            "- The diagnostic knob is the observed independent per-engine completion stream plus circular phase offset; no shipped simulator class is changed.",
            "- `phi0_penalty` is still far below the needed penalty; shifting one replica's prefill activity exposes the missing max-coupling cost.",
            "- TP8 remains covered by Phase407's dp=1 early-exit control; Phase408 only tests DP2 replica asymmetry.",
            "- Load imbalance remains a secondary item where mean prefill rate or active fraction differs by at least 20 percent; Phase409 should model replica phase first and keep load imbalance explicit.",
            "",
            "## Boundary",
            "",
            "- GPU/SSH: not used.",
            "- Runtime/PerfDatabase/gate: not modified.",
            "- Default AIC: No-Go.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_phase408_md(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_phase408_md(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--phase407-csv", type=Path, default=PHASE407_CSV)
    parser.add_argument("--raw-root", type=Path, default=PHASE403_RAW_ROOT)
    args = parser.parse_args()

    rows = build_phase408_rows(
        phase407_csv=args.phase407_csv,
        raw_root=args.raw_root,
    )
    write_phase408_csv(args.csv, rows)
    write_phase408_md(args.md, rows)
    print(f"wrote {args.csv}")
    print(f"wrote {args.md}")


if __name__ == "__main__":
    main()
