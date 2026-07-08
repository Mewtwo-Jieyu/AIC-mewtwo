#!/usr/bin/env python3
"""Phase448: multi-replica admission fingerprint gate."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_REAL_FINGERPRINT = REPO_ROOT / "docs/iter_gap_investigation/phase447_sequence_fingerprint.csv"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase448_dp_admission_fingerprint.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase448_dp_admission_fingerprint.md"

SCENARIOS = [
    "K2.5-tp4ep8dp2-8k2k",
    "K2.5-tp4ep8dp2-32k3k",
]


@dataclass(frozen=True)
class MetricGate:
    scenario: str
    metric: str
    sim_value: float
    target_low: float
    target_high: float

    @property
    def passed(self) -> bool:
        return self.target_low <= self.sim_value <= self.target_high


def _load_validate_module():
    path = REPO_ROOT / "scripts" / "validate_cb_simulator.py"
    spec = importlib.util.spec_from_file_location("validate_cb_simulator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_tolerances(path: Path) -> dict[tuple[str, str], tuple[float, float]]:
    tolerances: dict[tuple[str, str], tuple[float, float]] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("section") != "fingerprint_tolerance":
                continue
            metric = row["metric"]
            if metric not in {
                "mixed_decode_batch_p10_p90",
                "mixed_bucket_tokens_p10_p90",
            }:
                continue
            tolerances[(row["scenario"], metric)] = (
                float(row["target_low"]),
                float(row["target_high"]),
            )
    return tolerances


def _percentile(values: list[int], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct / 100.0
    lo = int(index)
    hi = min(lo + 1, len(ordered) - 1)
    frac = index - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _phase(row: dict[str, float | int | bool]) -> str:
    if int(row["prefill_tokens"]) > 0 and int(row["decode_reqs"]) > 0:
        return "mixed_prefill"
    if int(row["prefill_tokens"]) > 0:
        return "pure_prefill"
    if int(row["decode_reqs"]) > 0:
        return "decode"
    return "empty"


def summarize_trace(
    scenario: str,
    trace: Iterable[dict[str, float | int | bool]],
    tolerances: dict[tuple[str, str], tuple[float, float]],
) -> tuple[list[dict[str, object]], list[MetricGate]]:
    rows = list(trace)
    mixed = [
        row for row in rows
        if int(row["prefill_tokens"]) > 0 and int(row["decode_reqs"]) > 0
    ]
    mixed_decode_batches = [int(row["decode_reqs"]) for row in mixed]
    mixed_buckets = [int(row["total_tokens"]) for row in mixed]
    metric_values = {
        "mixed_decode_batch_p50": (
            float(statistics.median(mixed_decode_batches)) if mixed_decode_batches else 0.0
        ),
        "mixed_bucket_tokens_p50": (
            float(statistics.median(mixed_buckets)) if mixed_buckets else 0.0
        ),
    }

    gate_specs = [
        ("mixed_decode_batch_p50", "mixed_decode_batch_p10_p90"),
        ("mixed_bucket_tokens_p50", "mixed_bucket_tokens_p10_p90"),
    ]
    gates: list[MetricGate] = []
    out_rows: list[dict[str, object]] = []
    for metric, tolerance_metric in gate_specs:
        low, high = tolerances[(scenario, tolerance_metric)]
        gate = MetricGate(
            scenario=scenario,
            metric=metric,
            sim_value=metric_values[metric],
            target_low=low,
            target_high=high,
        )
        gates.append(gate)
        out_rows.append({
            "scenario": scenario,
            "section": "gate",
            "metric": metric,
            "sim_value": gate.sim_value,
            "target_low": gate.target_low,
            "target_high": gate.target_high,
            "passed": gate.passed,
        })

    out_rows.extend([
        {
            "scenario": scenario,
            "section": "distribution",
            "metric": "mixed_decode_batch_p10",
            "sim_value": _percentile(mixed_decode_batches, 10),
            "target_low": "",
            "target_high": "",
            "passed": "",
        },
        {
            "scenario": scenario,
            "section": "distribution",
            "metric": "mixed_decode_batch_p90",
            "sim_value": _percentile(mixed_decode_batches, 90),
            "target_low": "",
            "target_high": "",
            "passed": "",
        },
        {
            "scenario": scenario,
            "section": "distribution",
            "metric": "mixed_step_count",
            "sim_value": len(mixed),
            "target_low": "",
            "target_high": "",
            "passed": "",
        },
    ])

    by_iter: dict[int, dict[int, str]] = defaultdict(dict)
    for row in rows:
        by_iter[int(row["local_iter"])][int(row["replica_id"])] = _phase(row)
    phase_pairs = Counter(
        "+".join([phases[0], phases[1]])
        for _, phases in sorted(by_iter.items())
        if 0 in phases and 1 in phases
    )
    total_pairs = sum(phase_pairs.values())
    for phase_pair, count in sorted(phase_pairs.items()):
        out_rows.append({
            "scenario": scenario,
            "section": "phase_joint",
            "metric": phase_pair,
            "sim_value": count / total_pairs if total_pairs else 0.0,
            "target_low": "",
            "target_high": "",
            "passed": "",
        })

    return out_rows, gates


def _run_scenario(validate, scenario: str) -> list[dict[str, float | int | bool]]:
    from aiconfigurator.sdk.backends.cb_simulator import CBSimulator

    point = next(pt for pt in validate.MULTI_CONFIG_DATA if pt.name == scenario)
    model, db, backend = validate._load_model_and_db(
        tp=point.tp,
        dp=point.dp,
        moe_tp=point.moe_tp,
        moe_ep=point.moe_ep,
    )
    cb_config = validate._make_multi_config_cb_config(
        point,
        overlap_factor=0.0,
        ep8_per_iteration_overhead_ms=validate.DEFAULT_EP8_PER_ITERATION_OVERHEAD_MS,
    )
    sim = CBSimulator(backend, model, db, cb_config)
    sim.run_multi_replica(
        isl=point.isl,
        osl=point.osl,
        concurrency=point.batch_size,
        data_parallel_size=point.dp,
        prefix=0,
        num_gpus=point.tp * point.dp,
    )
    return sim.get_last_schedule_trace()


def analyze(real_fingerprint: Path) -> tuple[list[dict[str, object]], list[MetricGate]]:
    validate = _load_validate_module()
    tolerances = _read_tolerances(real_fingerprint)
    rows: list[dict[str, object]] = []
    gates: list[MetricGate] = []
    for scenario in SCENARIOS:
        trace = _run_scenario(validate, scenario)
        scenario_rows, scenario_gates = summarize_trace(scenario, trace, tolerances)
        rows.extend(scenario_rows)
        gates.extend(scenario_gates)
    return rows, gates


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["scenario", "section", "metric", "sim_value", "target_low", "target_high", "passed"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, object]], gates: list[MetricGate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    passed = all(gate.passed for gate in gates)
    lines = [
        "# Phase448 DP admission fingerprint gate",
        "",
        f"结论: 指纹门{'通过' if passed else '未通过'}。`runtime_lockstep_allowed={str(passed).lower()}`。",
        "",
        "| scenario | metric | sim | target low | target high | passed |",
        "|---|---|---:|---:|---:|---|",
    ]
    for gate in gates:
        lines.append(
            f"| {gate.scenario} | {gate.metric} | {gate.sim_value:.3f} | "
            f"{gate.target_low:.3f} | {gate.target_high:.3f} | {gate.passed} |"
        )
    lines.extend([
        "",
        "## Phase Joint Snapshot",
        "",
        "| scenario | phase pair | share |",
        "|---|---|---:|",
    ])
    for row in rows:
        if row["section"] == "phase_joint":
            lines.append(
                f"| {row['scenario']} | {row['metric']} | {float(row['sim_value']):.4f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-fingerprint", type=Path, default=DEFAULT_REAL_FINGERPRINT)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows, gates = analyze(args.real_fingerprint)
    write_csv(args.csv_out, rows)
    write_md(args.md_out, rows, gates)
    print(f"wrote {args.csv_out}")
    print(f"wrote {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
