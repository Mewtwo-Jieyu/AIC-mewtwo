#!/usr/bin/env python3
"""Build Phase185 scheduler inputs for Phase178 holdout run-one commands."""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from build_phase178_budget_mechanism_holdout_manifest import EXPECTED_SCENARIOS


SOURCE = "phase185_phase178_scheduler_inputs"
FIELDNAMES = [
    "source",
    "scenario",
    "topology_key",
    "shape_key",
    "role",
    "tp",
    "dp",
    "ep",
    "isl",
    "osl",
    "batch_size",
    "max_num_batched_tokens",
    "max_num_seqs",
    "steady_state_time_ms",
    "phase178_env",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def _required(row: dict[str, str], field: str) -> str:
    value = row.get(field)
    if value is None or value == "":
        raise ValueError(f"missing {field} for {row.get('scenario', '<unknown>')}")
    return value


def _as_int(row: dict[str, str], field: str) -> int:
    value = _required(row, field)
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be int for {row['scenario']}: {value!r}") from exc


def _as_positive_float(row: dict[str, str], field: str) -> float:
    value = _required(row, field)
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be float for {row['scenario']}: {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{field} must be positive for {row['scenario']}: {parsed!r}")
    return parsed


def _expect_flag(row: dict[str, str], field: str, expected: str) -> None:
    value = _required(row, field).lower()
    if value != expected:
        raise ValueError(f"{field} must be {expected} for {row['scenario']}: {value!r}")


def _validate_row(row: dict[str, str]) -> dict[str, str]:
    scenario = _required(row, "scenario")
    if scenario not in EXPECTED_SCENARIOS:
        raise ValueError(f"unexpected scenario: {scenario}")
    expected = EXPECTED_SCENARIOS[scenario]

    if _required(row, "source") != SOURCE:
        raise ValueError(f"source must be {SOURCE} for {scenario}")
    for key in ("topology_key", "shape_key", "role"):
        if _required(row, key) != str(expected[key]):
            raise ValueError(f"{key} mismatch for {scenario}")
    for key in ("tp", "dp", "ep", "isl", "osl", "batch_size", "max_num_batched_tokens"):
        if _as_int(row, key) != int(expected[key]):
            raise ValueError(f"{key} mismatch for {scenario}")
    if _as_int(row, "max_num_seqs") <= 0:
        raise ValueError(f"max_num_seqs must be positive for {scenario}")

    steady_ms = _as_positive_float(row, "steady_state_time_ms")
    expected_env = f"PHASE178_STEADY_STATE_TIME_MS={_format_float(steady_ms)}"
    if _required(row, "phase178_env") != expected_env:
        raise ValueError(f"phase178_env mismatch for {scenario}")
    _expect_flag(row, "diagnostic_only", "true")
    _expect_flag(row, "valid_for_default", "false")
    _expect_flag(row, "perf_database", "false")

    return {field: _required(row, field) for field in FIELDNAMES}


def validate_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    scenario_counts = Counter(row.get("scenario", "") for row in rows)
    duplicate_scenarios = sorted(
        scenario for scenario, count in scenario_counts.items() if scenario and count > 1
    )
    if duplicate_scenarios:
        raise ValueError(f"duplicate scenario rows: {duplicate_scenarios}")

    expected_count = len(EXPECTED_SCENARIOS)
    if len(rows) != expected_count:
        raise ValueError(f"scheduler inputs must contain exactly {expected_count} rows")

    seen = {row.get("scenario", "") for row in rows}
    expected = set(EXPECTED_SCENARIOS)
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValueError(f"scheduler input scenario set mismatch: missing={missing} extra={extra}")

    validated = [_validate_row(row) for row in rows]
    validated.sort(key=lambda row: list(EXPECTED_SCENARIOS).index(row["scenario"]))
    return validated


def load_scheduler_inputs(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return validate_rows(list(csv.DictReader(f)))


def _row_from_scheduling(
    scenario: str,
    expected: dict[str, Any],
    scheduling: dict[str, Any],
    max_num_seqs: int,
) -> dict[str, str]:
    if "steady_state_time_ms" not in scheduling:
        raise ValueError(f"missing cb_sim_scheduling.steady_state_time_ms for {scenario}")
    steady_ms = float(scheduling["steady_state_time_ms"])
    if steady_ms <= 0:
        raise ValueError(f"steady_state_time_ms must be positive for {scenario}: {steady_ms!r}")
    steady = _format_float(steady_ms)
    return {
        "source": SOURCE,
        "scenario": scenario,
        "topology_key": str(expected["topology_key"]),
        "shape_key": str(expected["shape_key"]),
        "role": str(expected["role"]),
        "tp": str(expected["tp"]),
        "dp": str(expected["dp"]),
        "ep": str(expected["ep"]),
        "isl": str(expected["isl"]),
        "osl": str(expected["osl"]),
        "batch_size": str(expected["batch_size"]),
        "max_num_batched_tokens": str(expected["max_num_batched_tokens"]),
        "max_num_seqs": str(max_num_seqs),
        "steady_state_time_ms": steady,
        "phase178_env": f"PHASE178_STEADY_STATE_TIME_MS={steady}",
        "diagnostic_only": "true",
        "valid_for_default": "false",
        "perf_database": "false",
    }


def build_scheduler_inputs(
    *,
    overlap_factor: float = 0.0,
    ep8_per_iteration_overhead_ms: float = 90.0,
    max_num_seqs: int = 256,
) -> list[dict[str, str]]:
    from aiconfigurator.sdk import common
    from aiconfigurator.sdk.backends.cb_simulator import CBSimConfig
    from aiconfigurator.sdk.config import RuntimeConfig

    from validate_cb_simulator import _load_model_and_db

    loaded: dict[tuple[int, int, int, int], tuple[Any, Any, Any]] = {}
    rows: list[dict[str, str]] = []
    for scenario, expected in EXPECTED_SCENARIOS.items():
        tp = int(expected["tp"])
        dp = int(expected["dp"])
        ep = int(expected["ep"])
        key = (tp, dp, 1, ep)
        if key not in loaded:
            loaded[key] = _load_model_and_db(tp=tp, dp=dp, moe_tp=1, moe_ep=ep)
        model, db, backend = loaded[key]
        concurrency = int(expected["batch_size"])
        cb_config = CBSimConfig(
            max_num_batched_tokens=int(expected["max_num_batched_tokens"]),
            max_num_seqs=max_num_seqs,
            num_requests=max(200, concurrency * 3),
            warmup_requests=max(50, concurrency),
            overlap_factor=overlap_factor,
            per_iteration_overhead_ms=ep8_per_iteration_overhead_ms,
        )
        cb_summary = backend.run_agg(
            model,
            db,
            RuntimeConfig(
                batch_size=int(expected["batch_size"]),
                isl=int(expected["isl"]),
                osl=int(expected["osl"]),
            ),
            ctx_tokens=int(expected["max_num_batched_tokens"]),
            database_mode=common.DatabaseMode.HYBRID,
            method="cb_sim",
            cb_config=cb_config,
        )
        scheduling = cb_summary.get_per_ops_data().get("cb_sim_scheduling")
        if not isinstance(scheduling, dict):
            raise ValueError(f"missing cb_sim_scheduling for {scenario}")
        rows.append(_row_from_scheduling(scenario, expected, scheduling, max_num_seqs))
    return validate_rows(rows)


def write_scheduler_inputs(path: Path, rows: list[dict[str, str]]) -> None:
    validated = validate_rows(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(validated)


def write_scheduler_inputs_doc(path: Path, rows: list[dict[str, str]]) -> None:
    validated = validate_rows(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase185: Phase178 Scheduler Inputs",
        "",
        "## Decision",
        "",
        "| Item | Result |",
        "|---|---|",
        "| Scope | Local cb_sim scheduler input table for Phase178 holdout run-one |",
        "| Data source | `cb_sim_scheduling.steady_state_time_ms` from cb_sim per-ops data |",
        "| GPU benchmark | No-Go in Phase185 |",
        "| Default AIC | No-Go |",
        "| PerfDatabase | No-Go |",
        "",
        "## Usage",
        "",
        "Use the row for the scenario being run and export the exact env value before `run-one`:",
        "",
        "```bash",
        "PHASE178_STEADY_STATE_TIME_MS=<csv steady_state_time_ms> \\",
        "bash collector/vllm/run_phase178_budget_mechanism_holdout.sh run-one <scenario>",
        "```",
        "",
        "## Scheduler Inputs",
        "",
        "| scenario | topology_key | shape_key | max_bt | steady_state_time_ms | env |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in validated:
        lines.append(
            "| {scenario} | {topology_key} | {shape_key} | "
            "{max_num_batched_tokens} | {steady_state_time_ms} | `{phase178_env}` |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "These rows are scheduler diagnostic inputs only. They are not clean GPU timing,",
            "not default cb_sim data, and not PerfDatabase rows.",
            "",
            "diagnostic_only=true valid_for_default=false perf_database=false",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build Phase185 scheduler inputs for Phase178 holdout scenarios."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/iter_gap_investigation/phase185_phase178_scheduler_inputs.csv"),
    )
    parser.add_argument(
        "--doc-out",
        type=Path,
        default=Path("docs/iter_gap_investigation/phase185_phase178_scheduler_inputs.md"),
    )
    parser.add_argument("--overlap-factor", type=float, default=0.0)
    parser.add_argument("--ep8-per-iteration-overhead-ms", type=float, default=90.0)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    args = parser.parse_args(argv)

    rows = build_scheduler_inputs(
        overlap_factor=args.overlap_factor,
        ep8_per_iteration_overhead_ms=args.ep8_per_iteration_overhead_ms,
        max_num_seqs=args.max_num_seqs,
    )
    write_scheduler_inputs(args.out, rows)
    write_scheduler_inputs_doc(args.doc_out, rows)
    print(f"wrote_phase185_scheduler_inputs={args.out}")
    print(f"wrote_phase185_scheduler_doc={args.doc_out}")
    print(f"phase185_scheduler_input_rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
