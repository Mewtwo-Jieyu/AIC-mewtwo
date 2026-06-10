#!/usr/bin/env python3
"""Analyze Phase171 budget mechanism candidates from Phase165 diagnostics."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


EXPECTED_TOPOLOGIES = {"tp8_dp1_ep8", "tp4_dp2_ep8"}
SOURCE = "phase171_budget_mechanism_candidate"

FIELDNAMES = [
    "source",
    "topology_key",
    "shape_key",
    "baseline_breakdown_name",
    "budget_breakdown_name",
    "clean_budget_effect",
    "sim_budget_effect",
    "steady_state_time_ratio",
    "depenalized_budget_effect",
    "raw_error_ratio",
    "depenalized_error_ratio",
    "depenalized_is_closer",
    "exact_gap_upper_bound",
    "baseline_steady_state_time_ms",
    "budget_steady_state_time_ms",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty csv: {path}")
    return rows


def _required(row: dict[str, str], field: str, path: Path) -> str:
    value = row.get(field)
    if value is None or value == "":
        raise ValueError(f"missing {field} in {path}")
    return value


def _as_float(value: str, field: str, path: Path) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be float in {path}: {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{field} must be positive in {path}: {parsed!r}")
    return parsed


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def _expect_flag(row: dict[str, str], field: str, expected: str, path: Path) -> None:
    value = _required(row, field, path).lower()
    if value != expected:
        raise ValueError(f"{field} must be {expected} in {path}: {value!r}")


def _load_gap(path: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(path)
    topology_keys = [_required(row, "topology_key", path) for row in rows]
    duplicates = sorted(
        {topology_key for topology_key in topology_keys if topology_keys.count(topology_key) > 1}
    )
    if duplicates:
        raise ValueError(f"duplicate gap topology_key: {duplicates}")
    seen = set(topology_keys)
    if seen != EXPECTED_TOPOLOGIES:
        raise ValueError(
            "gap topology set mismatch: "
            f"missing={sorted(EXPECTED_TOPOLOGIES - seen)} "
            f"extra={sorted(seen - EXPECTED_TOPOLOGIES)}"
        )

    by_topology: dict[str, dict[str, str]] = {}
    for row in rows:
        _expect_flag(row, "diagnostic_only", "true", path)
        _expect_flag(row, "valid_for_default", "false", path)
        _expect_flag(row, "perf_database", "false", path)
        topology_key = _required(row, "topology_key", path)
        for field in (
            "shape_key",
            "baseline_breakdown_name",
            "budget_breakdown_name",
            "clean_budget_effect",
            "sim_budget_effect",
            "budget_gap",
        ):
            _required(row, field, path)
        by_topology[topology_key] = row
    return by_topology


def _load_breakdown(path: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(path)
    names = [_required(row, "name", path) for row in rows]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate breakdown name: {duplicates}")
    by_name: dict[str, dict[str, str]] = {}
    for row in rows:
        _expect_flag(row, "diagnostic_only", "true", path)
        _expect_flag(row, "valid_for_default", "false", path)
        _expect_flag(row, "perf_database", "false", path)
        name = _required(row, "name", path)
        _required(row, "steady_state_time_ms", path)
        by_name[name] = row
    return by_name


def _ratio(numerator: float, denominator: float, label: str) -> float:
    if denominator <= 0:
        raise ValueError(f"{label} denominator must be positive: {denominator!r}")
    return numerator / denominator


def _is_closer_to_one(candidate_ratio: float, baseline_ratio: float) -> bool:
    return abs(math.log(candidate_ratio)) < abs(math.log(baseline_ratio))


def analyze_mechanism(
    gap_csv: Path,
    budget_breakdown_csv: Path,
) -> list[dict[str, str]]:
    gap_by_topology = _load_gap(Path(gap_csv))
    breakdown_by_name = _load_breakdown(Path(budget_breakdown_csv))
    rows: list[dict[str, str]] = []

    for topology_key in sorted(EXPECTED_TOPOLOGIES):
        gap = gap_by_topology[topology_key]
        baseline_name = _required(gap, "baseline_breakdown_name", Path(gap_csv))
        budget_name = _required(gap, "budget_breakdown_name", Path(gap_csv))
        if baseline_name not in breakdown_by_name:
            raise ValueError(f"missing baseline breakdown row: {baseline_name}")
        if budget_name not in breakdown_by_name:
            raise ValueError(f"missing budget breakdown row: {budget_name}")

        baseline = breakdown_by_name[baseline_name]
        budget = breakdown_by_name[budget_name]
        clean_budget_effect = _as_float(
            _required(gap, "clean_budget_effect", Path(gap_csv)),
            "clean_budget_effect",
            Path(gap_csv),
        )
        sim_budget_effect = _as_float(
            _required(gap, "sim_budget_effect", Path(gap_csv)),
            "sim_budget_effect",
            Path(gap_csv),
        )
        exact_gap_upper_bound = _as_float(
            _required(gap, "budget_gap", Path(gap_csv)),
            "budget_gap",
            Path(gap_csv),
        )
        baseline_steady_ms = _as_float(
            _required(baseline, "steady_state_time_ms", Path(budget_breakdown_csv)),
            "steady_state_time_ms",
            Path(budget_breakdown_csv),
        )
        budget_steady_ms = _as_float(
            _required(budget, "steady_state_time_ms", Path(budget_breakdown_csv)),
            "steady_state_time_ms",
            Path(budget_breakdown_csv),
        )

        steady_ratio = _ratio(
            budget_steady_ms,
            baseline_steady_ms,
            f"steady_state_time_ratio {topology_key}",
        )
        depenalized_effect = sim_budget_effect * steady_ratio
        raw_error_ratio = _ratio(
            sim_budget_effect,
            clean_budget_effect,
            f"raw_error_ratio {topology_key}",
        )
        depenalized_error_ratio = _ratio(
            depenalized_effect,
            clean_budget_effect,
            f"depenalized_error_ratio {topology_key}",
        )

        rows.append(
            {
                "source": SOURCE,
                "topology_key": topology_key,
                "shape_key": _required(gap, "shape_key", Path(gap_csv)),
                "baseline_breakdown_name": baseline_name,
                "budget_breakdown_name": budget_name,
                "clean_budget_effect": _format_float(clean_budget_effect),
                "sim_budget_effect": _format_float(sim_budget_effect),
                "steady_state_time_ratio": _format_float(steady_ratio),
                "depenalized_budget_effect": _format_float(depenalized_effect),
                "raw_error_ratio": _format_float(raw_error_ratio),
                "depenalized_error_ratio": _format_float(depenalized_error_ratio),
                "depenalized_is_closer": str(
                    _is_closer_to_one(depenalized_error_ratio, raw_error_ratio)
                ).lower(),
                "exact_gap_upper_bound": _format_float(exact_gap_upper_bound),
                "baseline_steady_state_time_ms": _format_float(baseline_steady_ms),
                "budget_steady_state_time_ms": _format_float(budget_steady_ms),
                "diagnostic_only": "true",
                "valid_for_default": "false",
                "perf_database": "false",
            }
        )
    return rows


def write_candidate_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if len(rows) != len(EXPECTED_TOPOLOGIES):
        raise ValueError(f"candidate csv must contain {len(EXPECTED_TOPOLOGIES)} rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_candidate_doc(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase171: Budget Mechanism Candidate",
        "",
        "## Decision",
        "",
        "| Item | Result |",
        "|---|---|",
        "| Scope | Diagnostic mechanism candidate only |",
        "| Default AIC | No-Go |",
        "| PerfDatabase | No-Go |",
        "| GPU benchmark | No-Go in Phase171-dev |",
        "| Exact gap upper bound | Diagnostic only, not a model multiplier |",
        "",
        "## Mechanism Check",
        "",
        "| topology_key | clean_budget_effect | sim_budget_effect | steady_state_time_ratio | depenalized_budget_effect | depenalized_error_ratio | exact_gap_upper_bound |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {topology_key} | {clean_budget_effect} | {sim_budget_effect} | "
            "{steady_state_time_ratio} | {depenalized_budget_effect} | "
            "{depenalized_error_ratio} | {exact_gap_upper_bound} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The depenalized estimate removes the high-budget steady-state time penalty "
            "from the existing cb_sim diagnostic output. If it moves closer to the "
            "clean budget effect, the next candidate is a mechanism term around "
            "budget scheduler steady-state cost, not a direct budget_gap multiplier.",
            "",
            "The exact_gap_upper_bound column is retained only as an exact diagnostic "
            "upper bound from Phase165. It is not valid for default AIC, not a "
            "PerfDatabase row, and not an interpolation or extrapolation rule.",
            "",
            "diagnostic_only=true valid_for_default=false perf_database=false",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Phase171 budget mechanism candidates."
    )
    parser.add_argument("--gap-csv", type=Path, required=True)
    parser.add_argument("--budget-breakdown-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--doc-out",
        type=Path,
        default=Path("docs/iter_gap_investigation/phase171_budget_mechanism_candidate.md"),
    )
    args = parser.parse_args()

    rows = analyze_mechanism(args.gap_csv, args.budget_breakdown_csv)
    write_candidate_csv(args.out, rows)
    write_candidate_doc(args.doc_out, rows)
    print(f"wrote_phase171_candidate={args.out}")
    print(f"phase171_candidate_rows={len(rows)}")
    print(f"wrote_phase171_candidate_doc={args.doc_out}")


if __name__ == "__main__":
    main()
