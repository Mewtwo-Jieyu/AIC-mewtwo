#!/usr/bin/env python3
"""Analyze Phase178 holdout budget mechanism pairs."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


SOURCE = "phase199_holdout_budget_mechanism_analysis"
INPUT_SOURCE = "phase178_budget_mechanism_holdout"

EXPECTED_SCENARIOS = {
    "tp8ep8-4k2k-bt4000": {
        "topology_key": "tp8_dp1_ep8",
        "shape_key": "isl4000_osl2000_batch128",
        "role": "control",
        "tp": 8,
        "dp": 1,
        "ep": 8,
        "isl": 4000,
        "osl": 2000,
        "batch_size": 128,
        "max_num_batched_tokens": 4000,
    },
    "tp8ep8-4k2k-bt65536": {
        "topology_key": "tp8_dp1_ep8",
        "shape_key": "isl4000_osl2000_batch128",
        "role": "holdout",
        "tp": 8,
        "dp": 1,
        "ep": 8,
        "isl": 4000,
        "osl": 2000,
        "batch_size": 128,
        "max_num_batched_tokens": 65536,
    },
    "tp4dp2ep8-4k2k-bt4000": {
        "topology_key": "tp4_dp2_ep8",
        "shape_key": "isl4000_osl2000_batch128",
        "role": "control",
        "tp": 4,
        "dp": 2,
        "ep": 8,
        "isl": 4000,
        "osl": 2000,
        "batch_size": 128,
        "max_num_batched_tokens": 4000,
    },
    "tp4dp2ep8-4k2k-bt65536": {
        "topology_key": "tp4_dp2_ep8",
        "shape_key": "isl4000_osl2000_batch128",
        "role": "holdout",
        "tp": 4,
        "dp": 2,
        "ep": 8,
        "isl": 4000,
        "osl": 2000,
        "batch_size": 128,
        "max_num_batched_tokens": 65536,
    },
    "tp8ep8-12k2k-bt12000": {
        "topology_key": "tp8_dp1_ep8",
        "shape_key": "isl12000_osl2000_batch128",
        "role": "control",
        "tp": 8,
        "dp": 1,
        "ep": 8,
        "isl": 12000,
        "osl": 2000,
        "batch_size": 128,
        "max_num_batched_tokens": 12000,
    },
    "tp8ep8-12k2k-bt65536": {
        "topology_key": "tp8_dp1_ep8",
        "shape_key": "isl12000_osl2000_batch128",
        "role": "holdout",
        "tp": 8,
        "dp": 1,
        "ep": 8,
        "isl": 12000,
        "osl": 2000,
        "batch_size": 128,
        "max_num_batched_tokens": 65536,
    },
    "tp4dp2ep8-12k2k-bt12000": {
        "topology_key": "tp4_dp2_ep8",
        "shape_key": "isl12000_osl2000_batch128",
        "role": "control",
        "tp": 4,
        "dp": 2,
        "ep": 8,
        "isl": 12000,
        "osl": 2000,
        "batch_size": 128,
        "max_num_batched_tokens": 12000,
    },
    "tp4dp2ep8-12k2k-bt65536": {
        "topology_key": "tp4_dp2_ep8",
        "shape_key": "isl12000_osl2000_batch128",
        "role": "holdout",
        "tp": 4,
        "dp": 2,
        "ep": 8,
        "isl": 12000,
        "osl": 2000,
        "batch_size": 128,
        "max_num_batched_tokens": 65536,
    },
}

PAIR_ORDER = [
    ("tp8_dp1_ep8", "isl4000_osl2000_batch128"),
    ("tp4_dp2_ep8", "isl4000_osl2000_batch128"),
    ("tp8_dp1_ep8", "isl12000_osl2000_batch128"),
    ("tp4_dp2_ep8", "isl12000_osl2000_batch128"),
]

FIELDNAMES = [
    "source",
    "topology_key",
    "shape_key",
    "control_scenario",
    "holdout_scenario",
    "control_max_bt",
    "holdout_max_bt",
    "control_output_tok_s",
    "holdout_output_tok_s",
    "clean_high_over_control",
    "clean_delta_from_1",
    "control_steady_state_time_ms",
    "holdout_steady_state_time_ms",
    "steady_state_time_ratio",
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


def _as_int(row: dict[str, str], field: str, path: Path) -> int:
    value = _required(row, field, path)
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be int in {path}: {value!r}") from exc


def _as_float(row: dict[str, str], field: str, path: Path) -> float:
    value = _required(row, field, path)
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be float in {path}: {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{field} must be positive in {path}: {parsed!r}")
    return parsed


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def _expect_str(row: dict[str, str], field: str, expected: str, path: Path) -> None:
    value = _required(row, field, path)
    if value != expected:
        raise ValueError(f"{field} must be {expected!r} in {path}: {value!r}")


def _expect_int(row: dict[str, str], field: str, expected: int, path: Path) -> None:
    value = _as_int(row, field, path)
    if value != expected:
        raise ValueError(f"{field} must be {expected} in {path}: {value!r}")


def _expect_flag(row: dict[str, str], field: str, expected: str, path: Path) -> None:
    value = _required(row, field, path).lower()
    if value != expected:
        raise ValueError(f"{field} must be {expected} in {path}: {value!r}")


def _validate_row(row: dict[str, str], path: Path) -> None:
    scenario = _required(row, "scenario", path)
    expected = EXPECTED_SCENARIOS.get(scenario)
    if expected is None:
        raise ValueError(f"unexpected scenario in {path}: {scenario!r}")

    _expect_str(row, "source", INPUT_SOURCE, path)
    _expect_flag(row, "diagnostic_only", "true", path)
    _expect_flag(row, "valid_for_default", "false", path)
    _expect_flag(row, "perf_database", "false", path)
    _expect_int(row, "request_success_count", 128, path)
    _expect_int(row, "request_fail_count", 0, path)

    for field in ("topology_key", "shape_key", "role"):
        expected_value = str(expected[field])
        value = _required(row, field, path)
        if value != expected_value:
            raise ValueError(
                f"control/holdout pair {field} mismatch for {scenario}: "
                f"expected {expected_value!r}, got {value!r}"
            )
    for field in (
        "tp",
        "dp",
        "ep",
        "isl",
        "osl",
        "batch_size",
        "max_num_batched_tokens",
    ):
        _expect_int(row, field, int(expected[field]), path)

    _as_float(row, "real_output_tok_s", path)
    _as_float(row, "steady_state_time_ms", path)


def _ratio(numerator: float, denominator: float, label: str) -> float:
    if denominator <= 0:
        raise ValueError(f"{label} denominator must be positive: {denominator!r}")
    return numerator / denominator


def analyze_holdout_budget_mechanism(manifest_path: Path) -> list[dict[str, str]]:
    path = Path(manifest_path)
    manifest_rows = _read_csv(path)
    if len(manifest_rows) != len(EXPECTED_SCENARIOS):
        raise ValueError(
            f"manifest must contain exactly 8 rows: got {len(manifest_rows)} in {path}"
        )

    scenarios = [_required(row, "scenario", path) for row in manifest_rows]
    duplicates = sorted({scenario for scenario in scenarios if scenarios.count(scenario) > 1})
    if duplicates:
        raise ValueError(f"duplicate scenario in {path}: {duplicates}")
    seen = set(scenarios)
    expected = set(EXPECTED_SCENARIOS)
    if seen != expected:
        raise ValueError(
            "scenario set mismatch: "
            f"missing={sorted(expected - seen)} extra={sorted(seen - expected)}"
        )

    by_pair_role: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in manifest_rows:
        _validate_row(row, path)
        pair_key = (_required(row, "topology_key", path), _required(row, "shape_key", path))
        role = _required(row, "role", path)
        roles = by_pair_role.setdefault(pair_key, {})
        if role in roles:
            raise ValueError(f"duplicate {role} row for control/holdout pair {pair_key}")
        roles[role] = row

    rows: list[dict[str, str]] = []
    for pair_key in PAIR_ORDER:
        roles = by_pair_role.get(pair_key)
        if roles is None or set(roles) != {"control", "holdout"}:
            raise ValueError(f"missing control/holdout pair for {pair_key}")
        control = roles["control"]
        holdout = roles["holdout"]

        control_output = _as_float(control, "real_output_tok_s", path)
        holdout_output = _as_float(holdout, "real_output_tok_s", path)
        control_steady_ms = _as_float(control, "steady_state_time_ms", path)
        holdout_steady_ms = _as_float(holdout, "steady_state_time_ms", path)
        clean_ratio = _ratio(holdout_output, control_output, f"clean_high_over_control {pair_key}")
        steady_ratio = _ratio(
            holdout_steady_ms,
            control_steady_ms,
            f"steady_state_time_ratio {pair_key}",
        )

        rows.append(
            {
                "source": SOURCE,
                "topology_key": pair_key[0],
                "shape_key": pair_key[1],
                "control_scenario": _required(control, "scenario", path),
                "holdout_scenario": _required(holdout, "scenario", path),
                "control_max_bt": _required(control, "max_num_batched_tokens", path),
                "holdout_max_bt": _required(holdout, "max_num_batched_tokens", path),
                "control_output_tok_s": _format_float(control_output),
                "holdout_output_tok_s": _format_float(holdout_output),
                "clean_high_over_control": _format_float(clean_ratio),
                "clean_delta_from_1": _format_float(clean_ratio - 1.0),
                "control_steady_state_time_ms": _format_float(control_steady_ms),
                "holdout_steady_state_time_ms": _format_float(holdout_steady_ms),
                "steady_state_time_ratio": _format_float(steady_ratio),
                "diagnostic_only": "true",
                "valid_for_default": "false",
                "perf_database": "false",
            }
        )
    return rows


def write_analysis_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if len(rows) != len(PAIR_ORDER):
        raise ValueError(f"phase199 analysis must contain {len(PAIR_ORDER)} rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_interpretation_doc(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase199: Holdout Budget Mechanism Interpretation",
        "",
        "## Decision",
        "",
        "| Item | Result |",
        "|---|---|",
        "| Scope | diagnostic mechanism candidate only |",
        "| Default AIC | No-Go |",
        "| PerfDatabase | No-Go |",
        "| GPU benchmark | No-Go in Phase199 |",
        "| Boundary flags | diagnostic_only=true valid_for_default=false perf_database=false |",
        "",
        "## Pair Analysis",
        "",
        "| topology_key | shape_key | control_max_bt | holdout_max_bt | clean_high_over_control | clean_delta_from_1 | steady_state_time_ratio |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {topology_key} | {shape_key} | {control_max_bt} | {holdout_max_bt} | "
            "{clean_high_over_control} | {clean_delta_from_1} | "
            "{steady_state_time_ratio} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The holdout rows show very large steady-state time ratios while clean "
            "high/control throughput stays near 1.0 or above 1.0 in three of four pairs.",
            "That supports the diagnostic hypothesis that cb_sim over-penalizes "
            "high-budget steady-state scheduler cost.",
            "The tp4_dp2_ep8 12k2k pair is below 1.0 by about 0.6%, which is too small "
            "and too shape-specific to justify a raw budget_gap multiplier.",
            "The 4k2k and 12k2k behavior differs, so any future mechanism item must be "
            "keyed by topology_key + shape_key + max_bt; it is not a global constant.",
            "These results remain diagnostic-only and are not valid for Default AIC or PerfDatabase.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--doc-out",
        type=Path,
        default=Path(
            "docs/iter_gap_investigation/phase199_holdout_budget_mechanism_interpretation.md"
        ),
    )
    args = parser.parse_args()

    rows = analyze_holdout_budget_mechanism(args.manifest)
    write_analysis_csv(args.out, rows)
    write_interpretation_doc(args.doc_out, rows)
    print(f"wrote_phase199_analysis={args.out}")
    print(f"phase199_pair_rows={len(rows)}")
    print(f"wrote_phase199_interpretation={args.doc_out}")


if __name__ == "__main__":
    main()
