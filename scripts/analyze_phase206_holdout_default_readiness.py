#!/usr/bin/env python3
"""Analyze whether Phase199 holdout evidence is ready for default AIC."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


MANIFEST_SOURCE = "phase178_budget_mechanism_holdout"
ANALYSIS_SOURCE = "phase199_holdout_budget_mechanism_analysis"
EXPECTED_SCENARIOS = {
    "tp8ep8-4k2k-bt4000",
    "tp8ep8-4k2k-bt65536",
    "tp4dp2ep8-4k2k-bt4000",
    "tp4dp2ep8-4k2k-bt65536",
    "tp8ep8-12k2k-bt12000",
    "tp8ep8-12k2k-bt65536",
    "tp4dp2ep8-12k2k-bt12000",
    "tp4dp2ep8-12k2k-bt65536",
}
EXPECTED_PAIRS = {
    ("tp8_dp1_ep8", "isl4000_osl2000_batch128"),
    ("tp4_dp2_ep8", "isl4000_osl2000_batch128"),
    ("tp8_dp1_ep8", "isl12000_osl2000_batch128"),
    ("tp4_dp2_ep8", "isl12000_osl2000_batch128"),
}


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
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be float in {path}: {value!r}") from exc


def _as_positive_float(row: dict[str, str], field: str, path: Path) -> float:
    value = _as_float(row, field, path)
    if value <= 0:
        raise ValueError(f"{field} must be positive in {path}: {value!r}")
    return value


def _expect_flag(row: dict[str, str], field: str, expected: str, path: Path) -> None:
    value = _required(row, field, path).lower()
    if value != expected:
        raise ValueError(f"{field} must be {expected} in {path}: {value!r}")


def _require_close(field: str, actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-5):
        raise ValueError(f"{field} mismatch: got {actual!r}, expected {expected!r}")


def _scenario_index(manifest_rows: list[dict[str, str]], path: Path) -> dict[str, dict[str, str]]:
    if len(manifest_rows) != 8:
        raise ValueError(f"manifest must contain exactly 8 rows: got {len(manifest_rows)}")
    scenarios = [_required(row, "scenario", path) for row in manifest_rows]
    duplicates = sorted({scenario for scenario in scenarios if scenarios.count(scenario) > 1})
    if duplicates:
        raise ValueError(f"duplicate manifest scenario: {duplicates}")
    seen = set(scenarios)
    if seen != EXPECTED_SCENARIOS:
        raise ValueError(
            "manifest scenario set mismatch: "
            f"missing={sorted(EXPECTED_SCENARIOS - seen)} "
            f"extra={sorted(seen - EXPECTED_SCENARIOS)}"
        )

    by_scenario: dict[str, dict[str, str]] = {}
    for row in manifest_rows:
        source = _required(row, "source", path)
        if source != MANIFEST_SOURCE:
            raise ValueError(f"manifest source must be {MANIFEST_SOURCE}: {source!r}")
        _expect_flag(row, "diagnostic_only", "true", path)
        _expect_flag(row, "valid_for_default", "false", path)
        _expect_flag(row, "perf_database", "false", path)
        if _as_int(row, "request_success_count", path) != 128:
            raise ValueError("request_success_count must be 128")
        if _as_int(row, "request_fail_count", path) != 0:
            raise ValueError("request_fail_count must be 0")
        _as_positive_float(row, "real_output_tok_s", path)
        _as_positive_float(row, "steady_state_time_ms", path)
        by_scenario[_required(row, "scenario", path)] = row
    return by_scenario


def _validate_analysis_rows(
    rows: list[dict[str, str]],
    path: Path,
    manifest_by_scenario: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    if len(rows) != 4:
        raise ValueError(f"analysis must contain exactly 4 rows: got {len(rows)}")

    pairs = [(_required(row, "topology_key", path), _required(row, "shape_key", path)) for row in rows]
    duplicates = sorted({pair for pair in pairs if pairs.count(pair) > 1})
    if duplicates:
        raise ValueError(f"duplicate analysis pair: {duplicates}")
    seen_pairs = set(pairs)
    if seen_pairs != EXPECTED_PAIRS:
        raise ValueError(
            "analysis pair set mismatch: "
            f"missing={sorted(EXPECTED_PAIRS - seen_pairs)} "
            f"extra={sorted(seen_pairs - EXPECTED_PAIRS)}"
        )

    validated: list[dict[str, Any]] = []
    for row in rows:
        source = _required(row, "source", path)
        if source != ANALYSIS_SOURCE:
            raise ValueError(f"analysis source must be {ANALYSIS_SOURCE}: {source!r}")
        _expect_flag(row, "diagnostic_only", "true", path)
        _expect_flag(row, "valid_for_default", "false", path)
        _expect_flag(row, "perf_database", "false", path)

        control_scenario = _required(row, "control_scenario", path)
        holdout_scenario = _required(row, "holdout_scenario", path)
        if control_scenario not in manifest_by_scenario:
            raise ValueError(f"missing manifest control scenario: {control_scenario}")
        if holdout_scenario not in manifest_by_scenario:
            raise ValueError(f"missing manifest holdout scenario: {holdout_scenario}")
        control = manifest_by_scenario[control_scenario]
        holdout = manifest_by_scenario[holdout_scenario]

        control_output = _as_positive_float(row, "control_output_tok_s", path)
        holdout_output = _as_positive_float(row, "holdout_output_tok_s", path)
        control_steady_ms = _as_positive_float(row, "control_steady_state_time_ms", path)
        holdout_steady_ms = _as_positive_float(row, "holdout_steady_state_time_ms", path)
        clean_ratio = _as_positive_float(row, "clean_high_over_control", path)
        clean_delta = _as_float(row, "clean_delta_from_1", path)
        steady_ratio = _as_positive_float(row, "steady_state_time_ratio", path)

        _require_close(
            "control_output_tok_s",
            control_output,
            _as_positive_float(control, "real_output_tok_s", Path("manifest")),
        )
        _require_close(
            "holdout_output_tok_s",
            holdout_output,
            _as_positive_float(holdout, "real_output_tok_s", Path("manifest")),
        )
        _require_close(
            "control_steady_state_time_ms",
            control_steady_ms,
            _as_positive_float(control, "steady_state_time_ms", Path("manifest")),
        )
        _require_close(
            "holdout_steady_state_time_ms",
            holdout_steady_ms,
            _as_positive_float(holdout, "steady_state_time_ms", Path("manifest")),
        )
        _require_close("clean_high_over_control", clean_ratio, holdout_output / control_output)
        _require_close("clean_delta_from_1", clean_delta, clean_ratio - 1.0)
        _require_close("steady_state_time_ratio", steady_ratio, holdout_steady_ms / control_steady_ms)

        validated.append(
            {
                "topology_key": _required(row, "topology_key", path),
                "shape_key": _required(row, "shape_key", path),
                "control_max_bt": _as_int(row, "control_max_bt", path),
                "holdout_max_bt": _as_int(row, "holdout_max_bt", path),
                "clean_high_over_control": clean_ratio,
                "clean_delta_from_1": clean_delta,
                "steady_state_time_ratio": steady_ratio,
            }
        )
    return validated


def analyze_default_readiness(manifest_path: Path | str, analysis_path: Path | str) -> dict[str, Any]:
    manifest_csv = Path(manifest_path)
    analysis_csv = Path(analysis_path)
    manifest_by_scenario = _scenario_index(_read_csv(manifest_csv), manifest_csv)
    pairs = _validate_analysis_rows(_read_csv(analysis_csv), analysis_csv, manifest_by_scenario)

    by_topology: dict[str, list[dict[str, Any]]] = {}
    for row in pairs:
        by_topology.setdefault(row["topology_key"], []).append(row)
    shape_behavior_inconsistent = any(
        len(rows) == 2
        and abs(rows[0]["clean_high_over_control"] - rows[1]["clean_high_over_control"]) > 0.01
        for rows in by_topology.values()
    )
    steady_penalty_supported = any(
        0.95 <= row["clean_high_over_control"] <= 1.15
        and row["steady_state_time_ratio"] >= 2.0
        for row in pairs
    )
    tp4dp2_12k2k_below_one = any(
        row["topology_key"] == "tp4_dp2_ep8"
        and row["shape_key"] == "isl12000_osl2000_batch128"
        and row["clean_high_over_control"] < 1.0
        for row in pairs
    )

    reasons = [
        "Only 4 holdout pairs are available, so default AIC is No-Go.",
        "Large steady-state ratios with clean ratios near 1 support a diagnostic penalty hypothesis.",
        "4k2k and 12k2k behavior differs, so there is no global constant.",
        "tp4dp2 12k2k is below 1, so a raw multiplier is not allowed.",
        "Phase200 remains a diagnostic-only lookup, not default model logic.",
    ]
    return {
        "default_readiness": "No-Go",
        "pair_count": len(pairs),
        "manifest_row_count": len(manifest_by_scenario),
        "analysis_row_count": len(pairs),
        "flags": ("true", "false", "false"),
        "default_aic_allowed": False,
        "global_constant_allowed": False,
        "raw_multiplier_allowed": False,
        "diagnostic_lookup_only": True,
        "shape_behavior_inconsistent": shape_behavior_inconsistent,
        "steady_penalty_supported": steady_penalty_supported,
        "tp4dp2_12k2k_below_one": tp4dp2_12k2k_below_one,
        "pairs": pairs,
        "reasons": reasons,
    }


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def write_readiness_doc(path: Path | str, result: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase206: Holdout Default Readiness Audit",
        "",
        f"default_readiness={result['default_readiness']}",
        "",
        "## Decision",
        "",
        "| Item | Result |",
        "|---|---|",
        "| Default AIC | No-Go |",
        "| VLLMBackend.run_agg | Unchanged |",
        "| PerfDatabase | No-Go |",
        "| GPU benchmark | No-Go |",
        "| Allowed use | diagnostic-only lookup |",
        "",
        "## Readiness Gates",
        "",
        "| Gate | Result |",
        "|---|---|",
        f"| Holdout pair count | {result['pair_count']} |",
        "| Evidence sufficiency | Only 4 holdout pairs; default No-Go |",
        f"| Steady penalty diagnosis | {str(result['steady_penalty_supported']).lower()} |",
        f"| Shape behavior differs | {str(result['shape_behavior_inconsistent']).lower()}; no global constant |",
        f"| tp4dp2 12k2k below 1 | {str(result['tp4dp2_12k2k_below_one']).lower()}; raw multiplier disallowed |",
        "| Exact-key candidate | diagnostic-only lookup |",
        "",
        "## Pair Evidence",
        "",
        "| topology_key | shape_key | control_bt | holdout_bt | clean_high_over_control | steady_state_time_ratio |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in result["pairs"]:
        lines.append(
            "| {topology_key} | {shape_key} | {control_max_bt} | {holdout_max_bt} | "
            "{clean} | {steady} |".format(
                topology_key=row["topology_key"],
                shape_key=row["shape_key"],
                control_max_bt=row["control_max_bt"],
                holdout_max_bt=row["holdout_max_bt"],
                clean=_format_float(row["clean_high_over_control"]),
                steady=_format_float(row["steady_state_time_ratio"]),
            )
        )
    lines.extend(["", "## Reasons", ""])
    for reason in result["reasons"]:
        lines.append(f"- {reason}")
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/iter_gap_investigation/phase178_budget_mechanism_holdout_manifest.csv"),
    )
    parser.add_argument(
        "--analysis",
        type=Path,
        default=Path("docs/iter_gap_investigation/phase199_holdout_budget_mechanism_analysis.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/iter_gap_investigation/phase206_holdout_default_readiness.md"),
    )
    args = parser.parse_args()
    result = analyze_default_readiness(args.manifest, args.analysis)
    write_readiness_doc(args.out, result)
    print(f"default_readiness={result['default_readiness']}")
    print(f"phase206_pair_count={result['pair_count']}")
    print(f"wrote_phase206_readiness={args.out}")


if __name__ == "__main__":
    main()
