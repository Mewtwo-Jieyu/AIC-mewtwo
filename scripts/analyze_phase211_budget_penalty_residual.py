#!/usr/bin/env python3
"""Analyze Phase178/199 budget penalty residuals."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


SOURCE = "phase211_budget_penalty_residual"
MANIFEST_SOURCE = "phase178_budget_mechanism_holdout"
ANALYSIS_SOURCE = "phase199_holdout_budget_mechanism_analysis"
DEFAULT_READINESS = "No-Go"

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

PAIR_ORDER = [
    ("tp8_dp1_ep8", "isl4000_osl2000_batch128"),
    ("tp4_dp2_ep8", "isl4000_osl2000_batch128"),
    ("tp8_dp1_ep8", "isl12000_osl2000_batch128"),
    ("tp4_dp2_ep8", "isl12000_osl2000_batch128"),
]
EXPECTED_PAIRS = set(PAIR_ORDER)

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
    "clean_budget_effect",
    "clean_delta_from_1",
    "control_steady_state_time_ms",
    "holdout_steady_state_time_ms",
    "steady_state_time_ratio",
    "penalty_overstatement",
    "default_readiness",
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


def _as_any_float(row: dict[str, str], field: str, path: Path) -> float:
    value = _required(row, field, path)
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be float in {path}: {value!r}") from exc


def _expect_flag(row: dict[str, str], field: str, expected: str, path: Path) -> None:
    value = _required(row, field, path).lower()
    if value != expected:
        raise ValueError(f"{field} must be {expected} in {path}: {value!r}")


def _require_close(field: str, actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-5):
        raise ValueError(f"{field} mismatch: got {actual!r}, expected {expected!r}")


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def _manifest_index(rows: list[dict[str, str]], path: Path) -> dict[str, dict[str, str]]:
    if len(rows) != len(EXPECTED_SCENARIOS):
        raise ValueError(f"manifest must contain exactly 8 rows: got {len(rows)}")

    scenarios = [_required(row, "scenario", path) for row in rows]
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
    for row in rows:
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
        _as_float(row, "real_output_tok_s", path)
        _as_float(row, "steady_state_time_ms", path)
        by_scenario[_required(row, "scenario", path)] = row
    return by_scenario


def _validate_analysis_rows(
    rows: list[dict[str, str]],
    path: Path,
    manifest_by_scenario: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    if len(rows) != len(PAIR_ORDER):
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

    by_pair = {(_required(row, "topology_key", path), _required(row, "shape_key", path)): row for row in rows}
    return [by_pair[pair] for pair in PAIR_ORDER]


def _validate_analysis_row(
    row: dict[str, str],
    path: Path,
    manifest_by_scenario: dict[str, dict[str, str]],
) -> dict[str, float | str | int]:
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

    topology_key = _required(row, "topology_key", path)
    shape_key = _required(row, "shape_key", path)
    for manifest_row, scenario_label in ((control, "control"), (holdout, "holdout")):
        if _required(manifest_row, "topology_key", Path("manifest")) != topology_key:
            raise ValueError(f"{scenario_label} topology_key mismatch")
        if _required(manifest_row, "shape_key", Path("manifest")) != shape_key:
            raise ValueError(f"{scenario_label} shape_key mismatch")

    control_output = _as_float(row, "control_output_tok_s", path)
    holdout_output = _as_float(row, "holdout_output_tok_s", path)
    control_steady_ms = _as_float(row, "control_steady_state_time_ms", path)
    holdout_steady_ms = _as_float(row, "holdout_steady_state_time_ms", path)
    clean_ratio = _as_float(row, "clean_high_over_control", path)
    clean_delta = _as_any_float(row, "clean_delta_from_1", path)
    steady_ratio = _as_float(row, "steady_state_time_ratio", path)

    _require_close(
        "control_output_tok_s",
        control_output,
        _as_float(control, "real_output_tok_s", Path("manifest")),
    )
    _require_close(
        "holdout_output_tok_s",
        holdout_output,
        _as_float(holdout, "real_output_tok_s", Path("manifest")),
    )
    _require_close(
        "control_steady_state_time_ms",
        control_steady_ms,
        _as_float(control, "steady_state_time_ms", Path("manifest")),
    )
    _require_close(
        "holdout_steady_state_time_ms",
        holdout_steady_ms,
        _as_float(holdout, "steady_state_time_ms", Path("manifest")),
    )
    _require_close("clean_high_over_control", clean_ratio, holdout_output / control_output)
    _require_close("clean_delta_from_1", clean_delta, clean_ratio - 1.0)
    _require_close("steady_state_time_ratio", steady_ratio, holdout_steady_ms / control_steady_ms)

    return {
        "topology_key": topology_key,
        "shape_key": shape_key,
        "control_scenario": control_scenario,
        "holdout_scenario": holdout_scenario,
        "control_max_bt": _as_int(row, "control_max_bt", path),
        "holdout_max_bt": _as_int(row, "holdout_max_bt", path),
        "control_output_tok_s": control_output,
        "holdout_output_tok_s": holdout_output,
        "clean_budget_effect": clean_ratio,
        "clean_delta_from_1": clean_delta,
        "control_steady_state_time_ms": control_steady_ms,
        "holdout_steady_state_time_ms": holdout_steady_ms,
        "steady_state_time_ratio": steady_ratio,
    }


def analyze_budget_penalty_residual(
    manifest_path: Path | str,
    analysis_path: Path | str,
) -> list[dict[str, str]]:
    manifest_csv = Path(manifest_path)
    analysis_csv = Path(analysis_path)
    manifest_by_scenario = _manifest_index(_read_csv(manifest_csv), manifest_csv)
    ordered_analysis_rows = _validate_analysis_rows(
        _read_csv(analysis_csv),
        analysis_csv,
        manifest_by_scenario,
    )

    rows: list[dict[str, str]] = []
    for row in ordered_analysis_rows:
        validated = _validate_analysis_row(row, analysis_csv, manifest_by_scenario)
        clean_effect = float(validated["clean_budget_effect"])
        steady_ratio = float(validated["steady_state_time_ratio"])
        rows.append(
            {
                "source": SOURCE,
                "topology_key": str(validated["topology_key"]),
                "shape_key": str(validated["shape_key"]),
                "control_scenario": str(validated["control_scenario"]),
                "holdout_scenario": str(validated["holdout_scenario"]),
                "control_max_bt": str(validated["control_max_bt"]),
                "holdout_max_bt": str(validated["holdout_max_bt"]),
                "control_output_tok_s": _format_float(float(validated["control_output_tok_s"])),
                "holdout_output_tok_s": _format_float(float(validated["holdout_output_tok_s"])),
                "clean_budget_effect": _format_float(clean_effect),
                "clean_delta_from_1": _format_float(float(validated["clean_delta_from_1"])),
                "control_steady_state_time_ms": _format_float(
                    float(validated["control_steady_state_time_ms"])
                ),
                "holdout_steady_state_time_ms": _format_float(
                    float(validated["holdout_steady_state_time_ms"])
                ),
                "steady_state_time_ratio": _format_float(steady_ratio),
                "penalty_overstatement": _format_float(steady_ratio / clean_effect),
                "default_readiness": DEFAULT_READINESS,
                "diagnostic_only": "true",
                "valid_for_default": "false",
                "perf_database": "false",
            }
        )
    return rows


def write_residual_csv(path: Path | str, rows: list[dict[str, str]]) -> None:
    if len(rows) != len(PAIR_ORDER):
        raise ValueError(f"phase211 residual must contain {len(PAIR_ORDER)} rows")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_residual_doc(path: Path | str, rows: list[dict[str, str]]) -> None:
    if len(rows) != len(PAIR_ORDER):
        raise ValueError(f"phase211 residual doc must contain {len(PAIR_ORDER)} rows")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase211: Budget Penalty Residual",
        "",
        "## Decision",
        "",
        "| Item | Result |",
        "|---|---|",
        "| Default AIC | No-Go |",
        "| VLLMBackend.run_agg | Unchanged |",
        "| PerfDatabase | No-Go |",
        "| GPU benchmark | No-Go in Phase211 |",
        "| Allowed use | diagnostic-only analysis |",
        "| Boundary flags | diagnostic_only=true valid_for_default=false perf_database=false |",
        "",
        "## Pair Residuals",
        "",
        "| topology_key | shape_key | control_bt | holdout_bt | clean_budget_effect | steady_state_time_ratio | penalty_overstatement | default_readiness |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {topology_key} | {shape_key} | {control_max_bt} | {holdout_max_bt} | "
            "{clean_budget_effect} | {steady_state_time_ratio} | "
            "{penalty_overstatement} | {default_readiness} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "All 4 holdout pairs keep clean_budget_effect near 1 while steady_state_time_ratio is much larger.",
            "That means the scheduler penalty is overstated in cb_sim relative to observed clean GPU throughput.",
            "The pair behavior differs by topology and shape, so this is not a global constant.",
            "The next mechanism candidate must stay exact-keyed by topology_key + shape_key + budget pair.",
            "A raw multiplier or direct default AIC integration remains No-Go.",
            "",
        ]
    )
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
        default=Path("docs/iter_gap_investigation/phase211_budget_penalty_residual.csv"),
    )
    parser.add_argument(
        "--doc-out",
        type=Path,
        default=Path("docs/iter_gap_investigation/phase211_budget_penalty_residual.md"),
    )
    args = parser.parse_args()

    rows = analyze_budget_penalty_residual(args.manifest, args.analysis)
    write_residual_csv(args.out, rows)
    write_residual_doc(args.doc_out, rows)
    print(f"wrote_phase211_residual={args.out}")
    print(f"phase211_pair_rows={len(rows)}")
    print(f"default_readiness={DEFAULT_READINESS}")
    print(f"wrote_phase211_interpretation={args.doc_out}")


if __name__ == "__main__":
    main()
