#!/usr/bin/env python3
"""Compare diagnostic budget penalty mechanism hypotheses."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


SOURCE = "phase215_budget_penalty_mechanism"
INPUT_SOURCE = "phase211_budget_penalty_residual"
DEFAULT_READINESS = "No-Go"
RECOMMENDED_BOUNDARY = "diagnostic_only_exact_key"

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
    "clean_budget_effect",
    "steady_state_time_ratio",
    "steady_linear_error",
    "no_budget_penalty_error",
    "topology_shape_spread",
    "recommended_model_boundary",
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
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be int in {path}: {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{field} must be positive in {path}: {parsed!r}")
    return parsed


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


def _symmetric_error(predicted: float, observed: float, label: str) -> float:
    if predicted <= 0 or observed <= 0:
        raise ValueError(f"{label} values must be positive")
    ratio = predicted / observed
    return ratio if ratio >= 1.0 else 1.0 / ratio


def _validate_rows(rows: list[dict[str, str]], path: Path) -> list[dict[str, str]]:
    if len(rows) != len(PAIR_ORDER):
        raise ValueError(f"phase211 residual must contain exactly 4 rows: got {len(rows)}")

    pairs = [(_required(row, "topology_key", path), _required(row, "shape_key", path)) for row in rows]
    duplicates = sorted({pair for pair in pairs if pairs.count(pair) > 1})
    if duplicates:
        raise ValueError(f"duplicate pair: {duplicates}")
    seen_pairs = set(pairs)
    if seen_pairs != EXPECTED_PAIRS:
        raise ValueError(
            "pair set mismatch: "
            f"missing={sorted(EXPECTED_PAIRS - seen_pairs)} "
            f"extra={sorted(seen_pairs - EXPECTED_PAIRS)}"
        )

    by_pair = {(_required(row, "topology_key", path), _required(row, "shape_key", path)): row for row in rows}
    ordered = [by_pair[pair] for pair in PAIR_ORDER]
    for row in ordered:
        source = _required(row, "source", path)
        if source != INPUT_SOURCE:
            raise ValueError(f"source must be {INPUT_SOURCE}: {source!r}")
        if _required(row, "default_readiness", path) != DEFAULT_READINESS:
            raise ValueError(f"default_readiness must be {DEFAULT_READINESS}")
        _expect_flag(row, "diagnostic_only", "true", path)
        _expect_flag(row, "valid_for_default", "false", path)
        _expect_flag(row, "perf_database", "false", path)

        control_bt = _as_int(row, "control_max_bt", path)
        holdout_bt = _as_int(row, "holdout_max_bt", path)
        if holdout_bt != 65536:
            raise ValueError(f"holdout_max_bt must be 65536: {holdout_bt}")
        if control_bt >= holdout_bt:
            raise ValueError(f"control_max_bt must be below holdout_max_bt: {control_bt}")

        control_output = _as_float(row, "control_output_tok_s", path)
        holdout_output = _as_float(row, "holdout_output_tok_s", path)
        clean_effect = _as_float(row, "clean_budget_effect", path)
        clean_delta = _as_any_float(row, "clean_delta_from_1", path)
        control_steady = _as_float(row, "control_steady_state_time_ms", path)
        holdout_steady = _as_float(row, "holdout_steady_state_time_ms", path)
        steady_ratio = _as_float(row, "steady_state_time_ratio", path)
        penalty_overstatement = _as_float(row, "penalty_overstatement", path)

        _require_close("clean_budget_effect", clean_effect, holdout_output / control_output)
        _require_close("clean_delta_from_1", clean_delta, clean_effect - 1.0)
        _require_close("steady_state_time_ratio", steady_ratio, holdout_steady / control_steady)
        _require_close("penalty_overstatement", penalty_overstatement, steady_ratio / clean_effect)
    return ordered


def _topology_spreads(rows: list[dict[str, str]], path: Path) -> dict[str, float]:
    by_topology: dict[str, list[float]] = {}
    for row in rows:
        by_topology.setdefault(_required(row, "topology_key", path), []).append(
            _as_float(row, "clean_budget_effect", path)
        )
    spreads: dict[str, float] = {}
    for topology, effects in by_topology.items():
        if len(effects) != 2:
            raise ValueError(f"topology must have exactly 2 shapes: {topology}")
        spreads[topology] = max(effects) / min(effects)
    return spreads


def analyze_budget_penalty_mechanism(input_path: Path | str) -> list[dict[str, str]]:
    path = Path(input_path)
    input_rows = _validate_rows(_read_csv(path), path)
    spreads = _topology_spreads(input_rows, path)

    rows: list[dict[str, str]] = []
    for row in input_rows:
        topology_key = _required(row, "topology_key", path)
        clean_effect = _as_float(row, "clean_budget_effect", path)
        steady_ratio = _as_float(row, "steady_state_time_ratio", path)
        rows.append(
            {
                "source": SOURCE,
                "topology_key": topology_key,
                "shape_key": _required(row, "shape_key", path),
                "control_scenario": _required(row, "control_scenario", path),
                "holdout_scenario": _required(row, "holdout_scenario", path),
                "control_max_bt": _required(row, "control_max_bt", path),
                "holdout_max_bt": _required(row, "holdout_max_bt", path),
                "clean_budget_effect": _format_float(clean_effect),
                "steady_state_time_ratio": _format_float(steady_ratio),
                "steady_linear_error": _format_float(
                    _symmetric_error(steady_ratio, clean_effect, "steady_linear_error")
                ),
                "no_budget_penalty_error": _format_float(
                    _symmetric_error(1.0, clean_effect, "no_budget_penalty_error")
                ),
                "topology_shape_spread": _format_float(spreads[topology_key]),
                "recommended_model_boundary": RECOMMENDED_BOUNDARY,
                "default_readiness": DEFAULT_READINESS,
                "diagnostic_only": "true",
                "valid_for_default": "false",
                "perf_database": "false",
            }
        )
    return rows


def write_mechanism_csv(path: Path | str, rows: list[dict[str, str]]) -> None:
    if len(rows) != len(PAIR_ORDER):
        raise ValueError(f"phase215 mechanism must contain {len(PAIR_ORDER)} rows")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_mechanism_doc(path: Path | str, rows: list[dict[str, str]]) -> None:
    if len(rows) != len(PAIR_ORDER):
        raise ValueError(f"phase215 mechanism doc must contain {len(PAIR_ORDER)} rows")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase215: Budget Penalty Mechanism Comparison",
        "",
        "## Decision",
        "",
        "| Item | Result |",
        "|---|---|",
        "| Default AIC | No-Go |",
        "| VLLMBackend.run_agg | Unchanged |",
        "| PerfDatabase | No-Go |",
        "| GPU benchmark | No-Go in Phase215 |",
        "| Recommended boundary | diagnostic_only_exact_key |",
        "| Boundary flags | diagnostic_only=true valid_for_default=false perf_database=false |",
        "",
        "## Mechanism Comparison",
        "",
        "| topology_key | shape_key | clean_budget_effect | steady_state_time_ratio | steady_linear_error | no_budget_penalty_error | topology_shape_spread | recommended_model_boundary |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {topology_key} | {shape_key} | {clean_budget_effect} | "
            "{steady_state_time_ratio} | {steady_linear_error} | "
            "{no_budget_penalty_error} | {topology_shape_spread} | "
            "{recommended_model_boundary} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "high-budget steady ratio is large while clean effect stays near 1.",
            "global constant is No-Go because topology and shape behavior is not uniform.",
            "The next step can only be a diagnostic_only_exact_key candidate, not Default AIC.",
            "",
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("docs/iter_gap_investigation/phase211_budget_penalty_residual.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/iter_gap_investigation/phase215_budget_penalty_mechanism.csv"),
    )
    parser.add_argument(
        "--doc-out",
        type=Path,
        default=Path("docs/iter_gap_investigation/phase215_budget_penalty_mechanism.md"),
    )
    args = parser.parse_args()

    rows = analyze_budget_penalty_mechanism(args.input)
    write_mechanism_csv(args.out, rows)
    write_mechanism_doc(args.doc_out, rows)
    print(f"wrote_phase215_mechanism={args.out}")
    print(f"phase215_pair_rows={len(rows)}")
    print(f"default_readiness={DEFAULT_READINESS}")
    print(f"recommended_model_boundary={RECOMMENDED_BOUNDARY}")
    print(f"wrote_phase215_interpretation={args.doc_out}")


if __name__ == "__main__":
    main()
