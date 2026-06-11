#!/usr/bin/env python3
"""Final diagnostic-only default readiness audit for Phase222 evidence."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


SOURCE = "phase228_default_readiness_final"
INPUT_SOURCE = "phase222_budget_penalty_evidence_family"
DEFAULT_READINESS = "No-Go"
RECOMMENDED_BOUNDARY = "diagnostic_only_exact_key"
REASON = "no_default_due_to_shape_topology_spread"
EXPECTED_TOPOLOGY_COUNT = 2
EXPECTED_SHAPE_COUNT_PER_TOPOLOGY = 3
EXPECTED_ROW_COUNT = EXPECTED_TOPOLOGY_COUNT * EXPECTED_SHAPE_COUNT_PER_TOPOLOGY

FIELDNAMES = [
    "source",
    "evidence_rows",
    "topology_count",
    "shape_count_per_topology",
    "max_clean_effect_spread",
    "min_clean_effect",
    "max_clean_effect",
    "default_readiness",
    "recommended_boundary",
    "reason",
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


def _expect_flag(row: dict[str, str], field: str, expected: str, path: Path) -> None:
    value = _required(row, field, path).lower()
    if value != expected:
        raise ValueError(f"{field} must be {expected} in {path}: {value!r}")


def _require_close(field: str, actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-5):
        raise ValueError(f"{field} mismatch: got {actual!r}, expected {expected!r}")


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def _validate_evidence_rows(rows: list[dict[str, str]], path: Path) -> list[dict[str, str]]:
    if len(rows) != EXPECTED_ROW_COUNT:
        raise ValueError(
            "phase222 evidence family must contain exactly 6 rows: "
            f"got {len(rows)}"
        )

    pairs = [(_required(row, "topology_key", path), _required(row, "shape_key", path)) for row in rows]
    duplicates = sorted({pair for pair in pairs if pairs.count(pair) > 1})
    if duplicates:
        raise ValueError(f"duplicate topology/shape evidence: {duplicates}")

    by_topology: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        source = _required(row, "source", path)
        if source != INPUT_SOURCE:
            raise ValueError(f"source must be {INPUT_SOURCE}: {source!r}")
        if _required(row, "default_readiness", path) != DEFAULT_READINESS:
            raise ValueError(f"default_readiness must be {DEFAULT_READINESS}")
        if _required(row, "recommended_model_boundary", path) != RECOMMENDED_BOUNDARY:
            raise ValueError(f"recommended_model_boundary must be {RECOMMENDED_BOUNDARY}")
        _expect_flag(row, "diagnostic_only", "true", path)
        _expect_flag(row, "valid_for_default", "false", path)
        _expect_flag(row, "perf_database", "false", path)

        clean_effect = _as_float(row, "clean_budget_effect", path)
        min_effect = _as_float(row, "min_clean_effect", path)
        max_effect = _as_float(row, "max_clean_effect", path)
        spread = _as_float(row, "clean_effect_spread", path)
        shape_count = _as_int(row, "shape_count", path)
        if shape_count != EXPECTED_SHAPE_COUNT_PER_TOPOLOGY:
            raise ValueError(f"shape_count must be {EXPECTED_SHAPE_COUNT_PER_TOPOLOGY}: {shape_count}")
        if min_effect > clean_effect or clean_effect > max_effect:
            raise ValueError("clean_budget_effect must be within min/max clean effect")
        _require_close("clean_effect_spread", spread, max_effect / min_effect)
        by_topology.setdefault(_required(row, "topology_key", path), []).append(row)

    if len(by_topology) != EXPECTED_TOPOLOGY_COUNT:
        raise ValueError(f"topology_count must be {EXPECTED_TOPOLOGY_COUNT}: {len(by_topology)}")
    for topology_key, topology_rows in by_topology.items():
        if len(topology_rows) != EXPECTED_SHAPE_COUNT_PER_TOPOLOGY:
            raise ValueError(f"topology must have exactly 3 shapes: {topology_key}")
        topology_effects = [_as_float(row, "clean_budget_effect", path) for row in topology_rows]
        topology_min = min(topology_effects)
        topology_max = max(topology_effects)
        topology_spread = topology_max / topology_min
        for row in topology_rows:
            _require_close("min_clean_effect", _as_float(row, "min_clean_effect", path), topology_min)
            _require_close("max_clean_effect", _as_float(row, "max_clean_effect", path), topology_max)
            _require_close("clean_effect_spread", _as_float(row, "clean_effect_spread", path), topology_spread)
    return rows


def analyze_default_readiness_final(input_path: Path | str) -> list[dict[str, str]]:
    path = Path(input_path)
    rows = _validate_evidence_rows(_read_csv(path), path)
    clean_effects = [_as_float(row, "clean_budget_effect", path) for row in rows]
    spreads = [_as_float(row, "clean_effect_spread", path) for row in rows]

    return [
        {
            "source": SOURCE,
            "evidence_rows": str(len(rows)),
            "topology_count": str(len({_required(row, "topology_key", path) for row in rows})),
            "shape_count_per_topology": str(EXPECTED_SHAPE_COUNT_PER_TOPOLOGY),
            "max_clean_effect_spread": _format_float(max(spreads)),
            "min_clean_effect": _format_float(min(clean_effects)),
            "max_clean_effect": _format_float(max(clean_effects)),
            "default_readiness": DEFAULT_READINESS,
            "recommended_boundary": RECOMMENDED_BOUNDARY,
            "reason": REASON,
            "diagnostic_only": "true",
            "valid_for_default": "false",
            "perf_database": "false",
        }
    ]


def write_final_audit_csv(path: Path | str, rows: list[dict[str, str]]) -> None:
    if len(rows) != 1:
        raise ValueError(f"phase228 final audit must contain 1 row: got {len(rows)}")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_final_audit_doc(path: Path | str, rows: list[dict[str, str]]) -> None:
    if len(rows) != 1:
        raise ValueError(f"phase228 final audit doc must contain 1 row: got {len(rows)}")
    row = rows[0]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase228: Default Readiness Final Audit",
        "",
        "## Decision",
        "",
        "| Item | Result |",
        "|---|---|",
        "| Default AIC | No-Go |",
        "| VLLMBackend.run_agg | Unchanged |",
        "| PerfDatabase | No-Go |",
        "| GPU benchmark | No-Go in Phase228 |",
        "| Recommended boundary | diagnostic_only_exact_key |",
        "| Reason | no_default_due_to_shape_topology_spread |",
        "| Boundary flags | diagnostic_only=true valid_for_default=false perf_database=false |",
        "",
        "## Machine Audit",
        "",
        "| Field | Value |",
        "|---|---:|",
        f"| evidence_rows | {row['evidence_rows']} |",
        f"| topology_count | {row['topology_count']} |",
        f"| shape_count_per_topology | {row['shape_count_per_topology']} |",
        f"| max_clean_effect_spread | {row['max_clean_effect_spread']} |",
        f"| min_clean_effect | {row['min_clean_effect']} |",
        f"| max_clean_effect | {row['max_clean_effect']} |",
        "",
        "## Interpretation",
        "",
        "Default AIC remains No-Go because the evidence family still has topology and shape spread.",
        "The Phase222/Phase225 result is valid only as a diagnostic_only_exact_key boundary.",
        "The audit does not write PerfDatabase and does not change default simulator logic.",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("docs/iter_gap_investigation/phase222_budget_penalty_evidence_family.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/iter_gap_investigation/phase228_default_readiness_final.csv"),
    )
    parser.add_argument(
        "--doc-out",
        type=Path,
        default=Path("docs/iter_gap_investigation/phase228_default_readiness_final.md"),
    )
    args = parser.parse_args()

    rows = analyze_default_readiness_final(args.input)
    write_final_audit_csv(args.out, rows)
    write_final_audit_doc(args.doc_out, rows)
    print(f"wrote_phase228_final_audit={args.out}")
    print(f"phase228_audit_rows={len(rows)}")
    print(f"default_readiness={DEFAULT_READINESS}")
    print(f"recommended_boundary={RECOMMENDED_BOUNDARY}")
    print(f"reason={REASON}")
    print(f"wrote_phase228_interpretation={args.doc_out}")


if __name__ == "__main__":
    main()
