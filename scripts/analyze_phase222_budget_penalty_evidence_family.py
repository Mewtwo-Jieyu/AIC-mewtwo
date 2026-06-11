#!/usr/bin/env python3
"""Build the Phase222 budget penalty evidence family table."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


SOURCE = "phase222_budget_penalty_evidence_family"
PHASE164_SOURCE = "phase164_clean_gpu_benchmark"
PHASE164_EVIDENCE_SOURCE = "phase164_clean_gpu_budget_manifest"
PHASE215_SOURCE = "phase215_budget_penalty_mechanism"
RECOMMENDED_BOUNDARY = "diagnostic_only_exact_key"
DEFAULT_READINESS = "No-Go"

TOPOLOGY_ORDER = ["tp8_dp1_ep8", "tp4_dp2_ep8"]
SHAPE_ORDER = [
    "isl4000_osl2000_batch128",
    "isl8000_osl2000_batch128",
    "isl12000_osl2000_batch128",
]
EXPECTED_EVIDENCE_KEYS = {(topology, shape) for topology in TOPOLOGY_ORDER for shape in SHAPE_ORDER}
EXPECTED_PHASE215_KEYS = {
    ("tp8_dp1_ep8", "isl4000_osl2000_batch128"),
    ("tp4_dp2_ep8", "isl4000_osl2000_batch128"),
    ("tp8_dp1_ep8", "isl12000_osl2000_batch128"),
    ("tp4_dp2_ep8", "isl12000_osl2000_batch128"),
}
EXPECTED_PHASE164_KEYS = {
    ("tp8_dp1_ep8", 8000),
    ("tp8_dp1_ep8", 65536),
    ("tp4_dp2_ep8", 8000),
    ("tp4_dp2_ep8", 65536),
}

FIELDNAMES = [
    "source",
    "topology_key",
    "shape_key",
    "control_scenario",
    "holdout_scenario",
    "control_bt",
    "holdout_bt",
    "clean_budget_effect",
    "no_budget_penalty_error",
    "evidence_source",
    "min_clean_effect",
    "max_clean_effect",
    "clean_effect_spread",
    "shape_count",
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


def _as_non_negative_int(row: dict[str, str], field: str, path: Path) -> int:
    value = _required(row, field, path)
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be int in {path}: {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"{field} must be non-negative in {path}: {parsed!r}")
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


def _symmetric_error(predicted: float, observed: float, label: str) -> float:
    if predicted <= 0 or observed <= 0:
        raise ValueError(f"{label} values must be positive")
    ratio = predicted / observed
    return ratio if ratio >= 1.0 else 1.0 / ratio


def _topology_key(row: dict[str, str], path: Path) -> str:
    tp = _as_int(row, "tp", path)
    dp = _as_int(row, "dp", path)
    ep = _as_int(row, "ep", path)
    return f"tp{tp}_dp{dp}_ep{ep}"


def _shape_key(row: dict[str, str], path: Path) -> str:
    isl = _as_int(row, "isl", path)
    osl = _as_int(row, "osl", path)
    batch = _as_int(row, "batch_size", path)
    return f"isl{isl}_osl{osl}_batch{batch}"


def _validate_phase164_rows(rows: list[dict[str, str]], path: Path) -> dict[tuple[str, int], dict[str, str]]:
    if len(rows) != 4:
        raise ValueError(f"phase164 manifest must contain exactly 4 rows: got {len(rows)}")

    by_key: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        source = _required(row, "source", path)
        if source != PHASE164_SOURCE:
            raise ValueError(f"phase164 source must be {PHASE164_SOURCE}: {source!r}")
        _expect_flag(row, "diagnostic_only", "true", path)
        _expect_flag(row, "valid_for_default", "false", path)
        _expect_flag(row, "perf_database", "false", path)
        if _as_int(row, "request_success_count", path) != 128:
            raise ValueError("phase164 request_success_count must be 128")
        if _as_non_negative_int(row, "request_fail_count", path) != 0:
            raise ValueError("phase164 request_fail_count must be 0")
        if _shape_key(row, path) != "isl8000_osl2000_batch128":
            raise ValueError("phase164 evidence must be isl8000_osl2000_batch128")
        _as_float(row, "real_output_tok_s", path)
        key = (_topology_key(row, path), _as_int(row, "max_num_batched_tokens", path))
        if key in by_key:
            raise ValueError(f"duplicate phase164 topology/budget: {key}")
        by_key[key] = row

    if set(by_key) != EXPECTED_PHASE164_KEYS:
        raise ValueError(
            "phase164 key set mismatch: "
            f"missing={sorted(EXPECTED_PHASE164_KEYS - set(by_key))} "
            f"extra={sorted(set(by_key) - EXPECTED_PHASE164_KEYS)}"
        )
    return by_key


def _validate_phase215_rows(rows: list[dict[str, str]], path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if len(rows) != 4:
        raise ValueError(f"phase215 mechanism must contain exactly 4 rows: got {len(rows)}")

    by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        source = _required(row, "source", path)
        if source != PHASE215_SOURCE:
            raise ValueError(f"phase215 source must be {PHASE215_SOURCE}: {source!r}")
        if _required(row, "recommended_model_boundary", path) != RECOMMENDED_BOUNDARY:
            raise ValueError(f"recommended_model_boundary must be {RECOMMENDED_BOUNDARY}")
        if _required(row, "default_readiness", path) != DEFAULT_READINESS:
            raise ValueError(f"default_readiness must be {DEFAULT_READINESS}")
        _expect_flag(row, "diagnostic_only", "true", path)
        _expect_flag(row, "valid_for_default", "false", path)
        _expect_flag(row, "perf_database", "false", path)
        control_bt = _as_int(row, "control_max_bt", path)
        holdout_bt = _as_int(row, "holdout_max_bt", path)
        if holdout_bt != 65536:
            raise ValueError(f"phase215 holdout_max_bt must be 65536: {holdout_bt}")
        if control_bt >= holdout_bt:
            raise ValueError(f"phase215 control_max_bt must be below holdout_max_bt: {control_bt}")
        clean_effect = _as_float(row, "clean_budget_effect", path)
        no_penalty_error = _as_float(row, "no_budget_penalty_error", path)
        _require_close(
            "no_budget_penalty_error",
            no_penalty_error,
            _symmetric_error(1.0, clean_effect, "no_budget_penalty_error"),
        )
        key = (_required(row, "topology_key", path), _required(row, "shape_key", path))
        if key in by_key:
            raise ValueError(f"duplicate phase215 topology/shape: {key}")
        by_key[key] = row

    if set(by_key) != EXPECTED_PHASE215_KEYS:
        raise ValueError(
            "phase215 key set mismatch: "
            f"missing={sorted(EXPECTED_PHASE215_KEYS - set(by_key))} "
            f"extra={sorted(set(by_key) - EXPECTED_PHASE215_KEYS)}"
        )
    return by_key


def _phase215_evidence_row(row: dict[str, str], path: Path) -> dict[str, str]:
    return {
        "source": SOURCE,
        "topology_key": _required(row, "topology_key", path),
        "shape_key": _required(row, "shape_key", path),
        "control_scenario": _required(row, "control_scenario", path),
        "holdout_scenario": _required(row, "holdout_scenario", path),
        "control_bt": _required(row, "control_max_bt", path),
        "holdout_bt": _required(row, "holdout_max_bt", path),
        "clean_budget_effect": _format_float(_as_float(row, "clean_budget_effect", path)),
        "no_budget_penalty_error": _format_float(_as_float(row, "no_budget_penalty_error", path)),
        "evidence_source": PHASE215_SOURCE,
        "recommended_model_boundary": RECOMMENDED_BOUNDARY,
        "default_readiness": DEFAULT_READINESS,
        "diagnostic_only": "true",
        "valid_for_default": "false",
        "perf_database": "false",
    }


def _phase164_evidence_row(
    topology_key: str,
    control: dict[str, str],
    holdout: dict[str, str],
    path: Path,
) -> dict[str, str]:
    control_output = _as_float(control, "real_output_tok_s", path)
    holdout_output = _as_float(holdout, "real_output_tok_s", path)
    clean_effect = holdout_output / control_output
    return {
        "source": SOURCE,
        "topology_key": topology_key,
        "shape_key": "isl8000_osl2000_batch128",
        "control_scenario": _required(control, "scenario", path),
        "holdout_scenario": _required(holdout, "scenario", path),
        "control_bt": _required(control, "max_num_batched_tokens", path),
        "holdout_bt": _required(holdout, "max_num_batched_tokens", path),
        "clean_budget_effect": _format_float(clean_effect),
        "no_budget_penalty_error": _format_float(
            _symmetric_error(1.0, clean_effect, "no_budget_penalty_error")
        ),
        "evidence_source": PHASE164_EVIDENCE_SOURCE,
        "recommended_model_boundary": RECOMMENDED_BOUNDARY,
        "default_readiness": DEFAULT_READINESS,
        "diagnostic_only": "true",
        "valid_for_default": "false",
        "perf_database": "false",
    }


def _add_topology_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_topology: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_topology.setdefault(row["topology_key"], []).append(row)

    for topology, topology_rows in by_topology.items():
        if len(topology_rows) != 3:
            raise ValueError(f"topology must have exactly 3 shapes: {topology}")
        effects = [float(row["clean_budget_effect"]) for row in topology_rows]
        min_effect = min(effects)
        max_effect = max(effects)
        spread = max_effect / min_effect
        for row in topology_rows:
            row["min_clean_effect"] = _format_float(min_effect)
            row["max_clean_effect"] = _format_float(max_effect)
            row["clean_effect_spread"] = _format_float(spread)
            row["shape_count"] = "3"
    return rows


def analyze_budget_penalty_evidence_family(
    phase164_manifest_path: Path | str,
    phase215_mechanism_path: Path | str,
) -> list[dict[str, str]]:
    phase164_path = Path(phase164_manifest_path)
    phase215_path = Path(phase215_mechanism_path)
    phase164_by_key = _validate_phase164_rows(_read_csv(phase164_path), phase164_path)
    phase215_by_key = _validate_phase215_rows(_read_csv(phase215_path), phase215_path)

    by_key: dict[tuple[str, str], dict[str, str]] = {}
    for key, row in phase215_by_key.items():
        by_key[key] = _phase215_evidence_row(row, phase215_path)
    for topology_key in TOPOLOGY_ORDER:
        by_key[(topology_key, "isl8000_osl2000_batch128")] = _phase164_evidence_row(
            topology_key,
            phase164_by_key[(topology_key, 8000)],
            phase164_by_key[(topology_key, 65536)],
            phase164_path,
        )

    if set(by_key) != EXPECTED_EVIDENCE_KEYS:
        raise ValueError(
            "evidence family key set mismatch: "
            f"missing={sorted(EXPECTED_EVIDENCE_KEYS - set(by_key))} "
            f"extra={sorted(set(by_key) - EXPECTED_EVIDENCE_KEYS)}"
        )

    ordered = [by_key[(topology, shape)] for topology in TOPOLOGY_ORDER for shape in SHAPE_ORDER]
    return _add_topology_summary(ordered)


def write_evidence_family_csv(path: Path | str, rows: list[dict[str, str]]) -> None:
    if len(rows) != 6:
        raise ValueError(f"phase222 evidence family must contain 6 rows: got {len(rows)}")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_evidence_family_doc(path: Path | str, rows: list[dict[str, str]]) -> None:
    if len(rows) != 6:
        raise ValueError(f"phase222 evidence family doc must contain 6 rows: got {len(rows)}")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase222: Budget Penalty Evidence Family",
        "",
        "## Decision",
        "",
        "| Item | Result |",
        "|---|---|",
        "| Default AIC | No-Go |",
        "| VLLMBackend.run_agg | Unchanged |",
        "| PerfDatabase | No-Go |",
        "| GPU benchmark | No-Go in Phase222 |",
        "| Recommended boundary | diagnostic_only_exact_key |",
        "| Boundary flags | diagnostic_only=true valid_for_default=false perf_database=false |",
        "",
        "## Evidence Rows",
        "",
        "| topology_key | shape_key | control_bt | holdout_bt | clean_budget_effect | no_budget_penalty_error | evidence_source | clean_effect_spread |",
        "|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            "| {topology_key} | {shape_key} | {control_bt} | {holdout_bt} | "
            "{clean_budget_effect} | {no_budget_penalty_error} | {evidence_source} | "
            "{clean_effect_spread} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Phase164 clean evidence supplies the 8k2k rows; those rows are not interpolation.",
            "Phase215 supplies the 4k2k and 12k2k diagnostic mechanism evidence.",
            "Across both topologies, high-budget clean effect stays close to 1 compared with the cb_sim steady penalty family.",
            "Shape and topology spread remains visible, so Default AIC remains No-Go.",
            "The only defensible next boundary is a diagnostic_only_exact_key family, not a global constant or a default multiplier.",
            "",
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase164-manifest",
        type=Path,
        default=Path("docs/iter_gap_investigation/phase164_clean_gpu_budget_manifest.csv"),
    )
    parser.add_argument(
        "--phase215-mechanism",
        type=Path,
        default=Path("docs/iter_gap_investigation/phase215_budget_penalty_mechanism.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/iter_gap_investigation/phase222_budget_penalty_evidence_family.csv"),
    )
    parser.add_argument(
        "--doc-out",
        type=Path,
        default=Path("docs/iter_gap_investigation/phase222_budget_penalty_evidence_family.md"),
    )
    args = parser.parse_args()

    rows = analyze_budget_penalty_evidence_family(args.phase164_manifest, args.phase215_mechanism)
    write_evidence_family_csv(args.out, rows)
    write_evidence_family_doc(args.doc_out, rows)
    print(f"wrote_phase222_evidence_family={args.out}")
    print(f"phase222_evidence_rows={len(rows)}")
    print(f"default_readiness={DEFAULT_READINESS}")
    print(f"recommended_model_boundary={RECOMMENDED_BOUNDARY}")
    print(f"wrote_phase222_interpretation={args.doc_out}")


if __name__ == "__main__":
    main()
