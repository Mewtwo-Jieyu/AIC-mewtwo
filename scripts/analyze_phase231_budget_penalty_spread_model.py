#!/usr/bin/env python3
"""Compare simple diagnostic models for Phase222 budget penalty spread."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


SOURCE = "phase231_budget_penalty_spread_model"
SUMMARY_SOURCE = "phase231_budget_penalty_spread_model_summary"
EVIDENCE_SOURCE = "phase222_budget_penalty_evidence_family"
AUDIT_SOURCE = "phase228_default_readiness_final"
DEFAULT_READINESS = "No-Go"
RECOMMENDED_BOUNDARY = "diagnostic_only_exact_key"
REASON = "no_default_due_to_shape_topology_spread"

TOPOLOGY_ORDER = ["tp8_dp1_ep8", "tp4_dp2_ep8"]
SHAPE_ORDER = [
    "isl4000_osl2000_batch128",
    "isl8000_osl2000_batch128",
    "isl12000_osl2000_batch128",
]
EXPECTED_KEYS = {(topology, shape) for topology in TOPOLOGY_ORDER for shape in SHAPE_ORDER}
MODEL_NAMES = ["global_mean", "topology_mean", "shape_mean", "topology_plus_shape"]

FIELDNAMES = [
    "source",
    "topology_key",
    "shape_key",
    "control_bt",
    "holdout_bt",
    "clean_budget_effect",
    "global_mean_prediction",
    "topology_mean_prediction",
    "shape_mean_prediction",
    "topology_plus_shape_prediction",
    "global_mean_abs_error",
    "topology_mean_abs_error",
    "shape_mean_abs_error",
    "topology_plus_shape_abs_error",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]

SUMMARY_FIELDNAMES = [
    "source",
    "model_name",
    "parameter_count",
    "mean_abs_error",
    "max_abs_error",
    "max_relative_error",
    "loocv_mean_abs_error",
    "loocv_max_abs_error",
    "best_in_sample_form",
    "best_loocv_form",
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


def _expect_flag(row: dict[str, str], field: str, expected: str, path: Path) -> None:
    value = _required(row, field, path).lower()
    if value != expected:
        raise ValueError(f"{field} must be {expected} in {path}: {value!r}")


def _require_close(field: str, actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-5):
        raise ValueError(f"{field} mismatch: got {actual!r}, expected {expected!r}")


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot compute mean of empty values")
    return sum(values) / len(values)


def _validate_phase228_audit(rows: list[dict[str, str]], path: Path) -> None:
    if len(rows) != 1:
        raise ValueError(f"phase228 final audit must contain exactly 1 row: got {len(rows)}")
    row = rows[0]
    if _required(row, "source", path) != AUDIT_SOURCE:
        raise ValueError(f"phase228 source must be {AUDIT_SOURCE}")
    if _as_int(row, "evidence_rows", path) != 6:
        raise ValueError("phase228 evidence_rows must be 6")
    if _as_int(row, "topology_count", path) != 2:
        raise ValueError("phase228 topology_count must be 2")
    if _as_int(row, "shape_count_per_topology", path) != 3:
        raise ValueError("phase228 shape_count_per_topology must be 3")
    if _required(row, "default_readiness", path) != DEFAULT_READINESS:
        raise ValueError(f"default_readiness must be {DEFAULT_READINESS}")
    if _required(row, "recommended_boundary", path) != RECOMMENDED_BOUNDARY:
        raise ValueError(f"recommended_boundary must be {RECOMMENDED_BOUNDARY}")
    if _required(row, "reason", path) != REASON:
        raise ValueError(f"reason must be {REASON}")
    _expect_flag(row, "diagnostic_only", "true", path)
    _expect_flag(row, "valid_for_default", "false", path)
    _expect_flag(row, "perf_database", "false", path)


def _validate_evidence_rows(rows: list[dict[str, str]], path: Path) -> list[dict[str, str]]:
    if len(rows) != 6:
        raise ValueError(f"phase222 evidence family must contain exactly 6 rows: got {len(rows)}")

    pairs = [(_required(row, "topology_key", path), _required(row, "shape_key", path)) for row in rows]
    duplicates = sorted({pair for pair in pairs if pairs.count(pair) > 1})
    if duplicates:
        raise ValueError(f"duplicate topology/shape evidence: {duplicates}")
    by_topology: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if _required(row, "source", path) != EVIDENCE_SOURCE:
            raise ValueError(f"source must be {EVIDENCE_SOURCE}")
        if _required(row, "default_readiness", path) != DEFAULT_READINESS:
            raise ValueError(f"default_readiness must be {DEFAULT_READINESS}")
        if _required(row, "recommended_model_boundary", path) != RECOMMENDED_BOUNDARY:
            raise ValueError(f"recommended_model_boundary must be {RECOMMENDED_BOUNDARY}")
        _expect_flag(row, "diagnostic_only", "true", path)
        _expect_flag(row, "valid_for_default", "false", path)
        _expect_flag(row, "perf_database", "false", path)
        control_bt = _as_int(row, "control_bt", path)
        holdout_bt = _as_int(row, "holdout_bt", path)
        if holdout_bt != 65536:
            raise ValueError(f"holdout_bt must be 65536: {holdout_bt}")
        if control_bt >= holdout_bt:
            raise ValueError(f"control_bt must be below holdout_bt: {control_bt}")
        if _as_int(row, "shape_count", path) != 3:
            raise ValueError("shape_count must be 3")
        clean_effect = _as_float(row, "clean_budget_effect", path)
        min_effect = _as_float(row, "min_clean_effect", path)
        max_effect = _as_float(row, "max_clean_effect", path)
        spread = _as_float(row, "clean_effect_spread", path)
        if min_effect > clean_effect or clean_effect > max_effect:
            raise ValueError("clean_budget_effect must be within min/max clean effect")
        _require_close("clean_effect_spread", spread, max_effect / min_effect)
        by_topology.setdefault(_required(row, "topology_key", path), []).append(row)

    if set(by_topology) != set(TOPOLOGY_ORDER):
        raise ValueError(f"topology set mismatch: {sorted(by_topology)}")
    for topology_key, topology_rows in by_topology.items():
        shapes = {_required(row, "shape_key", path) for row in topology_rows}
        if len(shapes) != 3:
            raise ValueError(f"topology must have exactly 3 shapes: {topology_key}")

    if set(pairs) != EXPECTED_KEYS:
        missing = sorted(EXPECTED_KEYS - set(pairs))
        extra = sorted(set(pairs) - EXPECTED_KEYS)
        raise ValueError(f"phase222 topology/shape key set mismatch: missing={missing} extra={extra}")

    for topology_key, topology_rows in by_topology.items():
        effects = [_as_float(row, "clean_budget_effect", path) for row in topology_rows]
        topology_min = min(effects)
        topology_max = max(effects)
        topology_spread = topology_max / topology_min
        for row in topology_rows:
            _require_close("min_clean_effect", _as_float(row, "min_clean_effect", path), topology_min)
            _require_close("max_clean_effect", _as_float(row, "max_clean_effect", path), topology_max)
            _require_close("clean_effect_spread", _as_float(row, "clean_effect_spread", path), topology_spread)

    return sorted(rows, key=lambda row: (TOPOLOGY_ORDER.index(row["topology_key"]), SHAPE_ORDER.index(row["shape_key"])))


def _predictions(rows: list[dict[str, str]], path: Path) -> dict[tuple[str, str], dict[str, float]]:
    values = {
        (_required(row, "topology_key", path), _required(row, "shape_key", path)): _as_float(row, "clean_budget_effect", path)
        for row in rows
    }
    grand_mean = _mean(list(values.values()))
    topology_means = {
        topology: _mean([value for (row_topology, _), value in values.items() if row_topology == topology])
        for topology in TOPOLOGY_ORDER
    }
    shape_means = {
        shape: _mean([value for (_, row_shape), value in values.items() if row_shape == shape])
        for shape in SHAPE_ORDER
    }

    return {
        key: {
            "global_mean": grand_mean,
            "topology_mean": topology_means[key[0]],
            "shape_mean": shape_means[key[1]],
            "topology_plus_shape": topology_means[key[0]] + shape_means[key[1]] - grand_mean,
        }
        for key in values
    }


def _predict_for_key(
    values: dict[tuple[str, str], float],
    key: tuple[str, str],
    model: str,
) -> float:
    grand_mean = _mean(list(values.values()))
    topology_values = [value for (topology, _), value in values.items() if topology == key[0]]
    shape_values = [value for (_, shape), value in values.items() if shape == key[1]]
    topology_mean = _mean(topology_values)
    shape_mean = _mean(shape_values)
    if model == "global_mean":
        return grand_mean
    if model == "topology_mean":
        return topology_mean
    if model == "shape_mean":
        return shape_mean
    if model == "topology_plus_shape":
        return topology_mean + shape_mean - grand_mean
    raise ValueError(f"unknown model: {model}")


def _loocv_metrics(rows: list[dict[str, str]]) -> dict[str, tuple[float, float]]:
    values = {
        (row["topology_key"], row["shape_key"]): float(row["clean_budget_effect"])
        for row in rows
    }
    metrics: dict[str, tuple[float, float]] = {}
    for model in MODEL_NAMES:
        errors: list[float] = []
        for key, observed in values.items():
            train = {train_key: value for train_key, value in values.items() if train_key != key}
            prediction = _predict_for_key(train, key, model)
            errors.append(abs(prediction - observed))
        metrics[model] = (_mean(errors), max(errors))
    return metrics


def _summary_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    parameter_counts = {
        "global_mean": 1,
        "topology_mean": len(TOPOLOGY_ORDER),
        "shape_mean": len(SHAPE_ORDER),
        "topology_plus_shape": len(TOPOLOGY_ORDER) + len(SHAPE_ORDER) - 1,
    }
    metrics: dict[str, tuple[float, float, float]] = {}
    for model in MODEL_NAMES:
        errors = [float(row[f"{model}_abs_error"]) for row in rows]
        relatives = [error / float(row["clean_budget_effect"]) for error, row in zip(errors, rows)]
        metrics[model] = (_mean(errors), max(errors), max(relatives))

    loocv = _loocv_metrics(rows)
    best_in_sample = min(MODEL_NAMES, key=lambda model: metrics[model])
    best_loocv = min(MODEL_NAMES, key=lambda model: loocv[model])
    return [
        {
            "source": SUMMARY_SOURCE,
            "model_name": model,
            "parameter_count": str(parameter_counts[model]),
            "mean_abs_error": _format_float(metrics[model][0]),
            "max_abs_error": _format_float(metrics[model][1]),
            "max_relative_error": _format_float(metrics[model][2]),
            "loocv_mean_abs_error": _format_float(loocv[model][0]),
            "loocv_max_abs_error": _format_float(loocv[model][1]),
            "best_in_sample_form": best_in_sample,
            "best_loocv_form": best_loocv,
            "default_readiness": DEFAULT_READINESS,
            "diagnostic_only": "true",
            "valid_for_default": "false",
            "perf_database": "false",
        }
        for model in MODEL_NAMES
    ]


def analyze_budget_penalty_spread_model(
    evidence_path: Path | str,
    audit_path: Path | str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    evidence = Path(evidence_path)
    audit = Path(audit_path)
    _validate_phase228_audit(_read_csv(audit), audit)
    evidence_rows = _validate_evidence_rows(_read_csv(evidence), evidence)
    predictions = _predictions(evidence_rows, evidence)

    out_rows: list[dict[str, str]] = []
    for row in evidence_rows:
        topology = _required(row, "topology_key", evidence)
        shape = _required(row, "shape_key", evidence)
        clean_effect = _as_float(row, "clean_budget_effect", evidence)
        row_predictions = predictions[(topology, shape)]
        out_row = {
            "source": SOURCE,
            "topology_key": topology,
            "shape_key": shape,
            "control_bt": _required(row, "control_bt", evidence),
            "holdout_bt": _required(row, "holdout_bt", evidence),
            "clean_budget_effect": _format_float(clean_effect),
            "diagnostic_only": "true",
            "valid_for_default": "false",
            "perf_database": "false",
        }
        for model in MODEL_NAMES:
            prediction = row_predictions[model]
            out_row[f"{model}_prediction"] = _format_float(prediction)
            out_row[f"{model}_abs_error"] = _format_float(abs(prediction - clean_effect))
        out_rows.append(out_row)

    return out_rows, _summary_rows(out_rows)


def write_spread_model_csv(path: Path | str, rows: list[dict[str, str]]) -> None:
    if len(rows) != 6:
        raise ValueError(f"phase231 spread model must contain 6 rows: got {len(rows)}")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_spread_model_summary_csv(path: Path | str, rows: list[dict[str, str]]) -> None:
    if len(rows) != len(MODEL_NAMES):
        raise ValueError(f"phase231 spread summary must contain 4 rows: got {len(rows)}")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_spread_model_doc(
    path: Path | str,
    rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
) -> None:
    if len(rows) != 6:
        raise ValueError(f"phase231 doc expects 6 spread rows: got {len(rows)}")
    if len(summary_rows) != len(MODEL_NAMES):
        raise ValueError(f"phase231 doc expects 4 summary rows: got {len(summary_rows)}")
    best_in_sample = summary_rows[0]["best_in_sample_form"]
    best_loocv = summary_rows[0]["best_loocv_form"]
    by_model = {row["model_name"]: row for row in summary_rows}
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase231: Budget Penalty Spread Model",
        "",
        "## Decision",
        "",
        "| Item | Result |",
        "|---|---|",
        f"| Best in-sample form | {best_in_sample} |",
        f"| Best LOOCV form | {best_loocv} |",
        "| Default AIC | No-Go |",
        "| Recommended boundary | diagnostic_only_exact_key |",
        "| PerfDatabase | No-Go |",
        "| GPU benchmark | No-Go in Phase231 |",
        "| Boundary flags | diagnostic_only=true valid_for_default=false perf_database=false |",
        "",
        "## Model Errors",
        "",
        "| Model | Parameters | Mean abs error | Max abs error | Max relative error | LOOCV mean abs error | LOOCV max abs error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODEL_NAMES:
        row = by_model[model]
        lines.append(
            f"| {model} | {row['parameter_count']} | {row['mean_abs_error']} | "
            f"{row['max_abs_error']} | {row['max_relative_error']} | "
            f"{row['loocv_mean_abs_error']} | {row['loocv_max_abs_error']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The six-row evidence family is still diagnostic-only.",
            "The complex model fits in-sample better, but the generalization evidence is insufficient.",
            "The best_loocv_form remains a warning against promoting the in-sample fit into default AIC.",
            "Default AIC remains No-Go; the next safe boundary is exact-key diagnostic evidence, not VLLMBackend.run_agg or PerfDatabase wiring.",
            "",
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("docs/iter_gap_investigation/phase222_budget_penalty_evidence_family.csv"),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("docs/iter_gap_investigation/phase228_default_readiness_final.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/iter_gap_investigation/phase231_budget_penalty_spread_model.csv"),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("docs/iter_gap_investigation/phase231_budget_penalty_spread_model_summary.csv"),
    )
    parser.add_argument(
        "--doc-out",
        type=Path,
        default=Path("docs/iter_gap_investigation/phase231_budget_penalty_spread_model.md"),
    )
    args = parser.parse_args()

    rows, summary = analyze_budget_penalty_spread_model(args.input, args.audit)
    write_spread_model_csv(args.out, rows)
    write_spread_model_summary_csv(args.summary_out, summary)
    write_spread_model_doc(args.doc_out, rows, summary)
    print(f"wrote_phase231_spread_model={args.out}")
    print(f"phase231_spread_rows={len(rows)}")
    print(f"wrote_phase231_spread_summary={args.summary_out}")
    print(f"phase231_summary_rows={len(summary)}")
    print(f"best_in_sample_form={summary[0]['best_in_sample_form']}")
    print(f"best_loocv_form={summary[0]['best_loocv_form']}")
    print(f"default_readiness={DEFAULT_READINESS}")
    print(f"wrote_phase231_interpretation={args.doc_out}")


if __name__ == "__main__":
    main()
