#!/usr/bin/env python3
"""Fit diagnostic-only MoE WNA16 timing candidates for Phase 116."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


EXPECTED_TOKENS = (128, 248, 512, 1024)
MODEL_USE = "diagnostic_error_audit_only"
MODEL_FORM = "constant_plus_token_linear"
INPUT_SOURCE = "synthetic_random_hidden_states"
TARGET_METRIC = "cuda_event_ms_mean"

REQUIRED_FIELDS = {
    "tokens_actual",
    "run_count",
    "cuda_event_ms_mean",
    "wall_ms_mean",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
}

FORBIDDEN_FIELD_PARTS = (
    "latency",
    "residual",
    "profiler",
    "profiled",
    "nccl",
    "sync",
    "throughput",
)

FIT_AUDIT_FIELDS = [
    "model_use",
    "diagnostic_model_form",
    "target_metric",
    "shape_count",
    "tokens_min",
    "tokens_max",
    "diagnostic_intercept_ms",
    "diagnostic_slope_ms_per_token",
    "wall_sanity_intercept_ms",
    "wall_sanity_slope_ms_per_token",
    "cuda_event_shape_monotonic",
    "wall_shape_monotonic",
    "wall_cuda_slope_same_sign",
    "input_source",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]

HOLDOUT_ERROR_FIELDS = [
    "model_use",
    "diagnostic_model_form",
    "holdout_kind",
    "target_metric",
    "holdout_tokens_actual",
    "train_tokens_actual",
    "observed_cuda_event_ms_mean",
    "predicted_cuda_event_ms_mean",
    "diagnostic_abs_error_ms",
    "diagnostic_relative_error_percent",
    "input_source",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]


@dataclass(frozen=True)
class ShapeRow:
    tokens_actual: int
    run_count: int
    cuda_event_ms_mean: float
    wall_ms_mean: float
    diagnostic_only: bool
    valid_for_default: bool
    perf_database: bool


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit diagnostic-only MoE WNA16 model candidates for Phase 116."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("docs/iter_gap_investigation/phase114_shape_trend_summary.csv"),
    )
    parser.add_argument(
        "--fit-audit-out",
        type=Path,
        default=Path("docs/iter_gap_investigation/phase116_moe_wna16_fit_audit.csv"),
    )
    parser.add_argument(
        "--holdout-errors-out",
        type=Path,
        default=Path(
            "docs/iter_gap_investigation/phase116_moe_wna16_holdout_errors.csv"
        ),
    )
    return parser.parse_args(argv)


def _bool_value(value: str, field: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"{field} must be boolean-like, got {value!r}")


def _check_header(fieldnames: Sequence[str] | None, path: Path) -> list[str]:
    if fieldnames is None:
        raise ValueError(f"{path} has no CSV header")
    fields = list(fieldnames)
    forbidden = [
        field
        for field in fields
        if any(part in field.lower() for part in FORBIDDEN_FIELD_PARTS)
    ]
    if forbidden:
        raise ValueError(f"{path} contains forbidden fields: {', '.join(forbidden)}")
    missing = sorted(REQUIRED_FIELDS - set(fields))
    if missing:
        raise ValueError(f"{path} missing required fields: {', '.join(missing)}")
    return fields


def _int_value(row: dict[str, str], field: str, path: Path) -> int:
    try:
        return int(row[field])
    except ValueError as exc:
        raise ValueError(f"{path}: {field} must be an integer") from exc


def _float_value(row: dict[str, str], field: str, path: Path) -> float:
    try:
        return float(row[field])
    except ValueError as exc:
        raise ValueError(f"{path}: {field} must be a number") from exc


def read_shape_summary(path: Path) -> list[ShapeRow]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        _check_header(reader.fieldnames, path)
        rows = [
            ShapeRow(
                tokens_actual=_int_value(row, "tokens_actual", path),
                run_count=_int_value(row, "run_count", path),
                cuda_event_ms_mean=_float_value(row, "cuda_event_ms_mean", path),
                wall_ms_mean=_float_value(row, "wall_ms_mean", path),
                diagnostic_only=_bool_value(row["diagnostic_only"], "diagnostic_only"),
                valid_for_default=_bool_value(
                    row["valid_for_default"], "valid_for_default"
                ),
                perf_database=_bool_value(row["perf_database"], "perf_database"),
            )
            for row in reader
        ]
    if not rows:
        raise ValueError(f"{path} must contain at least one row")
    tokens = tuple(row.tokens_actual for row in rows)
    if tokens != EXPECTED_TOKENS:
        raise ValueError(
            f"{path} must contain exact token shapes {EXPECTED_TOKENS}, got {tokens}"
        )
    for row in rows:
        if not row.diagnostic_only:
            raise ValueError("diagnostic_only must be true")
        if row.valid_for_default:
            raise ValueError("valid_for_default must be false")
        if row.perf_database:
            raise ValueError("perf_database must be false")
    return rows


def _linear_fit(points: Sequence[tuple[int, float]]) -> tuple[float, float]:
    count = len(points)
    if count < 2:
        raise ValueError("linear fit requires at least two points")
    sum_x = float(sum(x for x, _ in points))
    sum_y = float(sum(y for _, y in points))
    sum_xx = float(sum(x * x for x, _ in points))
    sum_xy = float(sum(x * y for x, y in points))
    denominator = count * sum_xx - sum_x * sum_x
    if denominator == 0.0:
        raise ValueError("linear fit token values must not be identical")
    slope = (count * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / count
    return intercept, slope


def _predict(intercept: float, slope: float, tokens: int) -> float:
    return intercept + slope * tokens


def _is_monotonic(values: Sequence[float]) -> bool:
    return all(values[index] <= values[index + 1] for index in range(len(values) - 1))


def _format_float(value: float) -> str:
    return f"{value:.12g}"


def fit_diagnostic_dataset(
    rows: Sequence[ShapeRow],
) -> tuple[dict[str, str], list[dict[str, str]]]:
    cuda_points = [(row.tokens_actual, row.cuda_event_ms_mean) for row in rows]
    wall_points = [(row.tokens_actual, row.wall_ms_mean) for row in rows]
    cuda_intercept, cuda_slope = _linear_fit(cuda_points)
    wall_intercept, wall_slope = _linear_fit(wall_points)
    audit = {
        "model_use": MODEL_USE,
        "diagnostic_model_form": MODEL_FORM,
        "target_metric": TARGET_METRIC,
        "shape_count": str(len(rows)),
        "tokens_min": str(min(row.tokens_actual for row in rows)),
        "tokens_max": str(max(row.tokens_actual for row in rows)),
        "diagnostic_intercept_ms": _format_float(cuda_intercept),
        "diagnostic_slope_ms_per_token": _format_float(cuda_slope),
        "wall_sanity_intercept_ms": _format_float(wall_intercept),
        "wall_sanity_slope_ms_per_token": _format_float(wall_slope),
        "cuda_event_shape_monotonic": str(
            _is_monotonic([row.cuda_event_ms_mean for row in rows])
        ).lower(),
        "wall_shape_monotonic": str(
            _is_monotonic([row.wall_ms_mean for row in rows])
        ).lower(),
        "wall_cuda_slope_same_sign": str(
            (cuda_slope >= 0 and wall_slope >= 0) or (cuda_slope <= 0 and wall_slope <= 0)
        ).lower(),
        "input_source": INPUT_SOURCE,
        "diagnostic_only": "true",
        "valid_for_default": "false",
        "perf_database": "false",
    }
    errors: list[dict[str, str]] = []
    for held_out in rows:
        training = [row for row in rows if row.tokens_actual != held_out.tokens_actual]
        errors.append(
            _holdout_error_row(
                "leave_one_shape_out",
                training,
                held_out,
            )
        )
    held_out_1024 = next(row for row in rows if row.tokens_actual == 1024)
    training_without_1024 = [row for row in rows if row.tokens_actual != 1024]
    errors.append(
        _holdout_error_row(
            "extrapolation_1024",
            training_without_1024,
            held_out_1024,
        )
    )
    return audit, errors


def _holdout_error_row(
    holdout_kind: str,
    training: Sequence[ShapeRow],
    held_out: ShapeRow,
) -> dict[str, str]:
    intercept, slope = _linear_fit(
        [(row.tokens_actual, row.cuda_event_ms_mean) for row in training]
    )
    predicted = _predict(intercept, slope, held_out.tokens_actual)
    error = abs(predicted - held_out.cuda_event_ms_mean)
    relative = error / held_out.cuda_event_ms_mean * 100.0
    return {
        "model_use": MODEL_USE,
        "diagnostic_model_form": MODEL_FORM,
        "holdout_kind": holdout_kind,
        "target_metric": TARGET_METRIC,
        "holdout_tokens_actual": str(held_out.tokens_actual),
        "train_tokens_actual": "|".join(str(row.tokens_actual) for row in training),
        "observed_cuda_event_ms_mean": _format_float(held_out.cuda_event_ms_mean),
        "predicted_cuda_event_ms_mean": _format_float(predicted),
        "diagnostic_abs_error_ms": _format_float(error),
        "diagnostic_relative_error_percent": _format_float(relative),
        "input_source": INPUT_SOURCE,
        "diagnostic_only": "true",
        "valid_for_default": "false",
        "perf_database": "false",
    }


def _write_single_row_csv(path: Path, fieldnames: Sequence[str], row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerow(row)


def _write_rows_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    rows = read_shape_summary(args.input)
    audit, errors = fit_diagnostic_dataset(rows)
    _write_single_row_csv(args.fit_audit_out, FIT_AUDIT_FIELDS, audit)
    _write_rows_csv(args.holdout_errors_out, HOLDOUT_ERROR_FIELDS, errors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
