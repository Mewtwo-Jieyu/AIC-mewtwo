from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.fit_moe_wna16_diagnostic_phase116 import (
    fit_diagnostic_dataset,
    main,
    read_shape_summary,
)


INPUT_FIELDS = [
    "tokens_actual",
    "run_count",
    "cuda_event_ms_mean",
    "cuda_event_ms_min_run_mean",
    "cuda_event_ms_max_run_mean",
    "cuda_event_ms_run_range",
    "wall_ms_mean",
    "wall_ms_min_run_mean",
    "wall_ms_max_run_mean",
    "wall_ms_run_range",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]


def _write_input(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _row(tokens: int, cuda: float, wall: float | None = None) -> dict[str, object]:
    wall_value = cuda + 0.02 if wall is None else wall
    return {
        "tokens_actual": tokens,
        "run_count": 1,
        "cuda_event_ms_mean": cuda,
        "cuda_event_ms_min_run_mean": cuda,
        "cuda_event_ms_max_run_mean": cuda,
        "cuda_event_ms_run_range": 0.0,
        "wall_ms_mean": wall_value,
        "wall_ms_min_run_mean": wall_value,
        "wall_ms_max_run_mean": wall_value,
        "wall_ms_run_range": 0.0,
        "diagnostic_only": "true",
        "valid_for_default": "false",
        "perf_database": "false",
    }


def test_read_shape_summary_requires_exact_shapes_and_flags(tmp_path: Path) -> None:
    path = tmp_path / "summary.csv"
    _write_input(
        path,
        [
            _row(128, 2.28),
            _row(248, 3.48),
            _row(512, 6.12),
            _row(1024, 11.24),
        ],
    )

    rows = read_shape_summary(path)

    assert [row.tokens_actual for row in rows] == [128, 248, 512, 1024]

    bad_flag = tmp_path / "bad_flag.csv"
    _write_input(
        bad_flag,
        [
            _row(128, 2.28),
            _row(248, 3.48),
            {**_row(512, 6.12), "valid_for_default": "true"},
            _row(1024, 11.24),
        ],
    )
    with pytest.raises(ValueError, match="valid_for_default"):
        read_shape_summary(bad_flag)


def test_read_shape_summary_rejects_missing_and_forbidden_columns(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing.csv"
    with missing_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[field for field in INPUT_FIELDS if field != "cuda_event_ms_mean"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerow(_row(128, 2.28))

    with pytest.raises(ValueError, match="missing required fields"):
        read_shape_summary(missing_path)

    forbidden_path = tmp_path / "forbidden.csv"
    with forbidden_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INPUT_FIELDS + ["latency_ms"])
        writer.writeheader()
        writer.writerow({**_row(128, 2.28), "latency_ms": 1.0})

    with pytest.raises(ValueError, match="forbidden fields"):
        read_shape_summary(forbidden_path)


def test_fit_linear_and_leave_one_shape_out_errors_are_reported(
    tmp_path: Path,
) -> None:
    path = tmp_path / "summary.csv"
    _write_input(
        path,
        [
            _row(128, 1.64),
            _row(248, 2.84),
            _row(512, 5.48),
            _row(1024, 10.60),
        ],
    )

    audit, errors = fit_diagnostic_dataset(read_shape_summary(path))

    assert audit["model_use"] == "diagnostic_error_audit_only"
    assert audit["valid_for_default"] == "false"
    assert audit["perf_database"] == "false"
    assert float(audit["diagnostic_intercept_ms"]) == pytest.approx(0.36)
    assert float(audit["diagnostic_slope_ms_per_token"]) == pytest.approx(0.01)
    assert {row["holdout_kind"] for row in errors} == {
        "leave_one_shape_out",
        "extrapolation_1024",
    }
    assert max(float(row["diagnostic_abs_error_ms"]) for row in errors) == pytest.approx(
        0.0
    )


def test_main_writes_diagnostic_only_outputs_without_forbidden_headers(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "summary.csv"
    audit_out = tmp_path / "audit.csv"
    errors_out = tmp_path / "errors.csv"
    _write_input(
        input_path,
        [
            _row(128, 1.64),
            _row(248, 2.84),
            _row(512, 5.48),
            _row(1024, 10.60),
        ],
    )

    assert main(
        [
            "--input",
            str(input_path),
            "--fit-audit-out",
            str(audit_out),
            "--holdout-errors-out",
            str(errors_out),
        ]
    ) == 0

    forbidden_parts = ("latency", "residual", "profiled", "profiler", "nccl", "sync")
    with audit_out.open(newline="", encoding="utf-8") as f:
        audit_rows = list(csv.DictReader(f))
        assert audit_rows[0]["model_use"] == "diagnostic_error_audit_only"
        assert audit_rows[0]["valid_for_default"] == "false"
        assert audit_rows[0]["perf_database"] == "false"
        assert not any(
            part in field.lower()
            for field in audit_rows[0]
            for part in forbidden_parts
        )
    with errors_out.open(newline="", encoding="utf-8") as f:
        error_rows = list(csv.DictReader(f))
        assert error_rows
        assert all(row["valid_for_default"] == "false" for row in error_rows)
        assert all(row["perf_database"] == "false" for row in error_rows)
        assert not any(
            part in field.lower()
            for field in error_rows[0]
            for part in forbidden_parts
        )
