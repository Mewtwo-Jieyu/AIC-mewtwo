from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.build_moe_wna16_experimental_table_phase117 import (
    build_table_rows,
    main,
    query_exact_token,
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
        writer.writerows(rows)


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


def _valid_rows() -> list[dict[str, object]]:
    return [
        _row(128, 1.0),
        _row(248, 1.3),
        _row(512, 1.6),
        _row(1024, 2.2),
    ]


def test_build_table_rows_adds_exact_key_and_safety_fields(tmp_path: Path) -> None:
    input_path = tmp_path / "summary.csv"
    _write_input(input_path, _valid_rows())

    table = build_table_rows(read_shape_summary(input_path))

    assert [row["tokens_actual"] for row in table] == ["128", "248", "512", "1024"]
    assert {row["backend"] for row in table} == {"vllm"}
    assert {row["vllm_version"] for row in table} == {"0.19.0"}
    assert {row["device"] for row in table} == {"NVIDIA_H200"}
    assert {row["module"] for row in table} == {"DeepseekV2MoE"}
    assert {row["experts_impl"] for row in table} == {"SharedFusedMoE"}
    assert {row["kernel"] for row in table} == {"WNA16"}
    assert {row["topology"] for row in table} == {"tp4dp2ep8"}
    assert {row["query_policy"] for row in table} == {"exact_match_only"}
    assert {row["model_use"] for row in table} == {
        "experimental_table_prototype_only"
    }
    assert {row["valid_for_default"] for row in table} == {"false"}
    assert {row["perf_database"] for row in table} == {"false"}


def test_input_validation_rejects_missing_shape_flags_and_forbidden_headers(
    tmp_path: Path,
) -> None:
    bad_shape = tmp_path / "bad_shape.csv"
    _write_input(
        bad_shape,
        [_row(128, 1.0), _row(248, 1.3), _row(512, 1.6), _row(768, 2.0)],
    )
    with pytest.raises(ValueError, match="exact token shapes"):
        read_shape_summary(bad_shape)

    bad_flag = tmp_path / "bad_flag.csv"
    _write_input(
        bad_flag,
        [
            _row(128, 1.0),
            _row(248, 1.3),
            {**_row(512, 1.6), "perf_database": "true"},
            _row(1024, 2.2),
        ],
    )
    with pytest.raises(ValueError, match="perf_database"):
        read_shape_summary(bad_flag)

    forbidden = tmp_path / "forbidden.csv"
    with forbidden.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INPUT_FIELDS + ["latency_ms"])
        writer.writeheader()
        writer.writerow({**_row(128, 1.0), "latency_ms": 1.0})
    with pytest.raises(ValueError, match="forbidden fields"):
        read_shape_summary(forbidden)


def test_exact_query_hits_measured_tokens_and_rejects_unmeasured_tokens(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "summary.csv"
    _write_input(input_path, _valid_rows())
    table = build_table_rows(read_shape_summary(input_path))

    assert query_exact_token(table, 128)["cuda_event_ms_mean"] == "1"
    assert query_exact_token(table, 1024)["cuda_event_ms_mean"] == "2.2"
    with pytest.raises(KeyError, match="exact token key missing"):
        query_exact_token(table, 256)
    with pytest.raises(KeyError, match="exact token key missing"):
        query_exact_token(table, 768)


def test_main_writes_table_without_forbidden_headers(tmp_path: Path) -> None:
    input_path = tmp_path / "summary.csv"
    output_path = tmp_path / "table.csv"
    _write_input(input_path, _valid_rows())

    assert main(["--input", str(input_path), "--out", str(output_path)]) == 0

    with output_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fields = rows[0].keys()
    forbidden_parts = (
        "latency",
        "residual",
        "profiled",
        "profiler",
        "nccl",
        "sync",
        "throughput",
    )
    assert len(rows) == 4
    assert not any(
        part in field.lower()
        for field in fields
        for part in forbidden_parts
    )
    assert rows[0]["query_policy"] == "exact_match_only"
    assert rows[0]["valid_for_default"] == "false"
    assert rows[0]["perf_database"] == "false"
