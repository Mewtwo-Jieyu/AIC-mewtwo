from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.build_moe_wna16_shape_coverage_phase125 import (
    build_manifest_rows,
    main,
    read_coverage,
)


FIELDS = ["tokens_actual", "occurrence_count", "phase117_exact_key_covered"]


def _write_coverage(
    path: Path,
    rows: list[dict[str, object]],
    fields: list[str] | None = None,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _row(tokens: int, count: int, covered: object = "False") -> dict[str, object]:
    return {
        "tokens_actual": tokens,
        "occurrence_count": count,
        "phase117_exact_key_covered": covered,
    }


def test_all_miss_rows_become_shape_evidence_actions(tmp_path: Path) -> None:
    coverage_path = tmp_path / "coverage.csv"
    _write_coverage(
        coverage_path,
        [
            _row(1, 6960),
            _row(16, 959280),
            _row(8192, 480),
        ],
    )

    manifest = build_manifest_rows(read_coverage(coverage_path))

    assert [row["tokens_actual"] for row in manifest] == ["1", "16", "8192"]
    assert [row["occurrence_count"] for row in manifest] == [
        "6960",
        "959280",
        "480",
    ]
    assert {row["phase117_exact_key_covered"] for row in manifest} == {"false"}
    assert {row["phase125_action"] for row in manifest} == {
        "needs_exact_shape_evidence"
    }
    assert {row["valid_for_time_gate"] for row in manifest} == {"false"}
    assert {row["valid_for_default"] for row in manifest} == {"false"}
    assert {row["perf_database"] for row in manifest} == {"false"}


def test_partial_hit_rows_keep_exact_hit_as_already_covered(tmp_path: Path) -> None:
    coverage_path = tmp_path / "coverage.csv"
    _write_coverage(
        coverage_path,
        [
            _row(128, 10, "True"),
            _row(241, 3, "False"),
        ],
    )

    manifest = build_manifest_rows(read_coverage(coverage_path))

    assert manifest == [
        {
            "tokens_actual": "128",
            "occurrence_count": "10",
            "phase117_exact_key_covered": "true",
            "phase125_action": "already_has_phase117_exact_shape",
            "valid_for_time_gate": "false",
            "valid_for_default": "false",
            "perf_database": "false",
            "notes": "descriptor_only",
        },
        {
            "tokens_actual": "241",
            "occurrence_count": "3",
            "phase117_exact_key_covered": "false",
            "phase125_action": "needs_exact_shape_evidence",
            "valid_for_time_gate": "false",
            "valid_for_default": "false",
            "perf_database": "false",
            "notes": "no_interpolation_or_extrapolation",
        },
    ]


def test_duplicate_tokens_fail(tmp_path: Path) -> None:
    coverage_path = tmp_path / "coverage.csv"
    _write_coverage(coverage_path, [_row(16, 1), _row(16, 2)])

    with pytest.raises(ValueError, match="duplicate tokens_actual"):
        read_coverage(coverage_path)


def test_illegal_boolean_fails(tmp_path: Path) -> None:
    coverage_path = tmp_path / "coverage.csv"
    _write_coverage(coverage_path, [_row(16, 1, "maybe")])

    with pytest.raises(ValueError, match="phase117_exact_key_covered"):
        read_coverage(coverage_path)


def test_empty_table_fails(tmp_path: Path) -> None:
    coverage_path = tmp_path / "coverage.csv"
    _write_coverage(coverage_path, [])

    with pytest.raises(ValueError, match="must contain rows"):
        read_coverage(coverage_path)


def test_forbidden_field_fails(tmp_path: Path) -> None:
    coverage_path = tmp_path / "coverage.csv"
    _write_coverage(
        coverage_path,
        [_row(16, 1)],
        fields=FIELDS + ["latency_ms"],
    )

    with pytest.raises(ValueError, match="forbidden field"):
        read_coverage(coverage_path)


def test_timing_field_fails(tmp_path: Path) -> None:
    coverage_path = tmp_path / "coverage.csv"
    _write_coverage(
        coverage_path,
        [_row(16, 1)],
        fields=FIELDS + ["timing"],
    )

    with pytest.raises(ValueError, match="forbidden field"):
        read_coverage(coverage_path)


def test_main_writes_manifest_and_preserves_total_count(tmp_path: Path) -> None:
    coverage_path = tmp_path / "coverage.csv"
    out_path = tmp_path / "manifest.csv"
    _write_coverage(
        coverage_path,
        [
            _row(1, 6960),
            _row(15, 240),
            _row(16, 959280),
            _row(241, 240),
            _row(1808, 240),
            _row(2048, 240),
            _row(8192, 480),
        ],
    )

    assert main(["--coverage", str(coverage_path), "--out", str(out_path)]) == 0

    with out_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 7
    assert sum(int(row["occurrence_count"]) for row in rows) == 967680
    assert {row["phase125_action"] for row in rows} == {
        "needs_exact_shape_evidence"
    }
