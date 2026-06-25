from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT / "scripts" / "analyze_phase370_route_a_perfdb_schema_readiness_gate.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase370_route_a_perfdb_schema_readiness_gate",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


PHASE366 = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase366_ep8_comm_minimal_shape_sweep_result.csv"
)
PHASE369 = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase369_fusedmoe_runner_shape_sweep_result.csv"
)
CHECKED_IN_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase370_route_a_perfdb_schema_readiness_gate.csv"
)
EXPECTED_BUCKETS = "1/15/16/241/1808/2048/8192"
EXPECTED_ROW_TYPES = [
    "phase366_ep8_comm_shape_sweep_passed",
    "phase369_fusedmoe_runner_shape_sweep_passed",
    "shared_bucket_alignment_passed",
    "smoke_bucket_128_excluded",
    "kernel_source_metadata_only",
    "perfdb_schema_design_ready",
    "perfdb_write_blocked",
    "default_aic_blocked",
]


def test_phase370_outputs_exact_gate_rows() -> None:
    rows = analyzer.analyze_phase370_route_a_perfdb_schema_readiness_gate(
        PHASE366, PHASE369
    )

    assert [row["row_type"] for row in rows] == EXPECTED_ROW_TYPES
    assert len(rows) == 8


def test_phase370_marks_only_schema_design_ready_not_perfdb_write() -> None:
    rows = analyzer.analyze_phase370_route_a_perfdb_schema_readiness_gate(
        PHASE366, PHASE369
    )
    schema = next(row for row in rows if row["row_type"] == "perfdb_schema_design_ready")
    write = next(row for row in rows if row["row_type"] == "perfdb_write_blocked")
    default = next(row for row in rows if row["row_type"] == "default_aic_blocked")

    assert schema["verdict"] == "ready_for_phase371_schema_design_spec"
    assert schema["next_allowed_phase"] == "phase371_perfdb_schema_design_spec"
    assert write["verdict"] == "blocked_no_perfdb_row_or_curve"
    assert write["perfdb_write_allowed"] == "false"
    assert default["verdict"] == "blocked_default_aic_no_go"
    assert default["default_readiness"] == "No-Go"


def test_phase370_requires_aligned_buckets_and_excludes_128() -> None:
    rows = analyzer.analyze_phase370_route_a_perfdb_schema_readiness_gate(
        PHASE366, PHASE369
    )
    alignment = next(row for row in rows if row["row_type"] == "shared_bucket_alignment_passed")
    smoke = next(row for row in rows if row["row_type"] == "smoke_bucket_128_excluded")

    assert alignment["bucket_tokens"] == EXPECTED_BUCKETS
    assert alignment["ep8_comm_bucket_tokens"] == EXPECTED_BUCKETS
    assert alignment["fusedmoe_bucket_tokens"] == EXPECTED_BUCKETS
    assert smoke["excluded_bucket_tokens"] == "128"
    assert smoke["verdict"] == "smoke_bucket_excluded_from_schema_readiness"


def test_phase370_preserves_evidence_boundaries() -> None:
    rows = analyzer.analyze_phase370_route_a_perfdb_schema_readiness_gate(
        PHASE366, PHASE369
    )
    phase366 = rows[0]
    phase369 = rows[1]
    metadata = next(row for row in rows if row["row_type"] == "kernel_source_metadata_only")

    assert phase366["input_evidence"] == "phase366_ep8_comm_minimal_shape_sweep_result"
    assert phase366["input_status"] == "passed_7_bucket_diagnostic"
    assert phase369["input_evidence"] == "phase369_fusedmoe_runner_shape_sweep_result"
    assert phase369["input_status"] == "passed_7_bucket_diagnostic"
    assert metadata["kernel_source_metadata"] == "CompressedTensorsWNA16MarlinMoEMethod:Marlin"
    assert metadata["kernel_source_lookup_key"] == "false"


def test_phase370_preserves_no_go_diagnostic_flags() -> None:
    rows = analyzer.analyze_phase370_route_a_perfdb_schema_readiness_gate(
        PHASE366, PHASE369
    )

    for row in rows:
        assert row["gpu_allowed"] == "false"
        assert row["default_readiness"] == "No-Go"
        assert row["diagnostic_only"] == "true"
        assert row["valid_for_default"] == "false"
        assert row["perf_database"] == "false"
        assert row["curve_fit_allowed"] == "false"
        assert row["interpolation_allowed"] == "false"
        assert row["extrapolation_allowed"] == "false"


def test_checked_in_csv_matches_analyzer_output() -> None:
    expected = analyzer.analyze_phase370_route_a_perfdb_schema_readiness_gate(
        PHASE366, PHASE369
    )

    with CHECKED_IN_CSV.open(newline="", encoding="utf-8") as handle:
        checked_in = list(csv.DictReader(handle))

    assert checked_in == expected


def test_phase370_rejects_phase366_bucket_mismatch(tmp_path: Path) -> None:
    with PHASE366.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    assert fieldnames is not None
    rows[0]["bucket_tokens"] = "128"

    bad = tmp_path / "phase366.csv"
    with bad.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="Phase366 buckets"):
        analyzer.analyze_phase370_route_a_perfdb_schema_readiness_gate(bad, PHASE369)


def test_phase370_rejects_phase369_not_finite(tmp_path: Path) -> None:
    with PHASE369.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    assert fieldnames is not None
    rows[0]["output_all_finite"] = "false"

    bad = tmp_path / "phase369.csv"
    with bad.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="output_all_finite"):
        analyzer.analyze_phase370_route_a_perfdb_schema_readiness_gate(PHASE366, bad)


def test_output_rejects_perfdb_or_default_upgrade(tmp_path: Path) -> None:
    rows = analyzer.analyze_phase370_route_a_perfdb_schema_readiness_gate(
        PHASE366, PHASE369
    )
    rows[0]["perf_database"] = "true"

    with pytest.raises(ValueError, match="perf_database"):
        analyzer.write_phase370_csv(tmp_path / "bad.csv", rows)


def test_writers_emit_csv_and_md(tmp_path: Path) -> None:
    rows = analyzer.analyze_phase370_route_a_perfdb_schema_readiness_gate(
        PHASE366, PHASE369
    )
    csv_path = tmp_path / "phase370.csv"
    md_path = tmp_path / "phase370.md"

    analyzer.write_phase370_csv(csv_path, rows)
    analyzer.write_phase370_md(md_path, rows)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 8
    assert set(written[0]) == set(analyzer.FIELDNAMES)

    doc = md_path.read_text(encoding="utf-8")
    assert "ready for Phase371 PerfDatabase schema design spec" in doc
    assert "not ready for PerfDatabase write" in doc
    assert "Default AIC | No-Go" in doc
    assert "PerfDatabase | not written" in doc
    assert "1/15/16/241/1808/2048/8192" in doc
    assert "128" in doc
