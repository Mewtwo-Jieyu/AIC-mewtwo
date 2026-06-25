from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT / "scripts" / "analyze_phase371_route_a_perfdb_schema_design_spec.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase371_route_a_perfdb_schema_design_spec",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


PHASE370 = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase370_route_a_perfdb_schema_readiness_gate.csv"
)
CHECKED_IN_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase371_route_a_perfdb_schema_design_spec.csv"
)
EXPECTED_ROW_TYPES = [
    "schema_scope",
    "module_row_fusedmoe_runner_compute",
    "module_row_ep8_comm_dispatch_combine",
    "key_dimensions",
    "kernel_source_policy",
    "bucket_policy",
    "prohibited_actions",
    "next_phase",
]
EXPECTED_KEY_DIMENSIONS = (
    "model;hardware;vllm_version;topology;bucket_tokens;module_boundary;quant_runtime"
)
EXPECTED_BUCKETS = "1/15/16/241/1808/2048/8192"


def test_phase371_outputs_exact_schema_design_rows() -> None:
    rows = analyzer.analyze_phase371_route_a_perfdb_schema_design_spec(PHASE370)

    assert [row["row_type"] for row in rows] == EXPECTED_ROW_TYPES
    assert len(rows) == 8


def test_phase371_defines_module_level_rows_only() -> None:
    rows = analyzer.analyze_phase371_route_a_perfdb_schema_design_spec(PHASE370)
    scope = next(row for row in rows if row["row_type"] == "schema_scope")
    modules = [row for row in rows if row["row_type"].startswith("module_row_")]

    assert scope["row_boundary"] == "module_level_row_not_end_to_end_row"
    assert scope["end_to_end_row_allowed"] == "false"
    assert [row["module_boundary"] for row in modules] == [
        "fusedmoe_runner_compute",
        "ep8_comm_dispatch_combine",
    ]
    assert all(row["schema_row_allowed"] == "true" for row in modules)


def test_phase371_locks_key_dimensions_and_kernel_metadata_policy() -> None:
    rows = analyzer.analyze_phase371_route_a_perfdb_schema_design_spec(PHASE370)
    key_row = next(row for row in rows if row["row_type"] == "key_dimensions")
    kernel = next(row for row in rows if row["row_type"] == "kernel_source_policy")

    assert key_row["key_dimensions"] == EXPECTED_KEY_DIMENSIONS
    assert key_row["kernel_source_lookup_key"] == "false"
    assert kernel["kernel_source_metadata"] == "metadata_only"
    assert kernel["kernel_source_lookup_key"] == "false"


def test_phase371_locks_bucket_whitelist_and_excludes_128() -> None:
    rows = analyzer.analyze_phase371_route_a_perfdb_schema_design_spec(PHASE370)
    bucket = next(row for row in rows if row["row_type"] == "bucket_policy")

    assert bucket["allowed_bucket_tokens"] == EXPECTED_BUCKETS
    assert bucket["excluded_bucket_tokens"] == "128"
    assert bucket["bucket_interpolation_allowed"] == "false"
    assert bucket["bucket_extrapolation_allowed"] == "false"


def test_phase371_blocks_runtime_changes_perfdb_write_and_default_aic() -> None:
    rows = analyzer.analyze_phase371_route_a_perfdb_schema_design_spec(PHASE370)
    prohibited = next(row for row in rows if row["row_type"] == "prohibited_actions")
    next_phase = next(row for row in rows if row["row_type"] == "next_phase")

    assert prohibited["perfdb_write_allowed"] == "false"
    assert prohibited["loader_query_change_allowed"] == "false"
    assert prohibited["default_aic_allowed"] == "false"
    assert next_phase["next_allowed_phase"] == "phase372_loader_query_change_plan"
    assert next_phase["loader_query_change_allowed"] == "false"


def test_phase371_preserves_no_go_diagnostic_flags() -> None:
    rows = analyzer.analyze_phase371_route_a_perfdb_schema_design_spec(PHASE370)

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
    expected = analyzer.analyze_phase371_route_a_perfdb_schema_design_spec(PHASE370)

    with CHECKED_IN_CSV.open(newline="", encoding="utf-8") as handle:
        checked_in = list(csv.DictReader(handle))

    assert checked_in == expected


def test_phase370_input_must_be_schema_design_ready(tmp_path: Path) -> None:
    with PHASE370.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    assert fieldnames is not None
    for row in rows:
        if row["row_type"] == "perfdb_schema_design_ready":
            row["perfdb_schema_design_ready"] = "false"
            break

    bad = tmp_path / "phase370.csv"
    with bad.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="perfdb_schema_design_ready"):
        analyzer.analyze_phase371_route_a_perfdb_schema_design_spec(bad)


def test_output_rejects_perfdb_or_loader_query_upgrade(tmp_path: Path) -> None:
    rows = analyzer.analyze_phase371_route_a_perfdb_schema_design_spec(PHASE370)
    rows[0]["perf_database"] = "true"

    with pytest.raises(ValueError, match="perf_database"):
        analyzer.write_phase371_csv(tmp_path / "bad.csv", rows)

    rows = analyzer.analyze_phase371_route_a_perfdb_schema_design_spec(PHASE370)
    rows[-1]["loader_query_change_allowed"] = "true"
    with pytest.raises(ValueError, match="loader_query_change_allowed"):
        analyzer.write_phase371_csv(tmp_path / "bad.csv", rows)


def test_writers_emit_csv_and_md(tmp_path: Path) -> None:
    rows = analyzer.analyze_phase371_route_a_perfdb_schema_design_spec(PHASE370)
    csv_path = tmp_path / "phase371.csv"
    md_path = tmp_path / "phase371.md"

    analyzer.write_phase371_csv(csv_path, rows)
    analyzer.write_phase371_md(md_path, rows)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 8
    assert set(written[0]) == set(analyzer.FIELDNAMES)

    doc = md_path.read_text(encoding="utf-8")
    assert "module-level row only" in doc
    assert "fusedmoe_runner_compute" in doc
    assert "ep8_comm_dispatch_combine" in doc
    assert EXPECTED_KEY_DIMENSIONS in doc
    assert "kernel source remains metadata only" in doc
    assert "Phase372" in doc
    assert "Default AIC | No-Go" in doc
    assert "PerfDatabase | not written" in doc
