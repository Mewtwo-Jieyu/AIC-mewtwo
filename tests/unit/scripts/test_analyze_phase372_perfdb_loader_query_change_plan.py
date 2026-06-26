from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase372_perfdb_loader_query_change_plan.py"
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase372_perfdb_loader_query_change_plan",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


PHASE371 = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase371_route_a_perfdb_schema_design_spec.csv"
)
CHECKED_IN_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase372_perfdb_loader_query_change_plan.csv"
)
EXPECTED_ROW_TYPES = [
    "phase371_schema_spec_prerequisite",
    "change_scope_plan_only",
    "vllm_module_lookup_boundary",
    "allowed_module_boundaries",
    "key_dimensions_contract",
    "kernel_source_metadata_policy",
    "bucket_whitelist",
    "prohibited_actions",
    "next_phase",
]
EXPECTED_KEY_DIMENSIONS = (
    "model;hardware;vllm_version;topology;bucket_tokens;module_boundary;quant_runtime"
)
EXPECTED_BUCKETS = "1/15/16/241/1808/2048/8192"
EXPECTED_MODULES = "fusedmoe_runner_compute;ep8_comm_dispatch_combine"


def test_phase372_outputs_exact_loader_query_plan_rows() -> None:
    rows = analyzer.analyze_phase372_perfdb_loader_query_change_plan(PHASE371)

    assert [row["row_type"] for row in rows] == EXPECTED_ROW_TYPES
    assert len(rows) == 9


def test_phase372_requires_phase371_schema_design_spec(tmp_path: Path) -> None:
    with PHASE371.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    assert fieldnames is not None
    rows[-1]["next_allowed_phase"] = "phase372_wrong_target"

    bad = tmp_path / "phase371.csv"
    with bad.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="phase372_loader_query_change_plan"):
        analyzer.analyze_phase372_perfdb_loader_query_change_plan(bad)


def test_phase372_is_plan_only_and_does_not_change_runtime() -> None:
    rows = analyzer.analyze_phase372_perfdb_loader_query_change_plan(PHASE371)
    scope = next(row for row in rows if row["row_type"] == "change_scope_plan_only")
    lookup = next(row for row in rows if row["row_type"] == "vllm_module_lookup_boundary")

    assert scope["change_kind"] == "plan_only"
    assert scope["loader_query_implemented"] == "false"
    assert scope["runtime_changed"] == "false"
    assert lookup["lookup_design"] == "add_vllm_module_level_lookup"
    assert lookup["replace_existing_trtllm_path"] == "false"


def test_phase372_locks_allowed_modules_key_dimensions_and_kernel_policy() -> None:
    rows = analyzer.analyze_phase372_perfdb_loader_query_change_plan(PHASE371)
    modules = next(row for row in rows if row["row_type"] == "allowed_module_boundaries")
    keys = next(row for row in rows if row["row_type"] == "key_dimensions_contract")
    kernel = next(row for row in rows if row["row_type"] == "kernel_source_metadata_policy")

    assert modules["module_boundaries"] == EXPECTED_MODULES
    assert modules["module_boundary_extension_allowed"] == "false"
    assert keys["key_dimensions"] == EXPECTED_KEY_DIMENSIONS
    assert kernel["kernel_source_metadata"] == "metadata_only"
    assert kernel["kernel_source_lookup_key"] == "false"


def test_phase372_locks_bucket_whitelist_and_excludes_128() -> None:
    rows = analyzer.analyze_phase372_perfdb_loader_query_change_plan(PHASE371)
    bucket = next(row for row in rows if row["row_type"] == "bucket_whitelist")

    assert bucket["allowed_bucket_tokens"] == EXPECTED_BUCKETS
    assert bucket["excluded_bucket_tokens"] == "128"
    assert bucket["bucket_interpolation_allowed"] == "false"
    assert bucket["bucket_extrapolation_allowed"] == "false"


def test_phase372_blocks_perfdb_write_fit_default_and_gpu() -> None:
    rows = analyzer.analyze_phase372_perfdb_loader_query_change_plan(PHASE371)
    prohibited = next(row for row in rows if row["row_type"] == "prohibited_actions")

    assert prohibited["perfdb_write_allowed"] == "false"
    assert prohibited["curve_fit_allowed"] == "false"
    assert prohibited["interpolation_allowed"] == "false"
    assert prohibited["extrapolation_allowed"] == "false"
    assert prohibited["default_aic_allowed"] == "false"
    assert all(row["gpu_allowed"] == "false" for row in rows)


def test_phase372_preserves_no_go_diagnostic_flags() -> None:
    rows = analyzer.analyze_phase372_perfdb_loader_query_change_plan(PHASE371)

    for row in rows:
        assert row["default_readiness"] == "No-Go"
        assert row["diagnostic_only"] == "true"
        assert row["valid_for_default"] == "false"
        assert row["perf_database"] == "false"


def test_checked_in_csv_matches_analyzer_output() -> None:
    expected = analyzer.analyze_phase372_perfdb_loader_query_change_plan(PHASE371)

    with CHECKED_IN_CSV.open(newline="", encoding="utf-8") as handle:
        checked_in = list(csv.DictReader(handle))

    assert checked_in == expected


def test_output_rejects_loader_query_implementation_or_perfdb_write(tmp_path: Path) -> None:
    rows = analyzer.analyze_phase372_perfdb_loader_query_change_plan(PHASE371)
    rows[1]["loader_query_implemented"] = "true"

    with pytest.raises(ValueError, match="loader_query_implemented"):
        analyzer.write_phase372_csv(tmp_path / "bad.csv", rows)

    rows = analyzer.analyze_phase372_perfdb_loader_query_change_plan(PHASE371)
    rows[7]["perfdb_write_allowed"] = "true"
    with pytest.raises(ValueError, match="perfdb_write_allowed"):
        analyzer.write_phase372_csv(tmp_path / "bad.csv", rows)


def test_writers_emit_csv_and_md(tmp_path: Path) -> None:
    rows = analyzer.analyze_phase372_perfdb_loader_query_change_plan(PHASE371)
    csv_path = tmp_path / "phase372.csv"
    md_path = tmp_path / "phase372.md"

    analyzer.write_phase372_csv(csv_path, rows)
    analyzer.write_phase372_md(md_path, rows)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 9
    assert set(written[0]) == set(analyzer.FIELDNAMES)

    doc = md_path.read_text(encoding="utf-8")
    assert "plan-only" in doc
    assert "add vLLM module-level lookup design" in doc
    assert "does not replace the existing TRT-LLM path" in doc
    assert EXPECTED_MODULES in doc
    assert EXPECTED_KEY_DIMENSIONS in doc
    assert "kernel source remains metadata only" in doc
    assert "bucket `128` remains excluded" in doc
    assert "Phase373" in doc
    assert "Default AIC | No-Go" in doc
    assert "PerfDatabase | not written" in doc
