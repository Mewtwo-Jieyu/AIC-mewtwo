from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase373_loader_query_implementation_spec.py"
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase373_loader_query_implementation_spec",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


PHASE372 = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase372_perfdb_loader_query_change_plan.csv"
)
CHECKED_IN_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase373_loader_query_implementation_spec.csv"
)
EXPECTED_ROW_TYPES = [
    "phase372_plan_prerequisite",
    "data_file_contract",
    "future_code_entry_common",
    "future_code_entry_perf_database",
    "loader_contract",
    "query_contract",
    "existing_path_guard",
    "key_dimensions_contract",
    "module_boundary_contract",
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


def test_phase373_outputs_exact_implementation_spec_rows() -> None:
    rows = analyzer.analyze_phase373_loader_query_implementation_spec(PHASE372)

    assert [row["row_type"] for row in rows] == EXPECTED_ROW_TYPES
    assert len(rows) == 13


def test_phase373_requires_phase372_plan_only_input(tmp_path: Path) -> None:
    with PHASE372.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    assert fieldnames is not None
    rows[1]["loader_query_implemented"] = "true"

    bad = tmp_path / "phase372.csv"
    with bad.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="loader_query_implemented"):
        analyzer.analyze_phase373_loader_query_implementation_spec(bad)


def test_phase373_designs_new_vllm_module_data_file_only() -> None:
    rows = analyzer.analyze_phase373_loader_query_implementation_spec(PHASE372)
    data_file = next(row for row in rows if row["row_type"] == "data_file_contract")

    assert data_file["new_data_file"] == "vllm_module_perf.txt"
    assert data_file["existing_data_file_reused"] == "false"
    assert "moe_perf.txt" not in data_file["new_data_file"]


def test_phase373_identifies_future_code_entries_without_src_change() -> None:
    rows = analyzer.analyze_phase373_loader_query_implementation_spec(PHASE372)
    entries = [row["future_code_entry"] for row in rows if row["future_code_entry"]]

    assert entries == [
        "src/aiconfigurator/sdk/common.py",
        "src/aiconfigurator/sdk/perf_database.py",
    ]
    assert all(row["src_change_allowed"] == "false" for row in rows)


def test_phase373_designs_loader_and_query_without_implementation() -> None:
    rows = analyzer.analyze_phase373_loader_query_implementation_spec(PHASE372)
    loader = next(row for row in rows if row["row_type"] == "loader_contract")
    query = next(row for row in rows if row["row_type"] == "query_contract")

    assert loader["new_loader"] == "load_vllm_module_data"
    assert loader["new_loader_implemented"] == "false"
    assert query["new_query"] == "query_vllm_module"
    assert query["new_query_implemented"] == "false"


def test_phase373_preserves_existing_moe_trtllm_sglang_paths() -> None:
    rows = analyzer.analyze_phase373_loader_query_implementation_spec(PHASE372)
    guard = next(row for row in rows if row["row_type"] == "existing_path_guard")

    assert guard["existing_query_replaced"] == "false"
    assert guard["trtllm_sglang_path_changed"] == "false"
    assert "query_moe" in guard["decision"]


def test_phase373_locks_key_modules_kernel_policy_and_buckets() -> None:
    rows = analyzer.analyze_phase373_loader_query_implementation_spec(PHASE372)
    keys = next(row for row in rows if row["row_type"] == "key_dimensions_contract")
    modules = next(row for row in rows if row["row_type"] == "module_boundary_contract")
    kernel = next(row for row in rows if row["row_type"] == "kernel_source_metadata_policy")
    bucket = next(row for row in rows if row["row_type"] == "bucket_whitelist")

    assert keys["key_dimensions"] == EXPECTED_KEY_DIMENSIONS
    assert modules["module_boundaries"] == EXPECTED_MODULES
    assert kernel["kernel_source_metadata"] == "metadata_only"
    assert kernel["kernel_source_lookup_key"] == "false"
    assert bucket["allowed_bucket_tokens"] == EXPECTED_BUCKETS
    assert bucket["excluded_bucket_tokens"] == "128"
    assert bucket["bucket_interpolation_allowed"] == "false"
    assert bucket["bucket_extrapolation_allowed"] == "false"


def test_phase373_blocks_perfdb_rows_fit_gpu_and_default_aic() -> None:
    rows = analyzer.analyze_phase373_loader_query_implementation_spec(PHASE372)
    prohibited = next(row for row in rows if row["row_type"] == "prohibited_actions")

    assert prohibited["perfdb_row_write_allowed"] == "false"
    assert prohibited["curve_fit_allowed"] == "false"
    assert prohibited["interpolation_allowed"] == "false"
    assert prohibited["extrapolation_allowed"] == "false"
    assert prohibited["default_aic_allowed"] == "false"
    assert all(row["gpu_allowed"] == "false" for row in rows)


def test_phase373_preserves_no_go_diagnostic_flags() -> None:
    rows = analyzer.analyze_phase373_loader_query_implementation_spec(PHASE372)

    for row in rows:
        assert row["default_readiness"] == "No-Go"
        assert row["diagnostic_only"] == "true"
        assert row["valid_for_default"] == "false"
        assert row["perf_database"] == "false"


def test_checked_in_csv_matches_analyzer_output() -> None:
    expected = analyzer.analyze_phase373_loader_query_implementation_spec(PHASE372)

    with CHECKED_IN_CSV.open(newline="", encoding="utf-8") as handle:
        checked_in = list(csv.DictReader(handle))

    assert checked_in == expected


def test_output_rejects_src_change_or_perfdb_row_write(tmp_path: Path) -> None:
    rows = analyzer.analyze_phase373_loader_query_implementation_spec(PHASE372)
    rows[3]["src_change_allowed"] = "true"

    with pytest.raises(ValueError, match="src_change_allowed"):
        analyzer.write_phase373_csv(tmp_path / "bad.csv", rows)

    rows = analyzer.analyze_phase373_loader_query_implementation_spec(PHASE372)
    rows[11]["perfdb_row_write_allowed"] = "true"
    with pytest.raises(ValueError, match="perfdb_row_write_allowed"):
        analyzer.write_phase373_csv(tmp_path / "bad.csv", rows)


def test_writers_emit_csv_and_md(tmp_path: Path) -> None:
    rows = analyzer.analyze_phase373_loader_query_implementation_spec(PHASE372)
    csv_path = tmp_path / "phase373.csv"
    md_path = tmp_path / "phase373.md"

    analyzer.write_phase373_csv(csv_path, rows)
    analyzer.write_phase373_md(md_path, rows)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 13
    assert set(written[0]) == set(analyzer.FIELDNAMES)

    doc = md_path.read_text(encoding="utf-8")
    assert "implementation spec only" in doc
    assert "vllm_module_perf.txt" in doc
    assert "load_vllm_module_data" in doc
    assert "query_vllm_module" in doc
    assert "does not replace `query_moe(...)`" in doc
    assert EXPECTED_KEY_DIMENSIONS in doc
    assert EXPECTED_MODULES in doc
    assert "kernel source remains metadata only" in doc
    assert "bucket `128` remains excluded" in doc
    assert "Phase374" in doc
    assert "Default AIC | No-Go" in doc
    assert "PerfDatabase | not written" in doc
