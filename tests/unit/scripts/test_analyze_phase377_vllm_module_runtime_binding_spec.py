from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT / "scripts" / "analyze_phase377_vllm_module_runtime_binding_spec.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase377_vllm_module_runtime_binding_spec",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


PHASE376_TABLE = (
    REPO_ROOT
    / "src/aiconfigurator/systems/data/h200_sxm/vllm/0.19.0/vllm_module_perf.txt"
)
CHECKED_IN_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase377_vllm_module_runtime_binding_spec.csv"
)
EXPECTED_BUCKETS = "1/15/16/241/1808/2048/8192"
EXPECTED_ROW_TYPES = [
    "phase376_table_prerequisite",
    "moe_query_runtime_binding",
    "moe_dispatch_query_runtime_binding",
    "runtime_key_source_contract",
    "bucket_exact_only_contract",
    "kernel_source_metadata_policy",
    "prohibited_runtime_behaviors",
    "next_phase",
]


def test_phase377_outputs_exact_runtime_binding_rows() -> None:
    rows = analyzer.analyze_phase377_vllm_module_runtime_binding_spec(PHASE376_TABLE)

    assert [row["row_type"] for row in rows] == EXPECTED_ROW_TYPES
    assert len(rows) == 8


def test_phase377_requires_phase376_real_table_14_exact_keys() -> None:
    rows = analyzer.analyze_phase377_vllm_module_runtime_binding_spec(PHASE376_TABLE)
    prerequisite = rows[0]

    assert prerequisite["phase376_table"] == str(PHASE376_TABLE)
    assert prerequisite["phase376_exact_key_count"] == "14"
    assert prerequisite["module_boundaries"] == (
        "fusedmoe_runner_compute;ep8_comm_dispatch_combine"
    )
    assert prerequisite["allowed_bucket_tokens"] == EXPECTED_BUCKETS


def test_phase377_binds_runtime_operations_to_two_module_boundaries() -> None:
    rows = analyzer.analyze_phase377_vllm_module_runtime_binding_spec(PHASE376_TABLE)
    moe = rows[1]
    dispatch = rows[2]

    assert moe["runtime_operation"] == "MoE.query(...)"
    assert moe["module_boundary"] == "fusedmoe_runner_compute"
    assert dispatch["runtime_operation"] == "MoEDispatch.query(...)"
    assert dispatch["module_boundary"] == "ep8_comm_dispatch_combine"
    assert {moe["module_boundary"], dispatch["module_boundary"]} == {
        "fusedmoe_runner_compute",
        "ep8_comm_dispatch_combine",
    }


def test_phase377_locks_runtime_key_sources_and_version_scope() -> None:
    rows = analyzer.analyze_phase377_vllm_module_runtime_binding_spec(PHASE376_TABLE)
    key = next(row for row in rows if row["row_type"] == "runtime_key_source_contract")

    assert key["hardware_source"] == "database.system"
    assert key["hardware_required"] == "h200_sxm"
    assert key["vllm_version_source"] == "database.version"
    assert key["vllm_version_required"] == "0.19.0"
    assert key["topology_required"] == "tp4dp2ep8"
    assert key["quant_runtime_required"] == "CompressedTensorsWNA16MarlinMoEMethod"


def test_phase377_allows_only_exact_buckets_and_excludes_128() -> None:
    rows = analyzer.analyze_phase377_vllm_module_runtime_binding_spec(PHASE376_TABLE)
    bucket = next(row for row in rows if row["row_type"] == "bucket_exact_only_contract")

    assert bucket["allowed_bucket_tokens"] == EXPECTED_BUCKETS
    assert bucket["excluded_bucket_tokens"] == "128"
    assert bucket["bucket_selection"] == "exact_only"
    assert bucket["nearest_bucket_allowed"] == "false"
    assert bucket["interpolation_allowed"] == "false"
    assert bucket["extrapolation_allowed"] == "false"


def test_phase377_preserves_metadata_only_kernel_source_and_no_go_flags() -> None:
    rows = analyzer.analyze_phase377_vllm_module_runtime_binding_spec(PHASE376_TABLE)
    metadata = next(row for row in rows if row["row_type"] == "kernel_source_metadata_policy")

    assert metadata["kernel_source_metadata"] == "metadata_only"
    assert metadata["kernel_source_lookup_key"] == "false"
    for row in rows:
        assert row["runtime_changed"] == "false"
        assert row["operations_py_changed"] == "false"
        assert row["gpu_allowed"] == "false"
        assert row["default_readiness"] == "No-Go"
        assert row["diagnostic_only"] == "true"
        assert row["valid_for_default"] == "false"
        assert row["perf_database"] == "false"


def test_phase377_rejects_bad_table_bucket_or_key_count(tmp_path: Path) -> None:
    with PHASE376_TABLE.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    assert fieldnames is not None
    rows[0]["bucket_tokens"] = "128"

    bad = tmp_path / "vllm_module_perf.txt"
    with bad.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="Phase376 buckets"):
        analyzer.analyze_phase377_vllm_module_runtime_binding_spec(bad)


def test_checked_in_csv_matches_analyzer_output() -> None:
    expected = analyzer.analyze_phase377_vllm_module_runtime_binding_spec(PHASE376_TABLE)

    with CHECKED_IN_CSV.open(newline="", encoding="utf-8") as handle:
        checked_in = list(csv.DictReader(handle))

    assert checked_in == expected


def test_output_rejects_interpolation_default_or_runtime_upgrade(tmp_path: Path) -> None:
    rows = analyzer.analyze_phase377_vllm_module_runtime_binding_spec(PHASE376_TABLE)
    rows[4]["interpolation_allowed"] = "true"

    with pytest.raises(ValueError, match="interpolation_allowed"):
        analyzer.write_phase377_csv(tmp_path / "bad.csv", rows)

    rows = analyzer.analyze_phase377_vllm_module_runtime_binding_spec(PHASE376_TABLE)
    rows[0]["valid_for_default"] = "true"
    with pytest.raises(ValueError, match="valid_for_default"):
        analyzer.write_phase377_csv(tmp_path / "bad.csv", rows)

    rows = analyzer.analyze_phase377_vllm_module_runtime_binding_spec(PHASE376_TABLE)
    rows[1]["runtime_changed"] = "true"
    with pytest.raises(ValueError, match="runtime_changed"):
        analyzer.write_phase377_csv(tmp_path / "bad.csv", rows)


def test_writers_emit_csv_and_md(tmp_path: Path) -> None:
    rows = analyzer.analyze_phase377_vllm_module_runtime_binding_spec(PHASE376_TABLE)
    csv_path = tmp_path / "phase377.csv"
    md_path = tmp_path / "phase377.md"

    analyzer.write_phase377_csv(csv_path, rows)
    analyzer.write_phase377_md(md_path, rows)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 8
    assert set(written[0]) == set(analyzer.FIELDNAMES)

    doc = md_path.read_text(encoding="utf-8")
    assert "MoE.query(...)" in doc
    assert "MoEDispatch.query(...)" in doc
    assert "fusedmoe_runner_compute" in doc
    assert "ep8_comm_dispatch_combine" in doc
    assert "h200_sxm" in doc
    assert "0.19.0" in doc
    assert "1/15/16/241/1808/2048/8192" in doc
    assert "bucket `128` remains rejected" in doc
    assert "Default AIC | No-Go" in doc
    assert "operations.py | unchanged" in doc
