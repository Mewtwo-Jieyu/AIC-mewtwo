from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "scripts"
    / "analyze_phase389_fusedmoe_paired_bucket_materialization_sufficiency_gate.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase389_fusedmoe_paired_bucket_materialization_sufficiency_gate",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


PHASE388_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase388_fusedmoe_paired_bucket_gpu_run_result.csv"
)
VLLM_MODULE_PERF = (
    REPO_ROOT
    / "src/aiconfigurator/systems/data/h200_sxm/vllm/0.19.0/vllm_module_perf.txt"
)
CHECKED_IN_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase389_fusedmoe_paired_bucket_materialization_sufficiency_gate.csv"
)

EXPECTED_ROW_TYPES = [
    "phase388_prerequisite",
    "existing_vllm_module_table_verified",
    "paired_fusedmoe_rows_planned",
    "future_row_count_verified",
    "exact_lookup_policy_locked",
    "kernel_source_metadata_only",
    "phase389_no_write_no_default",
    "next_phase",
]
PLANNED_BUCKETS = "2/30/32/482/3616/4096/16384"
EXISTING_BUCKETS = "1/15/16/241/1808/2048/8192"


def _rows() -> list[dict[str, str]]:
    return analyzer.analyze_phase389_fusedmoe_paired_bucket_materialization_sufficiency_gate(
        phase388_csv=PHASE388_CSV,
        vllm_module_perf=VLLM_MODULE_PERF,
    )


def test_phase389_outputs_materialization_sufficiency_gate_rows() -> None:
    rows = _rows()

    assert [row["row_type"] for row in rows] == EXPECTED_ROW_TYPES
    assert len(rows) == 8
    assert {row["gate_type"] for row in rows} == {"materialization_sufficiency_gate"}


def test_phase389_reads_phase388_paired_bucket_result() -> None:
    prereq = next(row for row in _rows() if row["row_type"] == "phase388_prerequisite")

    assert prereq["phase388_prerequisite"] == "fusedmoe_paired_bucket_gpu_run_passed"
    assert prereq["planned_bucket_tokens"] == PLANNED_BUCKETS
    assert prereq["planned_added_row_count"] == "7"
    assert prereq["measurement_boundary"] == "fusedmoe_forward_runner_level"
    assert prereq["quant_runtime"] == "CompressedTensorsWNA16MarlinMoEMethod"


def test_phase389_verifies_existing_14_rows_and_future_21_rows() -> None:
    existing = next(row for row in _rows() if row["row_type"] == "existing_vllm_module_table_verified")
    planned = next(row for row in _rows() if row["row_type"] == "paired_fusedmoe_rows_planned")
    future = next(row for row in _rows() if row["row_type"] == "future_row_count_verified")

    assert existing["existing_row_count"] == "14"
    assert existing["existing_bucket_tokens"] == EXISTING_BUCKETS
    assert existing["existing_module_boundaries"] == (
        "fusedmoe_runner_compute;ep8_comm_dispatch_combine"
    )
    assert planned["planned_module_boundary"] == "fusedmoe_runner_compute"
    assert planned["planned_bucket_tokens"] == PLANNED_BUCKETS
    assert planned["planned_added_row_count"] == "7"
    assert future["existing_row_count"] == "14"
    assert future["planned_added_row_count"] == "7"
    assert future["future_row_count"] == "21"


def test_phase389_locks_exact_only_no_128_and_metadata_policy() -> None:
    exact = next(row for row in _rows() if row["row_type"] == "exact_lookup_policy_locked")
    metadata = next(row for row in _rows() if row["row_type"] == "kernel_source_metadata_only")

    assert exact["exact_lookup_only"] == "true"
    assert exact["nearest_lookup_allowed"] == "false"
    assert exact["interpolation_allowed"] == "false"
    assert exact["extrapolation_allowed"] == "false"
    assert exact["forbidden_bucket_tokens"] == "128"
    assert metadata["kernel_source_metadata"] == "CompressedTensorsWNA16MarlinMoEMethod:Marlin"
    assert metadata["kernel_source_lookup_key"] == "false"


def test_phase389_preserves_no_write_no_default_flags() -> None:
    for row in _rows():
        assert row["gpu_allowed"] == "false"
        assert row["ssh_allowed"] == "false"
        assert row["write_real_data_file"] == "false"
        assert row["default_aic_allowed"] == "false"
        assert row["diagnostic_only"] == "true"
        assert row["valid_for_default"] == "false"
        assert row["perf_database"] == "false"
        assert row["default_readiness"] == "No-Go"


def test_checked_in_csv_matches_analyzer_output() -> None:
    expected = _rows()

    with CHECKED_IN_CSV.open(newline="", encoding="utf-8") as handle:
        checked_in = list(csv.DictReader(handle))

    assert checked_in == expected


def test_phase388_input_must_be_complete_paired_bucket_pass(tmp_path: Path) -> None:
    with PHASE388_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    assert fieldnames is not None
    rows[0]["bucket_tokens"] = "128"

    bad = tmp_path / "phase388.csv"
    with bad.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="Phase388 buckets"):
        analyzer.analyze_phase389_fusedmoe_paired_bucket_materialization_sufficiency_gate(
            bad,
            VLLM_MODULE_PERF,
        )


def test_existing_table_must_have_14_rows_before_planning(tmp_path: Path) -> None:
    with VLLM_MODULE_PERF.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    assert fieldnames is not None
    bad = tmp_path / "vllm_module_perf.txt"
    with bad.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows[:-1])

    with pytest.raises(ValueError, match="existing vllm_module_perf row count"):
        analyzer.analyze_phase389_fusedmoe_paired_bucket_materialization_sufficiency_gate(
            PHASE388_CSV,
            bad,
        )


def test_writer_rejects_missing_bucket_128_or_default_upgrade(tmp_path: Path) -> None:
    rows = _rows()
    rows[2]["planned_bucket_tokens"] = "2/30/32"
    with pytest.raises(ValueError, match="planned_bucket_tokens"):
        analyzer.write_phase389_csv(tmp_path / "bad.csv", rows)

    rows = _rows()
    rows[2]["planned_bucket_tokens"] = "2/30/128"
    with pytest.raises(ValueError, match="planned_bucket_tokens"):
        analyzer.write_phase389_csv(tmp_path / "bad.csv", rows)

    rows = _rows()
    rows[0]["valid_for_default"] = "true"
    with pytest.raises(ValueError, match="valid_for_default"):
        analyzer.write_phase389_csv(tmp_path / "bad.csv", rows)

    rows = _rows()
    rows[0]["perf_database"] = "true"
    with pytest.raises(ValueError, match="perf_database"):
        analyzer.write_phase389_csv(tmp_path / "bad.csv", rows)

    rows = _rows()
    rows[0]["write_real_data_file"] = "true"
    with pytest.raises(ValueError, match="write_real_data_file"):
        analyzer.write_phase389_csv(tmp_path / "bad.csv", rows)


def test_writers_emit_csv_and_md(tmp_path: Path) -> None:
    rows = _rows()
    csv_path = tmp_path / "phase389.csv"
    md_path = tmp_path / "phase389.md"

    analyzer.write_phase389_csv(csv_path, rows)
    analyzer.write_phase389_md(md_path, rows)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert written == rows
    assert set(written[0]) == set(analyzer.FIELDNAMES)

    doc = md_path.read_text(encoding="utf-8")
    assert "materialization sufficiency gate" in doc
    assert "future row count | 21" in doc
    assert "existing 14 rows are preserved" in doc
    assert "Phase390" in doc
    assert "Default AIC | No-Go" in doc
