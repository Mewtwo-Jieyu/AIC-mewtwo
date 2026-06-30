from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "scripts"
    / "analyze_phase386_fusedmoe_paired_bucket_gpu_run_decision_spec.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase386_fusedmoe_paired_bucket_gpu_run_decision_spec",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


PHASE385_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase385_fusedmoe_paired_bucket_gpu_data_spec.csv"
)
CHECKED_IN_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase386_fusedmoe_paired_bucket_gpu_run_decision_spec.csv"
)

EXPECTED_ROW_TYPES = [
    "phase385_prerequisite",
    "decision_type_locked",
    "target_bucket_set_verified",
    "measurement_boundary_verified",
    "environment_key_verified",
    "quant_runtime_verified",
    "stop_rules_locked",
    "artifact_contract_locked",
    "gpu_run_allowed_next_phase_only",
    "phase386_no_gpu_no_ssh",
    "perfdb_write_blocked",
    "default_aic_blocked",
    "next_phase",
]


def _rows() -> list[dict[str, str]]:
    return analyzer.analyze_phase386_fusedmoe_paired_bucket_gpu_run_decision_spec(
        phase385_csv=PHASE385_CSV,
    )


def test_phase386_outputs_gpu_run_decision_rows() -> None:
    rows = _rows()

    assert [row["row_type"] for row in rows] == EXPECTED_ROW_TYPES
    assert len(rows) == 13


def test_phase386_reads_phase385_without_turning_it_into_result() -> None:
    prereq = next(row for row in _rows() if row["row_type"] == "phase385_prerequisite")
    decision = next(row for row in _rows() if row["row_type"] == "decision_type_locked")

    assert prereq["phase385_prerequisite"] == "fusedmoe_paired_bucket_gpu_data_spec_passed"
    assert prereq["phase385_next_allowed_phase"] == "phase386_fusedmoe_paired_bucket_gpu_run_decision"
    assert decision["decision_type"] == "gpu_run_decision_spec"
    assert decision["gpu_result"] == "false"


def test_phase386_verifies_target_buckets_and_runtime_key() -> None:
    buckets = next(row for row in _rows() if row["row_type"] == "target_bucket_set_verified")
    boundary = next(row for row in _rows() if row["row_type"] == "measurement_boundary_verified")
    env = next(row for row in _rows() if row["row_type"] == "environment_key_verified")
    quant = next(row for row in _rows() if row["row_type"] == "quant_runtime_verified")

    assert buckets["target_bucket_tokens"] == "2/30/32/482/3616/4096/16384"
    assert buckets["target_bucket_count"] == "7"
    assert boundary["measurement_boundary"] == "fusedmoe_forward_runner_level"
    assert boundary["runner_boundary_api"] == "FusedMoE.forward()"
    assert env["hardware"] == "h200_sxm"
    assert env["vllm_version"] == "0.19.0"
    assert env["topology"] == "tp4dp2ep8"
    assert quant["quant_runtime"] == "CompressedTensorsWNA16MarlinMoEMethod"


def test_phase386_locks_stop_rules() -> None:
    row = next(row for row in _rows() if row["row_type"] == "stop_rules_locked")

    assert row["stop_rules"] == (
        "ptx_or_compat_error;"
        "forward_context_error;"
        "quant_runtime_mismatch;"
        "shape_or_output_non_finite;"
        "gpu_or_process_residue;"
        "missing_bucket"
    )


def test_phase386_allows_gpu_only_in_next_phase() -> None:
    run_gate = next(row for row in _rows() if row["row_type"] == "gpu_run_allowed_next_phase_only")
    current = next(row for row in _rows() if row["row_type"] == "phase386_no_gpu_no_ssh")
    next_phase = next(row for row in _rows() if row["row_type"] == "next_phase")

    assert run_gate["future_gpu_run_allowed"] == "true"
    assert current["gpu_allowed"] == "false"
    assert current["ssh_allowed"] == "false"
    assert next_phase["next_allowed_phase"] == "phase387_fusedmoe_paired_bucket_gpu_run"


def test_phase386_preserves_no_result_no_perfdb_no_default_flags() -> None:
    for row in _rows():
        assert row["gpu_allowed"] == "false"
        assert row["ssh_allowed"] == "false"
        assert row["runtime_change"] == "false"
        assert row["new_perfdb_data"] == "false"
        assert row["write_real_data_file"] == "false"
        assert row["gpu_result"] == "false"
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


def test_writers_reject_gpu_result_or_default_upgrade(tmp_path: Path) -> None:
    rows = _rows()
    rows[0]["gpu_result"] = "true"
    with pytest.raises(ValueError, match="gpu_result"):
        analyzer.write_phase386_csv(tmp_path / "bad.csv", rows)

    rows = _rows()
    rows[0]["gpu_allowed"] = "true"
    with pytest.raises(ValueError, match="gpu_allowed"):
        analyzer.write_phase386_csv(tmp_path / "bad.csv", rows)

    rows = _rows()
    rows[0]["valid_for_default"] = "true"
    with pytest.raises(ValueError, match="valid_for_default"):
        analyzer.write_phase386_csv(tmp_path / "bad.csv", rows)


def test_writers_emit_csv_and_md(tmp_path: Path) -> None:
    rows = _rows()
    csv_path = tmp_path / "phase386.csv"
    md_path = tmp_path / "phase386.md"

    analyzer.write_phase386_csv(csv_path, rows)
    analyzer.write_phase386_md(md_path, rows)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert written == rows
    assert set(written[0]) == set(analyzer.FIELDNAMES)

    doc = md_path.read_text(encoding="utf-8")
    assert "GPU run decision spec" in doc
    assert "This is not GPU result evidence" in doc
    assert "phase387_fusedmoe_paired_bucket_gpu_run" in doc
    assert "ptx_or_compat_error" in doc
