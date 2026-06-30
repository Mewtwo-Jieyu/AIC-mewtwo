from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase385_fusedmoe_paired_bucket_gpu_data_spec.py"
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase385_fusedmoe_paired_bucket_gpu_data_spec",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


PHASE384_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase384_paired_bucket_expansion_spec.csv"
CHECKED_IN_CSV = (
    REPO_ROOT / "docs/iter_gap_investigation/phase385_fusedmoe_paired_bucket_gpu_data_spec.csv"
)

EXPECTED_ROW_TYPES = [
    "phase384_prerequisite",
    "target_bucket_set_locked",
    "measurement_boundary_locked",
    "environment_key_locked",
    "quant_runtime_locked",
    "collection_artifact_contract",
    "no_ep8_recollection",
    "preserve_existing_data_rows",
    "lookup_policy_exact_only",
    "bucket_128_blocked",
    "gpu_run_deferred",
    "perfdb_write_blocked",
    "default_aic_blocked",
    "next_phase",
]


def _rows() -> list[dict[str, str]]:
    return analyzer.analyze_phase385_fusedmoe_paired_bucket_gpu_data_spec(
        phase384_csv=PHASE384_CSV,
    )


def test_phase385_outputs_gpu_data_spec_rows() -> None:
    rows = _rows()

    assert [row["row_type"] for row in rows] == EXPECTED_ROW_TYPES
    assert len(rows) == 14


def test_phase385_reads_phase384_paired_bucket_expansion_spec() -> None:
    row = next(row for row in _rows() if row["row_type"] == "phase384_prerequisite")

    assert row["phase384_prerequisite"] == "paired_bucket_expansion_spec_passed"
    assert row["phase384_next_allowed_phase"] == "phase385_fusedmoe_paired_bucket_gpu_data_spec"
    assert row["existing_rows_preserved"] == "true"
    assert row["existing_module_row_count"] == "14"


def test_phase385_locks_target_fusedmoe_paired_buckets_only() -> None:
    row = next(row for row in _rows() if row["row_type"] == "target_bucket_set_locked")

    assert row["target_bucket_tokens"] == "2/30/32/482/3616/4096/16384"
    assert row["target_bucket_count"] == "7"
    assert row["bucket_128_allowed"] == "false"
    assert "128" not in row["target_bucket_tokens"].split("/")


def test_phase385_locks_runner_boundary_and_environment_key() -> None:
    boundary = next(row for row in _rows() if row["row_type"] == "measurement_boundary_locked")
    env = next(row for row in _rows() if row["row_type"] == "environment_key_locked")
    quant = next(row for row in _rows() if row["row_type"] == "quant_runtime_locked")

    assert boundary["measurement_boundary"] == "fusedmoe_forward_runner_level"
    assert boundary["runner_boundary_api"] == "FusedMoE.forward()"
    assert env["hardware"] == "h200_sxm"
    assert env["vllm_version"] == "0.19.0"
    assert env["topology"] == "tp4dp2ep8"
    assert quant["quant_runtime"] == "CompressedTensorsWNA16MarlinMoEMethod"


def test_phase385_forbids_ep8_recollection_and_real_data_write() -> None:
    ep8 = next(row for row in _rows() if row["row_type"] == "no_ep8_recollection")
    preserve = next(row for row in _rows() if row["row_type"] == "preserve_existing_data_rows")

    assert ep8["ep8_recollection_allowed"] == "false"
    assert preserve["existing_rows_preserved"] == "true"
    assert preserve["delete_existing_rows"] == "false"
    assert preserve["write_real_data_file"] == "false"


def test_phase385_keeps_exact_only_and_no_go_flags() -> None:
    for row in _rows():
        assert row["gpu_allowed"] == "false"
        assert row["ssh_allowed"] == "false"
        assert row["runtime_change"] == "false"
        assert row["new_perfdb_data"] == "false"
        assert row["write_real_data_file"] == "false"
        assert row["default_aic_allowed"] == "false"
        assert row["diagnostic_only"] == "true"
        assert row["valid_for_default"] == "false"
        assert row["perf_database"] == "false"
        assert row["default_readiness"] == "No-Go"
        assert row["interpolation_allowed"] == "false"
        assert row["extrapolation_allowed"] == "false"
        assert row["nearest_bucket_allowed"] == "false"


def test_phase385_next_phase_is_gpu_run_decision_not_run() -> None:
    row = next(row for row in _rows() if row["row_type"] == "next_phase")

    assert row["next_allowed_phase"] == "phase386_fusedmoe_paired_bucket_gpu_run_decision"
    assert row["gpu_allowed"] == "false"


def test_checked_in_csv_matches_analyzer_output() -> None:
    expected = _rows()

    with CHECKED_IN_CSV.open(newline="", encoding="utf-8") as handle:
        checked_in = list(csv.DictReader(handle))

    assert checked_in == expected


def test_writers_reject_gpu_real_data_or_default_upgrade(tmp_path: Path) -> None:
    rows = _rows()
    rows[0]["gpu_allowed"] = "true"
    with pytest.raises(ValueError, match="gpu_allowed"):
        analyzer.write_phase385_csv(tmp_path / "bad.csv", rows)

    rows = _rows()
    rows[0]["write_real_data_file"] = "true"
    with pytest.raises(ValueError, match="write_real_data_file"):
        analyzer.write_phase385_csv(tmp_path / "bad.csv", rows)

    rows = _rows()
    rows[0]["valid_for_default"] = "true"
    with pytest.raises(ValueError, match="valid_for_default"):
        analyzer.write_phase385_csv(tmp_path / "bad.csv", rows)


def test_writers_emit_csv_and_md(tmp_path: Path) -> None:
    rows = _rows()
    csv_path = tmp_path / "phase385.csv"
    md_path = tmp_path / "phase385.md"

    analyzer.write_phase385_csv(csv_path, rows)
    analyzer.write_phase385_md(md_path, rows)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert written == rows
    assert set(written[0]) == set(analyzer.FIELDNAMES)

    doc = md_path.read_text(encoding="utf-8")
    assert "FusedMoE.forward() runner level" in doc
    assert "2/30/32/482/3616/4096/16384" in doc
    assert "Do not collect EP8 again" in doc
    assert "Do not run GPU in Phase385" in doc
