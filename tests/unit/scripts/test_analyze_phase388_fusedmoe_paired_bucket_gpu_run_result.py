from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "scripts"
    / "analyze_phase388_fusedmoe_paired_bucket_gpu_run_result.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase388_fusedmoe_paired_bucket_gpu_run_result",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


PHASE386_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase386_fusedmoe_paired_bucket_gpu_run_decision_spec.csv"
)
CHECKED_IN_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase388_fusedmoe_paired_bucket_gpu_run_result.csv"
)
EXPECTED_BUCKETS = ["2", "30", "32", "482", "3616", "4096", "16384"]
EXPECTED_LATENCIES = [
    "0.188482",
    "0.355168",
    "0.353423",
    "2.040215",
    "9.283537",
    "10.359648",
    "41.146568",
]


def _rows() -> list[dict[str, str]]:
    return analyzer.analyze_phase388_fusedmoe_paired_bucket_gpu_run_result(
        phase386_csv=PHASE386_CSV,
    )


def test_phase388_outputs_exact_paired_bucket_result_rows() -> None:
    rows = _rows()

    assert [row["bucket_tokens"] for row in rows] == EXPECTED_BUCKETS
    assert [row["latency_ms_median"] for row in rows] == EXPECTED_LATENCIES
    assert len(rows) == 7


def test_checked_in_csv_matches_analyzer_output() -> None:
    expected = _rows()

    with CHECKED_IN_CSV.open(newline="", encoding="utf-8") as handle:
        checked_in = list(csv.DictReader(handle))

    assert checked_in == expected


def test_phase388_preserves_phase387_artifact_and_source_sha() -> None:
    rows = _rows()

    for row in rows:
        assert row["phase387_artifact_dir"] == (
            "/mnt/shared-storage-user/ailab-sys/zhaojieyu/aic/"
            "phase387_fusedmoe_paired_bucket_gpu_run_2ee80b7"
        )
        assert row["source_sha"] == "2ee80b7c8e3a390227b7d19942f5ba6514e20ea1"
        assert row["worker"] == "worker-r28f2"


def test_phase388_preserves_runner_boundary_runtime_key_and_kernel_metadata() -> None:
    rows = _rows()

    for row in rows:
        assert row["hardware"] == "h200_sxm"
        assert row["vllm_version"] == "0.19.0"
        assert row["topology"] == "tp4dp2ep8"
        assert row["measurement_boundary"] == "fusedmoe_forward_runner_level"
        assert row["runner_boundary_api"] == "FusedMoE.forward()"
        assert row["quant_runtime"] == "CompressedTensorsWNA16MarlinMoEMethod"
        assert row["kernel_source_metadata"] == "CompressedTensorsWNA16MarlinMoEMethod:Marlin"


def test_phase388_preserves_output_shape_finite_and_cleanup_guards() -> None:
    rows = _rows()

    for row in rows:
        bucket = row["bucket_tokens"]
        assert row["ok"] == "true"
        assert row["output_shape"] == f"{bucket}x7168"
        assert row["output_all_finite"] == "true"
        assert row["cleanup"] == "true"
        assert row["gpu_process_residue"] == "false"
        assert row["failure_reason"] == ""
        assert float(row["latency_ms_median"]) > 0.0


def test_phase388_preserves_no_perfdb_no_default_flags() -> None:
    rows = _rows()

    for row in rows:
        assert row["diagnostic_only"] == "true"
        assert row["valid_for_default"] == "false"
        assert row["perf_database"] == "false"
        assert row["default_readiness"] == "No-Go"


def test_phase386_input_must_define_paired_bucket_gpu_run_decision(tmp_path: Path) -> None:
    with PHASE386_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    assert fieldnames is not None
    for row in rows:
        if row["row_type"] == "target_bucket_set_verified":
            row["target_bucket_tokens"] = "2/30/32"
            break

    bad = tmp_path / "phase386.csv"
    with bad.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="target_bucket_tokens"):
        analyzer.analyze_phase388_fusedmoe_paired_bucket_gpu_run_result(bad)


def test_output_rejects_128_or_missing_bucket(tmp_path: Path) -> None:
    rows = _rows()
    rows[0]["bucket_tokens"] = "128"
    with pytest.raises(ValueError, match="bucket sequence"):
        analyzer.write_phase388_csv(tmp_path / "bad.csv", rows)

    rows = _rows()[:-1]
    with pytest.raises(ValueError, match="exactly 7 rows"):
        analyzer.write_phase388_csv(tmp_path / "bad.csv", rows)


def test_output_rejects_shape_mismatch_or_default_upgrade(tmp_path: Path) -> None:
    rows = _rows()
    rows[0]["output_shape"] = "2x4096"
    with pytest.raises(ValueError, match="output_shape"):
        analyzer.write_phase388_csv(tmp_path / "bad.csv", rows)

    rows = _rows()
    rows[0]["perf_database"] = "true"
    with pytest.raises(ValueError, match="perf_database"):
        analyzer.write_phase388_csv(tmp_path / "bad.csv", rows)

    rows = _rows()
    rows[0]["valid_for_default"] = "true"
    with pytest.raises(ValueError, match="valid_for_default"):
        analyzer.write_phase388_csv(tmp_path / "bad.csv", rows)

    rows = _rows()
    rows[0]["default_readiness"] = "Ready"
    with pytest.raises(ValueError, match="default_readiness"):
        analyzer.write_phase388_csv(tmp_path / "bad.csv", rows)

    rows = _rows()
    rows[0]["result_interpretation"] = "perfdb_ready"
    with pytest.raises(ValueError, match="result_interpretation"):
        analyzer.write_phase388_csv(tmp_path / "bad.csv", rows)


def test_writers_emit_csv_and_md(tmp_path: Path) -> None:
    rows = _rows()
    csv_path = tmp_path / "phase388.csv"
    md_path = tmp_path / "phase388.md"

    analyzer.write_phase388_csv(csv_path, rows)
    analyzer.write_phase388_md(md_path, rows)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert written == rows
    assert set(written[0]) == set(analyzer.FIELDNAMES)

    doc = md_path.read_text(encoding="utf-8")
    assert "Phase387 FusedMoE paired bucket GPU run passed" in doc
    assert "FusedMoE.forward()" in doc
    assert "2`, `30`, `32`, `482`, `3616`, `4096`, `16384" in doc
    assert "does not write `vllm_module_perf.txt`" in doc
    assert "Default AIC | No-Go" in doc
