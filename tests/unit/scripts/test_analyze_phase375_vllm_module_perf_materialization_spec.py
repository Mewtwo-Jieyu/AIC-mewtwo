from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "scripts"
    / "analyze_phase375_vllm_module_perf_materialization_spec.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase375_vllm_module_perf_materialization_spec",
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
    / "docs/iter_gap_investigation/phase375_vllm_module_perf_materialization_spec.csv"
)
EXPECTED_BUCKETS = ["1", "15", "16", "241", "1808", "2048", "8192"]
EXPECTED_MODULES = [
    "fusedmoe_runner_compute",
    "ep8_comm_dispatch_combine",
]
EXPECTED_KEY_COLUMNS = (
    "model;hardware;vllm_version;topology;bucket_tokens;module_boundary;quant_runtime"
)


def test_phase375_outputs_exact_14_candidate_materialization_rows() -> None:
    rows = analyzer.analyze_phase375_vllm_module_perf_materialization_spec(
        PHASE366, PHASE369
    )

    assert len(rows) == 14
    assert {row["row_type"] for row in rows} == {"candidate_materialization_row"}
    assert [row["module_boundary"] for row in rows[:7]] == [
        "fusedmoe_runner_compute"
    ] * 7
    assert [row["module_boundary"] for row in rows[7:]] == [
        "ep8_comm_dispatch_combine"
    ] * 7
    assert [row["bucket_tokens"] for row in rows[:7]] == EXPECTED_BUCKETS
    assert [row["bucket_tokens"] for row in rows[7:]] == EXPECTED_BUCKETS


def test_phase375_materializes_from_phase366_and_phase369_only() -> None:
    rows = analyzer.analyze_phase375_vllm_module_perf_materialization_spec(
        PHASE366, PHASE369
    )
    fused = rows[:7]
    ep8 = rows[7:]

    assert {row["source_phase"] for row in fused} == {
        "phase369_fusedmoe_runner_shape_sweep_result"
    }
    assert {row["source_latency_field"] for row in fused} == {"latency_ms_median"}
    assert fused[0]["latency_ms"] == "0.187888"
    assert fused[-1]["latency_ms"] == "20.612520"

    assert {row["source_phase"] for row in ep8} == {
        "phase366_ep8_comm_minimal_shape_sweep_result"
    }
    assert {row["source_latency_field"] for row in ep8} == {"latency_ms"}
    assert ep8[0]["latency_ms"] == "0.157088"
    assert ep8[-1]["latency_ms"] == "2.237024"


def test_phase375_locks_vllm_module_file_schema_and_key_policy() -> None:
    rows = analyzer.analyze_phase375_vllm_module_perf_materialization_spec(
        PHASE366, PHASE369
    )

    for row in rows:
        assert row["materialization_target"] == "vllm_module_perf.txt"
        assert row["hardware"] == "h200_sxm"
        assert row["key_columns"] == EXPECTED_KEY_COLUMNS
        assert row["exact_lookup_only"] == "true"
        assert row["kernel_source_lookup_key"] == "false"
        assert row["power_default"] == "0.0"
        assert row["energy_formula"] == "power*latency"
        assert row["duplicate_key_policy"] == "fail_fast"
        assert row["missing_exact_key_policy"] == "fail_fast"


def test_phase375_rejects_bucket_128_or_unexpected_bucket(tmp_path: Path) -> None:
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
        analyzer.analyze_phase375_vllm_module_perf_materialization_spec(bad, PHASE369)


def test_phase375_rejects_phase369_quant_or_finite_drift(tmp_path: Path) -> None:
    with PHASE369.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    assert fieldnames is not None
    rows[0]["quant_runtime"] = "OtherQuantRuntime"

    bad = tmp_path / "phase369.csv"
    with bad.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="quant_runtime"):
        analyzer.analyze_phase375_vllm_module_perf_materialization_spec(PHASE366, bad)


def test_phase375_preserves_diagnostic_no_write_no_default_flags() -> None:
    rows = analyzer.analyze_phase375_vllm_module_perf_materialization_spec(
        PHASE366, PHASE369
    )

    for row in rows:
        assert row["gpu_allowed"] == "false"
        assert row["write_real_data_allowed"] == "false"
        assert row["next_allowed_phase"] == "phase376_write_vllm_module_perf_rows"
        assert row["default_readiness"] == "No-Go"
        assert row["diagnostic_only"] == "true"
        assert row["valid_for_default"] == "false"
        assert row["perf_database"] == "false"
        assert row["curve_fit_allowed"] == "false"
        assert row["interpolation_allowed"] == "false"
        assert row["extrapolation_allowed"] == "false"


def test_checked_in_csv_matches_analyzer_output() -> None:
    expected = analyzer.analyze_phase375_vllm_module_perf_materialization_spec(
        PHASE366, PHASE369
    )

    with CHECKED_IN_CSV.open(newline="", encoding="utf-8") as handle:
        checked_in = list(csv.DictReader(handle))

    assert checked_in == expected


def test_output_rejects_real_data_write_or_default_upgrade(tmp_path: Path) -> None:
    rows = analyzer.analyze_phase375_vllm_module_perf_materialization_spec(
        PHASE366, PHASE369
    )
    rows[0]["write_real_data_allowed"] = "true"

    with pytest.raises(ValueError, match="write_real_data_allowed"):
        analyzer.write_phase375_csv(tmp_path / "bad.csv", rows)

    rows = analyzer.analyze_phase375_vllm_module_perf_materialization_spec(
        PHASE366, PHASE369
    )
    rows[0]["perf_database"] = "true"
    with pytest.raises(ValueError, match="perf_database"):
        analyzer.write_phase375_csv(tmp_path / "bad.csv", rows)


def test_writers_emit_csv_and_md(tmp_path: Path) -> None:
    rows = analyzer.analyze_phase375_vllm_module_perf_materialization_spec(
        PHASE366, PHASE369
    )
    csv_path = tmp_path / "phase375.csv"
    md_path = tmp_path / "phase375.md"

    analyzer.write_phase375_csv(csv_path, rows)
    analyzer.write_phase375_md(md_path, rows)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 14
    assert set(written[0]) == set(analyzer.FIELDNAMES)

    doc = md_path.read_text(encoding="utf-8")
    assert "14 candidate rows" in doc
    assert "vllm_module_perf.txt" in doc
    assert "h200_sxm" in doc
    assert "fusedmoe_runner_compute" in doc
    assert "ep8_comm_dispatch_combine" in doc
    assert "bucket `128` remains excluded" in doc
    assert "kernel source remains metadata only" in doc
    assert "Phase376" in doc
    assert "Default AIC | No-Go" in doc
    assert "PerfDatabase | not written" in doc
