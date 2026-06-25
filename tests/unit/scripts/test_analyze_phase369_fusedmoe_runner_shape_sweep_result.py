from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT / "scripts" / "analyze_phase369_fusedmoe_runner_shape_sweep_result.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase369_fusedmoe_runner_shape_sweep_result",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


PHASE367 = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase367_fusedmoe_runner_shape_sweep_spec.csv"
)
CHECKED_IN_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase369_fusedmoe_runner_shape_sweep_result.csv"
)
EXPECTED_BUCKETS = ["1", "15", "16", "241", "1808", "2048", "8192"]
EXPECTED_LATENCIES = [
    "0.187888",
    "0.242813",
    "0.239764",
    "1.889204",
    "4.729947",
    "5.344776",
    "20.612520",
]


def test_phase369_outputs_exact_seven_bucket_rows() -> None:
    rows = analyzer.analyze_phase369_fusedmoe_runner_shape_sweep_result(PHASE367)

    assert [row["bucket_tokens"] for row in rows] == EXPECTED_BUCKETS
    assert [row["latency_ms_median"] for row in rows] == EXPECTED_LATENCIES
    assert len(rows) == 7


def test_checked_in_csv_matches_analyzer_output() -> None:
    expected = analyzer.analyze_phase369_fusedmoe_runner_shape_sweep_result(PHASE367)

    with CHECKED_IN_CSV.open(newline="", encoding="utf-8") as handle:
        checked_in = list(csv.DictReader(handle))

    assert checked_in == expected


def test_phase369_rejects_128_as_shape_sweep_result() -> None:
    rows = analyzer.analyze_phase369_fusedmoe_runner_shape_sweep_result(PHASE367)

    assert "128" not in {row["bucket_tokens"] for row in rows}
    assert all(
        row["result_interpretation"]
        == "diagnostic_fusedmoe_runner_shape_sweep_not_perfdb_curve"
        for row in rows
    )


def test_phase369_preserves_runner_boundary_quant_and_kernel_metadata() -> None:
    rows = analyzer.analyze_phase369_fusedmoe_runner_shape_sweep_result(PHASE367)

    for row in rows:
        assert row["worker"] == "worker-gn6kz"
        assert row["vllm_version"] == "0.19.0"
        assert row["measurement_boundary"] == "fusedmoe_forward_runner_level"
        assert row["runner_boundary_api"] == "FusedMoE.forward()"
        assert row["quant_runtime"] == "CompressedTensorsWNA16MarlinMoEMethod"
        assert row["kernel_source_metadata"] == "CompressedTensorsWNA16MarlinMoEMethod:Marlin"


def test_phase369_preserves_output_shape_finite_and_cleanup_guards() -> None:
    rows = analyzer.analyze_phase369_fusedmoe_runner_shape_sweep_result(PHASE367)

    for row in rows:
        bucket = row["bucket_tokens"]
        assert row["ok"] == "true"
        assert row["output_shape"] == f"{bucket}x7168"
        assert row["output_all_finite"] == "true"
        assert row["cleanup"] == "true"
        assert row["gpu_process_residue"] == "false"
        assert row["failure_reason"] == ""
        assert float(row["latency_ms_median"]) > 0.0


def test_phase369_preserves_no_curve_no_go_flags() -> None:
    rows = analyzer.analyze_phase369_fusedmoe_runner_shape_sweep_result(PHASE367)

    for row in rows:
        assert row["curve_fit_allowed"] == "false"
        assert row["interpolation_allowed"] == "false"
        assert row["extrapolation_allowed"] == "false"
        assert row["default_readiness"] == "No-Go"
        assert row["diagnostic_only"] == "true"
        assert row["valid_for_default"] == "false"
        assert row["perf_database"] == "false"


def test_phase367_input_must_define_fusedmoe_shape_sweep_spec(tmp_path: Path) -> None:
    with PHASE367.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    assert fieldnames is not None
    for row in rows:
        if row["row_type"] == "measurement_boundary":
            row["measurement_boundary"] = "bare_fused_experts"
            break

    bad = tmp_path / "phase367.csv"
    with bad.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="measurement_boundary"):
        analyzer.analyze_phase369_fusedmoe_runner_shape_sweep_result(bad)


def test_output_rejects_perfdb_or_default_curve_upgrade(tmp_path: Path) -> None:
    rows = analyzer.analyze_phase369_fusedmoe_runner_shape_sweep_result(PHASE367)
    rows[0]["perf_database"] = "true"

    with pytest.raises(ValueError, match="perf_database"):
        analyzer.write_phase369_csv(tmp_path / "bad.csv", rows)


def test_output_rejects_shape_mismatch(tmp_path: Path) -> None:
    rows = analyzer.analyze_phase369_fusedmoe_runner_shape_sweep_result(PHASE367)
    rows[0]["output_shape"] = "128x7168"

    with pytest.raises(ValueError, match="output_shape"):
        analyzer.write_phase369_csv(tmp_path / "bad.csv", rows)


def test_writers_emit_csv_and_md(tmp_path: Path) -> None:
    rows = analyzer.analyze_phase369_fusedmoe_runner_shape_sweep_result(PHASE367)
    csv_path = tmp_path / "phase369.csv"
    md_path = tmp_path / "phase369.md"

    analyzer.write_phase369_csv(csv_path, rows)
    analyzer.write_phase369_md(md_path, rows)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 7
    assert set(written[0]) == set(analyzer.FIELDNAMES)

    doc = md_path.read_text(encoding="utf-8")
    assert "Phase368 FusedMoE runner minimal shape sweep passed" in doc
    assert "FusedMoE.forward()" in doc
    assert "CompressedTensorsWNA16MarlinMoEMethod" in doc
    assert "`128` is not included" in doc
    assert "cannot be interpolated or extrapolated into a PerfDatabase curve" in doc
    assert "Default AIC | No-Go" in doc
    assert "PerfDatabase | not written" in doc
