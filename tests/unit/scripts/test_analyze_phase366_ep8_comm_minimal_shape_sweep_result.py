from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT / "scripts" / "analyze_phase366_ep8_comm_minimal_shape_sweep_result.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase366_ep8_comm_minimal_shape_sweep_result",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


PHASE364 = (
    REPO_ROOT / "docs/iter_gap_investigation/phase364_ep8_comm_shape_expansion_spec.csv"
)
EXPECTED_BUCKETS = ["1", "15", "16", "241", "1808", "2048", "8192"]
EXPECTED_LATENCIES = [
    "0.157088",
    "0.109920",
    "0.083776",
    "0.288960",
    "1.192704",
    "1.237760",
    "2.237024",
]


def test_phase366_outputs_exact_seven_bucket_rows() -> None:
    rows = analyzer.analyze_phase366_ep8_comm_minimal_shape_sweep_result(PHASE364)

    assert [row["bucket_tokens"] for row in rows] == EXPECTED_BUCKETS
    assert [row["latency_ms"] for row in rows] == EXPECTED_LATENCIES
    assert len(rows) == 7


def test_phase366_rejects_128_as_shape_sweep_bucket() -> None:
    rows = analyzer.analyze_phase366_ep8_comm_minimal_shape_sweep_result(PHASE364)

    assert "128" not in {row["bucket_tokens"] for row in rows}
    assert all(row["result_interpretation"] == "diagnostic_shape_sweep_not_perfdb_curve" for row in rows)


def test_phase366_preserves_runtime_backend_manager_capture() -> None:
    rows = analyzer.analyze_phase366_ep8_comm_minimal_shape_sweep_result(PHASE364)

    for row in rows:
        assert row["worker"] == "worker-gn6kz"
        assert row["vllm_version"] == "0.19.0"
        assert row["measurement_boundary"] == "vllm_ep_group_dispatch_router_logits_plus_combine"
        assert row["backend"] == "allgather_reducescatter"
        assert row["manager"] == "AgRsAll2AllManager"
        assert row["configured_backend"] == "allgather_reducescatter"


def test_phase366_preserves_cleanup_and_rank_guards() -> None:
    rows = analyzer.analyze_phase366_ep8_comm_minimal_shape_sweep_result(PHASE364)

    for row in rows:
        assert row["ok"] == "true"
        assert row["rank_error"] == "0"
        assert row["cleanup"] == "true"
        assert row["gpu_process_residue"] == "false"
        assert row["failure_reason"] == ""
        assert float(row["latency_ms"]) > 0.0


def test_phase366_preserves_no_curve_no_go_flags() -> None:
    rows = analyzer.analyze_phase366_ep8_comm_minimal_shape_sweep_result(PHASE364)

    for row in rows:
        assert row["curve_fit_allowed"] == "false"
        assert row["interpolation_allowed"] == "false"
        assert row["extrapolation_allowed"] == "false"
        assert row["default_readiness"] == "No-Go"
        assert row["diagnostic_only"] == "true"
        assert row["valid_for_default"] == "false"
        assert row["perf_database"] == "false"


def test_phase364_input_must_define_real_buckets_without_128(tmp_path: Path) -> None:
    with PHASE364.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    assert fieldnames is not None
    for row in rows:
        if row["row_type"] == "real_bucket" and row["bucket_tokens"] == "1":
            row["bucket_tokens"] = "128"
            break

    bad = tmp_path / "phase364.csv"
    with bad.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="real buckets"):
        analyzer.analyze_phase366_ep8_comm_minimal_shape_sweep_result(bad)


def test_output_rejects_default_or_perfdb_curve_upgrade(tmp_path: Path) -> None:
    rows = analyzer.analyze_phase366_ep8_comm_minimal_shape_sweep_result(PHASE364)
    rows[0]["curve_fit_allowed"] = "true"

    with pytest.raises(ValueError, match="curve_fit_allowed"):
        analyzer.write_phase366_csv(tmp_path / "bad.csv", rows)


def test_writers_emit_csv_and_md(tmp_path: Path) -> None:
    rows = analyzer.analyze_phase366_ep8_comm_minimal_shape_sweep_result(PHASE364)
    csv_path = tmp_path / "phase366.csv"
    md_path = tmp_path / "phase366.md"

    analyzer.write_phase366_csv(csv_path, rows)
    analyzer.write_phase366_md(md_path, rows)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 7
    assert set(written[0]) == set(analyzer.FIELDNAMES)

    doc = md_path.read_text(encoding="utf-8")
    assert "Phase365 EP8 comm minimal shape sweep passed" in doc
    assert "allgather_reducescatter" in doc
    assert "AgRsAll2AllManager" in doc
    assert "`128` is not included" in doc
    assert "cannot be interpolated or extrapolated into a PerfDatabase curve" in doc
    assert "Default AIC | No-Go" in doc
    assert "PerfDatabase | not written" in doc
