from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT / "scripts" / "analyze_phase364_ep8_comm_shape_expansion_spec.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase364_ep8_comm_shape_expansion_spec",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


PHASE363 = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase363_ep8_comm_single_point_sufficiency_gate.csv"
)


def test_phase364_outputs_shape_expansion_spec_rows() -> None:
    rows = analyzer.analyze_phase364_ep8_comm_shape_expansion_spec(PHASE363)

    assert [row["row_type"] for row in rows] == [
        "prerequisite",
        "smoke_bucket",
        "real_bucket",
        "real_bucket",
        "real_bucket",
        "real_bucket",
        "real_bucket",
        "real_bucket",
        "real_bucket",
        "backend_gate",
        "environment_gate",
        "stop_rules",
        "next_phase",
    ]
    assert rows[0]["decision"] == "phase363_single_point_insufficient_curve_fit_disallowed"
    assert rows[1]["bucket_tokens"] == "128"
    assert rows[1]["bucket_role"] == "runner_comm_smoke_only_not_real_coverage"
    assert rows[-1]["next_allowed_phase"] == "phase365_minimal_shape_sweep"


def test_phase364_locks_phase124_real_bucket_sequence() -> None:
    rows = analyzer.analyze_phase364_ep8_comm_shape_expansion_spec(PHASE363)
    real_buckets = [row for row in rows if row["row_type"] == "real_bucket"]

    assert [row["bucket_tokens"] for row in real_buckets] == [
        "1",
        "15",
        "16",
        "241",
        "1808",
        "2048",
        "8192",
    ]
    assert all(row["bucket_role"] == "phase124_real_bucket" for row in real_buckets)
    assert all(row["decision"] == "required_for_minimal_shape_sweep" for row in real_buckets)


def test_phase364_keeps_128_smoke_out_of_real_coverage() -> None:
    rows = analyzer.analyze_phase364_ep8_comm_shape_expansion_spec(PHASE363)
    real_bucket_tokens = {
        row["bucket_tokens"] for row in rows if row["row_type"] == "real_bucket"
    }

    assert "128" not in real_bucket_tokens
    smoke = next(row for row in rows if row["row_type"] == "smoke_bucket")
    assert smoke["curve_fit_allowed"] == "false"
    assert smoke["perfdb_curve_allowed"] == "false"


def test_phase364_requires_runtime_backend_manager_and_cuda_compat() -> None:
    rows = analyzer.analyze_phase364_ep8_comm_shape_expansion_spec(PHASE363)
    backend = next(row for row in rows if row["row_type"] == "backend_gate")
    env = next(row for row in rows if row["row_type"] == "environment_gate")

    assert backend["backend_required"] == "allgather_reducescatter"
    assert backend["manager_required"] == "AgRsAll2AllManager"
    assert backend["decision"] == "runtime_capture_backend_and_manager_per_bucket"
    assert (
        env["ld_library_path_prefix"]
        == "/usr/local/cuda-12.9/compat:/usr/local/nvidia/lib64"
    )
    assert env["vllm_enable_cuda_compatibility"] == "1"


def test_phase364_stop_rules_fail_fast_without_retry_or_fallback() -> None:
    rows = analyzer.analyze_phase364_ep8_comm_shape_expansion_spec(PHASE363)
    stop = next(row for row in rows if row["row_type"] == "stop_rules")

    assert stop["decision"] == "fail_fast_no_retry_no_fallback"
    for token in [
        "ptx_failure",
        "context_failure",
        "unknown_backend",
        "gpu_or_process_residue",
        "backend_drift_from_allgather_reducescatter",
    ]:
        assert token in stop["stop_rule"]


def test_phase364_records_future_gpu_entry_but_keeps_gpu_disallowed() -> None:
    rows = analyzer.analyze_phase364_ep8_comm_shape_expansion_spec(PHASE363)
    next_phase = next(row for row in rows if row["row_type"] == "next_phase")

    assert next_phase["future_gpu_ssh"] == (
        "ssh -CAXY "
        "ws-faaf0de74ef9a14d-worker-gn6kz.zhaojieyu+root.ailab-sys.pod"
        "@h.pjlab.org.cn"
    )
    assert next_phase["gpu_allowed"] == "false"
    assert next_phase["decision"] == "phase365_minimal_shape_sweep_only_not_full_gpu_matrix"


def test_phase364_preserves_no_go_diagnostic_flags() -> None:
    rows = analyzer.analyze_phase364_ep8_comm_shape_expansion_spec(PHASE363)

    for row in rows:
        assert row["gpu_allowed"] == "false"
        assert row["default_readiness"] == "No-Go"
        assert row["diagnostic_only"] == "true"
        assert row["valid_for_default"] == "false"
        assert row["perf_database"] == "false"


def test_phase363_input_must_keep_single_point_insufficient_gate(tmp_path: Path) -> None:
    with PHASE363.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    assert fieldnames is not None
    for row in rows:
        if row["gate"] == "single_point_sufficiency":
            row["verdict"] = "sufficient_for_perfdb_curve"

    bad = tmp_path / "phase363.csv"
    with bad.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="single_point_sufficiency"):
        analyzer.analyze_phase364_ep8_comm_shape_expansion_spec(bad)


def test_output_rejects_gpu_or_perfdb_upgrade(tmp_path: Path) -> None:
    rows = analyzer.analyze_phase364_ep8_comm_shape_expansion_spec(PHASE363)
    rows[0]["gpu_allowed"] = "true"

    with pytest.raises(ValueError, match="gpu_allowed"):
        analyzer.write_phase364_csv(tmp_path / "bad.csv", rows)


def test_writers_emit_csv_and_md(tmp_path: Path) -> None:
    rows = analyzer.analyze_phase364_ep8_comm_shape_expansion_spec(PHASE363)
    csv_path = tmp_path / "phase364.csv"
    md_path = tmp_path / "phase364.md"

    analyzer.write_phase364_csv(csv_path, rows)
    analyzer.write_phase364_md(md_path, rows)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 13
    assert set(written[0]) == set(analyzer.FIELDNAMES)

    doc = md_path.read_text(encoding="utf-8")
    assert "Phase363 proved the EP8 single point is insufficient" in doc
    assert "`128` remains a runner/comm smoke point" in doc
    assert "`1`, `15`, `16`, `241`, `1808`, `2048`, `8192`" in doc
    assert "runtime capture `backend` and `manager`" in doc
    assert "VLLM_ENABLE_CUDA_COMPATIBILITY=1" in doc
    assert "Phase365 minimal shape sweep" in doc
    assert "Default AIC | No-Go" in doc
    assert "PerfDatabase | not written" in doc
