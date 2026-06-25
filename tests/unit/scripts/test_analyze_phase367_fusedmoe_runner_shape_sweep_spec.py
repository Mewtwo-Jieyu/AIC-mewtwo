from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT / "scripts" / "analyze_phase367_fusedmoe_runner_shape_sweep_spec.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase367_fusedmoe_runner_shape_sweep_spec",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


PHASE366 = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase366_ep8_comm_minimal_shape_sweep_result.csv"
)
EXPECTED_BUCKETS = ["1", "15", "16", "241", "1808", "2048", "8192"]


def test_phase367_outputs_exact_spec_rows() -> None:
    rows = analyzer.analyze_phase367_fusedmoe_runner_shape_sweep_spec(PHASE366)

    assert [row["row_type"] for row in rows] == [
        "prerequisite",
        "measurement_boundary",
        "smoke_bucket",
        "shape_bucket",
        "shape_bucket",
        "shape_bucket",
        "shape_bucket",
        "shape_bucket",
        "shape_bucket",
        "shape_bucket",
        "quant_runtime",
        "kernel_metadata",
        "environment_gate",
        "stop_rules",
        "next_phase",
    ]
    assert rows[0]["decision"] == "phase366_ep8_comm_shape_sweep_pass_required"
    assert rows[1]["measurement_boundary"] == "FusedMoE.forward()"
    assert rows[-1]["next_allowed_phase"] == "phase368_fusedmoe_runner_minimal_shape_sweep"


def test_phase367_locks_real_buckets_and_excludes_128() -> None:
    rows = analyzer.analyze_phase367_fusedmoe_runner_shape_sweep_spec(PHASE366)
    shape_buckets = [row for row in rows if row["row_type"] == "shape_bucket"]

    assert [row["bucket_tokens"] for row in shape_buckets] == EXPECTED_BUCKETS
    assert "128" not in {row["bucket_tokens"] for row in shape_buckets}
    assert all(row["bucket_role"] == "phase124_real_bucket" for row in shape_buckets)
    smoke = next(row for row in rows if row["row_type"] == "smoke_bucket")
    assert smoke["bucket_tokens"] == "128"
    assert smoke["bucket_role"] == "fusedmoe_runner_smoke_only_not_shape_sweep"


def test_phase367_requires_fusedmoe_runner_boundary_and_quant_runtime() -> None:
    rows = analyzer.analyze_phase367_fusedmoe_runner_shape_sweep_spec(PHASE366)
    boundary = next(row for row in rows if row["row_type"] == "measurement_boundary")
    quant = next(row for row in rows if row["row_type"] == "quant_runtime")
    metadata = next(row for row in rows if row["row_type"] == "kernel_metadata")

    assert boundary["measurement_boundary"] == "FusedMoE.forward()"
    assert boundary["runner_level_boundary"] == "true"
    assert quant["quant_runtime"] == "CompressedTensorsWNA16MarlinMoEMethod"
    assert metadata["kernel_source_metadata"] == "true"
    assert metadata["kernel_source_lookup_key"] == "false"


def test_phase367_requires_cuda_compat_and_phase368_worker_entry() -> None:
    rows = analyzer.analyze_phase367_fusedmoe_runner_shape_sweep_spec(PHASE366)
    env = next(row for row in rows if row["row_type"] == "environment_gate")
    next_phase = next(row for row in rows if row["row_type"] == "next_phase")

    assert (
        env["ld_library_path_prefix"]
        == "/usr/local/cuda-12.9/compat:/usr/local/nvidia/lib64"
    )
    assert env["vllm_enable_cuda_compatibility"] == "1"
    assert next_phase["future_gpu_ssh"] == (
        "ssh -CAXY "
        "ws-faaf0de74ef9a14d-worker-gn6kz.zhaojieyu+root.ailab-sys.pod"
        "@h.pjlab.org.cn"
    )
    assert next_phase["gpu_allowed"] == "false"


def test_phase367_stop_rules_are_fail_fast() -> None:
    rows = analyzer.analyze_phase367_fusedmoe_runner_shape_sweep_spec(PHASE366)
    stop = next(row for row in rows if row["row_type"] == "stop_rules")

    assert stop["decision"] == "fail_fast_no_retry_no_bare_kernel_fallback"
    for token in [
        "ptx_failure",
        "forward_context_failure",
        "quant_runtime_drift",
        "gpu_or_process_residue",
    ]:
        assert token in stop["stop_rule"]


def test_phase367_preserves_no_go_diagnostic_flags() -> None:
    rows = analyzer.analyze_phase367_fusedmoe_runner_shape_sweep_spec(PHASE366)

    for row in rows:
        assert row["gpu_allowed"] == "false"
        assert row["default_readiness"] == "No-Go"
        assert row["diagnostic_only"] == "true"
        assert row["valid_for_default"] == "false"
        assert row["perf_database"] == "false"
        assert row["curve_fit_allowed"] == "false"
        assert row["interpolation_allowed"] == "false"
        assert row["extrapolation_allowed"] == "false"


def test_phase366_input_must_be_passed_ep8_comm_shape_sweep(tmp_path: Path) -> None:
    with PHASE366.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    assert fieldnames is not None
    rows[0]["ok"] = "false"

    bad = tmp_path / "phase366.csv"
    with bad.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="ok"):
        analyzer.analyze_phase367_fusedmoe_runner_shape_sweep_spec(bad)


def test_output_rejects_perfdb_or_default_upgrade(tmp_path: Path) -> None:
    rows = analyzer.analyze_phase367_fusedmoe_runner_shape_sweep_spec(PHASE366)
    rows[0]["perf_database"] = "true"

    with pytest.raises(ValueError, match="perf_database"):
        analyzer.write_phase367_csv(tmp_path / "bad.csv", rows)


def test_writers_emit_csv_and_md(tmp_path: Path) -> None:
    rows = analyzer.analyze_phase367_fusedmoe_runner_shape_sweep_spec(PHASE366)
    csv_path = tmp_path / "phase367.csv"
    md_path = tmp_path / "phase367.md"

    analyzer.write_phase367_csv(csv_path, rows)
    analyzer.write_phase367_md(md_path, rows)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 15
    assert set(written[0]) == set(analyzer.FIELDNAMES)

    doc = md_path.read_text(encoding="utf-8")
    assert "FusedMoE.forward() runner level" in doc
    assert "`1`, `15`, `16`, `241`, `1808`, `2048`, `8192`" in doc
    assert "`128` remains smoke-only" in doc
    assert "CompressedTensorsWNA16MarlinMoEMethod" in doc
    assert "kernel source is metadata only" in doc
    assert "Default AIC | No-Go" in doc
    assert "PerfDatabase | not written" in doc
