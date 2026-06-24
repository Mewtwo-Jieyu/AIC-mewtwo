from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT / "scripts" / "analyze_phase360_ep8_alltoall_single_point_spec.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase360_ep8_alltoall_single_point_spec",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


PHASE359B = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase359b_fusedmoe_cuda_compat_smoke_result.csv"
)


def test_phase360_outputs_exact_five_spec_rows() -> None:
    rows = analyzer.analyze_phase360_ep8_alltoall_single_point_spec(PHASE359B)

    assert [row["spec_row"] for row in rows] == [
        "phase359b_runner_smoke_prerequisite",
        "ep8_comm_boundary",
        "runtime_backend_capture",
        "single_point_artifact_contract",
        "stop_rules",
    ]
    assert rows[0]["decision"] == "requires_phase359b_pass_and_cuda_compat_preload"
    assert rows[1]["decision"] == "measure_vllm_0190_runner_level_ep_comm_path"
    assert rows[2]["decision"] == "capture_actual_runtime_backend_not_route_label"
    assert rows[3]["decision"] == "emit_one_diagnostic_csv_row_only"
    assert rows[4]["decision"] == "fail_fast_no_retry_no_bare_kernel_fallback"


def test_phase360_preserves_no_go_diagnostic_flags() -> None:
    rows = analyzer.analyze_phase360_ep8_alltoall_single_point_spec(PHASE359B)

    for row in rows:
        assert row["gpu_allowed"] == "false"
        assert row["default_readiness"] == "No-Go"
        assert row["diagnostic_only"] == "true"
        assert row["valid_for_default"] == "false"
        assert row["perf_database"] == "false"


def test_phase360_requires_phase359b_smoke_pass_and_compat_preload() -> None:
    rows = analyzer.analyze_phase360_ep8_alltoall_single_point_spec(PHASE359B)
    prerequisite = rows[0]

    assert prerequisite["required_phase359b_event"] == (
        "fusedmoe_forward_runner_smoke_passed"
    )
    assert prerequisite["required_phase359b_ok"] == "true"
    assert prerequisite["required_cuda_compat_preload"] == (
        "LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:/usr/local/nvidia/lib64"
    )
    assert prerequisite["required_vllm_enable_cuda_compatibility"] == "1"


def test_phase360_comm_boundary_rejects_wideep_and_total_fusedmoe_latency() -> None:
    rows = analyzer.analyze_phase360_ep8_alltoall_single_point_spec(PHASE359B)
    boundary = rows[1]

    assert boundary["measurement_boundary"] == "vllm_0190_runner_level_ep_comm_path"
    assert boundary["schema_reuse"] == "reject_wideep_schema"
    assert boundary["fusedmoe_total_latency"] == "not_comm_boundary"
    assert "WideEP" not in boundary["measurement_contract"]
    assert "FusedMoE total latency" not in boundary["measurement_contract"]


def test_phase360_requires_runtime_backend_capture() -> None:
    rows = analyzer.analyze_phase360_ep8_alltoall_single_point_spec(PHASE359B)
    backend = rows[2]

    assert backend["route_label"] == "ep8_alltoall_single_point_spec"
    assert backend["runtime_backend_required"] == (
        "allgather_reducescatter;deepep_*;flashinfer_*;other_explicit_backend"
    )
    assert backend["generic_backend_allowed"] == "false"
    assert "not_alltoall_string_only" in backend["measurement_contract"]


def test_phase360_artifact_contract_and_stop_rules_are_strict() -> None:
    rows = analyzer.analyze_phase360_ep8_alltoall_single_point_spec(PHASE359B)
    artifact = rows[3]
    stop_rules = rows[4]

    assert artifact["artifact_contract"] == (
        "one_row_csv:worker,vllm_version,backend,shape,latency_ms,ok,cleanup"
    )
    assert artifact["next_allowed_phase"] == "phase361_ep8_comm_single_point_gpu_run"
    assert stop_rules["stop_rule"] == (
        "ptx_failure;context_failure;unknown_backend;gpu_or_process_residue"
    )
    assert stop_rules["retry_policy"] == "no_retry_no_bare_kernel_fallback"
    assert stop_rules["next_allowed_phase"] == (
        "phase361_only_if_spec_inputs_are_available"
    )


def test_writers_emit_csv_and_md(tmp_path: Path) -> None:
    rows = analyzer.analyze_phase360_ep8_alltoall_single_point_spec(PHASE359B)
    csv_path = tmp_path / "phase360.csv"
    md_path = tmp_path / "phase360.md"

    analyzer.write_phase360_csv(csv_path, rows)
    analyzer.write_phase360_md(md_path, rows)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 5
    assert set(written[0]) == set(analyzer.FIELDNAMES)

    doc = md_path.read_text(encoding="utf-8")
    assert "Phase360 is a run spec, not performance evidence" in doc
    assert "Phase361 is the first allowed GPU single-point run" in doc
    assert "WideEP schema is not reused" in doc
    assert "FusedMoE total latency is not the EP8 comm boundary" in doc
    assert "actual runtime backend must be captured" in doc
    assert "Default AIC | No-Go" in doc
    assert "PerfDatabase | not written" in doc


def test_output_rejects_default_or_perfdb_upgrade(tmp_path: Path) -> None:
    rows = analyzer.analyze_phase360_ep8_alltoall_single_point_spec(PHASE359B)
    rows[2]["perf_database"] = "true"

    with pytest.raises(ValueError, match="perf_database"):
        analyzer.write_phase360_csv(tmp_path / "bad.csv", rows)


def test_phase359b_input_must_be_runner_smoke_pass(tmp_path: Path) -> None:
    with PHASE359B.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    assert fieldnames is not None
    rows[-1]["ok"] = "false"

    bad = tmp_path / "phase359b.csv"
    with bad.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="ok"):
        analyzer.analyze_phase360_ep8_alltoall_single_point_spec(bad)
