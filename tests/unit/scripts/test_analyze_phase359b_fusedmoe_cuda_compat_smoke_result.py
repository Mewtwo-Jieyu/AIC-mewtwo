from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "scripts"
    / "analyze_phase359b_fusedmoe_cuda_compat_smoke_result.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase359b_fusedmoe_cuda_compat_smoke_result",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


PHASE356 = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase356_route_a_fusedmoe_runner_boundary_decision.csv"
)


def test_phase359b_outputs_four_rows() -> None:
    rows = analyzer.analyze_phase359b_fusedmoe_cuda_compat_smoke_result(PHASE356)

    assert [row["event"] for row in rows] == [
        "initial_direct_python_smoke_failed",
        "process_start_cuda_compat_preload",
        "per_forward_context_required",
        "fusedmoe_forward_runner_smoke_passed",
    ]
    assert rows[-1]["ok"] == "true"
    assert rows[-1]["latency_ms_median"] == "1.039261"
    assert rows[-1]["output_shape"] == "128x7168"
    assert rows[-1]["output_all_finite"] == "true"


def test_phase359b_preserves_diagnostic_flags() -> None:
    rows = analyzer.analyze_phase359b_fusedmoe_cuda_compat_smoke_result(PHASE356)

    for row in rows:
        assert row["default_readiness"] == "No-Go"
        assert row["diagnostic_only"] == "true"
        assert row["valid_for_default"] == "false"
        assert row["perf_database"] == "false"


def test_phase359b_records_cuda_compat_preload_requirement() -> None:
    rows = analyzer.analyze_phase359b_fusedmoe_cuda_compat_smoke_result(PHASE356)
    preload = rows[1]

    assert preload["decision"] == "ptx_cleared_by_process_start_cuda_compat_preload"
    assert preload["ld_library_path_prefix"] == (
        "/usr/local/cuda-12.9/compat:/usr/local/nvidia/lib64"
    )
    assert preload["vllm_enable_cuda_compatibility"] == "1"
    assert "direct_smoke_process_start" in preload["interpretation"]


def test_phase359b_records_forward_context_requirement() -> None:
    rows = analyzer.analyze_phase359b_fusedmoe_cuda_compat_smoke_result(PHASE356)
    context = rows[2]

    assert context["decision"] == "requires_per_forward_set_forward_context"
    assert context["forward_context"] == "per_forward_set_forward_context_none_num_tokens_128"
    assert context["static_all_moe_layers"] == (
        "phase358_cuda_compat_per_forward_context_retry"
    )
    assert "vllm_serve_model_runner_lifecycle" in context["interpretation"]


def test_phase359b_keeps_kernel_source_as_metadata_only() -> None:
    rows = analyzer.analyze_phase359b_fusedmoe_cuda_compat_smoke_result(PHASE356)

    for row in rows:
        assert row["kernel_source_metadata"] == (
            "CompressedTensorsWNA16MarlinMoEMethod:Marlin"
        )
        assert row["kernel_source_lookup_key"] == "false"
        assert row["perf_database_row"] == "false"


def test_writers_emit_csv_and_md(tmp_path: Path) -> None:
    rows = analyzer.analyze_phase359b_fusedmoe_cuda_compat_smoke_result(PHASE356)
    csv_path = tmp_path / "phase359b.csv"
    md_path = tmp_path / "phase359b.md"

    analyzer.write_phase359b_csv(csv_path, rows)
    analyzer.write_phase359b_md(md_path, rows)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 4
    assert set(written[0]) == set(analyzer.FIELDNAMES)

    doc = md_path.read_text(encoding="utf-8")
    assert "VLLM_ENABLE_CUDA_COMPATIBILITY=1 is effective" in doc
    assert "direct smoke is not equivalent to vLLM serve worker subprocess" in doc
    assert "kernel_source stays measurement metadata" in doc
    assert "not a PerfDatabase row" in doc
    assert "Default AIC | No-Go" in doc
    assert "default-ready" not in doc


def test_output_rejects_default_or_perfdb_upgrade(tmp_path: Path) -> None:
    rows = analyzer.analyze_phase359b_fusedmoe_cuda_compat_smoke_result(PHASE356)
    rows[-1]["valid_for_default"] = "true"

    with pytest.raises(ValueError, match="valid_for_default"):
        analyzer.write_phase359b_csv(tmp_path / "bad.csv", rows)


def test_phase356_input_must_keep_fusedmoe_runner_boundary(tmp_path: Path) -> None:
    with PHASE356.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    assert fieldnames is not None
    rows[0]["measurement_boundary"] = "bare_fused_experts_probe"
    bad = tmp_path / "phase356.csv"
    with bad.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="measurement_boundary"):
        analyzer.analyze_phase359b_fusedmoe_cuda_compat_smoke_result(bad)
