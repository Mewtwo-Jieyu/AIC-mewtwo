from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "scripts"
    / "analyze_phase356_route_a_fusedmoe_runner_boundary_decision.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase356_route_a_fusedmoe_runner_boundary_decision",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


PHASE349 = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase349_route_a_moe_measurement_api_contract.csv"
)
PHASE351 = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase351_route_a_kernel_source_key_contract.csv"
)


def _copy_csv_with_mutation(src: Path, dst: Path, mutate) -> Path:
    with src.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    assert fieldnames is not None
    mutate(rows)
    with dst.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return dst


def test_fusedmoe_runner_boundary_decision_outputs_one_row() -> None:
    rows = analyzer.analyze_route_a_fusedmoe_runner_boundary_decision(
        PHASE349,
        PHASE351,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["source"] == "phase356_route_a_fusedmoe_runner_boundary_decision"
    assert row["worker"] == "worker-892rz"
    assert row["vllm_version"] == "0.19.0"
    assert row["source_root"] == "/usr/local/lib/python3.12/dist-packages/vllm"
    assert row["measurement_boundary"] == "fusedmoe_forward_runner_level"
    assert row["runner_boundary_api"] == "FusedMoE.forward()"
    assert row["locked_kernel_path_guard_required"] == "false"
    assert row["locked_kernel_path_guard_decision"] == "superseded_not_applicable"
    assert row["kernel_source_lookup_key"] == "false"
    assert row["kernel_source_metadata"] == "true"
    assert row["runtime_dispatch_deterministic"] == "true"
    assert row["runtime_dispatch_scope"] == (
        "fixed_model_config_hw_vllm_version_tuple"
    )
    assert row["gpu_smoke_readiness"] == (
        "blocked_until_fusedmoe_forward_tensor_shapes_defined"
    )
    assert row["next_allowed_phase"] == "phase358_moe_tensor_shape_smoke_spec"
    assert row["gpu_allowed"] == "false"
    assert row["default_readiness"] == "No-Go"
    assert row["diagnostic_only"] == "true"
    assert row["valid_for_default"] == "false"
    assert row["perf_database"] == "false"


def test_phase355_evidence_is_preserved_without_blocking_runner_boundary() -> None:
    row = analyzer.analyze_route_a_fusedmoe_runner_boundary_decision(
        PHASE349,
        PHASE351,
    )[0]

    assert row["source_check_evidence"] == (
        "KimiMoE;FusedMoE;DefaultMoERunner;torch.ops.vllm.moe_forward"
    )
    assert row["source_check_interpretation"] == (
        "single_kernel_guard_not_applicable_runner_measurement_retained"
    )
    assert row["blocking_reasons"] == (
        "quant_method_runtime_selection;unquantized_backend_runtime_selection;"
        "shape_specific_fallback;multiple_expert_kernel_classes"
    )
    assert row["candidate_kernel_paths"] == (
        "Triton;Cutlass;DeepGemm;FlashInfer;Marlin;TRTLLM"
    )
    assert row["kernel_source_metadata"] == "true"
    assert row["kernel_source_lookup_key"] == "false"


def test_writers_emit_csv_and_doc(tmp_path: Path) -> None:
    rows = analyzer.analyze_route_a_fusedmoe_runner_boundary_decision(
        PHASE349,
        PHASE351,
    )
    csv_path = tmp_path / "out.csv"
    doc_path = tmp_path / "out.md"

    analyzer.write_route_a_fusedmoe_runner_boundary_decision_csv(csv_path, rows)
    analyzer.write_route_a_fusedmoe_runner_boundary_decision_doc(doc_path, rows)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 1
    assert set(written[0]) == set(analyzer.FIELDNAMES)
    assert written[0]["measurement_boundary"] == "fusedmoe_forward_runner_level"

    doc = doc_path.read_text(encoding="utf-8")
    assert "Default AIC | No-Go" in doc
    assert "GPU allowed | false" in doc
    assert "## Decision Evidence (Phase355 Source Check Evidence)" in doc
    assert "FusedMoE.forward() runner level" in doc
    assert "locked kernel path guard is not applicable" in doc
    assert "kernel_source stays measurement metadata" in doc
    assert "PerfDatabase | false" in doc
    assert "default-ready" not in doc


def test_old_blocked_source_check_wording_is_rejected(tmp_path: Path) -> None:
    rows = analyzer.analyze_route_a_fusedmoe_runner_boundary_decision(
        PHASE349,
        PHASE351,
    )
    rows[0]["measurement_boundary"] = "source_check_blocked"

    with pytest.raises(ValueError, match="measurement_boundary"):
        analyzer.write_route_a_fusedmoe_runner_boundary_decision_csv(
            tmp_path / "bad.csv",
            rows,
        )


def test_old_runtime_capture_next_action_is_rejected(tmp_path: Path) -> None:
    rows = analyzer.analyze_route_a_fusedmoe_runner_boundary_decision(
        PHASE349,
        PHASE351,
    )
    rows[0]["next_allowed_phase"] = (
        "define_kernel_source_runtime_capture_value_set_contract"
    )

    with pytest.raises(ValueError, match="next_allowed_phase"):
        analyzer.write_route_a_fusedmoe_runner_boundary_decision_csv(
            tmp_path / "bad.csv",
            rows,
        )


def test_phase349_input_must_keep_fusedmoe_measurement_boundary(tmp_path: Path) -> None:
    def mutate(rows: list[dict[str, str]]) -> None:
        for row in rows:
            if row["contract_row"] == "primary_measurement_boundary":
                row["decision"] = "bare_fused_experts_default_boundary"

    phase349 = _copy_csv_with_mutation(PHASE349, tmp_path / "phase349.csv", mutate)

    with pytest.raises(ValueError, match="primary_measurement_boundary"):
        analyzer.analyze_route_a_fusedmoe_runner_boundary_decision(
            phase349,
            PHASE351,
        )


def test_phase351_input_must_preserve_kernel_source_as_metadata_candidate(
    tmp_path: Path,
) -> None:
    def mutate(rows: list[dict[str, str]]) -> None:
        for row in rows:
            if row["contract_row"] == "kernel_source_identity":
                row["decision"] = "rejected"

    phase351 = _copy_csv_with_mutation(PHASE351, tmp_path / "phase351.csv", mutate)

    with pytest.raises(ValueError, match="kernel_source_identity"):
        analyzer.analyze_route_a_fusedmoe_runner_boundary_decision(
            PHASE349,
            phase351,
        )


def test_output_rejects_default_gpu_or_perfdb_upgrade(tmp_path: Path) -> None:
    rows = analyzer.analyze_route_a_fusedmoe_runner_boundary_decision(
        PHASE349,
        PHASE351,
    )
    rows[0]["perf_database"] = "true"

    with pytest.raises(ValueError, match="perf_database"):
        analyzer.write_route_a_fusedmoe_runner_boundary_decision_csv(
            tmp_path / "bad.csv",
            rows,
        )
