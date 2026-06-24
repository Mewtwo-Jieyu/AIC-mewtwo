from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT / "scripts" / "analyze_phase349_route_a_moe_measurement_api_contract.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase349_route_a_moe_measurement_api_contract",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


PHASE347 = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase347_route_a_schema_source_reconciliation.csv"
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


def test_moe_measurement_api_contract_outputs_seven_rows() -> None:
    rows = analyzer.analyze_route_a_moe_measurement_api_contract(PHASE347)

    assert [row["contract_row"] for row in rows] == [
        "primary_measurement_boundary",
        "bare_fused_experts_probe",
        "required_input_tensors",
        "required_quant_fields",
        "kernel_source_capture",
        "measurement_output",
        "gpu_smoke_readiness",
    ]
    assert [row["decision"] for row in rows] == [
        "candidate_fusedmoe_default_runner_boundary",
        "auxiliary_kernel_probe_not_perfdb_row",
        "blocked_until_hidden_states_router_weights_expert_weights_defined",
        "blocked_until_dtype_quant_method_kernel_source_defined",
        "required_for_any_measurement_row",
        "latency_ms_only_no_default_prediction",
        "blocked_until_api_contract_and_kernel_source_clear",
    ]
    assert {row["source"] for row in rows} == {
        "phase349_route_a_moe_measurement_api_contract"
    }
    assert {row["gpu_allowed"] for row in rows} == {"false"}
    assert {row["default_readiness"] for row in rows} == {"No-Go"}
    assert {row["diagnostic_only"] for row in rows} == {"true"}
    assert {row["valid_for_default"] for row in rows} == {"false"}
    assert {row["perf_database"] for row in rows} == {"false"}
    assert rows[0]["measurement_boundary"] == "fusedmoe_default_runner"
    assert rows[1]["measurement_boundary"] == "bare_fused_experts"
    assert rows[-1]["next_required_action"] == (
        "clear_moe_api_contract_and_kernel_source_before_gpu_smoke"
    )


def test_writers_emit_csv_and_doc(tmp_path: Path) -> None:
    rows = analyzer.analyze_route_a_moe_measurement_api_contract(PHASE347)
    csv_path = tmp_path / "out.csv"
    doc_path = tmp_path / "out.md"

    analyzer.write_route_a_moe_measurement_api_contract_csv(csv_path, rows)
    analyzer.write_route_a_moe_measurement_api_contract_doc(doc_path, rows)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 7
    assert set(written[0]) == set(analyzer.FIELDNAMES)

    doc = doc_path.read_text(encoding="utf-8")
    assert "Default AIC | No-Go" in doc
    assert "GPU allowed | false" in doc
    assert "bare fused_experts is only an auxiliary kernel probe" in doc
    assert "not a PerfDatabase row" in doc
    assert "latency_ms only" in doc
    assert "default-ready" not in doc


def test_phase347_input_must_keep_gpu_smoke_blocked(tmp_path: Path) -> None:
    def mutate(rows: list[dict[str, str]]) -> None:
        for row in rows:
            if row["gate"] == "gpu_smoke_design":
                row["decision"] = "cleared_for_gpu_smoke"

    phase347 = _copy_csv_with_mutation(PHASE347, tmp_path / "phase347.csv", mutate)

    with pytest.raises(ValueError, match="gpu_smoke_design"):
        analyzer.analyze_route_a_moe_measurement_api_contract(phase347)


def test_phase347_input_must_keep_moe_api_contract_blocked(tmp_path: Path) -> None:
    def mutate(rows: list[dict[str, str]]) -> None:
        for row in rows:
            if row["gate"] == "moe_measurement_api_contract":
                row["decision"] = "cleared"

    phase347 = _copy_csv_with_mutation(PHASE347, tmp_path / "phase347.csv", mutate)

    with pytest.raises(ValueError, match="moe_measurement_api_contract"):
        analyzer.analyze_route_a_moe_measurement_api_contract(phase347)


def test_output_rejects_default_or_perfdb_upgrade(tmp_path: Path) -> None:
    rows = analyzer.analyze_route_a_moe_measurement_api_contract(PHASE347)
    rows[0]["perf_database"] = "true"

    with pytest.raises(ValueError, match="perf_database"):
        analyzer.write_route_a_moe_measurement_api_contract_csv(
            tmp_path / "bad.csv", rows
        )
