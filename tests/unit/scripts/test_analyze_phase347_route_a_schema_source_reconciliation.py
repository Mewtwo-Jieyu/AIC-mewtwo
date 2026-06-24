from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT / "scripts" / "analyze_phase347_route_a_schema_source_reconciliation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase347_route_a_schema_source_reconciliation",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


PHASE344 = REPO_ROOT / "docs/iter_gap_investigation/phase344_route_a_source_schema_audit.csv"


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


def test_route_a_schema_source_reconciliation_outputs_seven_rows() -> None:
    rows = analyzer.analyze_route_a_schema_source_reconciliation(PHASE344)

    assert [row["gate"] for row in rows] == [
        "remote_vllm_version_source_root",
        "moe_measurement_api_contract",
        "kernel_source_key_contract",
        "vllm_ep8_comm_schema_contract",
        "attention_gemm_reuse_policy",
        "token_shape_fidelity_input",
        "gpu_smoke_design",
    ]
    assert [row["decision"] for row in rows] == [
        "cleared_baseline_only",
        "blocked_define_fusedmoe_or_fused_experts_api",
        "blocked_add_or_lock_kernel_source",
        "blocked_define_vllm_ep8_comm_scope",
        "blocked_assumption_pending_0190_validation",
        "retained_prerun_only_no_trace_feedback",
        "blocked_until_contracts_clear",
    ]
    assert {row["source"] for row in rows} == {
        "phase347_route_a_schema_source_reconciliation"
    }
    assert {row["gpu_allowed"] for row in rows} == {"false"}
    assert {row["default_readiness"] for row in rows} == {"No-Go"}
    assert {row["diagnostic_only"] for row in rows} == {"true"}
    assert {row["valid_for_default"] for row in rows} == {"false"}
    assert {row["perf_database"] for row in rows} == {"false"}
    assert rows[0]["phase346_fact"] == (
        "remote_vllm_version_0_19_0_source_root_cleared"
    )
    assert rows[-1]["next_required_action"] == (
        "write_schema_source_reconciliation_before_any_gpu_smoke_spec"
    )


def test_writers_emit_csv_and_doc(tmp_path: Path) -> None:
    rows = analyzer.analyze_route_a_schema_source_reconciliation(PHASE344)
    csv_path = tmp_path / "out.csv"
    doc_path = tmp_path / "out.md"

    analyzer.write_route_a_schema_source_reconciliation_csv(csv_path, rows)
    analyzer.write_route_a_schema_source_reconciliation_doc(doc_path, rows)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 7
    assert set(written[0]) == set(analyzer.FIELDNAMES)

    doc = doc_path.read_text(encoding="utf-8")
    assert "Default AIC | No-Go" in doc
    assert "GPU allowed | false" in doc
    assert "Phase346 cleared only the remote vLLM baseline" in doc
    assert "MoE measurement API contract" in doc
    assert "kernel_source schema contract" in doc
    assert "vLLM EP8 comm schema" in doc
    assert "not a GPU smoke spec" in doc
    assert "default-ready" not in doc


def test_phase344_input_must_keep_source_schema_blockers(tmp_path: Path) -> None:
    def mutate(rows: list[dict[str, str]]) -> None:
        for row in rows:
            if row["gate"] == "vllm_ep8_comm_scope":
                row["decision"] = "cleared"

    phase344 = _copy_csv_with_mutation(PHASE344, tmp_path / "phase344.csv", mutate)

    with pytest.raises(ValueError, match="vllm_ep8_comm_scope"):
        analyzer.analyze_route_a_schema_source_reconciliation(phase344)


def test_output_rejects_default_or_perfdb_upgrade(tmp_path: Path) -> None:
    rows = analyzer.analyze_route_a_schema_source_reconciliation(PHASE344)
    rows[0]["valid_for_default"] = "true"

    with pytest.raises(ValueError, match="valid_for_default"):
        analyzer.write_route_a_schema_source_reconciliation_csv(
            tmp_path / "bad.csv", rows
        )


def test_output_requires_exactly_seven_rows(tmp_path: Path) -> None:
    rows = analyzer.analyze_route_a_schema_source_reconciliation(PHASE344)

    with pytest.raises(ValueError, match="exactly 7"):
        analyzer.write_route_a_schema_source_reconciliation_csv(
            tmp_path / "bad.csv", rows[:6]
        )
