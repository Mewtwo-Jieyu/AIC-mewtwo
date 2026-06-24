from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT / "scripts" / "analyze_phase351_route_a_kernel_source_key_contract.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase351_route_a_kernel_source_key_contract",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


PHASE349 = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase349_route_a_moe_measurement_api_contract.csv"
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


def test_kernel_source_key_contract_outputs_six_rows() -> None:
    rows = analyzer.analyze_route_a_kernel_source_key_contract(PHASE349)

    assert [row["contract_row"] for row in rows] == [
        "kernel_source_identity",
        "logical_moe_key_without_kernel_source",
        "lock_single_kernel_path_option",
        "perfdb_schema_update_option",
        "measurement_row_eligibility",
        "gpu_smoke_readiness",
    ]
    assert [row["decision"] for row in rows] == [
        "required_key_dimension",
        "rejected_ambiguous_identity",
        "allowed_only_with_source_check_guard",
        "candidate_requires_loader_query_contract_change",
        "blocked_until_kernel_source_identity_defined",
        "blocked_until_kernel_source_contract_clears",
    ]
    assert {row["source"] for row in rows} == {
        "phase351_route_a_kernel_source_key_contract"
    }
    assert {row["gpu_allowed"] for row in rows} == {"false"}
    assert {row["default_readiness"] for row in rows} == {"No-Go"}
    assert {row["diagnostic_only"] for row in rows} == {"true"}
    assert {row["valid_for_default"] for row in rows} == {"false"}
    assert {row["perf_database"] for row in rows} == {"false"}
    assert rows[0]["key_contract"] == (
        "include_kernel_source_in_route_a_moe_measurement_identity"
    )
    assert rows[1]["key_contract"] == (
        "reject_quant_distribution_topk_experts_hidden_inter_moe_tp_moe_ep_num_tokens_without_kernel_source"
    )
    assert rows[-1]["next_required_action"] == (
        "choose_schema_update_or_locked_kernel_path_before_gpu_smoke"
    )


def test_writers_emit_csv_and_doc(tmp_path: Path) -> None:
    rows = analyzer.analyze_route_a_kernel_source_key_contract(PHASE349)
    csv_path = tmp_path / "out.csv"
    doc_path = tmp_path / "out.md"

    analyzer.write_route_a_kernel_source_key_contract_csv(csv_path, rows)
    analyzer.write_route_a_kernel_source_key_contract_doc(doc_path, rows)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 6
    assert set(written[0]) == set(analyzer.FIELDNAMES)

    doc = doc_path.read_text(encoding="utf-8")
    assert "Default AIC | No-Go" in doc
    assert "GPU allowed | false" in doc
    assert "include kernel_source in the key" in doc
    assert "lock exactly one kernel path" in doc
    assert "ambiguous old vLLM MoE key" in doc
    assert "does not update PerfDatabase" in doc
    assert "default-ready" not in doc


def test_phase349_input_must_require_kernel_source_capture(tmp_path: Path) -> None:
    def mutate(rows: list[dict[str, str]]) -> None:
        for row in rows:
            if row["contract_row"] == "kernel_source_capture":
                row["decision"] = "optional"

    phase349 = _copy_csv_with_mutation(PHASE349, tmp_path / "phase349.csv", mutate)

    with pytest.raises(ValueError, match="kernel_source_capture"):
        analyzer.analyze_route_a_kernel_source_key_contract(phase349)


def test_phase349_input_must_keep_gpu_smoke_blocked(tmp_path: Path) -> None:
    def mutate(rows: list[dict[str, str]]) -> None:
        for row in rows:
            if row["contract_row"] == "gpu_smoke_readiness":
                row["decision"] = "cleared_for_gpu_smoke"

    phase349 = _copy_csv_with_mutation(PHASE349, tmp_path / "phase349.csv", mutate)

    with pytest.raises(ValueError, match="gpu_smoke_readiness"):
        analyzer.analyze_route_a_kernel_source_key_contract(phase349)


def test_output_rejects_perfdb_or_default_upgrade(tmp_path: Path) -> None:
    rows = analyzer.analyze_route_a_kernel_source_key_contract(PHASE349)
    rows[0]["valid_for_default"] = "true"

    with pytest.raises(ValueError, match="valid_for_default"):
        analyzer.write_route_a_kernel_source_key_contract_csv(
            tmp_path / "bad.csv", rows
        )


def test_output_requires_exactly_six_rows(tmp_path: Path) -> None:
    rows = analyzer.analyze_route_a_kernel_source_key_contract(PHASE349)

    with pytest.raises(ValueError, match="exactly 6"):
        analyzer.write_route_a_kernel_source_key_contract_csv(
            tmp_path / "bad.csv", rows[:5]
        )
