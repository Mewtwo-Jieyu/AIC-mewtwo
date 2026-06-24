from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase342_route_a_module_perf_table_spec.py"
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase342_route_a_module_perf_table_spec",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


PHASE339 = REPO_ROOT / "docs/iter_gap_investigation/phase339_prerun_model_feasibility.csv"


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


def test_route_a_spec_outputs_five_rows() -> None:
    rows = analyzer.analyze_route_a_module_perf_table_spec(PHASE339)

    assert [row["candidate"] for row in rows] == [
        "route_a_module_perf_table_candidate",
        "route_b_runtime_shape_input_layer",
        "legacy_attention_gemm_reuse",
        "vllm_ep8_comm_modeling",
        "full_trace_feedback_to_perf_table",
    ]
    assert [row["decision"] for row in rows] == [
        "retained_as_primary_candidate",
        "absorbed_as_token_shape_source",
        "assumption_pending_validation",
        "schema_pending_source_check",
        "rejected_data_leakage",
    ]
    assert {row["source"] for row in rows} == {"phase342_route_a_module_perf_table_spec"}
    assert {row["gpu_allowed"] for row in rows} == {"false"}
    assert {row["default_readiness"] for row in rows} == {"No-Go"}
    assert {row["diagnostic_only"] for row in rows} == {"true"}
    assert {row["valid_for_default"] for row in rows} == {"false"}
    assert {row["perf_database"] for row in rows} == {"false"}
    assert rows[0]["next_phase"] == "phase343_vllm_0190_moe_source_api_check"
    assert rows[4]["blocking_condition"] == "data_leakage_from_full_trace_feedback"


def test_writers_emit_csv_and_doc(tmp_path: Path) -> None:
    rows = analyzer.analyze_route_a_module_perf_table_spec(PHASE339)
    csv_path = tmp_path / "out.csv"
    doc_path = tmp_path / "out.md"

    analyzer.write_route_a_module_perf_table_spec_csv(csv_path, rows)
    analyzer.write_route_a_module_perf_table_spec_doc(doc_path, rows)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 5
    assert set(written[0]) == set(analyzer.FIELDNAMES)

    doc = doc_path.read_text(encoding="utf-8")
    assert "Route A | vLLM 0.19.0 module-level perf table" in doc
    assert "Route B | absorbed as token-shape input layer" in doc
    assert "Full trace feedback | rejected_data_leakage" in doc
    assert "mean error <=20%, max error <=30%, direction must be correct" in doc
    assert "Default AIC | No-Go" in doc
    assert "GPU allowed | false" in doc
    assert "not a default model" in doc
    assert "default-ready" not in doc


def test_phase339_default_readiness_must_stay_no_go(tmp_path: Path) -> None:
    def mutate(rows: list[dict[str, str]]) -> None:
        rows[0]["default_readiness"] = "Ready"

    phase339 = _copy_csv_with_mutation(PHASE339, tmp_path / "phase339.csv", mutate)

    with pytest.raises(ValueError, match="default_readiness"):
        analyzer.analyze_route_a_module_perf_table_spec(phase339)


def test_phase339_postrun_trace_candidate_must_remain_rejected(tmp_path: Path) -> None:
    def mutate(rows: list[dict[str, str]]) -> None:
        for row in rows:
            if row["candidate"] == "postrun_trace_feature_candidate":
                row["verdict"] = "retained"

    phase339 = _copy_csv_with_mutation(PHASE339, tmp_path / "phase339.csv", mutate)

    with pytest.raises(ValueError, match="postrun_trace_feature_candidate"):
        analyzer.analyze_route_a_module_perf_table_spec(phase339)


def test_writer_requires_exactly_five_rows(tmp_path: Path) -> None:
    rows = analyzer.analyze_route_a_module_perf_table_spec(PHASE339)

    with pytest.raises(ValueError, match="exactly 5"):
        analyzer.write_route_a_module_perf_table_spec_csv(tmp_path / "bad.csv", rows[:4])
