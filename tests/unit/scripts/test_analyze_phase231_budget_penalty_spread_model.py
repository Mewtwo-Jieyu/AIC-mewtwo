from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "analyze_phase231_budget_penalty_spread_model.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase231_budget_penalty_spread_model",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


EVIDENCE_FIELDNAMES = [
    "source",
    "topology_key",
    "shape_key",
    "control_scenario",
    "holdout_scenario",
    "control_bt",
    "holdout_bt",
    "clean_budget_effect",
    "no_budget_penalty_error",
    "evidence_source",
    "min_clean_effect",
    "max_clean_effect",
    "clean_effect_spread",
    "shape_count",
    "recommended_model_boundary",
    "default_readiness",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]

AUDIT_FIELDNAMES = [
    "source",
    "evidence_rows",
    "topology_count",
    "shape_count_per_topology",
    "max_clean_effect_spread",
    "min_clean_effect",
    "max_clean_effect",
    "default_readiness",
    "recommended_boundary",
    "reason",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]


def _evidence_rows() -> list[dict[str, str]]:
    base = {
        "source": "phase222_budget_penalty_evidence_family",
        "holdout_bt": "65536",
        "no_budget_penalty_error": "1.000000",
        "evidence_source": "synthetic_phase222",
        "shape_count": "3",
        "recommended_model_boundary": "diagnostic_only_exact_key",
        "default_readiness": "No-Go",
        "diagnostic_only": "true",
        "valid_for_default": "false",
        "perf_database": "false",
    }
    return [
        {
            **base,
            "topology_key": "tp8_dp1_ep8",
            "shape_key": "isl4000_osl2000_batch128",
            "control_scenario": "tp8ep8-4k2k-bt4000",
            "holdout_scenario": "tp8ep8-4k2k-bt65536",
            "control_bt": "4000",
            "clean_budget_effect": "1.000000",
            "min_clean_effect": "1.000000",
            "max_clean_effect": "1.400000",
            "clean_effect_spread": "1.400000",
        },
        {
            **base,
            "topology_key": "tp8_dp1_ep8",
            "shape_key": "isl8000_osl2000_batch128",
            "control_scenario": "tp8ep8-bt8000",
            "holdout_scenario": "tp8ep8-bt65536",
            "control_bt": "8000",
            "clean_budget_effect": "1.200000",
            "min_clean_effect": "1.000000",
            "max_clean_effect": "1.400000",
            "clean_effect_spread": "1.400000",
        },
        {
            **base,
            "topology_key": "tp8_dp1_ep8",
            "shape_key": "isl12000_osl2000_batch128",
            "control_scenario": "tp8ep8-12k2k-bt12000",
            "holdout_scenario": "tp8ep8-12k2k-bt65536",
            "control_bt": "12000",
            "clean_budget_effect": "1.400000",
            "min_clean_effect": "1.000000",
            "max_clean_effect": "1.400000",
            "clean_effect_spread": "1.400000",
        },
        {
            **base,
            "topology_key": "tp4_dp2_ep8",
            "shape_key": "isl4000_osl2000_batch128",
            "control_scenario": "tp4dp2ep8-4k2k-bt4000",
            "holdout_scenario": "tp4dp2ep8-4k2k-bt65536",
            "control_bt": "4000",
            "clean_budget_effect": "1.100000",
            "min_clean_effect": "1.100000",
            "max_clean_effect": "1.500000",
            "clean_effect_spread": "1.363636",
        },
        {
            **base,
            "topology_key": "tp4_dp2_ep8",
            "shape_key": "isl8000_osl2000_batch128",
            "control_scenario": "tp4dp2ep8-bt8000",
            "holdout_scenario": "tp4dp2ep8-bt65536",
            "control_bt": "8000",
            "clean_budget_effect": "1.300000",
            "min_clean_effect": "1.100000",
            "max_clean_effect": "1.500000",
            "clean_effect_spread": "1.363636",
        },
        {
            **base,
            "topology_key": "tp4_dp2_ep8",
            "shape_key": "isl12000_osl2000_batch128",
            "control_scenario": "tp4dp2ep8-12k2k-bt12000",
            "holdout_scenario": "tp4dp2ep8-12k2k-bt65536",
            "control_bt": "12000",
            "clean_budget_effect": "1.500000",
            "min_clean_effect": "1.100000",
            "max_clean_effect": "1.500000",
            "clean_effect_spread": "1.363636",
        },
    ]


def _audit_rows() -> list[dict[str, str]]:
    return [
        {
            "source": "phase228_default_readiness_final",
            "evidence_rows": "6",
            "topology_count": "2",
            "shape_count_per_topology": "3",
            "max_clean_effect_spread": "1.500000",
            "min_clean_effect": "1.000000",
            "max_clean_effect": "1.500000",
            "default_readiness": "No-Go",
            "recommended_boundary": "diagnostic_only_exact_key",
            "reason": "no_default_due_to_shape_topology_spread",
            "diagnostic_only": "true",
            "valid_for_default": "false",
            "perf_database": "false",
        }
    ]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_inputs(tmp_path: Path, evidence_rows=None, audit_rows=None) -> tuple[Path, Path]:
    evidence = tmp_path / "phase222.csv"
    audit = tmp_path / "phase228.csv"
    _write_csv(evidence, EVIDENCE_FIELDNAMES, evidence_rows or _evidence_rows())
    _write_csv(audit, AUDIT_FIELDNAMES, audit_rows or _audit_rows())
    return evidence, audit


def test_spread_model_outputs_predictions_and_summary(tmp_path: Path) -> None:
    evidence, audit = _write_inputs(tmp_path)
    out_csv = tmp_path / "phase231.csv"
    summary_csv = tmp_path / "phase231_summary.csv"
    out_doc = tmp_path / "phase231.md"

    rows, summary = analyzer.analyze_budget_penalty_spread_model(evidence, audit)
    analyzer.write_spread_model_csv(out_csv, rows)
    analyzer.write_spread_model_summary_csv(summary_csv, summary)
    analyzer.write_spread_model_doc(out_doc, rows, summary)

    assert len(rows) == 6
    assert len(summary) == 4
    first = rows[0]
    assert first["source"] == "phase231_budget_penalty_spread_model"
    assert first["clean_budget_effect"] == "1.000000"
    assert first["global_mean_prediction"] == "1.250000"
    assert first["topology_mean_prediction"] == "1.200000"
    assert first["shape_mean_prediction"] == "1.050000"
    assert first["topology_plus_shape_prediction"] == "1.000000"
    assert first["global_mean_abs_error"] == "0.250000"
    assert first["topology_plus_shape_abs_error"] == "0.000000"
    assert {row["diagnostic_only"] for row in rows} == {"true"}
    assert {row["valid_for_default"] for row in rows} == {"false"}
    assert {row["perf_database"] for row in rows} == {"false"}

    by_model = {row["model_name"]: row for row in summary}
    assert set(by_model) == {"global_mean", "topology_mean", "shape_mean", "topology_plus_shape"}
    assert by_model["topology_plus_shape"]["mean_abs_error"] == "0.000000"
    assert by_model["global_mean"]["parameter_count"] == "1"
    assert by_model["topology_mean"]["parameter_count"] == "2"
    assert by_model["shape_mean"]["parameter_count"] == "3"
    assert by_model["topology_plus_shape"]["parameter_count"] == "4"
    assert by_model["global_mean"]["loocv_mean_abs_error"] == "0.180000"
    assert by_model["global_mean"]["loocv_max_abs_error"] == "0.300000"
    assert by_model["topology_plus_shape"]["loocv_mean_abs_error"] == "0.053333"
    assert by_model["topology_plus_shape"]["loocv_max_abs_error"] == "0.100000"
    assert {row["best_in_sample_form"] for row in summary} == {"topology_plus_shape"}
    assert {row["best_loocv_form"] for row in summary} == {"topology_plus_shape"}
    assert {row["default_readiness"] for row in summary} == {"No-Go"}
    assert out_csv.read_text(encoding="utf-8").count("\n") == 7
    assert summary_csv.read_text(encoding="utf-8").count("\n") == 5
    doc = out_doc.read_text(encoding="utf-8")
    assert "complex model fits in-sample better" in doc
    assert "generalization evidence is insufficient" in doc
    assert "best_loocv_form" in doc
    assert "diagnostic_only_exact_key" in doc


@pytest.mark.parametrize("rows", [_evidence_rows()[:-1], _evidence_rows() + [_evidence_rows()[0]]])
def test_evidence_row_count_guard_fails_fast(tmp_path: Path, rows: list[dict[str, str]]) -> None:
    evidence, audit = _write_inputs(tmp_path, evidence_rows=rows)

    with pytest.raises(ValueError, match="phase222 evidence family must contain exactly 6 rows"):
        analyzer.analyze_budget_penalty_spread_model(evidence, audit)


def test_duplicate_topology_shape_guard_fails_fast(tmp_path: Path) -> None:
    rows = _evidence_rows()
    rows[1]["shape_key"] = rows[0]["shape_key"]
    evidence, audit = _write_inputs(tmp_path, evidence_rows=rows)

    with pytest.raises(ValueError, match="duplicate topology/shape evidence"):
        analyzer.analyze_budget_penalty_spread_model(evidence, audit)


def test_each_topology_must_have_three_shapes(tmp_path: Path) -> None:
    rows = _evidence_rows()
    rows[2]["topology_key"] = "tp4_dp2_ep8"
    rows[2]["shape_key"] = "isl16000_osl2000_batch128"
    evidence, audit = _write_inputs(tmp_path, evidence_rows=rows)

    with pytest.raises(ValueError, match="topology must have exactly 3 shapes"):
        analyzer.analyze_budget_penalty_spread_model(evidence, audit)


@pytest.mark.parametrize("field", ["diagnostic_only", "valid_for_default", "perf_database"])
def test_evidence_flag_guards_fail_fast(tmp_path: Path, field: str) -> None:
    rows = _evidence_rows()
    rows[0][field] = "false" if field == "diagnostic_only" else "true"
    evidence, audit = _write_inputs(tmp_path, evidence_rows=rows)

    with pytest.raises(ValueError, match=field):
        analyzer.analyze_budget_penalty_spread_model(evidence, audit)


def test_phase228_audit_must_be_no_go(tmp_path: Path) -> None:
    audit_rows = _audit_rows()
    audit_rows[0]["default_readiness"] = "Go"
    evidence, audit = _write_inputs(tmp_path, audit_rows=audit_rows)

    with pytest.raises(ValueError, match="default_readiness must be No-Go"):
        analyzer.analyze_budget_penalty_spread_model(evidence, audit)


def test_actual_csvs_generate_six_rows_and_no_go_summary() -> None:
    root = Path(__file__).resolve().parents[3]

    rows, summary = analyzer.analyze_budget_penalty_spread_model(
        root / "docs/iter_gap_investigation/phase222_budget_penalty_evidence_family.csv",
        root / "docs/iter_gap_investigation/phase228_default_readiness_final.csv",
    )

    assert len(rows) == 6
    assert len(summary) == 4
    assert {row["default_readiness"] for row in summary} == {"No-Go"}
    assert {row["diagnostic_only"] for row in rows} == {"true"}
    assert {row["valid_for_default"] for row in rows} == {"false"}
    assert {row["perf_database"] for row in rows} == {"false"}
    assert {row["best_in_sample_form"] for row in summary} == {"topology_plus_shape"}
    assert {row["best_loocv_form"] for row in summary} == {"global_mean"}
