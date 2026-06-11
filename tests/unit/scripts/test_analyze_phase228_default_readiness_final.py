from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "analyze_phase228_default_readiness_final.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase228_default_readiness_final",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


FIELDNAMES = [
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


def _rows() -> list[dict[str, str]]:
    base = {
        "source": "phase222_budget_penalty_evidence_family",
        "holdout_bt": "65536",
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
            "clean_budget_effect": "1.012212",
            "no_budget_penalty_error": "1.012212",
            "evidence_source": "phase215_budget_penalty_mechanism",
            "min_clean_effect": "0.987382",
            "max_clean_effect": "1.074279",
            "clean_effect_spread": "1.088007",
        },
        {
            **base,
            "topology_key": "tp8_dp1_ep8",
            "shape_key": "isl8000_osl2000_batch128",
            "control_scenario": "tp8ep8-bt8000",
            "holdout_scenario": "tp8ep8-bt65536",
            "control_bt": "8000",
            "clean_budget_effect": "0.987382",
            "no_budget_penalty_error": "1.012779",
            "evidence_source": "phase164_clean_gpu_budget_manifest",
            "min_clean_effect": "0.987382",
            "max_clean_effect": "1.074279",
            "clean_effect_spread": "1.088007",
        },
        {
            **base,
            "topology_key": "tp8_dp1_ep8",
            "shape_key": "isl12000_osl2000_batch128",
            "control_scenario": "tp8ep8-12k2k-bt12000",
            "holdout_scenario": "tp8ep8-12k2k-bt65536",
            "control_bt": "12000",
            "clean_budget_effect": "1.074279",
            "no_budget_penalty_error": "1.074279",
            "evidence_source": "phase215_budget_penalty_mechanism",
            "min_clean_effect": "0.987382",
            "max_clean_effect": "1.074279",
            "clean_effect_spread": "1.088007",
        },
        {
            **base,
            "topology_key": "tp4_dp2_ep8",
            "shape_key": "isl4000_osl2000_batch128",
            "control_scenario": "tp4dp2ep8-4k2k-bt4000",
            "holdout_scenario": "tp4dp2ep8-4k2k-bt65536",
            "control_bt": "4000",
            "clean_budget_effect": "1.133187",
            "no_budget_penalty_error": "1.133187",
            "evidence_source": "phase215_budget_penalty_mechanism",
            "min_clean_effect": "0.993809",
            "max_clean_effect": "1.167497",
            "clean_effect_spread": "1.174770",
        },
        {
            **base,
            "topology_key": "tp4_dp2_ep8",
            "shape_key": "isl8000_osl2000_batch128",
            "control_scenario": "tp4dp2ep8-bt8000",
            "holdout_scenario": "tp4dp2ep8-bt65536",
            "control_bt": "8000",
            "clean_budget_effect": "1.167497",
            "no_budget_penalty_error": "1.167497",
            "evidence_source": "phase164_clean_gpu_budget_manifest",
            "min_clean_effect": "0.993809",
            "max_clean_effect": "1.167497",
            "clean_effect_spread": "1.174770",
        },
        {
            **base,
            "topology_key": "tp4_dp2_ep8",
            "shape_key": "isl12000_osl2000_batch128",
            "control_scenario": "tp4dp2ep8-12k2k-bt12000",
            "holdout_scenario": "tp4dp2ep8-12k2k-bt65536",
            "control_bt": "12000",
            "clean_budget_effect": "0.993809",
            "no_budget_penalty_error": "1.006230",
            "evidence_source": "phase215_budget_penalty_mechanism",
            "min_clean_effect": "0.993809",
            "max_clean_effect": "1.167497",
            "clean_effect_spread": "1.174770",
        },
    ]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_input(tmp_path: Path, rows=None) -> Path:
    path = tmp_path / "phase222.csv"
    _write_csv(path, rows or _rows())
    return path


def test_final_audit_emits_one_no_go_row(tmp_path: Path) -> None:
    input_csv = _write_input(tmp_path)
    out_csv = tmp_path / "phase228.csv"
    out_doc = tmp_path / "phase228.md"

    rows = analyzer.analyze_default_readiness_final(input_csv)
    analyzer.write_final_audit_csv(out_csv, rows)
    analyzer.write_final_audit_doc(out_doc, rows)

    assert len(rows) == 1
    row = rows[0]
    assert row["source"] == "phase228_default_readiness_final"
    assert row["evidence_rows"] == "6"
    assert row["topology_count"] == "2"
    assert row["shape_count_per_topology"] == "3"
    assert row["max_clean_effect_spread"] == "1.174770"
    assert row["min_clean_effect"] == "0.987382"
    assert row["max_clean_effect"] == "1.167497"
    assert row["default_readiness"] == "No-Go"
    assert row["recommended_boundary"] == "diagnostic_only_exact_key"
    assert row["reason"] == "no_default_due_to_shape_topology_spread"
    assert row["diagnostic_only"] == "true"
    assert row["valid_for_default"] == "false"
    assert row["perf_database"] == "false"
    assert out_csv.read_text(encoding="utf-8").count("\n") == 2
    doc = out_doc.read_text(encoding="utf-8")
    assert "Default AIC remains No-Go" in doc
    assert "no_default_due_to_shape_topology_spread" in doc
    assert "diagnostic_only_exact_key" in doc


@pytest.mark.parametrize("rows", [_rows()[:-1], _rows() + [_rows()[0]]])
def test_row_count_guard_fails_fast(tmp_path: Path, rows: list[dict[str, str]]) -> None:
    input_csv = _write_input(tmp_path, rows)

    with pytest.raises(ValueError, match="phase222 evidence family must contain exactly 6 rows"):
        analyzer.analyze_default_readiness_final(input_csv)


def test_each_topology_must_have_three_shapes(tmp_path: Path) -> None:
    rows = _rows()
    rows[2]["topology_key"] = "tp4_dp2_ep8"
    rows[2]["shape_key"] = "isl16000_osl2000_batch128"
    input_csv = _write_input(tmp_path, rows)

    with pytest.raises(ValueError, match="topology must have exactly 3 shapes"):
        analyzer.analyze_default_readiness_final(input_csv)


@pytest.mark.parametrize("field", ["diagnostic_only", "valid_for_default", "perf_database"])
def test_flag_guards_fail_fast(tmp_path: Path, field: str) -> None:
    rows = _rows()
    rows[0][field] = "false" if field == "diagnostic_only" else "true"
    input_csv = _write_input(tmp_path, rows)

    with pytest.raises(ValueError, match=field):
        analyzer.analyze_default_readiness_final(input_csv)


def test_no_go_when_spread_exists_and_any_clean_effect_below_one(tmp_path: Path) -> None:
    input_csv = _write_input(tmp_path)

    row = analyzer.analyze_default_readiness_final(input_csv)[0]

    assert float(row["max_clean_effect_spread"]) > 1.0
    assert float(row["min_clean_effect"]) < 1.0
    assert row["default_readiness"] == "No-Go"
    assert row["valid_for_default"] == "false"


def test_actual_phase222_csv_generates_no_go_final_audit() -> None:
    root = Path(__file__).resolve().parents[3]

    rows = analyzer.analyze_default_readiness_final(
        root / "docs/iter_gap_investigation/phase222_budget_penalty_evidence_family.csv"
    )

    assert len(rows) == 1
    assert rows[0]["evidence_rows"] == "6"
    assert rows[0]["default_readiness"] == "No-Go"
    assert rows[0]["valid_for_default"] == "false"
