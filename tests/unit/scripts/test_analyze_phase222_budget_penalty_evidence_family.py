from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "analyze_phase222_budget_penalty_evidence_family.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase222_budget_penalty_evidence_family",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


PHASE164_FIELDNAMES = [
    "source",
    "scenario",
    "tp",
    "dp",
    "ep",
    "isl",
    "osl",
    "batch_size",
    "max_num_batched_tokens",
    "request_success_count",
    "request_fail_count",
    "real_output_tok_s",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]

PHASE215_FIELDNAMES = [
    "source",
    "topology_key",
    "shape_key",
    "control_scenario",
    "holdout_scenario",
    "control_max_bt",
    "holdout_max_bt",
    "clean_budget_effect",
    "steady_state_time_ratio",
    "steady_linear_error",
    "no_budget_penalty_error",
    "topology_shape_spread",
    "recommended_model_boundary",
    "default_readiness",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]


def _phase164_rows() -> list[dict[str, str]]:
    base = {
        "source": "phase164_clean_gpu_benchmark",
        "isl": "8000",
        "osl": "2000",
        "batch_size": "128",
        "request_success_count": "128",
        "request_fail_count": "0",
        "diagnostic_only": "true",
        "valid_for_default": "false",
        "perf_database": "false",
    }
    return [
        {**base, "scenario": "tp8ep8-bt8000", "tp": "8", "dp": "1", "ep": "8", "max_num_batched_tokens": "8000", "real_output_tok_s": "100.000000"},
        {**base, "scenario": "tp8ep8-bt65536", "tp": "8", "dp": "1", "ep": "8", "max_num_batched_tokens": "65536", "real_output_tok_s": "98.000000"},
        {**base, "scenario": "tp4dp2ep8-bt8000", "tp": "4", "dp": "2", "ep": "8", "max_num_batched_tokens": "8000", "real_output_tok_s": "200.000000"},
        {**base, "scenario": "tp4dp2ep8-bt65536", "tp": "4", "dp": "2", "ep": "8", "max_num_batched_tokens": "65536", "real_output_tok_s": "240.000000"},
    ]


def _phase215_rows() -> list[dict[str, str]]:
    base = {
        "source": "phase215_budget_penalty_mechanism",
        "holdout_max_bt": "65536",
        "steady_state_time_ratio": "4.000000",
        "steady_linear_error": "4.000000",
        "topology_shape_spread": "1.000000",
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
            "control_max_bt": "4000",
            "clean_budget_effect": "1.050000",
            "no_budget_penalty_error": "1.050000",
        },
        {
            **base,
            "topology_key": "tp4_dp2_ep8",
            "shape_key": "isl4000_osl2000_batch128",
            "control_scenario": "tp4dp2ep8-4k2k-bt4000",
            "holdout_scenario": "tp4dp2ep8-4k2k-bt65536",
            "control_max_bt": "4000",
            "clean_budget_effect": "1.200000",
            "no_budget_penalty_error": "1.200000",
        },
        {
            **base,
            "topology_key": "tp8_dp1_ep8",
            "shape_key": "isl12000_osl2000_batch128",
            "control_scenario": "tp8ep8-12k2k-bt12000",
            "holdout_scenario": "tp8ep8-12k2k-bt65536",
            "control_max_bt": "12000",
            "clean_budget_effect": "1.100000",
            "no_budget_penalty_error": "1.100000",
        },
        {
            **base,
            "topology_key": "tp4_dp2_ep8",
            "shape_key": "isl12000_osl2000_batch128",
            "control_scenario": "tp4dp2ep8-12k2k-bt12000",
            "holdout_scenario": "tp4dp2ep8-12k2k-bt65536",
            "control_max_bt": "12000",
            "clean_budget_effect": "0.990000",
            "no_budget_penalty_error": "1.010101",
        },
    ]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_inputs(tmp_path: Path, phase164_rows=None, phase215_rows=None) -> tuple[Path, Path]:
    phase164 = tmp_path / "phase164.csv"
    phase215 = tmp_path / "phase215.csv"
    _write_csv(phase164, PHASE164_FIELDNAMES, phase164_rows or _phase164_rows())
    _write_csv(phase215, PHASE215_FIELDNAMES, phase215_rows or _phase215_rows())
    return phase164, phase215


def test_evidence_family_combines_phase164_and_phase215_without_interpolation(tmp_path: Path) -> None:
    phase164, phase215 = _write_inputs(tmp_path)
    out_csv = tmp_path / "family.csv"
    out_doc = tmp_path / "family.md"

    rows = analyzer.analyze_budget_penalty_evidence_family(phase164, phase215)
    analyzer.write_evidence_family_csv(out_csv, rows)
    analyzer.write_evidence_family_doc(out_doc, rows)

    assert len(rows) == 6
    assert {(row["topology_key"], row["shape_key"]) for row in rows} == {
        ("tp8_dp1_ep8", "isl4000_osl2000_batch128"),
        ("tp8_dp1_ep8", "isl8000_osl2000_batch128"),
        ("tp8_dp1_ep8", "isl12000_osl2000_batch128"),
        ("tp4_dp2_ep8", "isl4000_osl2000_batch128"),
        ("tp4_dp2_ep8", "isl8000_osl2000_batch128"),
        ("tp4_dp2_ep8", "isl12000_osl2000_batch128"),
    }
    eight_k_rows = [row for row in rows if row["shape_key"] == "isl8000_osl2000_batch128"]
    assert {row["evidence_source"] for row in eight_k_rows} == {"phase164_clean_gpu_budget_manifest"}
    assert {row["control_bt"] for row in eight_k_rows} == {"8000"}
    assert {row["holdout_bt"] for row in eight_k_rows} == {"65536"}
    assert rows[1]["clean_budget_effect"] == "0.980000"
    assert rows[1]["no_budget_penalty_error"] == "1.020408"
    assert rows[4]["clean_budget_effect"] == "1.200000"
    assert {row["recommended_model_boundary"] for row in rows} == {"diagnostic_only_exact_key"}
    assert {row["default_readiness"] for row in rows} == {"No-Go"}
    assert {row["diagnostic_only"] for row in rows} == {"true"}
    assert {row["valid_for_default"] for row in rows} == {"false"}
    assert {row["perf_database"] for row in rows} == {"false"}
    assert {row["shape_count"] for row in rows} == {"3"}
    assert out_csv.read_text(encoding="utf-8").count("\n") == 7
    doc = out_doc.read_text(encoding="utf-8")
    assert "Phase164 clean evidence" in doc
    assert "not interpolation" in doc
    assert "Default AIC remains No-Go" in doc


@pytest.mark.parametrize("rows", [_phase164_rows()[:-1], _phase164_rows() + [_phase164_rows()[0]]])
def test_phase164_row_count_guard_fails_fast(tmp_path: Path, rows: list[dict[str, str]]) -> None:
    phase164, phase215 = _write_inputs(tmp_path, phase164_rows=rows)

    with pytest.raises(ValueError, match="phase164 manifest must contain exactly 4 rows"):
        analyzer.analyze_budget_penalty_evidence_family(phase164, phase215)


@pytest.mark.parametrize("rows", [_phase215_rows()[:-1], _phase215_rows() + [_phase215_rows()[0]]])
def test_phase215_row_count_guard_fails_fast(tmp_path: Path, rows: list[dict[str, str]]) -> None:
    phase164, phase215 = _write_inputs(tmp_path, phase215_rows=rows)

    with pytest.raises(ValueError, match="phase215 mechanism must contain exactly 4 rows"):
        analyzer.analyze_budget_penalty_evidence_family(phase164, phase215)


@pytest.mark.parametrize("field", ["diagnostic_only", "valid_for_default", "perf_database"])
def test_flag_guards_fail_fast(tmp_path: Path, field: str) -> None:
    rows = _phase215_rows()
    rows[0][field] = "false" if field == "diagnostic_only" else "true"
    phase164, phase215 = _write_inputs(tmp_path, phase215_rows=rows)

    with pytest.raises(ValueError, match=field):
        analyzer.analyze_budget_penalty_evidence_family(phase164, phase215)


def test_each_topology_must_have_three_shapes(tmp_path: Path) -> None:
    phase164, phase215 = _write_inputs(tmp_path)
    rows = analyzer.analyze_budget_penalty_evidence_family(phase164, phase215)

    with pytest.raises(ValueError, match="topology must have exactly 3 shapes"):
        analyzer._add_topology_summary(rows[:-1])


def test_actual_csvs_generate_six_diagnostic_rows() -> None:
    root = Path(__file__).resolve().parents[3]

    rows = analyzer.analyze_budget_penalty_evidence_family(
        root / "docs/iter_gap_investigation/phase164_clean_gpu_budget_manifest.csv",
        root / "docs/iter_gap_investigation/phase215_budget_penalty_mechanism.csv",
    )

    assert len(rows) == 6
    assert {row["shape_count"] for row in rows} == {"3"}
    assert {row["default_readiness"] for row in rows} == {"No-Go"}
    assert {row["recommended_model_boundary"] for row in rows} == {"diagnostic_only_exact_key"}
    assert [row for row in rows if row["shape_key"] == "isl8000_osl2000_batch128"][0]["evidence_source"] == "phase164_clean_gpu_budget_manifest"
