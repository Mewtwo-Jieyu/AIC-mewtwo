from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "analyze_phase215_budget_penalty_mechanism.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase215_budget_penalty_mechanism",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


PAIR_ROWS = [
    ("tp8_dp1_ep8", "isl4000_osl2000_batch128", "tp8ep8-4k2k-bt4000", "tp8ep8-4k2k-bt65536", 4000, 100.0, 105.0, 10.0, 40.0),
    ("tp4_dp2_ep8", "isl4000_osl2000_batch128", "tp4dp2ep8-4k2k-bt4000", "tp4dp2ep8-4k2k-bt65536", 4000, 200.0, 240.0, 20.0, 100.0),
    ("tp8_dp1_ep8", "isl12000_osl2000_batch128", "tp8ep8-12k2k-bt12000", "tp8ep8-12k2k-bt65536", 12000, 80.0, 88.0, 25.0, 50.0),
    ("tp4_dp2_ep8", "isl12000_osl2000_batch128", "tp4dp2ep8-12k2k-bt12000", "tp4dp2ep8-12k2k-bt65536", 12000, 300.0, 297.0, 30.0, 120.0),
]


def _phase211_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for topology, shape, control, holdout, control_bt, control_out, holdout_out, control_ms, holdout_ms in PAIR_ROWS:
        clean_effect = holdout_out / control_out
        steady_ratio = holdout_ms / control_ms
        rows.append(
            {
                "source": "phase211_budget_penalty_residual",
                "topology_key": topology,
                "shape_key": shape,
                "control_scenario": control,
                "holdout_scenario": holdout,
                "control_max_bt": str(control_bt),
                "holdout_max_bt": "65536",
                "control_output_tok_s": f"{control_out:.6f}",
                "holdout_output_tok_s": f"{holdout_out:.6f}",
                "clean_budget_effect": f"{clean_effect:.6f}",
                "clean_delta_from_1": f"{clean_effect - 1.0:.6f}",
                "control_steady_state_time_ms": f"{control_ms:.6f}",
                "holdout_steady_state_time_ms": f"{holdout_ms:.6f}",
                "steady_state_time_ratio": f"{steady_ratio:.6f}",
                "penalty_overstatement": f"{steady_ratio / clean_effect:.6f}",
                "default_readiness": "No-Go",
                "diagnostic_only": "true",
                "valid_for_default": "false",
                "perf_database": "false",
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_input(tmp_path: Path, rows=None) -> Path:
    path = tmp_path / "phase211.csv"
    _write_csv(path, rows or _phase211_rows())
    return path


def test_budget_penalty_mechanism_compares_four_diagnostic_pairs(tmp_path: Path) -> None:
    input_csv = _write_input(tmp_path)
    out_csv = tmp_path / "mechanism.csv"
    out_doc = tmp_path / "mechanism.md"

    rows = analyzer.analyze_budget_penalty_mechanism(input_csv)
    analyzer.write_mechanism_csv(out_csv, rows)
    analyzer.write_mechanism_doc(out_doc, rows)

    assert len(rows) == 4
    assert rows[0]["source"] == "phase215_budget_penalty_mechanism"
    assert rows[0]["steady_linear_error"] == "3.809524"
    assert rows[0]["no_budget_penalty_error"] == "1.050000"
    assert rows[0]["topology_shape_spread"] == "1.047619"
    assert rows[0]["recommended_model_boundary"] == "diagnostic_only_exact_key"
    assert {row["diagnostic_only"] for row in rows} == {"true"}
    assert {row["valid_for_default"] for row in rows} == {"false"}
    assert {row["perf_database"] for row in rows} == {"false"}
    assert out_csv.read_text(encoding="utf-8").count("\n") == 5
    doc = out_doc.read_text(encoding="utf-8")
    assert "high-budget steady ratio is large while clean effect stays near 1" in doc
    assert "global constant is No-Go" in doc
    assert "diagnostic_only_exact_key" in doc
    assert "Default AIC | No-Go" in doc


@pytest.mark.parametrize("rows", [_phase211_rows()[:-1], _phase211_rows() + [_phase211_rows()[0]]])
def test_row_count_guard_fails_fast(tmp_path: Path, rows: list[dict[str, str]]) -> None:
    input_csv = _write_input(tmp_path, rows)

    with pytest.raises(ValueError, match="phase211 residual must contain exactly 4 rows"):
        analyzer.analyze_budget_penalty_mechanism(input_csv)


@pytest.mark.parametrize("field", ["diagnostic_only", "valid_for_default", "perf_database"])
def test_flag_guards_fail_fast(tmp_path: Path, field: str) -> None:
    rows = _phase211_rows()
    rows[0][field] = "false" if field == "diagnostic_only" else "true"
    input_csv = _write_input(tmp_path, rows)

    with pytest.raises(ValueError, match=field):
        analyzer.analyze_budget_penalty_mechanism(input_csv)


def test_ratio_guard_fails_fast(tmp_path: Path) -> None:
    rows = _phase211_rows()
    rows[0]["penalty_overstatement"] = "1.000000"
    input_csv = _write_input(tmp_path, rows)

    with pytest.raises(ValueError, match="penalty_overstatement mismatch"):
        analyzer.analyze_budget_penalty_mechanism(input_csv)


def test_actual_phase211_csv_generates_exact_key_boundary() -> None:
    root = Path(__file__).resolve().parents[3]
    rows = analyzer.analyze_budget_penalty_mechanism(
        root / "docs/iter_gap_investigation/phase211_budget_penalty_residual.csv"
    )

    assert len(rows) == 4
    assert {row["default_readiness"] for row in rows} == {"No-Go"}
    assert {row["recommended_model_boundary"] for row in rows} == {"diagnostic_only_exact_key"}
    assert float(rows[0]["steady_linear_error"]) > float(rows[0]["no_budget_penalty_error"])
