from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "analyze_phase211_budget_penalty_residual.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase211_budget_penalty_residual",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


PAIR_ROWS = [
    ("tp8_dp1_ep8", "isl4000_osl2000_batch128", "tp8ep8-4k2k-bt4000", "tp8ep8-4k2k-bt65536", 4000, 100.0, 105.0, 10.0, 40.0),
    ("tp4_dp2_ep8", "isl4000_osl2000_batch128", "tp4dp2ep8-4k2k-bt4000", "tp4dp2ep8-4k2k-bt65536", 4000, 200.0, 240.0, 20.0, 100.0),
    ("tp8_dp1_ep8", "isl12000_osl2000_batch128", "tp8ep8-12k2k-bt12000", "tp8ep8-12k2k-bt65536", 12000, 80.0, 84.0, 25.0, 50.0),
    ("tp4_dp2_ep8", "isl12000_osl2000_batch128", "tp4dp2ep8-12k2k-bt12000", "tp4dp2ep8-12k2k-bt65536", 12000, 300.0, 297.0, 30.0, 120.0),
]

SCENARIO_META = {
    "tp8ep8-4k2k-bt4000": ("control", 8, 1, 8, 4000, 4000),
    "tp8ep8-4k2k-bt65536": ("holdout", 8, 1, 8, 4000, 65536),
    "tp4dp2ep8-4k2k-bt4000": ("control", 4, 2, 8, 4000, 4000),
    "tp4dp2ep8-4k2k-bt65536": ("holdout", 4, 2, 8, 4000, 65536),
    "tp8ep8-12k2k-bt12000": ("control", 8, 1, 8, 12000, 12000),
    "tp8ep8-12k2k-bt65536": ("holdout", 8, 1, 8, 12000, 65536),
    "tp4dp2ep8-12k2k-bt12000": ("control", 4, 2, 8, 12000, 12000),
    "tp4dp2ep8-12k2k-bt65536": ("holdout", 4, 2, 8, 12000, 65536),
}


def _analysis_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for topology, shape, control, holdout, control_bt, control_out, holdout_out, control_ms, holdout_ms in PAIR_ROWS:
        clean_ratio = holdout_out / control_out
        rows.append(
            {
                "source": "phase199_holdout_budget_mechanism_analysis",
                "topology_key": topology,
                "shape_key": shape,
                "control_scenario": control,
                "holdout_scenario": holdout,
                "control_max_bt": str(control_bt),
                "holdout_max_bt": "65536",
                "control_output_tok_s": f"{control_out:.6f}",
                "holdout_output_tok_s": f"{holdout_out:.6f}",
                "clean_high_over_control": f"{clean_ratio:.6f}",
                "clean_delta_from_1": f"{clean_ratio - 1.0:.6f}",
                "control_steady_state_time_ms": f"{control_ms:.6f}",
                "holdout_steady_state_time_ms": f"{holdout_ms:.6f}",
                "steady_state_time_ratio": f"{holdout_ms / control_ms:.6f}",
                "diagnostic_only": "true",
                "valid_for_default": "false",
                "perf_database": "false",
            }
        )
    return rows


def _manifest_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    by_scenario = {}
    for topology, shape, control, holdout, _control_bt, control_out, holdout_out, control_ms, holdout_ms in PAIR_ROWS:
        by_scenario[control] = (topology, shape, control_out, control_ms)
        by_scenario[holdout] = (topology, shape, holdout_out, holdout_ms)
    for scenario, (topology, shape, output, steady_ms) in by_scenario.items():
        role, tp, dp, ep, isl, max_bt = SCENARIO_META[scenario]
        rows.append(
            {
                "source": "phase178_budget_mechanism_holdout",
                "scenario": scenario,
                "topology_key": topology,
                "shape_key": shape,
                "role": role,
                "tp": str(tp),
                "dp": str(dp),
                "ep": str(ep),
                "isl": str(isl),
                "osl": "2000",
                "batch_size": "128",
                "max_num_batched_tokens": str(max_bt),
                "request_success_count": "128",
                "request_fail_count": "0",
                "real_output_tok_s": f"{output:.6f}",
                "steady_state_time_ms": f"{steady_ms:.6f}",
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


def _write_inputs(tmp_path: Path, manifest_rows=None, analysis_rows=None) -> tuple[Path, Path]:
    manifest = tmp_path / "manifest.csv"
    analysis = tmp_path / "analysis.csv"
    _write_csv(manifest, manifest_rows or _manifest_rows())
    _write_csv(analysis, analysis_rows or _analysis_rows())
    return manifest, analysis


def test_budget_penalty_residual_rows_are_diagnostic_no_go(tmp_path: Path) -> None:
    manifest, analysis = _write_inputs(tmp_path)
    out_csv = tmp_path / "residual.csv"
    out_doc = tmp_path / "residual.md"

    rows = analyzer.analyze_budget_penalty_residual(manifest, analysis)
    analyzer.write_residual_csv(out_csv, rows)
    analyzer.write_residual_doc(out_doc, rows)

    assert len(rows) == 4
    assert rows[0]["source"] == "phase211_budget_penalty_residual"
    assert rows[0]["clean_budget_effect"] == "1.050000"
    assert rows[0]["steady_state_time_ratio"] == "4.000000"
    assert rows[0]["penalty_overstatement"] == "3.809524"
    assert rows[0]["default_readiness"] == "No-Go"
    assert {row["diagnostic_only"] for row in rows} == {"true"}
    assert {row["valid_for_default"] for row in rows} == {"false"}
    assert {row["perf_database"] for row in rows} == {"false"}
    assert out_csv.read_text(encoding="utf-8").count("\n") == 5
    doc = out_doc.read_text(encoding="utf-8")
    assert "scheduler penalty is overstated" in doc
    assert "not a global constant" in doc
    assert "Default AIC | No-Go" in doc


@pytest.mark.parametrize(
    ("manifest_rows", "analysis_rows", "message"),
    [
        (_manifest_rows()[:-1], None, "manifest must contain exactly 8 rows"),
        (None, _analysis_rows()[:-1], "analysis must contain exactly 4 rows"),
    ],
)
def test_row_count_guards_fail_fast(tmp_path: Path, manifest_rows, analysis_rows, message: str) -> None:
    manifest, analysis = _write_inputs(tmp_path, manifest_rows, analysis_rows)

    with pytest.raises(ValueError, match=message):
        analyzer.analyze_budget_penalty_residual(manifest, analysis)


@pytest.mark.parametrize("field", ["diagnostic_only", "valid_for_default", "perf_database"])
def test_flag_guards_fail_fast(tmp_path: Path, field: str) -> None:
    rows = _analysis_rows()
    rows[0][field] = "false" if field == "diagnostic_only" else "true"
    manifest, analysis = _write_inputs(tmp_path, analysis_rows=rows)

    with pytest.raises(ValueError, match=field):
        analyzer.analyze_budget_penalty_residual(manifest, analysis)


def test_ratio_guard_fails_fast(tmp_path: Path) -> None:
    rows = _analysis_rows()
    rows[0]["steady_state_time_ratio"] = "9.000000"
    manifest, analysis = _write_inputs(tmp_path, analysis_rows=rows)

    with pytest.raises(ValueError, match="steady_state_time_ratio mismatch"):
        analyzer.analyze_budget_penalty_residual(manifest, analysis)


def test_actual_phase199_inputs_generate_four_no_go_rows() -> None:
    root = Path(__file__).resolve().parents[3]
    rows = analyzer.analyze_budget_penalty_residual(
        root / "docs/iter_gap_investigation/phase178_budget_mechanism_holdout_manifest.csv",
        root / "docs/iter_gap_investigation/phase199_holdout_budget_mechanism_analysis.csv",
    )

    assert len(rows) == 4
    assert {row["default_readiness"] for row in rows} == {"No-Go"}
    assert rows[3]["topology_key"] == "tp4_dp2_ep8"
    assert rows[3]["shape_key"] == "isl12000_osl2000_batch128"
    assert rows[3]["clean_budget_effect"] == "0.993809"
    assert float(rows[3]["penalty_overstatement"]) > 4.0
