from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "analyze_phase206_holdout_default_readiness.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase206_holdout_default_readiness",
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
        ratio = holdout_out / control_out
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
                "clean_high_over_control": f"{ratio:.6f}",
                "clean_delta_from_1": f"{ratio - 1.0:.6f}",
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
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_inputs(tmp_path: Path, manifest_rows=None, analysis_rows=None) -> tuple[Path, Path]:
    manifest = tmp_path / "manifest.csv"
    analysis = tmp_path / "analysis.csv"
    _write_csv(manifest, manifest_rows or _manifest_rows())
    _write_csv(analysis, analysis_rows or _analysis_rows())
    return manifest, analysis


def test_default_readiness_is_no_go_for_four_diagnostic_pairs(tmp_path: Path) -> None:
    manifest, analysis = _write_inputs(tmp_path)
    out_doc = tmp_path / "readiness.md"

    result = analyzer.analyze_default_readiness(manifest, analysis)
    analyzer.write_readiness_doc(out_doc, result)

    assert result["default_readiness"] == "No-Go"
    assert result["pair_count"] == 4
    assert result["default_aic_allowed"] is False
    assert result["global_constant_allowed"] is False
    assert result["raw_multiplier_allowed"] is False
    assert result["diagnostic_lookup_only"] is True
    assert result["shape_behavior_inconsistent"] is True
    assert result["tp4dp2_12k2k_below_one"] is True
    assert result["flags"] == ("true", "false", "false")
    text = out_doc.read_text(encoding="utf-8")
    assert "default_readiness=No-Go" in text
    assert "Default AIC | No-Go" in text
    assert "Only 4 holdout pairs" in text
    assert "no global constant" in text
    assert "raw multiplier" in text
    assert "diagnostic-only lookup" in text


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
        analyzer.analyze_default_readiness(manifest, analysis)


@pytest.mark.parametrize("field", ["diagnostic_only", "valid_for_default", "perf_database"])
def test_flag_guards_fail_fast(tmp_path: Path, field: str) -> None:
    rows = _analysis_rows()
    rows[0][field] = "false" if field == "diagnostic_only" else "true"
    manifest, analysis = _write_inputs(tmp_path, analysis_rows=rows)

    with pytest.raises(ValueError, match=field):
        analyzer.analyze_default_readiness(manifest, analysis)


def test_ratio_guard_fails_fast(tmp_path: Path) -> None:
    rows = _analysis_rows()
    rows[0]["clean_high_over_control"] = "2.000000"
    manifest, analysis = _write_inputs(tmp_path, analysis_rows=rows)

    with pytest.raises(ValueError, match="clean_high_over_control mismatch"):
        analyzer.analyze_default_readiness(manifest, analysis)


def test_cross_input_mismatch_fails_fast(tmp_path: Path) -> None:
    rows = _analysis_rows()
    rows[0]["control_output_tok_s"] = "101.000000"
    manifest, analysis = _write_inputs(tmp_path, analysis_rows=rows)

    with pytest.raises(ValueError, match="control_output_tok_s mismatch"):
        analyzer.analyze_default_readiness(manifest, analysis)


def test_actual_phase199_inputs_remain_no_go() -> None:
    root = Path(__file__).resolve().parents[3]
    result = analyzer.analyze_default_readiness(
        root / "docs/iter_gap_investigation/phase178_budget_mechanism_holdout_manifest.csv",
        root / "docs/iter_gap_investigation/phase199_holdout_budget_mechanism_analysis.csv",
    )

    assert result["default_readiness"] == "No-Go"
    assert result["pair_count"] == 4
    assert result["tp4dp2_12k2k_below_one"] is True
    assert result["diagnostic_lookup_only"] is True
