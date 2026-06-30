from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase392_e2e_attribution_gate.py"
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase392_e2e_attribution_gate", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)

CHECKED_IN_CSV = (
    REPO_ROOT / "docs/iter_gap_investigation/phase392_e2e_attribution_gate.csv"
)
CHECKED_IN_MD = (
    REPO_ROOT / "docs/iter_gap_investigation/phase392_e2e_attribution_gate.md"
)

EXPECTED_ROW_TYPES = [
    "baseline_query_moe_0_12",
    "baseline_query_moe_0_12",
    "baseline_query_moe_0_12",
    "module_bound_0_19_snap_oh0",
    "module_bound_0_19_snap_oh0",
    "module_bound_0_19_snap_oh0",
    "module_bound_0_19_snap_oh90_decode",
    "reachability_exact_only",
    "residual_attribution",
    "verdict",
    "next_phase",
]


def _rows() -> list[dict[str, str]]:
    return analyzer.analyze_phase392_e2e_attribution_gate()


def test_row_order_and_count() -> None:
    rows = _rows()
    assert [r["row_type"] for r in rows] == EXPECTED_ROW_TYPES


def test_all_rows_use_fieldnames() -> None:
    for row in _rows():
        assert set(row) == set(analyzer.FIELDNAMES)


def test_error_ratio_is_self_consistent() -> None:
    for row in _rows():
        if not row["error_ratio"]:
            continue
        gt = float(row["ground_truth_median_ms"])
        pred = float(row["predicted_median_ms"])
        recomputed = analyzer._error_ratio(gt, pred)
        assert abs(recomputed - float(row["error_ratio"])) <= 0.011


def test_prefill_baseline_under_predicts_and_module_improves() -> None:
    rows = _rows()
    by = {
        (r["phase"], r["comparison_mode"]): r for r in rows if r["error_ratio"]
    }
    base = float(by[("prefill", "baseline_query_moe_0_12")]["error_ratio"])
    mod = float(by[("prefill", "module_bound_0_19_snap")]["error_ratio"])
    assert base >= 3.0
    assert mod < base
    assert mod <= 1.5


def test_mixed_module_bound_near_gate() -> None:
    rows = _rows()
    by = {
        (r["phase"], r["comparison_mode"]): r for r in rows if r["error_ratio"]
    }
    mixed_mod = float(by[("mixed", "module_bound_0_19_snap")]["error_ratio"])
    assert mixed_mod <= 1.3


def test_decode_baseline_accurate_module_regresses() -> None:
    rows = _rows()
    by = {
        (r["phase"], r["comparison_mode"]): r for r in rows if r["error_ratio"]
    }
    base = float(by[("pure_decode", "baseline_query_moe_0_12")]["error_ratio"])
    mod = float(by[("pure_decode", "module_bound_0_19_snap")]["error_ratio"])
    assert base <= 1.3
    assert mod > base


def test_reachability_records_continuous_spread_blocker() -> None:
    row = next(r for r in _rows() if r["row_type"] == "reachability_exact_only")
    assert row["module_binding_triggered"] == "true"
    assert row["diagnostic_snap_used"] == "true"
    assert "ep8" in row["cbsim_token_spread"]
    assert "fusedmoe" in row["cbsim_token_spread"]
    assert row["exact_reachable_iters"]


def test_no_go_discipline_flags() -> None:
    forbidden_false = [
        "nearest_lookup_allowed",
        "interpolation_allowed",
        "extrapolation_allowed",
        "fudge_factor_tuning_used",
        "runtime_modified",
        "write_real_data_file",
        "gpu_allowed",
        "ssh_allowed",
        "default_aic_allowed",
        "valid_for_default",
        "perf_database",
    ]
    for row in _rows():
        for field in forbidden_false:
            assert row[field] == "false", (row["row_type"], field)
        assert row["exact_lookup_only"] == "true"
        assert row["diagnostic_only"] == "true"
        assert row["default_readiness"] == "No-Go"


def test_verdict_points_to_route_b_and_decode_measurement() -> None:
    verdict = next(r for r in _rows() if r["row_type"] == "verdict")
    assert "mixed" in verdict["verdict"]
    assert verdict["next_route"] == analyzer.NEXT_ROUTE
    assert "route_b" in analyzer.NEXT_ROUTE
    assert "decode" in analyzer.NEXT_ROUTE


def test_checked_in_csv_matches_analyzer(tmp_path: Path) -> None:
    if not CHECKED_IN_CSV.exists():
        pytest.skip("phase392 csv not generated yet")
    out = tmp_path / "phase392.csv"
    analyzer.write_phase392_csv(out, _rows())
    expected = out.read_text(encoding="utf-8")
    actual = CHECKED_IN_CSV.read_text(encoding="utf-8")
    assert actual == expected


def test_checked_in_csv_is_parseable_and_has_header() -> None:
    if not CHECKED_IN_CSV.exists():
        pytest.skip("phase392 csv not generated yet")
    with CHECKED_IN_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == analyzer.FIELDNAMES
        rows = list(reader)
    assert [r["row_type"] for r in rows] == EXPECTED_ROW_TYPES


def test_checked_in_md_has_verdict() -> None:
    if not CHECKED_IN_MD.exists():
        pytest.skip("phase392 md not generated yet")
    text = CHECKED_IN_MD.read_text(encoding="utf-8")
    assert "Phase392" in text
    assert "mixed" in text
    assert "Route B" in text or "route_b" in text
