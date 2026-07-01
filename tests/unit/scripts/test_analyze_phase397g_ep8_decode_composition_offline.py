from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT / "scripts" / "analyze_phase397g_ep8_decode_composition_offline.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase397g_ep8_decode_composition_offline", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analyzer
SPEC.loader.exec_module(analyzer)

CHECKED_IN_CSV = (
    REPO_ROOT / "docs/iter_gap_investigation/phase397g_ep8_decode_composition_offline.csv"
)
CHECKED_IN_MD = (
    REPO_ROOT / "docs/iter_gap_investigation/phase397g_ep8_decode_composition_offline.md"
)
OVERHEAD_RAW = (
    REPO_ROOT / "docs/iter_gap_investigation/phase397g_overhead_sensitivity_raw.csv"
)
ATTRIBUTION_RAW = (
    REPO_ROOT / "docs/iter_gap_investigation/phase397g_residual_attribution_raw.csv"
)


def _rows() -> list[dict[str, str]]:
    return analyzer.analyze_phase397g()


def test_first_and_last_rows() -> None:
    rows = _rows()
    assert rows[0]["row_type"] == "ep8_overhead_semantics"
    assert [r["row_type"] for r in rows[-2:]] == ["verdict", "next_phase"]


def test_all_rows_use_fieldnames() -> None:
    for row in _rows():
        assert set(row) == set(analyzer.FIELDNAMES)


def test_required_row_types_present() -> None:
    types = {r["row_type"] for r in _rows()}
    for expected in (
        "ep8_overhead_semantics",
        "overhead_sensitivity",
        "puredecode_ep8_probe",
        "dp_normalization_check",
        "residual_attribution",
        "candidate_fixes",
        "verdict",
        "next_phase",
    ):
        assert expected in types


def test_row_counts_match_configs() -> None:
    rows = _rows()
    n = len(analyzer.ATTR)
    for per_config in ("overhead_sensitivity", "puredecode_ep8_probe", "residual_attribution"):
        assert sum(r["row_type"] == per_config for r in rows) == n


def test_verdict_only_guards_all_false() -> None:
    for row in _rows():
        assert row["runtime_modified"] == "false"
        assert row["diagnostic_only"] == "true"
        assert row["valid_for_default"] == "false"
        assert row["default_readiness"] == "No-Go"
        for guard in (
            "fudge_factor_tuning_used",
            "scope_gating_used",
            "write_real_data_file",
            "gpu_allowed",
            "ssh_allowed",
            "perf_database",
            "nearest_lookup_allowed",
            "extrapolation_allowed",
            "default_aic_allowed",
        ):
            assert row[guard] == "false", (row["row_type"], guard)


def test_every_config_overpredicts_at_overhead_zero() -> None:
    # The load-bearing finding: the 90ms fudge was calibrated to pre-397f
    # under-count; with it removed every ep8 config over-predicts.
    for name, a in analyzer.ATTR.items():
        assert a.sim_ovh0_ratio > 1.0, name


def test_90ms_overhead_regresses_decode_heavy() -> None:
    a = analyzer.ATTR["tp8ep8-8k2k"]
    assert a.overhead_regressed
    assert a.abs_err_ovh90 > a.abs_err_ovh0


def test_dp2_matches_real_at_tp_times_dp_not_tp() -> None:
    a = analyzer.ATTR["tp4ep8dp2-8k2k"]
    # pure-decode compose lines up with real decode iter at num_gpus=tp*dp
    assert 0.9 <= a.per_iter_over_real_ng_tp_dp <= 1.1
    # ...and is off by ~2x at num_gpus=tp
    assert a.per_iter_no_ovh_ms / a.real_iter_ng_tp_ms < 0.75


def test_dp1_configs_have_equal_num_gpus_conventions() -> None:
    for name, a in analyzer.ATTR.items():
        if a.dp == 1:
            assert a.real_iter_ng_tp_ms == pytest.approx(a.real_iter_ng_tp_dp_ms)


def test_abs_err_symmetry() -> None:
    a = analyzer.Attr("x", 8, 1, "s", 100.0, 120.0, 120.0, 0.5, 2.0, "c")
    assert a.abs_err_ovh0 == pytest.approx(2.0)
    assert a.abs_err_ovh90 == pytest.approx(2.0)


def test_candidate_fixes_enumerate_structural_changes() -> None:
    row = next(r for r in _rows() if r["row_type"] == "candidate_fixes")
    text = (row["value_a"] + row["value_b"]).lower()
    assert "num_gpus" in text
    assert "overhead" in text
    assert "mixed" in text or "prefill" in text


def test_verdict_and_next_phase() -> None:
    verdict = next(r for r in _rows() if r["row_type"] == "verdict")
    assert "single" in verdict["verdict"]
    assert "No-Go" in verdict["value_b"]
    assert verdict["next_allowed_phase"] == analyzer.NEXT_PHASE
    nxt = next(r for r in _rows() if r["row_type"] == "next_phase")
    assert "ep8" in nxt["verdict"] or "num_gpus" in nxt["value_b"]
    assert analyzer.NEXT_PHASE == "phase397h_ep8_decode_composition_fix"


def test_checked_in_csv_matches_analyzer(tmp_path: Path) -> None:
    if not CHECKED_IN_CSV.exists():
        pytest.skip("phase397g csv not generated yet")
    out = tmp_path / "phase397g.csv"
    analyzer.write_phase397g_csv(out, _rows())
    assert CHECKED_IN_CSV.read_text(encoding="utf-8") == out.read_text(encoding="utf-8")


def test_checked_in_csv_parseable() -> None:
    if not CHECKED_IN_CSV.exists():
        pytest.skip("phase397g csv not generated yet")
    with CHECKED_IN_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == analyzer.FIELDNAMES
        rows = list(reader)
    assert rows[0]["row_type"] == "ep8_overhead_semantics"


def test_overhead_raw_present_and_consistent() -> None:
    assert OVERHEAD_RAW.exists()
    with OVERHEAD_RAW.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    for row in rows:
        # At overhead=0 every config over-predicts.
        assert float(row["ratio_ovh0"]) > 1.0


def test_attribution_raw_present_and_consistent() -> None:
    assert ATTRIBUTION_RAW.exists()
    with ATTRIBUTION_RAW.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(analyzer.ATTR)
    by_name = {r["config"]: r for r in rows}
    dp2 = by_name["tp4ep8dp2-8k2k"]
    assert 0.9 <= float(dp2["per_iter_over_real_ng_tp_dp"]) <= 1.1


def test_checked_in_md_has_verdict() -> None:
    if not CHECKED_IN_MD.exists():
        pytest.skip("phase397g md not generated yet")
    text = CHECKED_IN_MD.read_text(encoding="utf-8")
    assert "Phase397g" in text
    assert "step 4" in text
    assert "No-Go" in text
    assert "num_gpus=tp*dp" in text
