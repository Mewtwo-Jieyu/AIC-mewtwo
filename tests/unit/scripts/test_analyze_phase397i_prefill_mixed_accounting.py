from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase397i_prefill_mixed_accounting.py"
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase397i_prefill_mixed_accounting", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analyzer
SPEC.loader.exec_module(analyzer)

DOCS = REPO_ROOT / "docs/iter_gap_investigation"
CHECKED_IN_CSV = DOCS / "phase397i_prefill_mixed_accounting.csv"
CHECKED_IN_MD = DOCS / "phase397i_prefill_mixed_accounting.md"
MAXDROP_RAW = DOCS / "phase397i_maxdrop_attn_raw.csv"
WALLSHARE_RAW = DOCS / "phase397i_prefill_wallshare_raw.csv"
REAL_TPOT_RAW = DOCS / "phase397i_real_tpot_raw.csv"


def _rows() -> list[dict[str, str]]:
    return analyzer.analyze_phase397i()


def test_first_and_last_rows() -> None:
    rows = _rows()
    assert rows[0]["row_type"] == "hypothesis_under_test"
    assert [r["row_type"] for r in rows[-2:]] == ["verdict", "next_phase"]


def test_all_rows_use_fieldnames() -> None:
    for row in _rows():
        assert set(row) == set(analyzer.FIELDNAMES)


def test_required_row_types_present() -> None:
    types = {r["row_type"] for r in _rows()}
    for expected in (
        "hypothesis_under_test",
        "baseline_realign",
        "real_tpot",
        "decode_moe_oversized",
        "dp_tokencount_error",
        "refute_397h_composition",
        "mechanism_a_minor",
        "mechanism_b_confounded",
        "candidate_fixes",
        "verdict",
        "next_phase",
    ):
        assert expected in types


def test_guards_gpu_ssh_true_rest_false() -> None:
    for row in _rows():
        assert row["runtime_modified"] == "false"
        assert row["diagnostic_only"] == "true"
        assert row["valid_for_default"] == "false"
        assert row["default_readiness"] == "No-Go"
        # authorized read-only capture this round
        assert row["gpu_allowed"] == "true"
        assert row["ssh_allowed"] == "true"
        for guard in (
            "fudge_factor_tuning_used",
            "scope_gating_used",
            "write_real_data_file",
            "perf_database",
            "nearest_lookup_allowed",
            "extrapolation_allowed",
            "default_aic_allowed",
        ):
            assert row[guard] == "false", (row["row_type"], guard)


def test_baseline_realignment_flips_sign() -> None:
    for name, b in analyzer.REALIGN.items():
        assert b.flips_sign, name
        assert b.ratio_over_0p17 > 1.0
        assert b.ratio_over_0p19 < 1.0


def test_real_tpot_sim_iter_is_3x_too_slow() -> None:
    for name, t in analyzer.REAL_TPOT.items():
        assert t.sim_iter_over_real > 2.0, name
        assert t.sim_out_over_real < 1.0, name


def test_moe_alone_exceeds_real_decode_iter() -> None:
    # tp8dp1 has the correct decode token-count, yet sim MoE (~71.8ms) alone
    # exceeds the entire real 0.19.0 decode iter (~34ms).
    assert analyzer.REAL_TPOT["tp8ep8-8k2k"].real_decode_iter_ms < 71.8


def test_mechanism_a_minor_and_grows_with_isl() -> None:
    a = analyzer.MECH_A
    assert a["tp8ep8-32k3k"].dropped_frac_of_steady > a["tp8ep8-8k2k"].dropped_frac_of_steady
    assert a["tp4ep8dp2-32k3k"].dropped_frac_of_steady > a["tp4ep8dp2-8k2k"].dropped_frac_of_steady
    assert a["tp4ep8dp2-32k3k"].dropped_frac_of_steady <= 0.25


def test_verdict_reverses_direction_and_names_decode_fix() -> None:
    verdict = next(r for r in _rows() if r["row_type"] == "verdict")
    assert "UNDER-predicts" in verdict["value_a"]
    assert "decode iter" in verdict["verdict"]
    assert "No-Go" in verdict["value_b"]
    assert verdict["next_allowed_phase"] == analyzer.NEXT_PHASE
    assert analyzer.NEXT_PHASE == "phase397j_decode_iter_baseline_realign"


def test_refute_397h_composition() -> None:
    row = next(r for r in _rows() if r["row_type"] == "refute_397h_composition")
    assert "REFUTED" in row["verdict"]


def test_candidate_fixes_name_baseline_and_decode_moe() -> None:
    row = next(r for r in _rows() if r["row_type"] == "candidate_fixes")
    text = (row["value_a"] + row["value_b"]).lower()
    assert "0.19.0" in text
    assert "moe" in text
    assert "dp" in text


def test_checked_in_csv_matches_analyzer(tmp_path: Path) -> None:
    if not CHECKED_IN_CSV.exists():
        pytest.skip("phase397i csv not generated yet")
    out = tmp_path / "phase397i.csv"
    analyzer.write_phase397i_csv(out, _rows())
    assert CHECKED_IN_CSV.read_text(encoding="utf-8") == out.read_text(encoding="utf-8")


def test_checked_in_csv_parseable() -> None:
    if not CHECKED_IN_CSV.exists():
        pytest.skip("phase397i csv not generated yet")
    with CHECKED_IN_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == analyzer.FIELDNAMES
        rows = list(reader)
    assert rows[0]["row_type"] == "hypothesis_under_test"


def test_real_tpot_raw_present_and_consistent() -> None:
    if not REAL_TPOT_RAW.exists():
        pytest.skip("phase397i real_tpot raw not generated yet")
    with REAL_TPOT_RAW.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by = {r["config"] + "/" + r["engine_dp"]: r for r in rows}
    tp8 = by["tp8ep8-bt8000/DP0"]
    assert float(tp8["real_pure_decode_ms_mean"]) < 40.0
    assert float(tp8["sim_iter_over_real"]) > 2.0


def test_maxdrop_raw_present_and_isl_growing() -> None:
    if not MAXDROP_RAW.exists():
        pytest.skip("phase397i maxdrop raw not generated yet")
    with MAXDROP_RAW.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by = {r["config"]: r for r in rows}
    assert (
        float(by["K2.5-tp8ep8-32k3k"]["dropped_frac_of_steady"])
        > float(by["K2.5-tp8ep8-8k2k"]["dropped_frac_of_steady"])
    )


def test_checked_in_md_has_reversal() -> None:
    if not CHECKED_IN_MD.exists():
        pytest.skip("phase397i md not generated yet")
    text = CHECKED_IN_MD.read_text(encoding="utf-8")
    assert "Phase397i" in text
    assert "step 6" in text
    assert "No-Go" in text
    assert "0.19.0" in text
    assert "under-predict" in text.lower()
