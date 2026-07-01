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
TP_SCALING_RAW = (
    REPO_ROOT / "docs/iter_gap_investigation/phase397g_tp_scaling_probe_raw.csv"
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
        "tp_degree_check",
        "tp_scaling_probe",
        "residual_attribution",
        "candidate_fixes",
        "verdict",
        "next_phase",
    ):
        assert expected in types


def test_no_dp_normalization_row_type() -> None:
    # The retracted claim must not linger as a row type.
    assert "dp_normalization_check" not in {r["row_type"] for r in _rows()}


def test_row_counts_match_configs() -> None:
    rows = _rows()
    n = len(analyzer.ATTR)
    for per_config in ("overhead_sensitivity", "puredecode_ep8_probe", "residual_attribution"):
        assert sum(r["row_type"] == per_config for r in rows) == n
    assert sum(r["row_type"] == "tp_scaling_probe" for r in rows) == 5


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
    for name, a in analyzer.ATTR.items():
        assert a.sim_ovh0_ratio > 1.0, name


def test_90ms_overhead_regresses_decode_heavy() -> None:
    a = analyzer.ATTR["tp8ep8-8k2k"]
    assert a.overhead_regressed
    assert a.abs_err_ovh90 > a.abs_err_ovh0


def test_dp_cancels_in_normalization() -> None:
    # dp1 configs must have identical num_gpus conventions (dp trivially cancels),
    # and the tp_degree_check row must state the cancellation explicitly.
    for name, a in analyzer.ATTR.items():
        if a.dp == 1:
            assert a.real_iter_ng_tp_ms == pytest.approx(a.real_iter_ng_tp_dp_ms)
    row = next(r for r in _rows() if r["row_type"] == "tp_degree_check")
    assert "CANCELS" in row["value_a"]
    assert "not a normalization bug" in row["verdict"].lower() or "not a normalization" in row["value_a"].lower()


def test_tp4_real_iter_about_2x_tp8() -> None:
    real = analyzer.TP_SCALING["REAL_DECODE_ITER"]
    assert 1.8 <= real.ratio <= 2.2
    # ...while the sim total barely moves.
    assert analyzer.TP_SCALING["SIM_TOTAL"].ratio < 1.4


def test_sim_too_fast_worse_for_tp4() -> None:
    tp8 = analyzer.ATTR["tp8ep8-8k2k"]
    tp4 = analyzer.ATTR["tp4ep8dp2-8k2k"]
    # sim under-covers real iter for both, and much more for tp4.
    assert tp8.sim_too_fast_ratio < 1.0
    assert tp4.sim_too_fast_ratio < tp8.sim_too_fast_ratio
    assert tp4.owed_comm_ms > tp8.owed_comm_ms


def test_attention_barely_scales_with_tp() -> None:
    attn = analyzer.TP_SCALING["generation_attention"]
    assert 0.85 <= attn.ratio <= 1.15
    assert not attn.scales_with_tp


def test_moe_dominates_and_is_flat() -> None:
    moe = analyzer.TP_SCALING["generation_moe"]
    # MoE is the single largest op and does not scale ~2x with tp.
    assert moe.tp8_ms > analyzer.TP_SCALING["generation_attention"].tp8_ms
    assert moe.ratio < 1.4


def test_ep_dispatch_undermodeled_vs_owed_comm() -> None:
    pre = analyzer.TP_SCALING["generation_moe_pre_dispatch"]
    post = analyzer.TP_SCALING["generation_moe_post_dispatch"]
    modeled = pre.tp4_ms + post.tp4_ms
    owed = analyzer.ATTR["tp4ep8dp2-8k2k"].owed_comm_ms
    assert modeled < 0.15 * owed


def test_abs_err_symmetry() -> None:
    a = analyzer.Attr("x", 8, 1, "s", 100.0, 120.0, 120.0, 0.5, 2.0, "c")
    assert a.abs_err_ovh0 == pytest.approx(2.0)
    assert a.abs_err_ovh90 == pytest.approx(2.0)


def test_candidate_fixes_enumerate_ep_comm() -> None:
    row = next(r for r in _rows() if r["row_type"] == "candidate_fixes")
    text = (row["value_a"] + row["value_b"]).lower()
    assert "all-to-all" in text or "dispatch" in text
    assert "comm" in text
    assert "mixed" in text or "prefill" in text


def test_verdict_retracts_normalization_and_names_ep_comm() -> None:
    verdict = next(r for r in _rows() if r["row_type"] == "verdict")
    assert "not a normalization bug" in verdict["verdict"]
    assert "communication" in verdict["verdict"] or "comm" in verdict["value_a"].lower()
    assert "No-Go" in verdict["value_b"]
    assert verdict["next_allowed_phase"] == analyzer.NEXT_PHASE
    nxt = next(r for r in _rows() if r["row_type"] == "next_phase")
    assert "comm" in (nxt["value_a"] + nxt["verdict"]).lower()
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
        assert float(row["ratio_ovh0"]) > 1.0


def test_attribution_raw_present_and_consistent() -> None:
    assert ATTRIBUTION_RAW.exists()
    with ATTRIBUTION_RAW.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(analyzer.ATTR)
    by_name = {r["config"]: r for r in rows}
    # dp2 pure-decode compose is ~2x too fast vs the real (num_gpus=tp) iter.
    dp2 = by_name["tp4ep8dp2-8k2k"]
    assert 0.45 <= float(dp2["sim_too_fast_vs_real_iter_ng_tp"]) <= 0.55
    assert "ep_comm" in dp2["dominant_cause"]


def test_tp_scaling_raw_present_and_consistent() -> None:
    assert TP_SCALING_RAW.exists()
    with TP_SCALING_RAW.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_op = {r["op_name"]: r for r in rows}
    assert float(by_op["REAL_DECODE_ITER"]["ratio_tp4_over_tp8"]) > 1.8
    assert 0.85 <= float(by_op["generation_attention"]["ratio_tp4_over_tp8"]) <= 1.15


def test_checked_in_md_has_correction() -> None:
    if not CHECKED_IN_MD.exists():
        pytest.skip("phase397g md not generated yet")
    text = CHECKED_IN_MD.read_text(encoding="utf-8")
    assert "Phase397g" in text
    assert "step 4" in text
    assert "No-Go" in text
    assert "CORRECTION" in text
    assert "dp cancels" in text.lower() or "dp cancels" in text
