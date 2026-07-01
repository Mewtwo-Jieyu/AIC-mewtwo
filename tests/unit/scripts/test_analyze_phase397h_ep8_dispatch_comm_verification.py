from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT / "scripts" / "analyze_phase397h_ep8_dispatch_comm_verification.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase397h_ep8_dispatch_comm_verification", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analyzer
SPEC.loader.exec_module(analyzer)

CHECKED_IN_CSV = (
    REPO_ROOT / "docs/iter_gap_investigation/phase397h_ep8_dispatch_comm_verification.csv"
)
CHECKED_IN_MD = (
    REPO_ROOT / "docs/iter_gap_investigation/phase397h_ep8_dispatch_comm_verification.md"
)
DISPATCH_RAW = (
    REPO_ROOT / "docs/iter_gap_investigation/phase397h_dispatch_branch_raw.csv"
)
MLA_RAW = (
    REPO_ROOT / "docs/iter_gap_investigation/phase397h_mla_tp_probe_raw.csv"
)


def _rows() -> list[dict[str, str]]:
    return analyzer.analyze_phase397h()


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
        "dispatch_branch",
        "measured_comm_compare",
        "mla_tp_probe",
        "owed_magnitude_bounds",
        "isl_scaling_signature",
        "refutation",
        "candidate_fixes",
        "verdict",
        "next_phase",
    ):
        assert expected in types


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


def test_dispatch_is_additive_not_if_elif() -> None:
    # tp4dp2 must charge BOTH allreduce and the cross-dp comm term.
    d = analyzer.DISPATCH["tp4ep8dp2"]
    assert d.is_additive_dp
    assert d.attention_dp > 1
    # tp8dp1 has no cross-dp term.
    assert not analyzer.DISPATCH["tp8ep8dp1"].is_additive_dp


def test_measured_comm_is_small_vs_owed() -> None:
    d = analyzer.DISPATCH["tp4ep8dp2"]
    # measured all-layer comm ~13ms, far below owed 116ms; sim under-count minor.
    assert d.measured_comm_all_layer_ms is not None
    assert d.measured_comm_all_layer_ms < 25.0
    assert abs(d.sim_under_comm_ms) < 20.0


def test_sim_moe_reconciles_with_measured_module() -> None:
    calls = analyzer.DISPATCH["tp4ep8dp2"].implied_moe_calls
    assert calls is not None
    assert 30.0 <= calls <= 61.0


def test_mla_attention_flat_despite_heads_doubling() -> None:
    tp8 = analyzer.MLA["tp8ep8dp1"]
    tp4 = analyzer.MLA["tp4ep8dp2"]
    assert tp4.num_heads_per_gpu == 2 * tp8.num_heads_per_gpu
    ratio = tp4.attn_ms_bs128 / tp8.attn_ms_bs128
    assert 0.85 <= ratio <= 1.15


def test_aggregate_derived_iter_is_prefill_inflated() -> None:
    o = analyzer.OWED["tp4ep8dp2-8k2k"]
    assert o.measured_consistent
    assert o.aggregate_derived_iter_ms > 1.5 * o.sim_steady_iter_ms


def test_overprediction_grows_with_isl() -> None:
    assert (
        analyzer.OWED["tp8ep8-32k3k"].sim_ovh0_ratio
        > analyzer.OWED["tp8ep8-8k2k"].sim_ovh0_ratio
    )
    assert (
        analyzer.OWED["tp4ep8dp2-32k3k"].sim_ovh0_ratio
        > analyzer.OWED["tp4ep8dp2-8k2k"].sim_ovh0_ratio
    )


def test_refutation_and_verdict_reattribute_to_prefill() -> None:
    refute = next(r for r in _rows() if r["row_type"] == "refutation")
    assert "REFUTED" in refute["metric"]
    verdict = next(r for r in _rows() if r["row_type"] == "verdict")
    assert "NOT under-modeled" in verdict["value_a"]
    assert "prefill" in verdict["verdict"]
    assert "No-Go" in verdict["value_b"]
    assert verdict["next_allowed_phase"] == analyzer.NEXT_PHASE
    assert analyzer.NEXT_PHASE == "phase397i_prefill_mixed_accounting"


def test_candidate_fixes_pivot_to_prefill_and_flag_tpot() -> None:
    row = next(r for r in _rows() if r["row_type"] == "candidate_fixes")
    text = (row["value_a"] + row["value_b"]).lower()
    assert "prefill" in text or "mixed" in text
    assert "tpot" in text


def test_checked_in_csv_matches_analyzer(tmp_path: Path) -> None:
    if not CHECKED_IN_CSV.exists():
        pytest.skip("phase397h csv not generated yet")
    out = tmp_path / "phase397h.csv"
    analyzer.write_phase397h_csv(out, _rows())
    assert CHECKED_IN_CSV.read_text(encoding="utf-8") == out.read_text(encoding="utf-8")


def test_checked_in_csv_parseable() -> None:
    if not CHECKED_IN_CSV.exists():
        pytest.skip("phase397h csv not generated yet")
    with CHECKED_IN_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == analyzer.FIELDNAMES
        rows = list(reader)
    assert rows[0]["row_type"] == "hypothesis_under_test"


def test_dispatch_raw_present_and_consistent() -> None:
    assert DISPATCH_RAW.exists()
    with DISPATCH_RAW.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_name = {r["config"]: r for r in rows}
    tp4 = by_name["tp4ep8dp2"]
    assert "additive" in tp4["dispatch_branch"]
    assert float(tp4["measured_ep8comm_all_layer_ms"]) < 25.0


def test_mla_raw_present_and_consistent() -> None:
    assert MLA_RAW.exists()
    with MLA_RAW.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    # tp4 has 2x heads/gpu but the ratio vs tp8 stays ~1.0.
    tp4_rows = [r for r in rows if r["config"] == "tp4ep8dp2"]
    assert tp4_rows
    for r in tp4_rows:
        assert 0.85 <= float(r["ratio_vs_tp8_same_bs"]) <= 1.15


def test_checked_in_md_has_refutation() -> None:
    if not CHECKED_IN_MD.exists():
        pytest.skip("phase397h md not generated yet")
    text = CHECKED_IN_MD.read_text(encoding="utf-8")
    assert "Phase397h" in text
    assert "step 5" in text
    assert "No-Go" in text
    assert "REFUT" in text.upper()
    assert "prefill" in text.lower()

