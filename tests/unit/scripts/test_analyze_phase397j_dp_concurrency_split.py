from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase397j_dp_concurrency_split.py"
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase397j_dp_concurrency_split", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analyzer
SPEC.loader.exec_module(analyzer)

DOCS = REPO_ROOT / "docs/iter_gap_investigation"
CHECKED_IN_CSV = DOCS / "phase397j_dp_concurrency_split.csv"
CHECKED_IN_MD = DOCS / "phase397j_dp_concurrency_split.md"
MULTI_RAW = DOCS / "phase397j_multi_config_before_after_raw.csv"
TOKENCOUNT_RAW = DOCS / "phase397j_dp_tokencount_raw.csv"


def _rows() -> list[dict[str, str]]:
    return analyzer.analyze_phase397j()


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
        "dp_tokencount_fix",
        "multi_config_before_after",
        "ratio_fix",
        "multi_config_aggregate",
        "residual_out_of_scope",
        "cross_backend_followup",
        "verdict",
        "next_phase",
    ):
        assert expected in types


def test_guards_runtime_modified_true_gpu_ssh_false() -> None:
    for row in _rows():
        assert row["runtime_modified"] == "true"
        assert row["diagnostic_only"] == "true"
        assert row["valid_for_default"] == "false"
        assert row["default_readiness"] == "No-Go"
        assert row["gpu_allowed"] == "false"
        assert row["ssh_allowed"] == "false"
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


def test_dp1_configs_bit_for_bit_unchanged() -> None:
    for m in analyzer.MULTI:
        if m.dp == 1:
            assert m.sim_before == pytest.approx(m.sim_after, abs=1e-6), m.name
            assert m.dp1_bitforbit_unchanged
    tp8 = analyzer.TOKENCOUNT["tp8ep8-8k2k"]
    assert tp8.unchanged


def test_dp2_tokencount_halved_and_matches_real() -> None:
    tp4 = analyzer.TOKENCOUNT["tp4ep8dp2-8k2k"]
    assert tp4.per_replica_bs_before == 128
    assert tp4.per_replica_bs_after == 64
    assert tp4.moe_tokens_before == 256
    assert tp4.moe_tokens_after == 128
    assert tp4.global_bs_before == 256
    assert tp4.global_bs_after == 128
    assert tp4.moe_matches_real_after


def test_ratio_moves_toward_real_and_ranking_inverts() -> None:
    for r in analyzer.RATIOS:
        assert r.improves, r.shape
    r8 = next(r for r in analyzer.RATIOS if r.shape == "8k2k")
    # before: tp4dp2 looked faster (ratio < 1); after: tp8 >= tp4dp2 (ratio >= ~1)
    assert r8.sim_ratio_before < 0.75
    assert r8.sim_ratio_after >= 0.75
    assert r8.dist_after < r8.dist_before


def test_multi_config_max_error_improves() -> None:
    assert analyzer._multi_max_after() < analyzer._multi_max_before()
    # worst pre-fix config was the dp double-count over-prediction
    assert analyzer._multi_max_before() > 2.0
    assert analyzer._multi_max_after() < 2.0


def test_residual_shared_by_tp8_is_version() -> None:
    tp8 = next(m for m in analyzer.MULTI if m.name == "K2.5-tp8ep8-8k2k")
    # tp8 (dp=1) still under-predicts after the fix -> version residual, not dp bug
    assert tp8.err_after > 1.4
    assert tp8.dp1_bitforbit_unchanged
    row = next(r for r in _rows() if r["row_type"] == "residual_out_of_scope")
    assert "version" in row["value_b"].lower()


def test_verdict_names_dp_fix_and_version_residual() -> None:
    verdict = next(r for r in _rows() if r["row_type"] == "verdict")
    assert "FIXED" in verdict["value_a"]
    assert "version" in verdict["value_b"].lower()
    assert "No-Go" in verdict["value_b"]
    assert verdict["next_allowed_phase"] == analyzer.NEXT_PHASE


def test_checked_in_csv_matches_analyzer(tmp_path: Path) -> None:
    if not CHECKED_IN_CSV.exists():
        pytest.skip("phase397j csv not generated yet")
    out = tmp_path / "phase397j.csv"
    analyzer.write_phase397j_csv(out, _rows())
    assert CHECKED_IN_CSV.read_text(encoding="utf-8") == out.read_text(encoding="utf-8")


def test_checked_in_csv_parseable() -> None:
    if not CHECKED_IN_CSV.exists():
        pytest.skip("phase397j csv not generated yet")
    with CHECKED_IN_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == analyzer.FIELDNAMES
        rows = list(reader)
    assert rows[0]["row_type"] == "hypothesis_under_test"


def test_multi_raw_present_and_consistent() -> None:
    if not MULTI_RAW.exists():
        pytest.skip("phase397j multi raw not generated yet")
    with MULTI_RAW.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by = {r["name"]: r for r in rows}
    tp8 = by["K2.5-tp8ep8-8k2k"]
    assert tp8["dp1_bitforbit_unchanged"] == "true"
    assert float(tp8["sim_before"]) == pytest.approx(float(tp8["sim_after"]), abs=1e-6)
    tp4 = by["K2.5-tp4ep8dp2-8k2k"]
    assert float(tp4["sim_after"]) < float(tp4["sim_before"])


def test_tokencount_raw_present_and_matches_real() -> None:
    if not TOKENCOUNT_RAW.exists():
        pytest.skip("phase397j tokencount raw not generated yet")
    with TOKENCOUNT_RAW.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by = {r["name"]: r for r in rows}
    tp4 = by["tp4ep8dp2-8k2k"]
    assert tp4["moe_tokens_before"] == "256"
    assert tp4["moe_tokens_after"] == "128"
    assert tp4["moe_matches_real_after"] == "true"


def test_checked_in_md_has_fix_and_no_go() -> None:
    if not CHECKED_IN_MD.exists():
        pytest.skip("phase397j md not generated yet")
    text = CHECKED_IN_MD.read_text(encoding="utf-8")
    assert "Phase397j" in text
    assert "No-Go" in text
    assert "FIXED" in text
    assert "version" in text.lower()
