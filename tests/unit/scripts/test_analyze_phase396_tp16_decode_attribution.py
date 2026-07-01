from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase396_tp16_decode_attribution.py"
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase396_tp16_decode_attribution", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)

CHECKED_IN_CSV = (
    REPO_ROOT / "docs/iter_gap_investigation/phase396_tp16_decode_attribution.csv"
)
CHECKED_IN_MD = (
    REPO_ROOT / "docs/iter_gap_investigation/phase396_tp16_decode_attribution.md"
)


def _rows() -> list[dict[str, str]]:
    return analyzer.analyze_phase396_tp16_decode_attribution()


def test_first_and_last_rows() -> None:
    rows = _rows()
    assert rows[0]["row_type"] == "metric_reconciliation"
    assert [r["row_type"] for r in rows[-2:]] == ["verdict", "next_phase"]


def test_all_rows_use_fieldnames() -> None:
    for row in _rows():
        assert set(row) == set(analyzer.FIELDNAMES)


def test_required_row_types_present() -> None:
    types = {r["row_type"] for r in _rows()}
    for expected in (
        "metric_reconciliation",
        "phase395_correction",
        "steady_is_pure_decode",
        "decode_is_attention_bound",
        "overcount_magnitude",
        "verdict",
        "next_phase",
    ):
        assert expected in types


def test_one_decode_row_per_scenario() -> None:
    rows = _rows()
    n = len(analyzer.DECODE_COMPONENTS)
    assert sum(r["row_type"] == "decode_is_attention_bound" for r in rows) == n


def test_decode_is_attention_bound_all_scenarios() -> None:
    for row in _rows():
        if row["row_type"] == "decode_is_attention_bound":
            assert "attn_share=100%" in row["ratio_or_share"]


def test_overcount_ratio_matches_gate() -> None:
    row = next(r for r in _rows() if r["row_type"] == "overcount_magnitude")
    # 796039.6 / 524810 ~= 1.517, i.e. the ~1.52x throughput gate.
    assert row["ratio_or_share"] == "1.517x"
    assert 1.50 <= analyzer.OVERCOUNT_RATIO <= 1.53


def test_phase395_correction_recorded() -> None:
    row = next(r for r in _rows() if r["row_type"] == "phase395_correction")
    assert "still_hold" in row["verdict"]
    assert "wrong_basis" in row["verdict"]


def test_steady_is_pure_decode() -> None:
    row = next(r for r in _rows() if r["row_type"] == "steady_is_pure_decode")
    assert row["sim_value"] == "0.08"
    assert "decode_path_not_prefill" in row["verdict"]


def test_verdict_points_to_route_beta() -> None:
    verdict = next(r for r in _rows() if r["row_type"] == "verdict")
    assert "attention_bound" in verdict["verdict"]
    assert verdict["next_allowed_phase"] == analyzer.NEXT_PHASE
    nxt = next(r for r in _rows() if r["row_type"] == "next_phase")
    assert "route_beta" in nxt["verdict"]
    assert "gamma" in nxt["verdict"]


def test_verdict_only_discipline_flags() -> None:
    forbidden_false = [
        "runtime_modified",
        "nearest_lookup_allowed",
        "interpolation_allowed",
        "extrapolation_allowed",
        "fudge_factor_tuning_used",
        "scope_gating_used",
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
        assert row["diagnostic_only"] == "true"
        assert row["default_readiness"] == "No-Go"


def test_checked_in_csv_matches_analyzer(tmp_path: Path) -> None:
    if not CHECKED_IN_CSV.exists():
        pytest.skip("phase396 csv not generated yet")
    out = tmp_path / "phase396.csv"
    analyzer.write_phase396_csv(out, _rows())
    assert CHECKED_IN_CSV.read_text(encoding="utf-8") == out.read_text(
        encoding="utf-8"
    )


def test_checked_in_csv_parseable() -> None:
    if not CHECKED_IN_CSV.exists():
        pytest.skip("phase396 csv not generated yet")
    with CHECKED_IN_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == analyzer.FIELDNAMES
        rows = list(reader)
    assert rows[0]["row_type"] == "metric_reconciliation"


def test_checked_in_md_has_verdict() -> None:
    if not CHECKED_IN_MD.exists():
        pytest.skip("phase396 md not generated yet")
    text = CHECKED_IN_MD.read_text(encoding="utf-8")
    assert "Phase396" in text
    assert "attention-bound" in text
    assert "Route beta" in text
