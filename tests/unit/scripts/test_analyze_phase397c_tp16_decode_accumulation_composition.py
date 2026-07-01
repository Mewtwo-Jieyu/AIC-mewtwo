from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "scripts"
    / "analyze_phase397c_tp16_decode_accumulation_composition.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase397c_tp16_decode_accumulation_composition", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
# Register before exec so the module's @dataclass can resolve its own __module__.
sys.modules[SPEC.name] = analyzer
SPEC.loader.exec_module(analyzer)

CHECKED_IN_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase397c_tp16_decode_accumulation_composition.csv"
)
CHECKED_IN_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase397c_tp16_decode_accumulation_composition.md"
)
RAW_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase397c_canonical_wall_split_raw.csv"
)


def _rows() -> list[dict[str, str]]:
    return analyzer.analyze_phase397c()


def test_first_and_last_rows() -> None:
    rows = _rows()
    assert rows[0]["row_type"] == "correction_retraction"
    assert [r["row_type"] for r in rows[-2:]] == ["verdict", "next_phase"]


def test_all_rows_use_fieldnames() -> None:
    for row in _rows():
        assert set(row) == set(analyzer.FIELDNAMES)


def test_required_row_types_present() -> None:
    types = {r["row_type"] for r in _rows()}
    for expected in (
        "correction_retraction",
        "mechanism_ruled_out",
        "canonical_wall_split",
        "decode_iter_composition",
        "trapezoid_vs_periter_reconciliation",
        "root_cause_skip_trapezoid_seed",
        "compensating_error",
        "attribution",
        "ruled_out",
        "verdict",
        "next_phase",
    ):
        assert expected in types


def test_per_scenario_row_counts() -> None:
    rows = _rows()
    n = len(analyzer.SCENARIOS)
    for per_scenario_type in (
        "canonical_wall_split",
        "decode_iter_composition",
        "trapezoid_vs_periter_reconciliation",
        "root_cause_skip_trapezoid_seed",
        "attribution",
    ):
        assert sum(r["row_type"] == per_scenario_type for r in rows) == n


def test_retraction_records_mtp_retracted() -> None:
    row = next(r for r in _rows() if r["row_type"] == "correction_retraction")
    assert "RETRACTED" in row["verdict"]
    assert "num_nextn_predict_layers" in row["value_a"]


def test_canonical_wall_is_decode_trapezoid_dominated() -> None:
    for row in _rows():
        if row["row_type"] == "canonical_wall_split":
            assert float(row["ratio"]) >= 0.90


def test_trapezoid_over_counts_vs_periter_sum() -> None:
    for row in _rows():
        if row["row_type"] == "trapezoid_vs_periter_reconciliation":
            assert float(row["ratio"]) > 2.0


def test_root_cause_seed_is_mixed_not_puredecode() -> None:
    rows = [r for r in _rows() if r["row_type"] == "root_cause_skip_trapezoid_seed"]
    assert rows
    for row in rows:
        # first_lat (mixed) must be far larger than the true pure-decode iter.
        assert float(row["ratio"]) > 3.0


def test_decode_wall_exceeds_real_implied_total_wall() -> None:
    # The decode-skip-trapezoid wall alone exceeds the real total wall at every
    # gate scenario -> the over-count lives in decode accumulation, not prefill.
    for row in _rows():
        if row["row_type"] == "attribution":
            dec_over_real = float(row["value_b"].split("=")[1])
            assert dec_over_real > 1.0


def test_attribution_matches_scenarios() -> None:
    for row in _rows():
        if row["row_type"] != "attribution":
            continue
        sc = analyzer.SCENARIOS[row["scenario"]]
        expected_over = sc.real_tok_s_gpu / sc.sim_tok_s_gpu
        assert float(row["value_a"].split("=")[1]) == pytest.approx(
            expected_over, rel=1e-3
        )


def test_verdict_names_skip_trapezoid_and_compensation() -> None:
    verdict = next(r for r in _rows() if r["row_type"] == "verdict")
    assert "skip_trapezoid" in verdict["verdict"]
    assert "compensating" in verdict["verdict"]
    assert verdict["next_allowed_phase"] == analyzer.NEXT_PHASE
    nxt = next(r for r in _rows() if r["row_type"] == "next_phase")
    assert "GATED" in nxt["verdict"]
    assert "BOTH" in nxt["verdict"]


def test_discipline_flags_offline_verdict_only() -> None:
    forbidden_false = [
        "runtime_modified",
        "nearest_lookup_allowed",
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
        pytest.skip("phase397c csv not generated yet")
    out = tmp_path / "phase397c.csv"
    analyzer.write_phase397c_csv(out, _rows())
    assert CHECKED_IN_CSV.read_text(encoding="utf-8") == out.read_text(
        encoding="utf-8"
    )


def test_checked_in_csv_parseable() -> None:
    if not CHECKED_IN_CSV.exists():
        pytest.skip("phase397c csv not generated yet")
    with CHECKED_IN_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == analyzer.FIELDNAMES
        rows = list(reader)
    assert rows[0]["row_type"] == "correction_retraction"


def test_raw_evidence_csv_present_and_schema() -> None:
    assert RAW_CSV.exists(), "raw canonical wall-split evidence must be retained"
    with RAW_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert len(rows) == len(analyzer.SCENARIOS)
    for row in rows:
        # Raw evidence must corroborate the >90% decode-wall claim.
        assert float(row["decode_wall_pct"]) >= 90.0
        assert float(row["trapezoid_over_expand_ratio"]) > 2.0
        assert float(row["skip_first_lat_mixed_ms"]) > float(
            row["puredecode_single_iter_ms"]
        )


def test_checked_in_md_has_route_delta_verdict() -> None:
    if not CHECKED_IN_MD.exists():
        pytest.skip("phase397c md not generated yet")
    text = CHECKED_IN_MD.read_text(encoding="utf-8")
    assert "Phase397c" in text
    assert "skip-trapezoid" in text
    assert "Route delta" in text
