from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT / "scripts" / "analyze_phase397b_tp16_mla_decode_vs_kv_remeasure.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase397b_tp16_mla_decode_vs_kv_remeasure", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)

CHECKED_IN_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase397b_tp16_mla_decode_vs_kv_remeasure.csv"
)
CHECKED_IN_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase397b_tp16_mla_decode_vs_kv_remeasure.md"
)
RAW_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase397b_mla_num_heads4_raw_measurement.csv"
)


def _rows() -> list[dict[str, str]]:
    return analyzer.analyze_phase397b()


def test_first_and_last_rows() -> None:
    rows = _rows()
    assert rows[0]["row_type"] == "measurement_provenance"
    assert [r["row_type"] for r in rows[-2:]] == ["verdict", "next_phase"]


def test_all_rows_use_fieldnames() -> None:
    for row in _rows():
        assert set(row) == set(analyzer.FIELDNAMES)


def test_required_row_types_present() -> None:
    types = {r["row_type"] for r in _rows()}
    for expected in (
        "measurement_provenance",
        "route_beta_expectation",
        "pointwise_ratio",
        "operating_point",
        "ruled_out",
        "verdict",
        "next_phase",
    ):
        assert expected in types


def test_row_counts_for_repeated_types() -> None:
    rows = _rows()
    assert sum(r["row_type"] == "pointwise_ratio" for r in rows) == len(
        analyzer.POINTWISE
    )
    assert sum(r["row_type"] == "operating_point" for r in rows) == len(
        analyzer.OPERATING_POINTS
    )


def test_fresh_and_stored_datasets_are_full_grid() -> None:
    # 4 batches x 8 steps measured and paired.
    assert len(analyzer.FRESH) == 32
    assert len(analyzer.STORED) == 32
    assert set(analyzer.FRESH) == set(analyzer.STORED)


def test_gate_scenario_refutes_route_beta() -> None:
    rows = _rows()
    gate = next(
        r
        for r in rows
        if r["row_type"] == "operating_point"
        and r["scenario"] == analyzer.GATE_SCENARIO
    )
    # Fresh HIGHER than stored at the worst-over-count point -> opposite of beta.
    assert float(gate["ratio"]) > 1.0
    assert "opposite_of_route_beta" in gate["verdict"]


def test_no_operating_point_near_confirm_ratio() -> None:
    # Route beta would require ~0.66; none may be that low.
    for row in _rows():
        if row["row_type"] == "operating_point":
            assert float(row["ratio"]) >= analyzer.ROUTE_BETA_EXPECTED_RATIO + 0.05


def test_pointwise_ratios_match_raw_dicts() -> None:
    for row in _rows():
        if row["row_type"] != "pointwise_ratio":
            continue
        scenario = row["scenario"]  # e.g. b128_kv8191
        batch = int(scenario.split("_")[0][1:])
        step = int(scenario.split("kv")[1])
        expected = analyzer.FRESH[(batch, step)] / analyzer.STORED[(batch, step)]
        assert float(row["ratio"]) == pytest.approx(expected, rel=1e-3)


def test_interp_matches_manual_linear() -> None:
    # b128, KV 11500 between grid steps 8191 and 16383.
    val = analyzer._interp(analyzer.FRESH, 128, 11500)
    lo, hi = analyzer.FRESH[(128, 8191)], analyzer.FRESH[(128, 16383)]
    expected = lo + (hi - lo) * (11500 - 8191) / (16383 - 8191)
    assert val == pytest.approx(expected, rel=1e-9)


def test_ruled_out_records_refuted() -> None:
    row = next(r for r in _rows() if r["row_type"] == "ruled_out")
    assert "REFUTED" in row["verdict"]


def test_verdict_clears_table_and_points_to_composition() -> None:
    verdict = next(r for r in _rows() if r["row_type"] == "verdict")
    assert "decode_mla_vs_kv_table_values_CLEARED" in verdict["verdict"]
    assert "accumulation_composition" in verdict["verdict"]
    assert verdict["next_allowed_phase"] == analyzer.NEXT_PHASE
    nxt = next(r for r in _rows() if r["row_type"] == "next_phase")
    assert "route_delta" in nxt["verdict"]
    assert "NOT_the_table" in nxt["verdict"]
    assert "gamma" in nxt["verdict"]


def test_measurement_provenance_records_caveat() -> None:
    row = _rows()[0]
    assert "forward_mqa" in row["fresh_value"]
    assert f"vllm={analyzer.FRESH_VLLM_VERSION}" in row["fresh_value"]
    assert "LOWER_bound" in row["verdict"]


def test_discipline_flags_measurement_phase() -> None:
    # A measurement phase: gpu/ssh allowed; but no table/runtime mutation.
    forbidden_false = [
        "runtime_modified",
        "nearest_lookup_allowed",
        "extrapolation_allowed",
        "fudge_factor_tuning_used",
        "scope_gating_used",
        "write_real_data_file",
        "default_aic_allowed",
        "valid_for_default",
        "perf_database",
    ]
    for row in _rows():
        for field in forbidden_false:
            assert row[field] == "false", (row["row_type"], field)
        assert row["diagnostic_only"] == "true"
        assert row["gpu_allowed"] == "true"
        assert row["ssh_allowed"] == "true"
        assert row["default_readiness"] == "No-Go"


def test_checked_in_csv_matches_analyzer(tmp_path: Path) -> None:
    if not CHECKED_IN_CSV.exists():
        pytest.skip("phase397b csv not generated yet")
    out = tmp_path / "phase397b.csv"
    analyzer.write_phase397b_csv(out, _rows())
    assert CHECKED_IN_CSV.read_text(encoding="utf-8") == out.read_text(
        encoding="utf-8"
    )


def test_checked_in_csv_parseable() -> None:
    if not CHECKED_IN_CSV.exists():
        pytest.skip("phase397b csv not generated yet")
    with CHECKED_IN_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == analyzer.FIELDNAMES
        rows = list(reader)
    assert rows[0]["row_type"] == "measurement_provenance"


def test_raw_measurement_csv_present_and_schema() -> None:
    assert RAW_CSV.exists(), "raw single-GPU measurement must be retained"
    with RAW_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert len(rows) == 32
    for row in rows:
        assert row["version"] == analyzer.FRESH_VLLM_VERSION
        assert row["num_heads"] == str(analyzer.LOCAL_HEADS_TP16)
        assert row["op_name"] == "generation_mla"
        assert row["kv_cache_dtype"] == "float16"


def test_checked_in_md_has_refute_verdict() -> None:
    if not CHECKED_IN_MD.exists():
        pytest.skip("phase397b md not generated yet")
    text = CHECKED_IN_MD.read_text(encoding="utf-8")
    assert "Phase397b" in text
    assert "REFUTE" in text
    assert "accumulation / composition" in text
