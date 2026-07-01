from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT / "scripts" / "analyze_phase397a_tp16_decode_wall_decomposition.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase397a_tp16_decode_wall_decomposition", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)

CHECKED_IN_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase397a_tp16_decode_wall_decomposition.csv"
)
CHECKED_IN_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase397a_tp16_decode_wall_decomposition.md"
)


def _rows() -> list[dict[str, str]]:
    return analyzer.analyze_phase397a_tp16_decode_wall_decomposition()


def test_first_and_last_rows() -> None:
    rows = _rows()
    assert rows[0]["row_type"] == "model_constants"
    assert [r["row_type"] for r in rows[-2:]] == ["verdict", "next_phase"]


def test_all_rows_use_fieldnames() -> None:
    for row in _rows():
        assert set(row) == set(analyzer.FIELDNAMES)


def test_required_row_types_present() -> None:
    types = {r["row_type"] for r in _rows()}
    for expected in (
        "model_constants",
        "overcount_canonical",
        "wall_decomposition",
        "decode_is_attention_bound",
        "locate_experiment",
        "ruled_out",
        "verdict",
        "next_phase",
    ):
        assert expected in types


def test_one_row_per_scenario_for_repeated_types() -> None:
    rows = _rows()
    n = len(analyzer.SCENARIOS)
    for per_scenario in (
        "overcount_canonical",
        "wall_decomposition",
        "locate_experiment",
    ):
        assert sum(r["row_type"] == per_scenario for r in rows) == n


def test_model_constants_rule_out_layers_and_mtp() -> None:
    row = _rows()[0]
    assert "num_layers=61" in row["sim_value"]
    assert "mtp_scale_factor=1.0" in row["sim_value"]
    assert "local_heads_tp16=4" in row["sim_value"]
    assert "RULED_OUT" in row["verdict"]


def test_gate_scenario_overcount_matches_throughput_gate() -> None:
    rows = _rows()
    gate = next(
        r
        for r in rows
        if r["row_type"] == "overcount_canonical"
        and r["scenario"] == analyzer.GATE_SCENARIO
    )
    assert gate["ratio_or_share"] == "1.517x"


def test_wall_decomposition_decode_dominates() -> None:
    for row in _rows():
        if row["row_type"] == "wall_decomposition":
            assert "prefill=0.0%" in row["sim_value"]
            share = float(row["ratio_or_share"].split("=")[1].rstrip("%"))
            assert share >= 85.0


def test_decode_is_attention_bound() -> None:
    row = next(r for r in _rows() if r["row_type"] == "decode_is_attention_bound")
    assert "gen_attn" in row["sim_value"]
    assert "overlap_factor0_masks_gen_non_attn" in row["verdict"]


def test_locate_experiment_converges_within_6pct() -> None:
    rows = [r for r in _rows() if r["row_type"] == "locate_experiment"]
    assert len(rows) == len(analyzer.SCENARIOS)
    for row in rows:
        ratio = float(row["ratio_or_share"].rstrip("x"))
        assert 0.90 <= ratio <= 1.10


def test_verdict_points_to_mla_table_and_route_beta_single_gpu() -> None:
    verdict = next(r for r in _rows() if r["row_type"] == "verdict")
    assert "decode_mla_latency_vs_kv" in verdict["verdict"]
    assert "NOT_num_layers" in verdict["verdict"]
    assert "NOT_mtp" in verdict["verdict"]
    assert verdict["next_allowed_phase"] == analyzer.NEXT_PHASE
    nxt = next(r for r in _rows() if r["row_type"] == "next_phase")
    assert "single_gpu" in nxt["verdict"]
    assert "cross_node_tp16_NOT_required" in nxt["verdict"]
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
        pytest.skip("phase397a csv not generated yet")
    out = tmp_path / "phase397a.csv"
    analyzer.write_phase397a_csv(out, _rows())
    assert CHECKED_IN_CSV.read_text(encoding="utf-8") == out.read_text(
        encoding="utf-8"
    )


def test_checked_in_csv_parseable() -> None:
    if not CHECKED_IN_CSV.exists():
        pytest.skip("phase397a csv not generated yet")
    with CHECKED_IN_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == analyzer.FIELDNAMES
        rows = list(reader)
    assert rows[0]["row_type"] == "model_constants"


def test_checked_in_md_has_verdict() -> None:
    if not CHECKED_IN_MD.exists():
        pytest.skip("phase397a md not generated yet")
    text = CHECKED_IN_MD.read_text(encoding="utf-8")
    assert "Phase397a" in text
    assert "decode MLA-latency-vs-KV" in text
    assert "Cross-node tp16 is NOT required" in text
