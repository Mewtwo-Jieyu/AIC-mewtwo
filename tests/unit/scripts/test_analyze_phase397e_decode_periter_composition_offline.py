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
    / "analyze_phase397e_decode_periter_composition_offline.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase397e_decode_periter_composition_offline", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
# Register before exec so the module's @dataclass can resolve its own __module__.
sys.modules[SPEC.name] = analyzer
SPEC.loader.exec_module(analyzer)

CHECKED_IN_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase397e_decode_periter_composition_offline.csv"
)
CHECKED_IN_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase397e_decode_periter_composition_offline.md"
)
RAW_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase397e_periter_reconciliation_raw.csv"
)


def _rows() -> list[dict[str, str]]:
    return analyzer.analyze_phase397e()


def test_first_and_last_rows() -> None:
    rows = _rows()
    assert rows[0]["row_type"] == "table_semantics_verdict"
    assert [r["row_type"] for r in rows[-2:]] == ["verdict", "next_phase"]


def test_all_rows_use_fieldnames() -> None:
    for row in _rows():
        assert set(row) == set(analyzer.FIELDNAMES)


def test_required_row_types_present() -> None:
    types = {r["row_type"] for r in _rows()}
    for expected in (
        "table_semantics_verdict",
        "double_count_proof",
        "dead_code",
        "periter_reconciliation",
        "coupled_errors",
        "target_397f",
        "candidate_fixes",
        "verdict",
        "next_phase",
    ):
        assert expected in types


def test_per_scenario_row_counts() -> None:
    rows = _rows()
    n = len(analyzer.RECON)
    for per_scenario_type in ("double_count_proof", "periter_reconciliation", "target_397f"):
        assert sum(r["row_type"] == per_scenario_type for r in rows) == n


def test_double_count_is_heavy() -> None:
    for row in _rows():
        if row["row_type"] == "double_count_proof":
            assert float(row["ratio"]) > 4.0


def test_raw_sum_reconciles_within_15pct() -> None:
    for row in _rows():
        if row["row_type"] == "periter_reconciliation":
            assert 0.85 <= float(row["ratio"]) <= 1.15


def test_current_under_counts() -> None:
    for name, r in analyzer.RECON.items():
        assert r.per_iter_current_ms / r.real_decode_iter_ms < 0.5


def test_recon_derivations_match_dataclass() -> None:
    for name, r in analyzer.RECON.items():
        assert r.real_decode_iter_ms == pytest.approx(
            r.batch / (r.real_out_tok_s_gpu * r.num_gpus) * 1000.0
        )
        assert r.per_iter_raw_sum_ms == pytest.approx(
            r.gen_non_attn_raw_ms + r.gen_attn_ms
        )
        # overlap_factor=0 -> current is a max, not a sum.
        assert r.per_iter_current_ms == pytest.approx(
            max(r.gen_non_attn_div_tp_ms, r.gen_attn_ms)
        )


def test_table_semantics_is_per_rank() -> None:
    row = next(r for r in _rows() if r["row_type"] == "table_semantics_verdict")
    assert "per_rank" in row["value_a"]


def test_dead_code_flagged() -> None:
    row = next(r for r in _rows() if r["row_type"] == "dead_code")
    assert "_scale_generation_non_attention" in row["metric"]


def test_verdict_names_double_count_and_serial() -> None:
    verdict = next(r for r in _rows() if r["row_type"] == "verdict")
    assert "double-count" in verdict["verdict"]
    assert "serial" in verdict["verdict"]
    assert verdict["next_allowed_phase"] == analyzer.NEXT_PHASE
    nxt = next(r for r in _rows() if r["row_type"] == "next_phase")
    assert "serial sum" in nxt["verdict"]


def test_discipline_flags_verdict_only() -> None:
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
        pytest.skip("phase397e csv not generated yet")
    out = tmp_path / "phase397e.csv"
    analyzer.write_phase397e_csv(out, _rows())
    assert CHECKED_IN_CSV.read_text(encoding="utf-8") == out.read_text(
        encoding="utf-8"
    )


def test_checked_in_csv_parseable() -> None:
    if not CHECKED_IN_CSV.exists():
        pytest.skip("phase397e csv not generated yet")
    with CHECKED_IN_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == analyzer.FIELDNAMES
        rows = list(reader)
    assert rows[0]["row_type"] == "table_semantics_verdict"


def test_raw_evidence_csv_present_and_consistent() -> None:
    assert RAW_CSV.exists(), "raw probe evidence must be retained"
    with RAW_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(analyzer.RECON)
    for row in rows:
        raw = float(row["gen_non_attn_raw_ms"])
        div = float(row["gen_non_attn_div_tp_ms"])
        assert raw / div > 4.0
        raw_sum = float(row["per_iter_raw_sum_ms"])
        real = float(row["real_decode_iter_ms"])
        assert 0.85 <= raw_sum / real <= 1.15


def test_checked_in_md_has_route_delta_step2() -> None:
    if not CHECKED_IN_MD.exists():
        pytest.skip("phase397e md not generated yet")
    text = CHECKED_IN_MD.read_text(encoding="utf-8")
    assert "Phase397e" in text
    assert "per-rank" in text
    assert "step 2" in text
