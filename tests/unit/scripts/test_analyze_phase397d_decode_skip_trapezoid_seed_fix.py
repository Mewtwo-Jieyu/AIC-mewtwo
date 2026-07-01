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
    / "analyze_phase397d_decode_skip_trapezoid_seed_fix.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase397d_decode_skip_trapezoid_seed_fix", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
# Register before exec so the module's @dataclass can resolve its own __module__.
sys.modules[SPEC.name] = analyzer
SPEC.loader.exec_module(analyzer)

CHECKED_IN_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase397d_decode_skip_trapezoid_seed_fix.csv"
)
CHECKED_IN_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase397d_decode_skip_trapezoid_seed_fix.md"
)
RAW_CSV = (
    REPO_ROOT / "docs/iter_gap_investigation/phase397d_gate_before_after_raw.csv"
)


def _rows() -> list[dict[str, str]]:
    return analyzer.analyze_phase397d()


def test_first_and_last_rows() -> None:
    rows = _rows()
    assert rows[0]["row_type"] == "fix_provenance"
    assert [r["row_type"] for r in rows[-2:]] == ["verdict", "next_phase"]


def test_all_rows_use_fieldnames() -> None:
    for row in _rows():
        assert set(row) == set(analyzer.FIELDNAMES)


def test_required_row_types_present() -> None:
    types = {r["row_type"] for r in _rows()}
    for expected in (
        "fix_provenance",
        "gate_before_after",
        "direction_swing",
        "accuracy_claim",
        "verdict",
        "next_phase",
    ):
        assert expected in types


def test_gate_row_count() -> None:
    rows = _rows()
    assert sum(r["row_type"] == "gate_before_after" for r in rows) == len(
        analyzer.GATES
    )


def test_every_gate_crosses_unity() -> None:
    # before sim/real < 1 (too slow), after sim/real > 1 (too fast).
    for row in _rows():
        if row["row_type"] != "gate_before_after":
            continue
        before = float(row["value_a"].split("=")[1])
        after = float(row["value_b"].split("=")[1])
        assert before < 1.0 < after


def test_gate_ratios_match_raw_dataclass() -> None:
    for row in _rows():
        if row["row_type"] != "gate_before_after":
            continue
        g = analyzer.GATES[row["scenario"]]
        assert float(row["value_a"].split("=")[1]) == pytest.approx(
            g.sim_over_real_before, rel=1e-3
        )
        assert float(row["value_b"].split("=")[1]) == pytest.approx(
            g.sim_over_real_after, rel=1e-3
        )


def test_fix_carries_no_tuning() -> None:
    fix = next(r for r in _rows() if r["row_type"] == "fix_provenance")
    assert fix["fudge_factor_tuning_used"] == "false"
    assert ("pure-decode" in fix["verdict"]) or ("pure_decode" in fix["verdict"])


def test_discipline_flags_gated_runtime_fix() -> None:
    forbidden_false = [
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
        # This phase modifies runtime, and is a real (not diagnostic-only) fix.
        assert row["runtime_modified"] == "true"
        assert row["diagnostic_only"] == "false"
        assert row["default_readiness"] == "No-Go"


def test_verdict_names_trapezoid_seed_and_next_phase() -> None:
    verdict = next(r for r in _rows() if r["row_type"] == "verdict")
    assert "trapezoid seed" in verdict["verdict"]
    assert verdict["next_allowed_phase"] == analyzer.NEXT_PHASE
    nxt = next(r for r in _rows() if r["row_type"] == "next_phase")
    assert "generation_moe" in nxt["verdict"]


def test_checked_in_csv_matches_analyzer(tmp_path: Path) -> None:
    if not CHECKED_IN_CSV.exists():
        pytest.skip("phase397d csv not generated yet")
    out = tmp_path / "phase397d.csv"
    analyzer.write_phase397d_csv(out, _rows())
    assert CHECKED_IN_CSV.read_text(encoding="utf-8") == out.read_text(
        encoding="utf-8"
    )


def test_checked_in_csv_parseable() -> None:
    if not CHECKED_IN_CSV.exists():
        pytest.skip("phase397d csv not generated yet")
    with CHECKED_IN_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == analyzer.FIELDNAMES
        rows = list(reader)
    assert rows[0]["row_type"] == "fix_provenance"


def test_raw_evidence_csv_present_and_crosses_unity() -> None:
    assert RAW_CSV.exists(), "raw before/after evidence must be retained"
    with RAW_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(analyzer.GATES)
    for row in rows:
        assert float(row["sim_over_real_before"]) < 1.0
        assert float(row["sim_over_real_after"]) > 1.0


def test_checked_in_md_has_route_delta_step1() -> None:
    if not CHECKED_IN_MD.exists():
        pytest.skip("phase397d md not generated yet")
    text = CHECKED_IN_MD.read_text(encoding="utf-8")
    assert "Phase397d" in text
    assert "pure-decode" in text
    assert "step 1" in text
