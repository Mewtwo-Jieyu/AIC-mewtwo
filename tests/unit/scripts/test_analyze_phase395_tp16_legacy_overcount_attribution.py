from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "scripts"
    / "analyze_phase395_tp16_legacy_overcount_attribution.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase395_tp16_legacy_overcount_attribution", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)

CHECKED_IN_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase395_tp16_legacy_overcount_attribution.csv"
)
CHECKED_IN_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase395_tp16_legacy_overcount_attribution.md"
)


def _rows() -> list[dict[str, str]]:
    return analyzer.analyze_phase395_tp16_legacy_overcount_attribution()


def test_first_and_last_rows() -> None:
    rows = _rows()
    assert rows[0]["row_type"] == "pure_paths_unchanged"
    assert [r["row_type"] for r in rows[-2:]] == ["verdict", "next_phase"]


def test_all_rows_use_fieldnames() -> None:
    for row in _rows():
        assert set(row) == set(analyzer.FIELDNAMES)


def test_required_row_types_present() -> None:
    types = {r["row_type"] for r in _rows()}
    for expected in (
        "pure_paths_unchanged",
        "steady_throughput_delta",
        "mixed_component_change",
        "standing_error_precedes_merged",
        "decode_dominates_steady",
        "hypothesis_status",
        "verdict",
        "next_phase",
    ):
        assert expected in types


def test_one_row_per_scenario() -> None:
    rows = _rows()
    n = len(analyzer.SCENARIOS)
    assert sum(r["row_type"] == "steady_throughput_delta" for r in rows) == n
    assert sum(r["row_type"] == "mixed_component_change" for r in rows) == n


def test_steady_throughput_delta_is_negligible() -> None:
    for row in _rows():
        if row["row_type"] == "steady_throughput_delta":
            pct = float(row["delta_pct"].rstrip("%"))
            assert abs(pct) <= 1.0


def test_hypothesis_is_refuted() -> None:
    row = next(r for r in _rows() if r["row_type"] == "hypothesis_status")
    assert "refuted" in row["verdict"]


def test_standing_error_precedes_merged() -> None:
    row = next(r for r in _rows() if r["row_type"] == "standing_error_precedes_merged")
    assert row["split_value"] == "1.49x"
    assert row["merged_value"] == "1.52x"
    assert "standing" in row["verdict"]


def test_verdict_rejects_scope_gating() -> None:
    row = next(r for r in _rows() if r["row_type"] == "verdict")
    assert "do_not_scope_gate" in row["verdict"]
    assert row["next_allowed_phase"] == analyzer.NEXT_PHASE
    assert "phase396" in analyzer.NEXT_PHASE


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
        pytest.skip("phase395 csv not generated yet")
    out = tmp_path / "phase395.csv"
    analyzer.write_phase395_csv(out, _rows())
    assert CHECKED_IN_CSV.read_text(encoding="utf-8") == out.read_text(
        encoding="utf-8"
    )


def test_checked_in_csv_parseable() -> None:
    if not CHECKED_IN_CSV.exists():
        pytest.skip("phase395 csv not generated yet")
    with CHECKED_IN_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == analyzer.FIELDNAMES
        rows = list(reader)
    assert rows[0]["row_type"] == "pure_paths_unchanged"


def test_checked_in_md_has_verdict() -> None:
    if not CHECKED_IN_MD.exists():
        pytest.skip("phase395 md not generated yet")
    text = CHECKED_IN_MD.read_text(encoding="utf-8")
    assert "Phase395" in text
    assert "refuted" in text
    assert "standing" in text
