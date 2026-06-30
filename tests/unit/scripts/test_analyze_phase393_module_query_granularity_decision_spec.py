from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "scripts"
    / "analyze_phase393_module_query_granularity_decision_spec.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase393_module_query_granularity_decision_spec", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)

CHECKED_IN_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase393_module_query_granularity_decision_spec.csv"
)
CHECKED_IN_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase393_module_query_granularity_decision_spec.md"
)

EXPECTED_ROW_TYPES = [
    "root_cause",
    "evidence_phase124",
    "probe_split_granularity",
    "probe_merged_granularity",
    "candidate_merged_granularity",
    "candidate_bucketization_contract",
    "candidate_scheduler_alignment",
    "residual_non_saturated",
    "decode_residual_deferred",
    "verdict",
    "next_phase",
]


def _rows() -> list[dict[str, str]]:
    return analyzer.analyze_phase393_module_query_granularity_decision_spec()


def test_row_order() -> None:
    assert [r["row_type"] for r in _rows()] == EXPECTED_ROW_TYPES


def test_all_rows_use_fieldnames() -> None:
    for row in _rows():
        assert set(row) == set(analyzer.FIELDNAMES)


def test_split_misses_all_mixed() -> None:
    row = next(r for r in _rows() if r["row_type"] == "probe_split_granularity")
    assert row["split_mixed_exact_hit"] == f"0/{analyzer.MIXED_N}"


def test_merged_reaches_most_mixed() -> None:
    row = next(r for r in _rows() if r["row_type"] == "probe_merged_granularity")
    hit, total = row["merged_mixed_exact_hit"].split("/")
    assert int(total) == analyzer.MIXED_N
    assert int(hit) >= int(0.9 * analyzer.MIXED_N)
    assert int(hit) > analyzer.SPLIT_MIXED_HIT


def test_merged_is_recommended_primary() -> None:
    row = next(
        r for r in _rows() if r["row_type"] == "candidate_merged_granularity"
    )
    assert row["recommendation"] == "recommended_primary"
    assert row["candidate"] == "merged_batch_tokens"


def test_decode_residual_is_deferred_to_gpu() -> None:
    row = next(r for r in _rows() if r["row_type"] == "decode_residual_deferred")
    assert "deferred" in row["verdict"]
    assert "comm" in row["verdict"]


def test_no_go_discipline_flags() -> None:
    forbidden_false = [
        "nearest_lookup_allowed",
        "interpolation_allowed",
        "extrapolation_allowed",
        "fudge_factor_tuning_used",
        "runtime_modified",
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
        assert row["exact_lookup_only"] == "true"
        assert row["diagnostic_only"] == "true"
        assert row["default_readiness"] == "No-Go"


def test_verdict_next_phase() -> None:
    verdict = next(r for r in _rows() if r["row_type"] == "verdict")
    assert verdict["next_allowed_phase"] == analyzer.NEXT_PHASE
    assert "phase394" in analyzer.NEXT_PHASE


def test_checked_in_csv_matches_analyzer(tmp_path: Path) -> None:
    if not CHECKED_IN_CSV.exists():
        pytest.skip("phase393 csv not generated yet")
    out = tmp_path / "phase393.csv"
    analyzer.write_phase393_csv(out, _rows())
    assert CHECKED_IN_CSV.read_text(encoding="utf-8") == out.read_text(
        encoding="utf-8"
    )


def test_checked_in_csv_parseable() -> None:
    if not CHECKED_IN_CSV.exists():
        pytest.skip("phase393 csv not generated yet")
    with CHECKED_IN_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == analyzer.FIELDNAMES
        rows = list(reader)
    assert [r["row_type"] for r in rows] == EXPECTED_ROW_TYPES


def test_checked_in_md_has_verdict() -> None:
    if not CHECKED_IN_MD.exists():
        pytest.skip("phase393 md not generated yet")
    text = CHECKED_IN_MD.read_text(encoding="utf-8")
    assert "Phase393" in text
    assert "merged" in text
