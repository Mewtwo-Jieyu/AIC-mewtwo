from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT / "scripts" / "analyze_phase383_runtime_bucket_semantics_decision_spec.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase383_runtime_bucket_semantics_decision_spec",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


PHASE382_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase382_runtime_binding_probe_sufficiency_gate.csv"
)
CHECKED_IN_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase383_runtime_bucket_semantics_decision_spec.csv"
)

EXPECTED_ROW_TYPES = [
    "phase382_prerequisite",
    "current_runtime_bucket_semantics_confirmed",
    "runtime_bucket_semantics_change_rejected",
    "paired_bucket_data_expansion_selected",
    "existing_module_rows_preserved",
    "fusedmoe_paired_bucket_expansion_required",
    "ep8_bucket_set_preserved",
    "bucket_128_blocked",
    "perfdb_write_blocked",
    "default_aic_blocked",
    "next_phase",
]


def _rows() -> list[dict[str, str]]:
    return analyzer.analyze_phase383_runtime_bucket_semantics_decision_spec(
        phase382_csv=PHASE382_CSV,
    )


def test_phase383_outputs_decision_spec_rows() -> None:
    rows = _rows()

    assert [row["row_type"] for row in rows] == EXPECTED_ROW_TYPES
    assert len(rows) == 11


def test_phase383_accepts_phase382_formula_contract() -> None:
    prereq = next(row for row in _rows() if row["row_type"] == "phase382_prerequisite")
    formula = next(
        row for row in _rows() if row["row_type"] == "current_runtime_bucket_semantics_confirmed"
    )

    assert prereq["phase382_prerequisite"] == "full_model_exact_lookup_blocked"
    assert prereq["paired_lookup_candidate_count"] == "0"
    assert formula["ep8_bucket_formula"] == "max(1, raw_tokens//4)"
    assert formula["fusedmoe_bucket_formula"] == "max(1, raw_tokens//4)*2"
    assert formula["runtime_bucket_semantics_verdict"] == "correct_current_semantics"


def test_phase383_rejects_runtime_bucket_semantics_change() -> None:
    row = next(
        row for row in _rows() if row["row_type"] == "runtime_bucket_semantics_change_rejected"
    )

    assert row["runtime_bucket_semantics_change_allowed"] == "false"
    assert row["runtime_change"] == "false"
    assert row["runtime_change_rejection_reason"] == "would_diverge_from_real_fusedmoe_shape"


def test_phase383_selects_paired_bucket_data_expansion_first() -> None:
    decision = next(
        row for row in _rows() if row["row_type"] == "paired_bucket_data_expansion_selected"
    )
    preserve = next(row for row in _rows() if row["row_type"] == "existing_module_rows_preserved")
    expansion = next(
        row for row in _rows() if row["row_type"] == "fusedmoe_paired_bucket_expansion_required"
    )
    ep8 = next(row for row in _rows() if row["row_type"] == "ep8_bucket_set_preserved")

    assert decision["selected_route"] == "paired_bucket_data_expansion_first"
    assert preserve["existing_rows_preserved"] == "true"
    assert preserve["existing_module_row_count"] == "14"
    assert expansion["fusedmoe_paired_bucket_expansion"] == "2/30/32/482/3616/4096/16384"
    assert ep8["ep8_existing_buckets"] == "1/15/16/241/1808/2048/8192"


def test_phase383_blocks_bucket_128_and_nexts_to_phase384_spec() -> None:
    bucket = next(row for row in _rows() if row["row_type"] == "bucket_128_blocked")
    next_phase = next(row for row in _rows() if row["row_type"] == "next_phase")

    assert bucket["bucket_128_allowed"] == "false"
    assert next_phase["next_allowed_phase"] == "phase384_paired_bucket_expansion_spec"
    assert next_phase["gpu_allowed"] == "false"


def test_phase383_preserves_no_runtime_no_perfdb_no_default_flags() -> None:
    for row in _rows():
        assert row["gpu_allowed"] == "false"
        assert row["ssh_allowed"] == "false"
        assert row["runtime_change"] == "false"
        assert row["new_perfdb_data"] == "false"
        assert row["default_aic_allowed"] == "false"
        assert row["diagnostic_only"] == "true"
        assert row["valid_for_default"] == "false"
        assert row["perf_database"] == "false"
        assert row["default_readiness"] == "No-Go"
        assert row["interpolation_allowed"] == "false"
        assert row["extrapolation_allowed"] == "false"
        assert row["nearest_bucket_allowed"] == "false"


def test_checked_in_csv_matches_analyzer_output() -> None:
    expected = _rows()

    with CHECKED_IN_CSV.open(newline="", encoding="utf-8") as handle:
        checked_in = list(csv.DictReader(handle))

    assert checked_in == expected


def test_writers_reject_runtime_default_or_perfdb_upgrade(tmp_path: Path) -> None:
    rows = _rows()
    rows[0]["runtime_change"] = "true"
    with pytest.raises(ValueError, match="runtime_change"):
        analyzer.write_phase383_csv(tmp_path / "bad.csv", rows)

    rows = _rows()
    rows[0]["valid_for_default"] = "true"
    with pytest.raises(ValueError, match="valid_for_default"):
        analyzer.write_phase383_csv(tmp_path / "bad.csv", rows)

    rows = _rows()
    rows[0]["perf_database"] = "true"
    with pytest.raises(ValueError, match="perf_database"):
        analyzer.write_phase383_csv(tmp_path / "bad.csv", rows)


def test_writers_emit_csv_and_md(tmp_path: Path) -> None:
    rows = _rows()
    csv_path = tmp_path / "phase383.csv"
    md_path = tmp_path / "phase383.md"

    analyzer.write_phase383_csv(csv_path, rows)
    analyzer.write_phase383_md(md_path, rows)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert written == rows
    assert set(written[0]) == set(analyzer.FIELDNAMES)

    doc = md_path.read_text(encoding="utf-8")
    assert "paired_bucket_data_expansion_first" in doc
    assert "Do not change runtime bucket semantics" in doc
    assert "2/30/32/482/3616/4096/16384" in doc
    assert "phase384_paired_bucket_expansion_spec" in doc
