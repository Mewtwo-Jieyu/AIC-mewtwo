from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase329_default_model_form_candidates.py"
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase329_default_model_form_candidates",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


PHASE327 = REPO_ROOT / "docs/iter_gap_investigation/phase327_default_model_candidate_spec_audit.csv"


def _copy_csv_with_mutation(src: Path, dst: Path, mutate) -> Path:
    with src.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    assert fieldnames is not None
    mutate(rows)
    with dst.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return dst


def test_default_model_form_candidates_outputs_five_row_matrix() -> None:
    rows = analyzer.analyze_default_model_form_candidates(PHASE327)

    assert [row["candidate"] for row in rows] == [
        "global_correction",
        "budget_ceiling_model",
        "diagnostic_lookup_as_model",
        "lifecycle_canary_as_evidence",
        "topology_specific_cadence_boundary_candidate",
    ]
    assert {row["default_readiness"] for row in rows} == {"No-Go"}
    assert {row["diagnostic_only"] for row in rows} == {"true"}
    assert {row["valid_for_default"] for row in rows} == {"false"}
    assert {row["perf_database"] for row in rows} == {"false"}
    status_by_candidate = {row["candidate"]: row["candidate_status"] for row in rows}
    assert status_by_candidate == {
        "global_correction": "rejected",
        "budget_ceiling_model": "rejected",
        "diagnostic_lookup_as_model": "rejected",
        "lifecycle_canary_as_evidence": "rejected",
        "topology_specific_cadence_boundary_candidate": "candidate_needs_more_evidence",
    }
    assert "topology directions conflict" in rows[0]["reject_reason"]
    assert "configured budget ceiling" in rows[1]["reject_reason"]
    assert "lookup is not a prediction API" in rows[2]["reject_reason"]
    assert "runner lifecycle only" in rows[3]["reject_reason"]
    assert rows[4]["reject_reason"] == ""
    assert rows[4]["required_next_validation"] == "independent_topology_specific_holdout_with_error_threshold"


def test_writers_emit_csv_and_no_go_doc(tmp_path: Path) -> None:
    rows = analyzer.analyze_default_model_form_candidates(PHASE327)
    csv_path = tmp_path / "out.csv"
    doc_path = tmp_path / "out.md"

    analyzer.write_default_model_form_candidates_csv(csv_path, rows)
    analyzer.write_default_model_form_candidates_doc(doc_path, rows)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 5
    doc = doc_path.read_text(encoding="utf-8")
    assert "Default AIC | No-Go" in doc
    assert "topology_specific_cadence_boundary_candidate" in doc
    assert "candidate_needs_more_evidence" in doc
    assert "global correction is rejected" in doc
    assert "PerfDatabase | No-Go" in doc
    assert "default-ready" not in doc


def test_phase327_must_remain_no_go(tmp_path: Path) -> None:
    phase327 = _copy_csv_with_mutation(
        PHASE327,
        tmp_path / "phase327.csv",
        lambda rows: rows[0].__setitem__("default_readiness", "Go"),
    )

    with pytest.raises(ValueError, match="default_readiness"):
        analyzer.analyze_default_model_form_candidates(phase327)


def test_missing_default_gate_fails_fast(tmp_path: Path) -> None:
    def mutate(rows: list[dict[str, str]]) -> None:
        rows[:] = [row for row in rows if row["module"] != "default_model_candidate_gate"]

    phase327 = _copy_csv_with_mutation(PHASE327, tmp_path / "phase327.csv", mutate)

    with pytest.raises(ValueError, match="exactly 5"):
        analyzer.analyze_default_model_form_candidates(phase327)


def test_candidate_matrix_writer_requires_exact_five_rows(tmp_path: Path) -> None:
    rows = analyzer.analyze_default_model_form_candidates(PHASE327)

    with pytest.raises(ValueError, match="exactly 5"):
        analyzer.write_default_model_form_candidates_csv(tmp_path / "bad.csv", rows[:-1])
