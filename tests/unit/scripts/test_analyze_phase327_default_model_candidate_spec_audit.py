from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase327_default_model_candidate_spec_audit.py"
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase327_default_model_candidate_spec_audit",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


PHASE258 = REPO_ROOT / "docs/iter_gap_investigation/phase258_actual_scheduled_token_full_family.csv"
PHASE303 = REPO_ROOT / "docs/iter_gap_investigation/phase303_deeper_trace_topology_family.csv"
PHASE309 = REPO_ROOT / "docs/iter_gap_investigation/phase309_diagnostic_exact_key_api_consistency.csv"
PHASE323 = REPO_ROOT / "docs/iter_gap_investigation/phase323_runner_lifecycle_pair_canary.csv"


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


def test_default_model_candidate_spec_outputs_readiness_gap_rows() -> None:
    rows = analyzer.analyze_default_model_candidate_spec_audit(PHASE258, PHASE303, PHASE309, PHASE323)

    assert [row["module"] for row in rows] == [
        "phase258_actual_scheduled_token_evidence",
        "phase303_deeper_trace_topology_evidence",
        "phase305_phase309_diagnostic_exact_key_api",
        "phase323_runner_lifecycle_canary",
        "default_model_candidate_gate",
    ]
    assert {row["default_readiness"] for row in rows} == {"No-Go"}
    assert {row["diagnostic_only"] for row in rows} == {"true"}
    assert {row["valid_for_default"] for row in rows} == {"false"}
    assert {row["perf_database"] for row in rows} == {"false"}

    gate = rows[-1]
    assert gate["candidate_input"] == "explicit_model_form_plus_topology_shape_budget_keyspace"
    assert gate["candidate_output"] == "bounded_default_aic_prediction_or_fail_fast"
    assert gate["eligible_keys"] == "only_keys_with_model_spec_and_independent_holdout"
    assert gate["reject_keys"] == "unknown_or_missing_holdout_or_conflicting_topology_direction"
    assert gate["required_holdout"] == "independent_holdout_per_topology_shape_and_budget_family"
    assert gate["error_threshold"] == "must_be_defined_before_default_integration"
    assert gate["failure_policy"] == "fail_fast_no_prediction_no_perf_database_write"
    assert gate["verdict"] == "default_model_candidate_spec_gap"


def test_writers_emit_csv_and_no_go_doc(tmp_path: Path) -> None:
    rows = analyzer.analyze_default_model_candidate_spec_audit(PHASE258, PHASE303, PHASE309, PHASE323)
    csv_path = tmp_path / "out.csv"
    doc_path = tmp_path / "out.md"

    analyzer.write_default_model_candidate_spec_audit_csv(csv_path, rows)
    analyzer.write_default_model_candidate_spec_audit_doc(doc_path, rows)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 5
    assert written[-1]["default_readiness"] == "No-Go"
    doc = doc_path.read_text(encoding="utf-8")
    assert "Default AIC | No-Go" in doc
    assert "diagnostic lookup cannot be reused as a prediction API" in doc
    assert "Phase323 is lifecycle canary evidence, not model evidence" in doc
    assert "default-ready" not in doc


def test_phase309_unknown_key_guard_fails_fast(tmp_path: Path) -> None:
    phase309 = _copy_csv_with_mutation(
        PHASE309,
        tmp_path / "phase309.csv",
        lambda rows: rows[0].__setitem__("unknown_key_rejects", "false"),
    )

    with pytest.raises(ValueError, match="unknown_key"):
        analyzer.analyze_default_model_candidate_spec_audit(PHASE258, PHASE303, phase309, PHASE323)


def test_phase323_replacement_evidence_fails_fast(tmp_path: Path) -> None:
    phase323 = _copy_csv_with_mutation(
        PHASE323,
        tmp_path / "phase323.csv",
        lambda rows: rows[0].__setitem__("evidence_status", "replacement_evidence"),
    )

    with pytest.raises(ValueError, match="not_replacement_evidence"):
        analyzer.analyze_default_model_candidate_spec_audit(PHASE258, PHASE303, PHASE309, phase323)


def test_phase303_must_remain_topology_dependent(tmp_path: Path) -> None:
    def mutate(rows: list[dict[str, str]]) -> None:
        for row in rows:
            row["throughput_direction"] = "holdout_faster"

    phase303 = _copy_csv_with_mutation(PHASE303, tmp_path / "phase303.csv", mutate)

    with pytest.raises(ValueError, match="topology-dependent"):
        analyzer.analyze_default_model_candidate_spec_audit(PHASE258, phase303, PHASE309, PHASE323)
