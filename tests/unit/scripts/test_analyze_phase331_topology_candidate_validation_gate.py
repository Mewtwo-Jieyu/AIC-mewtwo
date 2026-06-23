from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase331_topology_candidate_validation_gate.py"
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase331_topology_candidate_validation_gate",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


PHASE329 = REPO_ROOT / "docs/iter_gap_investigation/phase329_default_model_form_candidates.csv"


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


def test_topology_candidate_validation_gate_outputs_one_row() -> None:
    rows = analyzer.analyze_topology_candidate_validation_gate(PHASE329)

    assert len(rows) == 1
    row = rows[0]
    assert row["source"] == "phase331_topology_candidate_validation_gate"
    assert row["candidate"] == "topology_specific_cadence_boundary_candidate"
    assert row["required_input_features"] == (
        "topology_key,shape_key,control_bt,holdout_bt,"
        "actual_scheduled_tokens,phase_mix,boundary_cadence"
    )
    assert row["eligible_scope"] == "topology_shape_budget_exact_key_only"
    assert row["required_holdout_matrix"] == (
        "independent_holdouts_covering_tp8_dp1_ep8_and_tp4_dp2_ep8_opposite_directions"
    )
    assert row["error_threshold_policy"] == "define_before_running_holdout_no_posthoc_threshold"
    assert row["failure_policy"] == "fail_fast_no_interpolation_no_extrapolation"
    assert row["promotion_target"] == "model_experiment_candidate_only"
    assert row["promotion_status"] == "blocked_pending_validation"
    assert row["default_readiness"] == "No-Go"
    assert row["diagnostic_only"] == "true"
    assert row["valid_for_default"] == "false"
    assert row["perf_database"] == "false"


def test_writers_emit_csv_and_doc(tmp_path: Path) -> None:
    rows = analyzer.analyze_topology_candidate_validation_gate(PHASE329)
    csv_path = tmp_path / "out.csv"
    doc_path = tmp_path / "out.md"

    analyzer.write_topology_candidate_validation_gate_csv(csv_path, rows)
    analyzer.write_topology_candidate_validation_gate_doc(doc_path, rows)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 1
    assert written[0]["promotion_status"] == "blocked_pending_validation"
    doc = doc_path.read_text(encoding="utf-8")
    assert "Default AIC | No-Go" in doc
    assert "blocked_pending_validation" in doc
    assert "must be defined before running holdout" in doc
    assert "No interpolation or extrapolation is allowed" in doc
    assert "global_correction" not in doc


def test_rejected_route_revival_fails_fast(tmp_path: Path) -> None:
    def mutate(rows: list[dict[str, str]]) -> None:
        for row in rows:
            if row["candidate"] == "global_correction":
                row["candidate_status"] = "candidate_needs_more_evidence"

    phase329 = _copy_csv_with_mutation(PHASE329, tmp_path / "phase329.csv", mutate)

    with pytest.raises(ValueError, match="rejected routes"):
        analyzer.analyze_topology_candidate_validation_gate(phase329)


def test_missing_retained_candidate_fails_fast(tmp_path: Path) -> None:
    def mutate(rows: list[dict[str, str]]) -> None:
        rows[:] = [row for row in rows if row["candidate"] != "topology_specific_cadence_boundary_candidate"]

    phase329 = _copy_csv_with_mutation(PHASE329, tmp_path / "phase329.csv", mutate)

    with pytest.raises(ValueError, match="topology_specific_cadence_boundary_candidate"):
        analyzer.analyze_topology_candidate_validation_gate(phase329)


def test_writer_requires_single_gate_row(tmp_path: Path) -> None:
    rows = analyzer.analyze_topology_candidate_validation_gate(PHASE329)

    with pytest.raises(ValueError, match="exactly 1"):
        analyzer.write_topology_candidate_validation_gate_csv(tmp_path / "bad.csv", rows + rows)
