from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase333_topology_holdout_matrix_spec.py"
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase333_topology_holdout_matrix_spec",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


PHASE331 = REPO_ROOT / "docs/iter_gap_investigation/phase331_topology_candidate_validation_gate.csv"


EXPECTED_PAIRS = [
    (
        "tp8_dp1_ep8",
        "isl12000_osl2000_batch128",
        "tp8ep8-12k2k-bt12000",
        "tp8ep8-12k2k-bt65536",
    ),
    (
        "tp4_dp2_ep8",
        "isl12000_osl2000_batch128",
        "tp4dp2ep8-12k2k-bt12000",
        "tp4dp2ep8-12k2k-bt65536",
    ),
    (
        "tp8_dp1_ep8",
        "isl4000_osl2000_batch128",
        "tp8ep8-4k2k-bt4000",
        "tp8ep8-4k2k-bt65536",
    ),
    (
        "tp4_dp2_ep8",
        "isl4000_osl2000_batch128",
        "tp4dp2ep8-4k2k-bt4000",
        "tp4dp2ep8-4k2k-bt65536",
    ),
]


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


def test_topology_holdout_matrix_spec_outputs_four_planned_rows() -> None:
    rows = analyzer.analyze_topology_holdout_matrix_spec(PHASE331)

    assert len(rows) == 4
    observed = [
        (
            row["topology_key"],
            row["shape_key"],
            row["control_scenario"],
            row["holdout_scenario"],
        )
        for row in rows
    ]
    assert observed == EXPECTED_PAIRS

    for row in rows:
        assert row["source"] == "phase333_topology_holdout_matrix_spec"
        assert row["candidate"] == "topology_specific_cadence_boundary_candidate"
        assert row["required_features"] == (
            "trace_integrity,worker_payload_alignment,phase_mix,"
            "boundary_cadence,actual_scheduled_tokens"
        )
        assert row["pass_condition"] == (
            "complete_trace_and_aligned_worker_rows_and_required_phase_cadence_boundary_fields_"
            "and_reproducible_direction_judgment"
        )
        assert row["fail_condition"] == (
            "missing_artifact_or_worker_payload_divergence_or_posthoc_threshold_or_"
            "unknown_key_or_direction_conflict_unresolved"
        )
        assert row["error_threshold"] == "must_be_defined_before_gpu_run"
        assert row["run_status"] == "not_run"
        assert row["default_readiness"] == "No-Go"
        assert row["diagnostic_only"] == "true"
        assert row["valid_for_default"] == "false"
        assert row["perf_database"] == "false"


def test_writers_emit_four_row_csv_and_doc(tmp_path: Path) -> None:
    rows = analyzer.analyze_topology_holdout_matrix_spec(PHASE331)
    csv_path = tmp_path / "out.csv"
    doc_path = tmp_path / "out.md"

    analyzer.write_topology_holdout_matrix_spec_csv(csv_path, rows)
    analyzer.write_topology_holdout_matrix_spec_doc(doc_path, rows)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 4
    assert {row["run_status"] for row in written} == {"not_run"}

    doc = doc_path.read_text(encoding="utf-8")
    assert "Default AIC | No-Go" in doc
    assert "must_be_defined_before_gpu_run" in doc
    assert "not a pass threshold" in doc
    assert "missing_artifact" in doc
    assert "direction_conflict_unresolved" in doc
    assert "default-ready" not in doc


def test_phase331_gate_must_still_block_promotion(tmp_path: Path) -> None:
    def mutate(rows: list[dict[str, str]]) -> None:
        rows[0]["promotion_status"] = "ready"

    phase331 = _copy_csv_with_mutation(PHASE331, tmp_path / "phase331.csv", mutate)

    with pytest.raises(ValueError, match="blocked_pending_validation"):
        analyzer.analyze_topology_holdout_matrix_spec(phase331)


def test_phase331_flags_must_stay_diagnostic_only(tmp_path: Path) -> None:
    def mutate(rows: list[dict[str, str]]) -> None:
        rows[0]["valid_for_default"] = "true"

    phase331 = _copy_csv_with_mutation(PHASE331, tmp_path / "phase331.csv", mutate)

    with pytest.raises(ValueError, match="valid_for_default"):
        analyzer.analyze_topology_holdout_matrix_spec(phase331)


def test_writer_requires_exactly_four_rows(tmp_path: Path) -> None:
    rows = analyzer.analyze_topology_holdout_matrix_spec(PHASE331)

    with pytest.raises(ValueError, match="exactly 4"):
        analyzer.write_topology_holdout_matrix_spec_csv(tmp_path / "bad.csv", rows[:3])
