from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase335_model_output_threshold_contract.py"
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase335_model_output_threshold_contract",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


PHASE333 = REPO_ROOT / "docs/iter_gap_investigation/phase333_topology_holdout_matrix_spec.csv"


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


def test_model_output_threshold_contract_outputs_three_rows() -> None:
    rows = analyzer.analyze_model_output_threshold_contract(PHASE333)

    assert [row["contract"] for row in rows] == [
        "direction_prediction_contract",
        "cadence_boundary_feature_contract",
        "numeric_error_threshold_contract",
    ]
    assert {row["source"] for row in rows} == {"phase335_model_output_threshold_contract"}
    assert {row["default_readiness"] for row in rows} == {"No-Go"}
    assert {row["diagnostic_only"] for row in rows} == {"true"}
    assert {row["valid_for_default"] for row in rows} == {"false"}
    assert {row["perf_database"] for row in rows} == {"false"}
    assert {row["gpu_run_allowed"] for row in rows} == {"false"}

    direction = rows[0]
    assert direction["model_output"] == "holdout_faster_or_holdout_slower_or_inconclusive"
    assert direction["pass_rule"] == "all_4_matrix_pairs_have_reproducible_direction_explanation"
    assert direction["fail_rule"] == "any_pair_direction_conflict_unresolved"
    assert direction["threshold_status"] == "direction_only_contract_registered"

    feature = rows[1]
    assert feature["required_inputs"] == (
        "actual_scheduled_tokens,phase_mix,boundary_cadence,"
        "trace_integrity,worker_payload_alignment"
    )
    assert feature["pass_rule"] == "uses_only_phase333_registered_features"
    assert feature["fail_rule"] == "posthoc_feature_introduced"
    assert feature["threshold_status"] == "feature_set_registered_no_posthoc_features"

    numeric = rows[2]
    assert numeric["model_output"] == "numeric_error_threshold"
    assert numeric["pass_rule"] == "numeric_formula_defined_before_gpu_run"
    assert numeric["fail_rule"] == "no_numeric_formula_or_threshold_changed_after_gpu_run"
    assert numeric["threshold_status"] == "blocked_until_numeric_model_form_exists"


def test_writers_emit_csv_and_doc(tmp_path: Path) -> None:
    rows = analyzer.analyze_model_output_threshold_contract(PHASE333)
    csv_path = tmp_path / "out.csv"
    doc_path = tmp_path / "out.md"

    analyzer.write_model_output_threshold_contract_csv(csv_path, rows)
    analyzer.write_model_output_threshold_contract_doc(doc_path, rows)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 3
    assert {row["gpu_run_allowed"] for row in written} == {"false"}
    assert written[2]["threshold_status"] == "blocked_until_numeric_model_form_exists"

    doc = doc_path.read_text(encoding="utf-8")
    assert "Default AIC | No-Go" in doc
    assert "GPU run allowed | false" in doc
    assert "holdout_faster / holdout_slower / inconclusive" in doc
    assert "No posthoc feature is allowed" in doc
    assert "blocked_until_numeric_model_form_exists" in doc
    assert "default-ready" not in doc


def test_phase333_matrix_must_remain_not_run(tmp_path: Path) -> None:
    def mutate(rows: list[dict[str, str]]) -> None:
        rows[0]["run_status"] = "run"

    phase333 = _copy_csv_with_mutation(PHASE333, tmp_path / "phase333.csv", mutate)

    with pytest.raises(ValueError, match="run_status"):
        analyzer.analyze_model_output_threshold_contract(phase333)


def test_phase333_threshold_placeholder_must_remain_pre_gpu(tmp_path: Path) -> None:
    def mutate(rows: list[dict[str, str]]) -> None:
        rows[0]["error_threshold"] = "0.05"

    phase333 = _copy_csv_with_mutation(PHASE333, tmp_path / "phase333.csv", mutate)

    with pytest.raises(ValueError, match="must_be_defined_before_gpu_run"):
        analyzer.analyze_model_output_threshold_contract(phase333)


def test_writer_requires_exactly_three_rows(tmp_path: Path) -> None:
    rows = analyzer.analyze_model_output_threshold_contract(PHASE333)

    with pytest.raises(ValueError, match="exactly 3"):
        analyzer.write_model_output_threshold_contract_csv(tmp_path / "bad.csv", rows[:2])
