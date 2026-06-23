from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase337_prerun_feature_eligibility.py"
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase337_prerun_feature_eligibility",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


PHASE335 = REPO_ROOT / "docs/iter_gap_investigation/phase335_model_output_threshold_contract.csv"


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


def test_prerun_feature_eligibility_outputs_five_rows() -> None:
    rows = analyzer.analyze_prerun_feature_eligibility(PHASE335)

    assert [row["feature"] for row in rows] == [
        "topology_key",
        "shape_key",
        "control_bt_holdout_bt",
        "actual_scheduled_tokens",
        "phase_mix_boundary_cadence",
    ]
    assert {row["source"] for row in rows} == {"phase337_prerun_feature_eligibility"}
    assert {row["default_readiness"] for row in rows} == {"No-Go"}
    assert {row["diagnostic_only"] for row in rows} == {"true"}
    assert {row["valid_for_default"] for row in rows} == {"false"}
    assert {row["perf_database"] for row in rows} == {"false"}

    pre_run = rows[:3]
    for row in pre_run:
        assert row["availability"] == "pre_run_available"
        assert row["eligible_for_default_prediction"] == "false"
        assert row["eligible_for_diagnostic_audit"] == "true"
        assert row["leakage_risk"] == "false"
        assert row["decision"] == "scope_key_only_not_standalone_throughput_predictor"

    post_run = rows[3:]
    for row in post_run:
        assert row["availability"] == "post_run_trace_derived"
        assert row["eligible_for_default_prediction"] == "false"
        assert row["eligible_for_diagnostic_audit"] == "true"
        assert row["leakage_risk"] == "true"
        assert row["decision"] == "diagnostic_audit_only_default_prediction_leakage"


def test_writers_emit_csv_and_doc(tmp_path: Path) -> None:
    rows = analyzer.analyze_prerun_feature_eligibility(PHASE335)
    csv_path = tmp_path / "out.csv"
    doc_path = tmp_path / "out.md"

    analyzer.write_prerun_feature_eligibility_csv(csv_path, rows)
    analyzer.write_prerun_feature_eligibility_doc(doc_path, rows)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 5
    assert [row["leakage_risk"] for row in written] == ["false", "false", "false", "true", "true"]

    doc = doc_path.read_text(encoding="utf-8")
    assert "Default AIC | No-Go" in doc
    assert "post-run trace features cannot be default prediction inputs" in doc
    assert "actual_scheduled_tokens" in doc
    assert "phase_mix_boundary_cadence" in doc
    assert "scope keys only" in doc
    assert "default-ready" not in doc


def test_phase335_must_stay_gpu_blocked(tmp_path: Path) -> None:
    def mutate(rows: list[dict[str, str]]) -> None:
        rows[0]["gpu_run_allowed"] = "true"

    phase335 = _copy_csv_with_mutation(PHASE335, tmp_path / "phase335.csv", mutate)

    with pytest.raises(ValueError, match="gpu_run_allowed"):
        analyzer.analyze_prerun_feature_eligibility(phase335)


def test_phase335_required_inputs_must_include_registered_trace_features(tmp_path: Path) -> None:
    def mutate(rows: list[dict[str, str]]) -> None:
        rows[1]["required_inputs"] = "actual_scheduled_tokens,phase_mix"

    phase335 = _copy_csv_with_mutation(PHASE335, tmp_path / "phase335.csv", mutate)

    with pytest.raises(ValueError, match="boundary_cadence"):
        analyzer.analyze_prerun_feature_eligibility(phase335)


def test_writer_requires_exactly_five_rows(tmp_path: Path) -> None:
    rows = analyzer.analyze_prerun_feature_eligibility(PHASE335)

    with pytest.raises(ValueError, match="exactly 5"):
        analyzer.write_prerun_feature_eligibility_csv(tmp_path / "bad.csv", rows[:4])
