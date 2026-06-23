from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase339_prerun_model_feasibility.py"
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase339_prerun_model_feasibility",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


PHASE337 = REPO_ROOT / "docs/iter_gap_investigation/phase337_prerun_feature_eligibility.csv"


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


def test_prerun_model_feasibility_outputs_three_rows() -> None:
    rows = analyzer.analyze_prerun_model_feasibility(PHASE337)

    assert [row["candidate"] for row in rows] == [
        "scope_key_only_candidate",
        "postrun_trace_feature_candidate",
        "topology_specific_cadence_boundary_candidate",
    ]
    assert [row["verdict"] for row in rows] == [
        "rejected_scope_only_no_throughput_prediction",
        "rejected_postrun_trace_leakage",
        "diagnostic_only_postrun_explanation",
    ]
    assert {row["source"] for row in rows} == {"phase339_prerun_model_feasibility"}
    assert {row["default_readiness"] for row in rows} == {"No-Go"}
    assert {row["gpu_run_allowed"] for row in rows} == {"false"}
    assert {row["diagnostic_only"] for row in rows} == {"true"}
    assert {row["valid_for_default"] for row in rows} == {"false"}
    assert {row["perf_database"] for row in rows} == {"false"}
    assert "cannot_predict_throughput" in rows[0]["reason"]
    assert "leakage_risk" in rows[1]["reason"]
    assert "post_run_only" in rows[2]["reason"]


def test_writers_emit_csv_and_doc(tmp_path: Path) -> None:
    rows = analyzer.analyze_prerun_model_feasibility(PHASE337)
    csv_path = tmp_path / "out.csv"
    doc_path = tmp_path / "out.md"

    analyzer.write_prerun_model_feasibility_csv(csv_path, rows)
    analyzer.write_prerun_model_feasibility_doc(doc_path, rows)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 3
    assert {row["gpu_run_allowed"] for row in written} == {"false"}

    doc = doc_path.read_text(encoding="utf-8")
    assert "Default AIC | No-Go" in doc
    assert "GPU run allowed | false" in doc
    assert "pre-run inputs only define exact-key scope" in doc
    assert "post-run trace features would leak the answer" in doc
    assert "not a default model" in doc
    assert "default-ready" not in doc


def test_phase337_postrun_features_must_remain_ineligible_for_default(tmp_path: Path) -> None:
    def mutate(rows: list[dict[str, str]]) -> None:
        for row in rows:
            if row["feature"] == "actual_scheduled_tokens":
                row["eligible_for_default_prediction"] = "true"

    phase337 = _copy_csv_with_mutation(PHASE337, tmp_path / "phase337.csv", mutate)

    with pytest.raises(ValueError, match="eligible_for_default_prediction"):
        analyzer.analyze_prerun_model_feasibility(phase337)


def test_phase337_scope_keys_must_remain_scope_only(tmp_path: Path) -> None:
    def mutate(rows: list[dict[str, str]]) -> None:
        rows[0]["decision"] = "predictor"

    phase337 = _copy_csv_with_mutation(PHASE337, tmp_path / "phase337.csv", mutate)

    with pytest.raises(ValueError, match="scope_key_only"):
        analyzer.analyze_prerun_model_feasibility(phase337)


def test_writer_requires_exactly_three_rows(tmp_path: Path) -> None:
    rows = analyzer.analyze_prerun_model_feasibility(PHASE337)

    with pytest.raises(ValueError, match="exactly 3"):
        analyzer.write_prerun_model_feasibility_csv(tmp_path / "bad.csv", rows[:2])
