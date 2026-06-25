from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "scripts"
    / "analyze_phase363_ep8_comm_single_point_sufficiency_gate.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase363_ep8_comm_single_point_sufficiency_gate",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


PHASE362 = (
    REPO_ROOT / "docs/iter_gap_investigation/phase362_ep8_comm_single_point_result.csv"
)


def test_phase363_outputs_exact_five_gate_rows() -> None:
    rows = analyzer.analyze_phase363_ep8_comm_single_point_sufficiency_gate(PHASE362)

    assert [row["gate"] for row in rows] == [
        "phase362_prerequisite",
        "single_point_sufficiency",
        "default_aic_gate",
        "shape_expansion_required",
        "future_gpu_entry",
    ]
    assert rows[0]["verdict"] == "phase362_single_point_pass_required"
    assert rows[1]["verdict"] == "insufficient_for_perfdb_curve"
    assert rows[2]["verdict"] == "No-Go"
    assert rows[3]["verdict"] == "shape_expansion_spec_required"
    assert rows[4]["verdict"] == "future_gpu_entry_recorded_only"


def test_phase363_requires_phase362_backend_capture() -> None:
    rows = analyzer.analyze_phase363_ep8_comm_single_point_sufficiency_gate(PHASE362)
    prerequisite = rows[0]

    assert prerequisite["phase362_ok"] == "true"
    assert prerequisite["required_backend"] == "allgather_reducescatter"
    assert prerequisite["observed_backend"] == "allgather_reducescatter"
    assert prerequisite["manager"] == "AgRsAll2AllManager"
    assert prerequisite["latency_ms"] == "0.590688"


def test_phase363_rejects_perfdb_curve_and_default_aic_promotion() -> None:
    rows = analyzer.analyze_phase363_ep8_comm_single_point_sufficiency_gate(PHASE362)
    sufficiency = rows[1]
    default_gate = rows[2]

    assert sufficiency["curve_fit_allowed"] == "false"
    assert sufficiency["interpolation_allowed"] == "false"
    assert sufficiency["extrapolation_allowed"] == "false"
    assert sufficiency["perfdb_curve_allowed"] == "false"
    assert "single_point_num_tokens_128_only" in sufficiency["blocking_reason"]
    assert default_gate["default_readiness"] == "No-Go"
    assert default_gate["blocking_reason"] == "missing_shape_coverage_and_error_model"


def test_phase363_records_future_gpu_entry_but_keeps_gpu_disallowed() -> None:
    rows = analyzer.analyze_phase363_ep8_comm_single_point_sufficiency_gate(PHASE362)
    future = rows[4]

    assert future["future_gpu_ssh"] == (
        "ssh -CAXY "
        "ws-faaf0de74ef9a14d-worker-gn6kz.zhaojieyu+root.ailab-sys.pod"
        "@h.pjlab.org.cn"
    )
    assert future["gpu_allowed"] == "false"
    assert future["next_allowed_phase"] == "phase364_shape_expansion_spec"


def test_phase363_preserves_no_go_diagnostic_flags() -> None:
    rows = analyzer.analyze_phase363_ep8_comm_single_point_sufficiency_gate(PHASE362)

    for row in rows:
        assert row["gpu_allowed"] == "false"
        assert row["default_readiness"] == "No-Go"
        assert row["diagnostic_only"] == "true"
        assert row["valid_for_default"] == "false"
        assert row["perf_database"] == "false"


def test_phase362_input_must_be_passed_allgather_result(tmp_path: Path) -> None:
    with PHASE362.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    assert fieldnames is not None
    rows[0]["backend"] = "alltoall"

    bad = tmp_path / "phase362.csv"
    with bad.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="backend"):
        analyzer.analyze_phase363_ep8_comm_single_point_sufficiency_gate(bad)


def test_output_rejects_default_or_perfdb_upgrade(tmp_path: Path) -> None:
    rows = analyzer.analyze_phase363_ep8_comm_single_point_sufficiency_gate(PHASE362)
    rows[1]["perfdb_curve_allowed"] = "true"

    with pytest.raises(ValueError, match="perfdb_curve_allowed"):
        analyzer.write_phase363_csv(tmp_path / "bad.csv", rows)


def test_writers_emit_csv_and_md(tmp_path: Path) -> None:
    rows = analyzer.analyze_phase363_ep8_comm_single_point_sufficiency_gate(PHASE362)
    csv_path = tmp_path / "phase363.csv"
    md_path = tmp_path / "phase363.md"

    analyzer.write_phase363_csv(csv_path, rows)
    analyzer.write_phase363_md(md_path, rows)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 5
    assert set(written[0]) == set(analyzer.FIELDNAMES)

    doc = md_path.read_text(encoding="utf-8")
    assert "Phase362 single point passed" in doc
    assert "`0.590688` ms only proves the EP8 comm single point is measurable" in doc
    assert "cannot be interpolated or extrapolated into a PerfDatabase curve" in doc
    assert "Phase364 shape expansion spec" in doc
    assert "Default AIC | No-Go" in doc
    assert "PerfDatabase | not written" in doc
