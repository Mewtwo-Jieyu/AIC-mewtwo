from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "analyze_phase303_deeper_trace_topology_family.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase303_deeper_trace_topology_family",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


PHASE287_HEADER = [
    "source",
    "pair_key",
    "topology_key",
    "shape_key",
    "control_scenario",
    "holdout_scenario",
    "control_bt",
    "holdout_bt",
    "output_ratio",
    "holdout_scheduled_p99",
    "holdout_scheduled_max",
    "holdout_max_fill",
    "verdict",
    "mechanism_conclusion",
    "default_readiness",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]

PHASE301_HEADER = [
    "source",
    "pair_key",
    "topology_key",
    "shape_key",
    "control_scenario",
    "holdout_scenario",
    "control_bt",
    "holdout_bt",
    "output_ratio",
    "holdout_scheduled_p99",
    "holdout_scheduled_max",
    "holdout_scheduled_max_fill",
    "verdict",
    "mechanism_conclusion",
    "default_readiness",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]


def _write_csv(path: Path, header: list[str], row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)


def _write_inputs(root: Path) -> tuple[Path, Path]:
    phase287 = root / "phase287_deeper_trace_partial_audit.csv"
    phase301 = root / "phase301_boundary_timeline_cadence_diagnostic.csv"
    _write_csv(
        phase287,
        PHASE287_HEADER,
        {
            "source": "phase287_deeper_trace_partial_audit",
            "pair_key": "tp8_dp1_ep8:isl12000_osl2000_batch128:control_bt12000:holdout_bt65536",
            "topology_key": "tp8_dp1_ep8",
            "shape_key": "isl12000_osl2000_batch128",
            "control_scenario": "tp8ep8-12k2k-bt12000",
            "holdout_scenario": "tp8ep8-12k2k-bt65536",
            "control_bt": "12000",
            "holdout_bt": "65536",
            "output_ratio": "0.969300",
            "holdout_scheduled_p99": "128",
            "holdout_scheduled_max": "12000",
            "holdout_max_fill": "0.183105",
            "verdict": "partial_only",
            "mechanism_conclusion": "boundary_mixed_overhead_partial_only",
            "default_readiness": "No-Go",
            "diagnostic_only": "true",
            "valid_for_default": "false",
            "perf_database": "false",
        },
    )
    _write_csv(
        phase301,
        PHASE301_HEADER,
        {
            "source": "phase301_boundary_timeline_cadence_diagnostic",
            "pair_key": "tp4_dp2_ep8:isl12000_osl2000_batch128:control_bt12000:holdout_bt65536",
            "topology_key": "tp4_dp2_ep8",
            "shape_key": "isl12000_osl2000_batch128",
            "control_scenario": "tp4dp2ep8-12k2k-bt12000",
            "holdout_scenario": "tp4dp2ep8-12k2k-bt65536",
            "control_bt": "12000",
            "holdout_bt": "65536",
            "output_ratio": "1.321996",
            "holdout_scheduled_p99": "128",
            "holdout_scheduled_max": "24736",
            "holdout_scheduled_max_fill": "0.188721",
            "verdict": "boundary_timeline_explains_direction",
            "mechanism_conclusion": "wall_span_iteration_cadence_diagnostic",
            "default_readiness": "No-Go",
            "diagnostic_only": "true",
            "valid_for_default": "false",
            "perf_database": "false",
        },
    )
    return phase287, phase301


def test_topology_family_outputs_two_rows(tmp_path: Path) -> None:
    phase287, phase301 = _write_inputs(tmp_path)

    rows = analyzer.analyze_deeper_trace_topology_family(phase287, phase301)

    assert [row["topology_key"] for row in rows] == ["tp8_dp1_ep8", "tp4_dp2_ep8"]
    assert [row["output_ratio"] for row in rows] == ["0.969300", "1.321996"]
    assert rows[0]["throughput_direction"] == "holdout_slower"
    assert rows[1]["throughput_direction"] == "holdout_faster"
    assert rows[0]["verdict"] == "partial_only"
    assert rows[1]["verdict"] == "boundary_timeline_explains_direction"
    assert {row["budget_ceiling_rejected"] for row in rows} == {"true"}
    assert {row["default_readiness"] for row in rows} == {"No-Go"}
    assert {row["diagnostic_only"] for row in rows} == {"true"}
    assert {row["valid_for_default"] for row in rows} == {"false"}
    assert {row["perf_database"] for row in rows} == {"false"}


def test_missing_input_fails_fast(tmp_path: Path) -> None:
    phase287, phase301 = _write_inputs(tmp_path)
    phase301.unlink()

    with pytest.raises(FileNotFoundError):
        analyzer.analyze_deeper_trace_topology_family(phase287, phase301)


def test_flags_fail_fast(tmp_path: Path) -> None:
    phase287, phase301 = _write_inputs(tmp_path)
    text = phase287.read_text(encoding="utf-8")
    phase287.write_text(text.replace(",true,false,false\n", ",true,true,false\n"), encoding="utf-8")

    with pytest.raises(ValueError, match="flag mismatch"):
        analyzer.analyze_deeper_trace_topology_family(phase287, phase301)


def test_ratio_direction_fails_fast(tmp_path: Path) -> None:
    phase287, phase301 = _write_inputs(tmp_path)
    text = phase287.read_text(encoding="utf-8")
    phase287.write_text(text.replace("0.969300", "1.100000"), encoding="utf-8")

    with pytest.raises(ValueError, match="ratio direction mismatch"):
        analyzer.analyze_deeper_trace_topology_family(phase287, phase301)


def test_write_csv_and_doc(tmp_path: Path) -> None:
    phase287, phase301 = _write_inputs(tmp_path)
    rows = analyzer.analyze_deeper_trace_topology_family(phase287, phase301)
    out_csv = tmp_path / "phase303.csv"
    out_doc = tmp_path / "phase303.md"

    analyzer.write_deeper_trace_topology_family_csv(out_csv, rows)
    analyzer.write_deeper_trace_topology_family_doc(out_doc, rows)

    persisted = list(csv.DictReader(out_csv.open(newline="", encoding="utf-8")))
    assert len(persisted) == 2
    assert persisted[0]["default_readiness"] == "No-Go"
    doc = out_doc.read_text(encoding="utf-8")
    assert "topology-dependent" in doc
    assert "Default AIC | No-Go" in doc
    assert "global correction" in doc
    assert "default-ready" not in doc
