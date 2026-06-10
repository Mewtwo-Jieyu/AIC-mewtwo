from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "analyze_phase171_budget_mechanism_candidate.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase171_budget_mechanism_candidate",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


GAP_FIELDNAMES = [
    "source",
    "topology_key",
    "shape_key",
    "baseline_scenario",
    "budget_scenario",
    "baseline_breakdown_name",
    "budget_breakdown_name",
    "tp",
    "dp",
    "ep",
    "isl",
    "osl",
    "batch_size",
    "baseline_max_num_batched_tokens",
    "budget_max_num_batched_tokens",
    "clean_budget_effect",
    "sim_budget_effect",
    "budget_gap",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]

BREAKDOWN_FIELDNAMES = [
    "name",
    "tp",
    "dp",
    "ep",
    "max_bt",
    "sim_output_tok_s_gpu",
    "steady_state_time_ms",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]


def _gap_rows() -> list[dict[str, str]]:
    return [
        {
            "source": "phase164_clean_gpu_benchmark",
            "topology_key": "tp8_dp1_ep8",
            "shape_key": "isl8000_osl2000_batch128",
            "baseline_scenario": "tp8ep8-bt8000",
            "budget_scenario": "tp8ep8-bt65536",
            "baseline_breakdown_name": "K2.5-tp8ep8-8k2k",
            "budget_breakdown_name": "K2.5-tp8ep8-8k2k-bt65536",
            "tp": "8",
            "dp": "1",
            "ep": "8",
            "isl": "8000",
            "osl": "2000",
            "batch_size": "128",
            "baseline_max_num_batched_tokens": "8000",
            "budget_max_num_batched_tokens": "65536",
            "clean_budget_effect": "1.000000",
            "sim_budget_effect": "0.250000",
            "budget_gap": "4.000000",
            "diagnostic_only": "true",
            "valid_for_default": "false",
            "perf_database": "false",
        },
        {
            "source": "phase164_clean_gpu_benchmark",
            "topology_key": "tp4_dp2_ep8",
            "shape_key": "isl8000_osl2000_batch128",
            "baseline_scenario": "tp4dp2ep8-bt8000",
            "budget_scenario": "tp4dp2ep8-bt65536",
            "baseline_breakdown_name": "K2.5-tp4ep8dp2-8k2k",
            "budget_breakdown_name": "K2.5-tp4ep8dp2-8k2k-bt65536",
            "tp": "4",
            "dp": "2",
            "ep": "8",
            "isl": "8000",
            "osl": "2000",
            "batch_size": "128",
            "baseline_max_num_batched_tokens": "8000",
            "budget_max_num_batched_tokens": "65536",
            "clean_budget_effect": "1.250000",
            "sim_budget_effect": "0.250000",
            "budget_gap": "5.000000",
            "diagnostic_only": "true",
            "valid_for_default": "false",
            "perf_database": "false",
        },
    ]


def _breakdown_rows() -> list[dict[str, str]]:
    return [
        {
            "name": "K2.5-tp8ep8-8k2k",
            "tp": "8",
            "dp": "1",
            "ep": "8",
            "max_bt": "8000",
            "sim_output_tok_s_gpu": "100.000000",
            "steady_state_time_ms": "1000.000000",
            "diagnostic_only": "true",
            "valid_for_default": "false",
            "perf_database": "false",
        },
        {
            "name": "K2.5-tp8ep8-8k2k-bt65536",
            "tp": "8",
            "dp": "1",
            "ep": "8",
            "max_bt": "65536",
            "sim_output_tok_s_gpu": "25.000000",
            "steady_state_time_ms": "4000.000000",
            "diagnostic_only": "true",
            "valid_for_default": "false",
            "perf_database": "false",
        },
        {
            "name": "K2.5-tp4ep8dp2-8k2k",
            "tp": "4",
            "dp": "2",
            "ep": "8",
            "max_bt": "8000",
            "sim_output_tok_s_gpu": "200.000000",
            "steady_state_time_ms": "2000.000000",
            "diagnostic_only": "true",
            "valid_for_default": "false",
            "perf_database": "false",
        },
        {
            "name": "K2.5-tp4ep8dp2-8k2k-bt65536",
            "tp": "4",
            "dp": "2",
            "ep": "8",
            "max_bt": "65536",
            "sim_output_tok_s_gpu": "50.000000",
            "steady_state_time_ms": "10000.000000",
            "diagnostic_only": "true",
            "valid_for_default": "false",
            "perf_database": "false",
        },
    ]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_mechanism_candidate_computes_depenalized_effect(tmp_path: Path) -> None:
    gap = tmp_path / "gap.csv"
    breakdown = tmp_path / "breakdown.csv"
    out_csv = tmp_path / "candidate.csv"
    out_doc = tmp_path / "candidate.md"
    _write_csv(gap, GAP_FIELDNAMES, _gap_rows())
    _write_csv(breakdown, BREAKDOWN_FIELDNAMES, _breakdown_rows())

    rows = analyzer.analyze_mechanism(gap, breakdown)
    analyzer.write_candidate_csv(out_csv, rows)
    analyzer.write_candidate_doc(out_doc, rows)

    written = _read_csv(out_csv)
    assert len(written) == 2
    by_topology = {row["topology_key"]: row for row in written}
    assert by_topology["tp8_dp1_ep8"]["steady_state_time_ratio"] == "4.000000"
    assert by_topology["tp8_dp1_ep8"]["depenalized_budget_effect"] == "1.000000"
    assert by_topology["tp8_dp1_ep8"]["depenalized_error_ratio"] == "1.000000"
    assert by_topology["tp8_dp1_ep8"]["exact_gap_upper_bound"] == "4.000000"
    assert by_topology["tp4_dp2_ep8"]["steady_state_time_ratio"] == "5.000000"
    assert by_topology["tp4_dp2_ep8"]["depenalized_budget_effect"] == "1.250000"
    assert by_topology["tp4_dp2_ep8"]["depenalized_error_ratio"] == "1.000000"
    assert by_topology["tp4_dp2_ep8"]["exact_gap_upper_bound"] == "5.000000"
    assert {row["diagnostic_only"] for row in written} == {"true"}
    assert {row["valid_for_default"] for row in written} == {"false"}
    assert {row["perf_database"] for row in written} == {"false"}
    text = out_doc.read_text(encoding="utf-8")
    assert "mechanism candidate" in text
    assert "exact diagnostic upper bound" in text
    assert "diagnostic_only=true valid_for_default=false perf_database=false" in text


def test_missing_topology_fails(tmp_path: Path) -> None:
    gap = tmp_path / "gap.csv"
    breakdown = tmp_path / "breakdown.csv"
    _write_csv(gap, GAP_FIELDNAMES, _gap_rows()[:1])
    _write_csv(breakdown, BREAKDOWN_FIELDNAMES, _breakdown_rows())

    with pytest.raises(ValueError, match="gap topology set mismatch"):
        analyzer.analyze_mechanism(gap, breakdown)


def test_duplicate_topology_fails(tmp_path: Path) -> None:
    gap = tmp_path / "gap.csv"
    breakdown = tmp_path / "breakdown.csv"
    _write_csv(gap, GAP_FIELDNAMES, [*_gap_rows(), _gap_rows()[0]])
    _write_csv(breakdown, BREAKDOWN_FIELDNAMES, _breakdown_rows())

    with pytest.raises(ValueError, match="duplicate gap topology_key"):
        analyzer.analyze_mechanism(gap, breakdown)


def test_boundary_flag_mismatch_fails(tmp_path: Path) -> None:
    gap = tmp_path / "gap.csv"
    breakdown = tmp_path / "breakdown.csv"
    rows = _gap_rows()
    rows[0]["valid_for_default"] = "true"
    _write_csv(gap, GAP_FIELDNAMES, rows)
    _write_csv(breakdown, BREAKDOWN_FIELDNAMES, _breakdown_rows())

    with pytest.raises(ValueError, match="valid_for_default"):
        analyzer.analyze_mechanism(gap, breakdown)


def test_missing_steady_state_field_fails(tmp_path: Path) -> None:
    gap = tmp_path / "gap.csv"
    breakdown = tmp_path / "breakdown.csv"
    rows = _breakdown_rows()
    for row in rows:
        row.pop("steady_state_time_ms")
    _write_csv(gap, GAP_FIELDNAMES, _gap_rows())
    _write_csv(
        breakdown,
        [field for field in BREAKDOWN_FIELDNAMES if field != "steady_state_time_ms"],
        rows,
    )

    with pytest.raises(ValueError, match="steady_state_time_ms"):
        analyzer.analyze_mechanism(gap, breakdown)
