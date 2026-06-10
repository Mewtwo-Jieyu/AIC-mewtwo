from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "analyze_phase165_clean_cb_sim_gap.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase165_clean_cb_sim_gap",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


MANIFEST_ROWS = [
    ("tp8ep8-bt8000", 8, 1, 8, 8000, "437.887553"),
    ("tp8ep8-bt65536", 8, 1, 8, 65536, "432.362445"),
    ("tp4dp2ep8-bt8000", 4, 2, 8, 8000, "343.644637"),
    ("tp4dp2ep8-bt65536", 4, 2, 8, 65536, "401.203958"),
]

BREAKDOWN_ROWS = [
    ("K2.5-tp8ep8-8k2k", 8, 1, 8, 8000, "100.000000", 4, 3),
    ("K2.5-tp8ep8-32k3k", 8, 1, 8, 32000, "80.000000", 6, 6),
    ("K2.5-tp4ep8dp2-8k2k", 4, 2, 8, 8000, "200.000000", 3, 4),
    ("K2.5-tp4ep8dp2-32k3k", 4, 2, 8, 32000, "70.000000", 5, 5),
    ("K2.5-tp8ep8-8k2k-bt65536", 8, 1, 8, 65536, "25.000000", 5, 2),
    ("K2.5-tp4ep8dp2-8k2k-bt65536", 4, 2, 8, 65536, "50.000000", 6, 1),
]


def _write_manifest(path: Path, rows=MANIFEST_ROWS) -> None:
    fieldnames = [
        "source",
        "scenario",
        "tp",
        "dp",
        "ep",
        "isl",
        "osl",
        "batch_size",
        "max_num_batched_tokens",
        "request_success_count",
        "request_fail_count",
        "real_output_tok_s_gpu",
        "real_total_tok_s_gpu",
        "diagnostic_only",
        "valid_for_default",
        "perf_database",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for scenario, tp, dp, ep, max_bt, output_gpu in rows:
            writer.writerow(
                {
                    "source": "phase164_clean_gpu_benchmark",
                    "scenario": scenario,
                    "tp": tp,
                    "dp": dp,
                    "ep": ep,
                    "isl": 8000,
                    "osl": 2000,
                    "batch_size": 128,
                    "max_num_batched_tokens": max_bt,
                    "request_success_count": 128,
                    "request_fail_count": 0,
                    "real_output_tok_s_gpu": output_gpu,
                    "real_total_tok_s_gpu": "1000.000000",
                    "diagnostic_only": "true",
                    "valid_for_default": "false",
                    "perf_database": "false",
                }
            )


def _write_breakdown(path: Path, rows=BREAKDOWN_ROWS) -> None:
    fieldnames = [
        "name",
        "tp",
        "dp",
        "ep",
        "max_bt",
        "real_output_tok_s_gpu",
        "sim_output_tok_s_gpu",
        "error_ratio",
        "rank",
        "real_rank",
        "avg_prefill_reqs_per_iter",
        "avg_decode_reqs_per_iter",
        "avg_tokens_per_iter",
        "peak_prefill_reqs_per_iter",
        "peak_decode_reqs_per_iter",
        "peak_tokens_per_iter",
        "steady_state_iterations",
        "steady_state_time_ms",
        "paired_baseline_name",
        "paired_baseline_max_bt",
        "paired_baseline_sim_output_tok_s_gpu",
        "paired_baseline_error_ratio",
        "paired_baseline_rank",
        "paired_baseline_avg_tokens_per_iter",
        "max_bt_vs_paired_baseline",
        "sim_vs_paired_baseline_ratio",
        "avg_tokens_vs_paired_baseline_ratio",
        "steady_state_time_vs_paired_baseline_ratio",
        "diagnostic_only",
        "valid_for_default",
        "perf_database",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, tp, dp, ep, max_bt, sim, rank, real_rank in rows:
            writer.writerow(
                {
                    "name": name,
                    "tp": tp,
                    "dp": dp,
                    "ep": ep,
                    "max_bt": max_bt,
                    "real_output_tok_s_gpu": "1.000000",
                    "sim_output_tok_s_gpu": sim,
                    "error_ratio": "1.000000",
                    "rank": rank,
                    "real_rank": real_rank,
                    "avg_prefill_reqs_per_iter": "1.000000",
                    "avg_decode_reqs_per_iter": "128.000000",
                    "avg_tokens_per_iter": str(max_bt),
                    "peak_prefill_reqs_per_iter": "2.000000",
                    "peak_decode_reqs_per_iter": "128.000000",
                    "peak_tokens_per_iter": str(max_bt + 128),
                    "steady_state_iterations": "8",
                    "steady_state_time_ms": str(max_bt * 2),
                    "paired_baseline_name": name.replace("-bt65536", ""),
                    "paired_baseline_max_bt": "8000",
                    "paired_baseline_sim_output_tok_s_gpu": "100.000000",
                    "paired_baseline_error_ratio": "1.000000",
                    "paired_baseline_rank": "4",
                    "paired_baseline_avg_tokens_per_iter": "8000.000000",
                    "max_bt_vs_paired_baseline": "1.000000",
                    "sim_vs_paired_baseline_ratio": "1.000000",
                    "avg_tokens_vs_paired_baseline_ratio": "1.000000",
                    "steady_state_time_vs_paired_baseline_ratio": "1.000000",
                    "diagnostic_only": "true",
                    "valid_for_default": "false",
                    "perf_database": "false",
                }
            )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_gap_analysis_joins_four_clean_rows_and_computes_budget_ratios(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    breakdown = tmp_path / "breakdown.csv"
    out_csv = tmp_path / "gap.csv"
    out_doc = tmp_path / "candidate.md"
    _write_manifest(manifest)
    _write_breakdown(breakdown)

    rows = analyzer.analyze_gap(manifest, breakdown)
    analyzer.write_gap_csv(out_csv, rows)
    analyzer.write_candidate_doc(out_doc, rows)

    written = _read_csv(out_csv)
    assert len(written) == 2
    by_topology = {row["topology_key"]: row for row in written}
    assert by_topology["tp8_dp1_ep8"]["clean_budget_effect"] == "0.987382"
    assert by_topology["tp8_dp1_ep8"]["sim_budget_effect"] == "0.250000"
    assert by_topology["tp8_dp1_ep8"]["budget_gap"] == "3.949529"
    assert by_topology["tp4_dp2_ep8"]["clean_budget_effect"] == "1.167497"
    assert by_topology["tp4_dp2_ep8"]["sim_budget_effect"] == "0.250000"
    assert by_topology["tp4_dp2_ep8"]["budget_gap"] == "4.669987"
    assert {row["diagnostic_only"] for row in written} == {"true"}
    assert {row["valid_for_default"] for row in written} == {"false"}
    assert {row["perf_database"] for row in written} == {"false"}
    text = out_doc.read_text(encoding="utf-8")
    assert "diagnostic_only=true" in text
    assert "valid_for_default=false" in text
    assert "perf_database=false" in text
    assert "topology_key + shape_key + max_num_batched_tokens" in text


def test_manifest_missing_scenario_fails(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    breakdown = tmp_path / "breakdown.csv"
    _write_manifest(manifest, rows=MANIFEST_ROWS[:-1])
    _write_breakdown(breakdown)

    with pytest.raises(ValueError, match="manifest scenario set mismatch"):
        analyzer.analyze_gap(manifest, breakdown)


def test_manifest_duplicate_scenario_fails(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    breakdown = tmp_path / "breakdown.csv"
    _write_manifest(manifest, rows=[*MANIFEST_ROWS, MANIFEST_ROWS[0]])
    _write_breakdown(breakdown)

    with pytest.raises(ValueError, match="duplicate manifest scenario"):
        analyzer.analyze_gap(manifest, breakdown)


def test_boundary_flag_mismatch_fails(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    breakdown = tmp_path / "breakdown.csv"
    _write_manifest(manifest)
    _write_breakdown(breakdown)
    rows = _read_csv(breakdown)
    rows[0]["diagnostic_only"] = "false"
    with breakdown.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="diagnostic_only"):
        analyzer.analyze_gap(manifest, breakdown)


def test_total_throughput_field_does_not_satisfy_output_field(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    breakdown = tmp_path / "breakdown.csv"
    _write_manifest(manifest)
    _write_breakdown(breakdown)
    rows = _read_csv(manifest)
    for row in rows:
        row.pop("real_output_tok_s_gpu")
        row["real_total_tok_s_gpu"] = "999.000000"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="real_output_tok_s_gpu"):
        analyzer.analyze_gap(manifest, breakdown)
