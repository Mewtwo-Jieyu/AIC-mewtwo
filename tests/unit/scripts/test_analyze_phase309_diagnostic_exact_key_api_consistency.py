from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "analyze_phase309_diagnostic_exact_key_api_consistency.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase309_diagnostic_exact_key_api_consistency",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


PHASE258_HEADER = [
    "source",
    "topology_key",
    "scenario",
    "role",
    "shape_key",
    "control_max_bt",
    "holdout_max_bt",
    "max_bt",
    "configured_budget_per_rank",
    "configured_budget_aggregate",
    "output_tok_s",
    "output_ratio_vs_control",
    "raw_trace_rows",
    "unique_iterations",
    "rank_min_p50_scheduled_total_tokens",
    "rank_min_p95_scheduled_total_tokens",
    "rank_min_p99_scheduled_total_tokens",
    "rank_min_max_scheduled_total_tokens",
    "rank_max_p50_scheduled_total_tokens",
    "rank_max_p95_scheduled_total_tokens",
    "rank_max_p99_scheduled_total_tokens",
    "rank_max_max_scheduled_total_tokens",
    "rank_sum_p50_scheduled_total_tokens",
    "rank_sum_p95_scheduled_total_tokens",
    "rank_sum_p99_scheduled_total_tokens",
    "rank_sum_max_scheduled_total_tokens",
    "rank_sum_mean_fill_ratio",
    "rank_sum_p99_fill_ratio",
    "rank_sum_max_fill_ratio",
    "mechanism_hypothesis",
    "default_readiness",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]

PHASE303_HEADER = [
    "source",
    "family_key",
    "topology_key",
    "shape_key",
    "control_scenario",
    "holdout_scenario",
    "control_bt",
    "holdout_bt",
    "output_ratio",
    "throughput_direction",
    "holdout_scheduled_p99",
    "holdout_scheduled_max",
    "holdout_max_fill",
    "budget_ceiling_rejected",
    "verdict",
    "mechanism_conclusion",
    "default_readiness",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]


def _write_csv(path: Path, header: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _phase258_row(
    *,
    topology_key: str,
    scenario: str,
    role: str,
    shape_key: str,
    control_bt: int,
    max_bt: int,
    aggregate: int,
    ratio: str,
    p99: int,
    max_scheduled: int,
    max_fill: str,
) -> dict[str, str]:
    return {
        "source": "phase258_actual_scheduled_token_full_family",
        "topology_key": topology_key,
        "scenario": scenario,
        "role": role,
        "shape_key": shape_key,
        "control_max_bt": str(control_bt),
        "holdout_max_bt": "65536",
        "max_bt": str(max_bt),
        "configured_budget_per_rank": str(max_bt),
        "configured_budget_aggregate": str(aggregate),
        "output_tok_s": "100.000000" if role == "control" else "110.000000",
        "output_ratio_vs_control": ratio,
        "raw_trace_rows": "16000",
        "unique_iterations": "2000",
        "rank_min_p50_scheduled_total_tokens": "64.000000",
        "rank_min_p95_scheduled_total_tokens": "64.000000",
        "rank_min_p99_scheduled_total_tokens": "64.000000",
        "rank_min_max_scheduled_total_tokens": str(max_bt),
        "rank_max_p50_scheduled_total_tokens": "64.000000",
        "rank_max_p95_scheduled_total_tokens": "64.000000",
        "rank_max_p99_scheduled_total_tokens": "64.000000",
        "rank_max_max_scheduled_total_tokens": str(max_bt),
        "rank_sum_p50_scheduled_total_tokens": "128.000000",
        "rank_sum_p95_scheduled_total_tokens": "128.000000",
        "rank_sum_p99_scheduled_total_tokens": f"{p99:.6f}",
        "rank_sum_max_scheduled_total_tokens": str(max_scheduled),
        "rank_sum_mean_fill_ratio": "0.001000",
        "rank_sum_p99_fill_ratio": f"{p99 / aggregate:.6f}",
        "rank_sum_max_fill_ratio": max_fill,
        "mechanism_hypothesis": "actual_scheduled_tokens_not_configured_budget",
        "default_readiness": "No-Go",
        "diagnostic_only": "true",
        "valid_for_default": "false",
        "perf_database": "false",
    }


def _phase258_rows() -> list[dict[str, str]]:
    specs = [
        ("tp8_dp1_ep8", "isl4000_osl2000_batch128", "tp8ep8-4k2k-bt4000", "tp8ep8-4k2k-bt65536", 4000, 4000, 4000, "0.061035"),
        ("tp4_dp2_ep8", "isl4000_osl2000_batch128", "tp4dp2ep8-4k2k-bt4000", "tp4dp2ep8-4k2k-bt65536", 4000, 8000, 8240, "0.062866"),
        ("tp8_dp1_ep8", "isl12000_osl2000_batch128", "tp8ep8-12k2k-bt12000", "tp8ep8-12k2k-bt65536", 12000, 12000, 12000, "0.183105"),
        ("tp4_dp2_ep8", "isl12000_osl2000_batch128", "tp4dp2ep8-12k2k-bt12000", "tp4dp2ep8-12k2k-bt65536", 12000, 24000, 24000, "0.183105"),
    ]
    rows: list[dict[str, str]] = []
    for topology, shape, control_scenario, holdout_scenario, control_bt, control_aggregate, holdout_max, holdout_fill in specs:
        rows.append(
            _phase258_row(
                topology_key=topology,
                scenario=control_scenario,
                role="control",
                shape_key=shape,
                control_bt=control_bt,
                max_bt=control_bt,
                aggregate=control_aggregate,
                ratio="1.000000",
                p99=128,
                max_scheduled=control_aggregate,
                max_fill="1.000000",
            )
        )
        holdout_aggregate = 131072 if topology == "tp4_dp2_ep8" else 65536
        rows.append(
            _phase258_row(
                topology_key=topology,
                scenario=holdout_scenario,
                role="holdout",
                shape_key=shape,
                control_bt=control_bt,
                max_bt=65536,
                aggregate=holdout_aggregate,
                ratio="1.100000",
                p99=128,
                max_scheduled=holdout_max,
                max_fill=holdout_fill,
            )
        )
    return rows


def _phase303_rows() -> list[dict[str, str]]:
    return [
        {
            "source": "phase303_deeper_trace_topology_family",
            "family_key": "deeper_trace_12k2k_topology_family",
            "topology_key": "tp8_dp1_ep8",
            "shape_key": "isl12000_osl2000_batch128",
            "control_scenario": "tp8ep8-12k2k-bt12000",
            "holdout_scenario": "tp8ep8-12k2k-bt65536",
            "control_bt": "12000",
            "holdout_bt": "65536",
            "output_ratio": "0.969300",
            "throughput_direction": "holdout_slower",
            "holdout_scheduled_p99": "128",
            "holdout_scheduled_max": "12000",
            "holdout_max_fill": "0.183105",
            "budget_ceiling_rejected": "true",
            "verdict": "partial_only",
            "mechanism_conclusion": "boundary_mixed_overhead_partial_only",
            "default_readiness": "No-Go",
            "diagnostic_only": "true",
            "valid_for_default": "false",
            "perf_database": "false",
        },
        {
            "source": "phase303_deeper_trace_topology_family",
            "family_key": "deeper_trace_12k2k_topology_family",
            "topology_key": "tp4_dp2_ep8",
            "shape_key": "isl12000_osl2000_batch128",
            "control_scenario": "tp4dp2ep8-12k2k-bt12000",
            "holdout_scenario": "tp4dp2ep8-12k2k-bt65536",
            "control_bt": "12000",
            "holdout_bt": "65536",
            "output_ratio": "1.321996",
            "throughput_direction": "holdout_faster",
            "holdout_scheduled_p99": "128",
            "holdout_scheduled_max": "24736",
            "holdout_max_fill": "0.188721",
            "budget_ceiling_rejected": "true",
            "verdict": "boundary_timeline_explains_direction",
            "mechanism_conclusion": "wall_span_iteration_cadence_diagnostic",
            "default_readiness": "No-Go",
            "diagnostic_only": "true",
            "valid_for_default": "false",
            "perf_database": "false",
        },
    ]


def _write_inputs(root: Path) -> tuple[Path, Path]:
    phase258 = root / "phase258_actual_scheduled_token_full_family.csv"
    phase303 = root / "phase303_deeper_trace_topology_family.csv"
    _write_csv(phase258, PHASE258_HEADER, _phase258_rows())
    _write_csv(phase303, PHASE303_HEADER, _phase303_rows())
    return phase258, phase303


def test_exact_key_api_consistency_outputs_two_rows(tmp_path: Path) -> None:
    phase258, phase303 = _write_inputs(tmp_path)

    rows = analyzer.analyze_diagnostic_exact_key_api_consistency(phase258, phase303)

    assert [row["api_family"] for row in rows] == [
        "actual_scheduled_token_family",
        "deeper_trace_topology_family",
    ]
    assert [row["expected_candidate_count"] for row in rows] == ["4", "2"]
    assert [row["loaded_candidate_count"] for row in rows] == ["4", "2"]
    assert {row["unknown_key_rejects"] for row in rows} == {"true"}
    assert {row["diagnostic_only_all"] for row in rows} == {"true"}
    assert {row["valid_for_default_any"] for row in rows} == {"false"}
    assert {row["perf_database_any"] for row in rows} == {"false"}
    assert {row["default_readiness"] for row in rows} == {"No-Go"}
    assert {row["verdict"] for row in rows} == {"diagnostic_exact_key_lookup_consistent"}


def test_missing_input_fails_fast(tmp_path: Path) -> None:
    phase258, phase303 = _write_inputs(tmp_path)
    phase303.unlink()

    with pytest.raises(FileNotFoundError):
        analyzer.analyze_diagnostic_exact_key_api_consistency(phase258, phase303)


def test_flag_mismatch_fails_fast(tmp_path: Path) -> None:
    phase258, phase303 = _write_inputs(tmp_path)
    text = phase258.read_text(encoding="utf-8")
    phase258.write_text(text.replace(",No-Go,true,false,false\n", ",No-Go,true,true,false\n", 1), encoding="utf-8")

    with pytest.raises(ValueError, match="valid_for_default"):
        analyzer.analyze_diagnostic_exact_key_api_consistency(phase258, phase303)


def test_candidate_count_mismatch_fails_fast(tmp_path: Path) -> None:
    phase258, phase303 = _write_inputs(tmp_path)
    rows = _phase303_rows()[:1]
    _write_csv(phase303, PHASE303_HEADER, rows)

    with pytest.raises(ValueError, match="2 rows"):
        analyzer.analyze_diagnostic_exact_key_api_consistency(phase258, phase303)


def test_write_csv_and_doc(tmp_path: Path) -> None:
    phase258, phase303 = _write_inputs(tmp_path)
    rows = analyzer.analyze_diagnostic_exact_key_api_consistency(phase258, phase303)
    out_csv = tmp_path / "phase309.csv"
    out_doc = tmp_path / "phase309.md"

    analyzer.write_diagnostic_exact_key_api_consistency_csv(out_csv, rows)
    analyzer.write_diagnostic_exact_key_api_consistency_doc(out_doc, rows)

    persisted = list(csv.DictReader(out_csv.open(newline="", encoding="utf-8")))
    assert len(persisted) == 2
    assert persisted[0]["api_family"] == "actual_scheduled_token_family"
    doc = out_doc.read_text(encoding="utf-8")
    assert "diagnostic exact-key lookup" in doc
    assert "Default AIC | No-Go" in doc
    assert "interpolation" in doc
    assert "extrapolation" in doc
    assert "default-ready" not in doc


def test_actual_phase_csvs_generate_consistency_rows() -> None:
    root = Path(__file__).resolve().parents[3]
    rows = analyzer.analyze_diagnostic_exact_key_api_consistency(
        root / "docs/iter_gap_investigation/phase258_actual_scheduled_token_full_family.csv",
        root / "docs/iter_gap_investigation/phase303_deeper_trace_topology_family.csv",
    )

    assert [row["loaded_candidate_count"] for row in rows] == ["4", "2"]
    assert {row["verdict"] for row in rows} == {"diagnostic_exact_key_lookup_consistent"}
