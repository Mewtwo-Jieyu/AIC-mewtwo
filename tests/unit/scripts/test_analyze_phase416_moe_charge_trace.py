import csv

import pytest

from scripts import analyze_phase416_moe_charge_trace as phase416


def test_phase416_identifies_moe_sol_and_dispatch_fallback_undercharge():
    rows = phase416.build_rows_from_trace(
        scenario=phase416.DEFAULT_SCENARIO,
        artifact_dir="fake/artifact",
        trace_records=[
            {
                "component": "moe_compute",
                "phase": "mixed_prefill",
                "op_input_tokens": 32006,
                "scaled_tokens": 16002,
                "expected_tokens": 32000,
                "query_path": "phase397v_int4_wo_calibrated_sol",
                "perfdb_table": "moe_perf_not_used",
                "returned_ms": 52.7,
                "roofline_lower_bound_ms": 455.8,
            },
            {
                "component": "ep_dispatch_combine",
                "phase": "mixed_prefill",
                "op_input_tokens": 32006,
                "scaled_tokens": 8001,
                "expected_tokens": 32000,
                "query_path": "fallback_tp_dp_collectives",
                "perfdb_table": "custom_allreduce+nccl",
                "returned_ms": 136.5,
                "roofline_lower_bound_ms": 489.3,
            },
        ],
        coverage_rows=[
            {
                "component": "moe_compute",
                "perfdb_table": "moe_perf",
                "coverage_min_tokens": 1,
                "coverage_max_tokens": 32768,
                "coverage_contains_scaled": True,
                "coverage_contains_expected": True,
            },
            {
                "component": "ep_dispatch_combine",
                "perfdb_table": "vllm_module_perf",
                "coverage_min_tokens": 1,
                "coverage_max_tokens": 8192,
                "coverage_contains_scaled": True,
                "coverage_contains_expected": False,
            },
        ],
        phase414_prefill_gap_ms=1000.0,
    )

    summary = [row for row in rows if row["row_type"] == "summary"][0]
    moe = [row for row in rows if row["row_type"] == "query_trace" and row["component"] == "moe_compute"][0]
    dispatch = [
        row
        for row in rows
        if row["row_type"] == "query_trace" and row["component"] == "ep_dispatch_combine"
    ][0]

    assert moe["query_path"] == "phase397v_int4_wo_calibrated_sol"
    assert moe["perfdb_table"] == "moe_perf_not_used"
    assert moe["token_ratio_actual_expected"] == "0.500062"
    assert dispatch["query_path"] == "fallback_tp_dp_collectives"
    assert dispatch["coverage_contains_expected"] == "false"
    assert summary["mechanism_verdict"] == "moe_sol_prefill_misapplied_plus_ep_dispatch_fallback_undercharge"
    assert summary["phase417_target"] == "split_prefill_moe_charge_and_restore_ep8_alltoall_charge"
    assert summary["phase405_penalty_read"] == "false"
    assert summary["valid_for_default"] == "false"
    assert summary["default_readiness"] == "No-Go"


def test_phase416_writer_rejects_runtime_or_default_claim(tmp_path):
    rows = [{field: "" for field in phase416.CSV_FIELDS}]
    rows[0].update(
        {
            "source": phase416.SOURCE,
            "row_type": "summary",
            "scenario": phase416.DEFAULT_SCENARIO,
            "phase405_penalty_read": "false",
            "gpu_allowed": "false",
            "ssh_allowed": "false",
            "runtime_modified": "true",
            "perf_database": "false",
            "valid_for_default": "false",
            "diagnostic_only": "true",
            "default_readiness": "No-Go",
        }
    )
    with pytest.raises(ValueError, match="runtime_modified"):
        phase416.write_phase416_csv(tmp_path / "bad.csv", rows)

    rows[0]["runtime_modified"] = "false"
    rows[0]["valid_for_default"] = "true"
    with pytest.raises(ValueError, match="valid_for_default"):
        phase416.write_phase416_csv(tmp_path / "bad.csv", rows)


def test_phase416_csv_roundtrip_keeps_trace_and_summary_rows(tmp_path):
    rows = phase416.build_rows_from_trace(
        scenario=phase416.DEFAULT_SCENARIO,
        artifact_dir="fake/artifact",
        trace_records=[
            {
                "component": "moe_compute",
                "phase": "decode",
                "op_input_tokens": 9,
                "scaled_tokens": 18,
                "expected_tokens": 18,
                "query_path": "phase397v_int4_wo_calibrated_sol",
                "perfdb_table": "moe_perf_not_used",
                "returned_ms": 1.0,
                "roofline_lower_bound_ms": 0.5,
            }
        ],
        coverage_rows=[],
        phase414_prefill_gap_ms=100.0,
    )
    out = tmp_path / "phase416.csv"
    phase416.write_phase416_csv(out, rows)
    with out.open(newline="") as f:
        loaded = list(csv.DictReader(f))
    assert [row["row_type"] for row in loaded] == ["query_trace", "summary"]


def test_phase416_fixed_verdict_renders_fixed_diagnosis():
    rows = phase416.build_rows_from_trace(
        scenario=phase416.DEFAULT_SCENARIO,
        artifact_dir="fake/artifact",
        trace_records=[
            {
                "component": "moe_compute",
                "phase": "mixed_prefill",
                "op_input_tokens": 32006,
                "scaled_tokens": 32000,
                "expected_tokens": 32000,
                "query_path": "moe_perf_lookup",
                "perfdb_table": "moe_perf",
                "returned_ms": 352.7,
                "roofline_lower_bound_ms": 455.8,
            },
            {
                "component": "ep_dispatch_combine",
                "phase": "mixed_prefill",
                "op_input_tokens": 32006,
                "scaled_tokens": 32000,
                "expected_tokens": 32000,
                "query_path": "ep8_alltoall_roofline",
                "perfdb_table": "roofline",
                "returned_ms": 497.5,
                "roofline_lower_bound_ms": 489.3,
            },
        ],
        coverage_rows=[],
        phase414_prefill_gap_ms=1000.0,
    )

    md = phase416.render_phase416_md(rows)

    assert "now queries `moe_perf.txt`" in md
    assert "EP8 all-to-all byte model" in md
    assert "Component roofline gate is now clean" in md
    assert "提前走 Phase397v" not in md


def test_phase416_corrected_mixed_step_only_lifts_undercharged_terms():
    rows = phase416.build_rows_from_trace(
        scenario=phase416.DEFAULT_SCENARIO,
        artifact_dir="fake/artifact",
        trace_records=[
            {
                "component": "moe_compute",
                "phase": "mixed_prefill",
                "op_input_tokens": 32006,
                "scaled_tokens": 32000,
                "expected_tokens": 32000,
                "query_path": "moe_perf_lookup",
                "perfdb_table": "moe_perf",
                "returned_ms": 352.7,
                "roofline_lower_bound_ms": 170.8,
            },
            {
                "component": "ep_dispatch_combine",
                "phase": "mixed_prefill",
                "op_input_tokens": 32006,
                "scaled_tokens": 32000,
                "expected_tokens": 32000,
                "query_path": "ep8_alltoall_roofline",
                "perfdb_table": "roofline",
                "returned_ms": 497.5,
                "roofline_lower_bound_ms": 489.3,
            },
        ],
        coverage_rows=[],
        phase414_prefill_gap_ms=1000.0,
        phase415_mixed_step_ms=1818.6,
    )

    summary = [row for row in rows if row["row_type"] == "summary"][0]

    assert summary["corrected_mixed_step_ms"] == "1818.600000"
