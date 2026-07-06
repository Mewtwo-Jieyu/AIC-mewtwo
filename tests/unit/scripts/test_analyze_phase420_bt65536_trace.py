import csv

import pytest

from scripts import analyze_phase420_bt65536_trace as phase420


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _trace(label, scenario, *, sim, step_ms, blocks, classification):
    return {
        "source": phase420.SOURCE,
        "row_type": "trace",
        "label": label,
        "scenario": scenario,
        "tp": "4" if "tp4" in scenario else "8",
        "dp": "2" if "dp2" in scenario else "1",
        "ep": "8",
        "max_bt": "65536",
        "real_output_tok_s_gpu": "155.952000" if "tp4" in scenario else "138.470000",
        "sim_output_tok_s_gpu": f"{sim:.6f}",
        "error_ratio": f"{phase420.abs_error(sim, 155.952 if 'tp4' in scenario else 138.47):.6f}",
        "num_gpu_blocks": str(blocks),
        "full_sequence_capacity_per_engine": "2.574400" if "tp4" in scenario else "34.355200",
        "per_replica_concurrency": "64" if "tp4" in scenario else "128",
        "configured_max_num_batched_tokens": "8000",
        "budget_full_prefill_reqs": "8",
        "initial_prefill_reqs": "1",
        "initial_prefill_tokens": "8000",
        "representative_step_ms": f"{step_ms:.6f}",
        "context_non_attention_ms": f"{step_ms - 100.0:.6f}",
        "context_attention_ms": "100.000000",
        "moe_query_path": "moe_perf_lookup" if label == "after" else "phase397v_int4_wo_calibrated_sol",
        "ep_query_path": "ep8_alltoall_fallback" if label == "after" else "fallback_tp_dp_collectives",
        "capacity_classification": classification,
    }


def test_phase420_builds_dirty_capacity_and_regression_verdict(tmp_path):
    before = tmp_path / "before.csv"
    after = tmp_path / "after.csv"
    _write_csv(
        before,
        phase420.TRACE_FIELDS,
        [
            _trace("before", phase420.DP2_BT_SCENARIO, sim=23.726143, step_ms=800.0, blocks=1609, classification="dirty_capacity_conflict"),
            _trace("before", phase420.TP8_BT_SCENARIO, sim=142.135372, step_ms=900.0, blocks=21472, classification="reference_dirty_capacity"),
        ],
    )
    _write_csv(
        after,
        phase420.TRACE_FIELDS,
        [
            _trace("after", phase420.DP2_BT_SCENARIO, sim=20.725982, step_ms=1000.0, blocks=1609, classification="dirty_capacity_conflict"),
            _trace("after", phase420.TP8_BT_SCENARIO, sim=116.324618, step_ms=1200.0, blocks=21472, classification="reference_dirty_capacity"),
        ],
    )

    rows = phase420.build_phase420_rows(before_csv=before, after_csv=after)
    summary = [row for row in rows if row["row_type"] == "summary"][0]
    dp2 = [row for row in rows if row["row_type"] == "diagnosis" and row["scenario"] == phase420.DP2_BT_SCENARIO][0]
    tp8 = [row for row in rows if row["row_type"] == "diagnosis" and row["scenario"] == phase420.TP8_BT_SCENARIO][0]

    assert dp2["dominant_cause"] == "bt65536_budget_not_wired_plus_dirty_capacity"
    assert tp8["dominant_cause"] == "phase417_single_prefill_step_cost_regression_after_budget_not_wired"
    assert summary["mechanism_verdict"] == "bt65536_budget_not_wired_plus_phase417_single_step_regression"
    assert summary["next_phase_target"] == "phase421_wire_bt65536_budget_then_retrace_query_cost"
    assert summary["default_readiness"] == "No-Go"
    assert summary["valid_for_default"] == "false"


def test_phase420_writer_rejects_default_claim(tmp_path):
    rows = [{field: "" for field in phase420.OUTPUT_FIELDS}]
    rows[0].update(
        {
            "source": phase420.SOURCE,
            "row_type": "summary",
            "phase405_penalty_read": "false",
            "gpu_allowed": "false",
            "ssh_allowed": "false",
            "runtime_modified": "false",
            "perf_database": "false",
            "valid_for_default": "true",
            "diagnostic_only": "true",
            "default_readiness": "No-Go",
        }
    )

    with pytest.raises(ValueError, match="valid_for_default"):
        phase420.write_phase420_csv(tmp_path / "bad.csv", rows)


def test_phase420_md_mentions_two_lines(tmp_path):
    before = tmp_path / "before.csv"
    after = tmp_path / "after.csv"
    _write_csv(before, phase420.TRACE_FIELDS, [_trace("before", phase420.DP2_BT_SCENARIO, sim=23.7, step_ms=800.0, blocks=1609, classification="dirty_capacity_conflict")])
    _write_csv(after, phase420.TRACE_FIELDS, [_trace("after", phase420.DP2_BT_SCENARIO, sim=20.7, step_ms=1000.0, blocks=1609, classification="dirty_capacity_conflict")])

    rows = phase420.build_phase420_rows(before_csv=before, after_csv=after)
    md = phase420.render_phase420_md(rows)

    assert "Phase420" in md
    assert "Default AIC: `No-Go`" in md
    assert phase420.DP2_BT_SCENARIO in md
