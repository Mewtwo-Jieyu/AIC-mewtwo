import csv

import pytest

from scripts import analyze_phase421_wire_bt65536 as phase421


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _validation_row(name, *, sim, error):
    real = 155.952 if "tp4" in name else 138.47
    return {
        "scenario": name,
        "tp": "4" if "tp4" in name else "8",
        "dp": "2" if "dp2" in name else "1",
        "ep": "8",
        "max_bt": "65536",
        "real_output_tok_s_gpu": f"{real:.6f}",
        "sim_output_tok_s_gpu": f"{sim:.6f}",
        "sim_real_ratio": f"{sim / real:.6f}",
        "error_ratio": f"{error:.6f}",
        "throughput_source": "cb_sim",
    }


def _trace_row(name, *, tokens, capacity):
    return {
        "source": "phase420_bt65536_trace",
        "row_type": "trace",
        "label": "after",
        "scenario": name,
        "tp": "4" if "tp4" in name else "8",
        "dp": "2" if "dp2" in name else "1",
        "ep": "8",
        "max_bt": "65536",
        "real_output_tok_s_gpu": "155.952000" if "tp4" in name else "138.470000",
        "sim_output_tok_s_gpu": "27.638441" if "tp4" in name else "155.069882",
        "error_ratio": "5.642576" if "tp4" in name else "1.119881",
        "num_gpu_blocks": "1609" if "tp4" in name else "21472",
        "full_sequence_capacity_per_engine": capacity,
        "per_replica_concurrency": "64" if "tp4" in name else "128",
        "configured_max_num_batched_tokens": "65536",
        "budget_full_prefill_reqs": "8",
        "initial_prefill_reqs": "3" if "tp4" in name else "9",
        "initial_prefill_tokens": str(tokens),
        "representative_step_ms": "650.457530" if "tp4" in name else "1100.792278",
        "context_non_attention_ms": "650.457530" if "tp4" in name else "1100.792278",
        "context_attention_ms": "94.741403" if "tp4" in name else "130.735166",
        "moe_query_path": "moe_perf_lookup" if "tp4" in name else "phase397v_sol_out_of_coverage",
        "ep_query_path": "ep8_alltoall_fallback",
        "capacity_classification": "dirty_capacity_conflict" if "tp4" in name else "reference_dirty_capacity",
    }


def test_phase421_builds_wiring_verdict(tmp_path):
    before = tmp_path / "before.csv"
    after = tmp_path / "after.csv"
    trace = tmp_path / "trace.csv"
    _write_csv(
        before,
        phase421.VALIDATION_FIELDS,
        [
            _validation_row(phase421.TP8_BT_SCENARIO, sim=116.324618, error=1.190376),
            _validation_row(phase421.DP2_BT_SCENARIO, sim=20.725982, error=7.524468),
        ],
    )
    _write_csv(
        after,
        phase421.VALIDATION_FIELDS,
        [
            _validation_row(phase421.TP8_BT_SCENARIO, sim=155.069882, error=1.119881),
            _validation_row(phase421.DP2_BT_SCENARIO, sim=27.638441, error=5.642576),
        ],
    )
    _write_csv(
        trace,
        phase421.TRACE_FIELDS,
        [
            _trace_row(phase421.TP8_BT_SCENARIO, tokens=65536, capacity="34.355200"),
            _trace_row(phase421.DP2_BT_SCENARIO, tokens=24000, capacity="2.574400"),
        ],
    )

    rows = phase421.build_phase421_rows(before_csv=before, after_csv=after, trace_csv=trace)
    summary = [row for row in rows if row["row_type"] == "summary"][0]
    tp8_trace = [row for row in rows if row["row_type"] == "trace" and row["scenario"] == phase421.TP8_BT_SCENARIO][0]
    dp2_trace = [row for row in rows if row["row_type"] == "trace" and row["scenario"] == phase421.DP2_BT_SCENARIO][0]

    assert tp8_trace["trace_status"] == "wired_full_budget"
    assert dp2_trace["trace_status"] == "wired_but_capacity_limited"
    assert summary["mechanism_verdict"] == "bt65536_wired_tp8_recovers_dp2_dirty_capacity_remains"
    assert summary["next_phase_target"] == "phase422_resolve_dirty_dp2_bt65536_capacity_before_large_step_cost"
    assert summary["default_readiness"] == "No-Go"


def test_phase421_writer_rejects_default_claim(tmp_path):
    rows = [{field: "" for field in phase421.OUTPUT_FIELDS}]
    rows[0].update(
        {
            "source": phase421.SOURCE,
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
        phase421.write_phase421_csv(tmp_path / "bad.csv", rows)
