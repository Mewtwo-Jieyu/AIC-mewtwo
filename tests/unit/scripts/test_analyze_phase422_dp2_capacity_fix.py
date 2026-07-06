import csv
from pathlib import Path

import pytest

import scripts.analyze_phase422_dp2_capacity_fix as phase422


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _phase421_csv(path: Path) -> None:
    _write_csv(
        path,
        [
            "row_type",
            "scenario",
            "tp",
            "dp",
            "ep",
            "max_bt",
            "real_output_tok_s_gpu",
            "sim_after_tok_s_gpu",
            "error_ratio_after",
        ],
        [
            {
                "row_type": "config",
                "scenario": "K2.5-tp8ep8-8k2k-bt65536",
                "tp": "8",
                "dp": "1",
                "ep": "8",
                "max_bt": "65536",
                "real_output_tok_s_gpu": "138.470000",
                "sim_after_tok_s_gpu": "155.069882",
                "error_ratio_after": "1.119881",
            },
            {
                "row_type": "config",
                "scenario": "K2.5-tp4ep8dp2-8k2k-bt65536",
                "tp": "4",
                "dp": "2",
                "ep": "8",
                "max_bt": "65536",
                "real_output_tok_s_gpu": "155.952000",
                "sim_after_tok_s_gpu": "27.638441",
                "error_ratio_after": "5.642576",
            },
        ],
    )


def _after_csv(path: Path) -> None:
    _write_csv(
        path,
        [
            "scenario",
            "tp",
            "dp",
            "ep",
            "max_bt",
            "real_output_tok_s_gpu",
            "sim_output_tok_s_gpu",
            "error_ratio",
        ],
        [
            {
                "scenario": "K2.5-tp8ep8-8k2k-bt65536",
                "tp": "8",
                "dp": "1",
                "ep": "8",
                "max_bt": "65536",
                "real_output_tok_s_gpu": "138.470000",
                "sim_output_tok_s_gpu": "155.069882",
                "error_ratio": "1.119881",
            },
            {
                "scenario": "K2.5-tp4ep8dp2-8k2k-bt65536",
                "tp": "4",
                "dp": "2",
                "ep": "8",
                "max_bt": "65536",
                "real_output_tok_s_gpu": "155.952000",
                "sim_output_tok_s_gpu": "27.638441",
                "error_ratio": "5.642576",
            },
        ],
    )


def _trace_csv(path: Path) -> None:
    _write_csv(
        path,
        [
            "row_type",
            "label",
            "scenario",
            "tp",
            "dp",
            "ep",
            "max_bt",
            "real_output_tok_s_gpu",
            "sim_output_tok_s_gpu",
            "error_ratio",
            "num_gpu_blocks",
            "full_sequence_capacity_per_engine",
            "configured_max_num_batched_tokens",
            "budget_full_prefill_reqs",
            "initial_prefill_reqs",
            "initial_prefill_tokens",
            "representative_step_ms",
            "moe_query_path",
            "ep_query_path",
            "capacity_classification",
        ],
        [
            {
                "row_type": "trace",
                "label": "after",
                "scenario": "K2.5-tp8ep8-8k2k-bt65536",
                "tp": "8",
                "dp": "1",
                "ep": "8",
                "max_bt": "65536",
                "real_output_tok_s_gpu": "138.470000",
                "sim_output_tok_s_gpu": "155.069882",
                "error_ratio": "1.119881",
                "num_gpu_blocks": "21472",
                "full_sequence_capacity_per_engine": "34.355200",
                "configured_max_num_batched_tokens": "65536",
                "budget_full_prefill_reqs": "8",
                "initial_prefill_reqs": "9",
                "initial_prefill_tokens": "65536",
                "representative_step_ms": "1100.792278",
                "moe_query_path": "phase397v_sol_out_of_coverage",
                "ep_query_path": "ep8_alltoall_fallback",
                "capacity_classification": "reference_dirty_capacity",
            },
            {
                "row_type": "trace",
                "label": "after",
                "scenario": "K2.5-tp4ep8dp2-8k2k-bt65536",
                "tp": "4",
                "dp": "2",
                "ep": "8",
                "max_bt": "65536",
                "real_output_tok_s_gpu": "155.952000",
                "sim_output_tok_s_gpu": "27.638441",
                "error_ratio": "5.642576",
                "num_gpu_blocks": "1609",
                "full_sequence_capacity_per_engine": "2.574400",
                "configured_max_num_batched_tokens": "65536",
                "budget_full_prefill_reqs": "8",
                "initial_prefill_reqs": "3",
                "initial_prefill_tokens": "24000",
                "representative_step_ms": "650.457530",
                "moe_query_path": "moe_perf_lookup",
                "ep_query_path": "ep8_alltoall_fallback",
                "capacity_classification": "dirty_capacity_conflict",
            },
        ],
    )


def test_phase422_build_marks_dp2_retry5e_capacity_as_dirty_reference(tmp_path: Path) -> None:
    phase421_csv = tmp_path / "phase421.csv"
    after_csv = tmp_path / "after.csv"
    trace_csv = tmp_path / "trace.csv"
    _phase421_csv(phase421_csv)
    _after_csv(after_csv)
    _trace_csv(trace_csv)

    rows = phase422.build_phase422_rows(
        phase421_csv=phase421_csv,
        after_csv=after_csv,
        trace_csv=trace_csv,
    )

    summary = next(row for row in rows if row["row_type"] == "summary")
    dp2_capacity = next(
        row for row in rows
        if row["row_type"] == "capacity" and row["scenario"] == phase422.DP2_BT_SCENARIO
    )
    dp2_trace = next(
        row for row in rows
        if row["row_type"] == "trace" and row["scenario"] == phase422.DP2_BT_SCENARIO
    )

    assert summary["mechanism_verdict"] == "dp2_bt65536_capacity_truth_is_dirty_reference_not_fixable_by_wiring"
    assert dp2_capacity["kv_cache_tokens"] == "25744"
    assert dp2_capacity["num_gpu_blocks"] == "1609"
    assert dp2_capacity["max_num_seqs"] == "128"
    assert dp2_trace["configured_max_num_batched_tokens"] == "65536"
    assert dp2_trace["initial_prefill_reqs"] == "3"
    assert dp2_trace["initial_prefill_tokens"] == "24000"
    assert dp2_trace["trace_status"] == "real_capacity_blocks_64k_step"


def test_phase422_writer_rejects_default_readiness(tmp_path: Path) -> None:
    phase421_csv = tmp_path / "phase421.csv"
    after_csv = tmp_path / "after.csv"
    trace_csv = tmp_path / "trace.csv"
    _phase421_csv(phase421_csv)
    _after_csv(after_csv)
    _trace_csv(trace_csv)
    rows = phase422.build_phase422_rows(
        phase421_csv=phase421_csv,
        after_csv=after_csv,
        trace_csv=trace_csv,
    )
    rows[0]["valid_for_default"] = "true"

    with pytest.raises(ValueError, match="valid_for_default"):
        phase422.write_phase422_csv(tmp_path / "bad.csv", rows)
