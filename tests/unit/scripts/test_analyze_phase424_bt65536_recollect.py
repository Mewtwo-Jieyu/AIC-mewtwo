import csv
import json
from pathlib import Path

import pytest

from scripts import analyze_phase424_bt65536_recollect as phase424


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_raw(root: Path) -> None:
    scenario_dir = root / phase424.SCENARIO
    scenario_dir.mkdir(parents=True)
    (scenario_dir / "meta.json").write_text(
        json.dumps(
            {
                "name": phase424.SCENARIO,
                "phase": "phase424",
                "tp": 4,
                "dp": 2,
                "ep": 8,
                "isl": 8000,
                "osl": 2000,
                "max_num_batched_tokens": 65536,
                "batch_size": 128,
                "world_size": 8,
                "prefix_caching": False,
                "prompt_variant_mode": "rotating",
                "gpu_memory_utilization": 0.8,
                "max_model_len": 131072,
                "max_num_seqs": 256,
            }
        )
    )
    (scenario_dir / "bench_result.json").write_text(
        json.dumps(
            {
                "ok_requests": 128,
                "failed_requests": 0,
                "output_tok_s": 911.281489504342,
                "total_tok_s": 4556.407447521709,
            }
        )
    )
    (scenario_dir / "metrics.jsonl").write_text("{}\n{}\n")
    (scenario_dir / "serve.log").write_text(
        "\n".join(
            [
                "INFO GPU KV cache size: 131,664 tokens",
                "INFO GPU KV cache size: 131,664 tokens",
                "(EngineCore_DP0 pid=1) INFO Iteration(2): 9 context requests, "
                "65533 context tokens, 3 generation requests, 3 generation tokens, "
                "iteration elapsed time: 6057.59 ms",
                "(EngineCore_DP1 pid=2) INFO Iteration(2): 9 context requests, "
                "65535 context tokens, 1 generation requests, 1 generation tokens, "
                "iteration elapsed time: 3911.81 ms",
            ]
        )
    )
    (root / "gpu_compute_apps_after.txt").write_text("")
    (root / "process_residual_after.txt").write_text("")


def _phase422_csv(path: Path) -> None:
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
            "phase422_sim_tok_s_gpu",
            "phase422_error_ratio",
        ],
        [
            {
                "row_type": "config",
                "scenario": phase424.SCENARIO,
                "tp": "4",
                "dp": "2",
                "ep": "8",
                "max_bt": "65536",
                "real_output_tok_s_gpu": "155.952000",
                "phase422_sim_tok_s_gpu": "27.638441",
                "phase422_error_ratio": "5.642576",
            },
            {
                "row_type": "config",
                "scenario": "K2.5-tp8ep8-8k2k",
                "tp": "8",
                "dp": "1",
                "ep": "8",
                "max_bt": "8000",
                "real_output_tok_s_gpu": "133.528000",
                "phase422_sim_tok_s_gpu": "166.971525",
                "phase422_error_ratio": "1.250461",
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
                "scenario": phase424.SCENARIO,
                "tp": "4",
                "dp": "2",
                "ep": "8",
                "max_bt": "65536",
                "real_output_tok_s_gpu": "113.910186",
                "sim_output_tok_s_gpu": "101.066831",
                "error_ratio": "1.127078",
            },
            {
                "scenario": "K2.5-tp8ep8-8k2k",
                "tp": "8",
                "dp": "1",
                "ep": "8",
                "max_bt": "8000",
                "real_output_tok_s_gpu": "133.528000",
                "sim_output_tok_s_gpu": "166.971525",
                "error_ratio": "1.250461",
            },
        ],
    )


def _trace_csv(path: Path) -> None:
    _write_csv(
        path,
        [
            "row_type",
            "scenario",
            "initial_prefill_tokens",
            "initial_prefill_reqs",
            "representative_step_ms",
            "context_non_attention_ms",
            "moe_query_path",
            "ep_query_path",
        ],
        [
            {
                "row_type": "trace",
                "scenario": phase424.SCENARIO,
                "initial_prefill_tokens": "65536",
                "initial_prefill_reqs": "9",
                "representative_step_ms": "1740.707002",
                "context_non_attention_ms": "1740.707002",
                "moe_query_path": "phase397v_sol_out_of_coverage",
                "ep_query_path": "ep8_alltoall_fallback",
            }
        ],
    )


def test_phase424_builds_clean_reference_replacement_verdict(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    phase422_csv = tmp_path / "phase422.csv"
    after_csv = tmp_path / "after.csv"
    trace_csv = tmp_path / "trace.csv"
    _write_raw(raw)
    _phase422_csv(phase422_csv)
    _after_csv(after_csv)
    _trace_csv(trace_csv)

    rows = phase424.build_phase424_rows(
        raw_root=raw,
        phase422_csv=phase422_csv,
        after_csv=after_csv,
        trace_csv=trace_csv,
    )

    summary = next(row for row in rows if row["row_type"] == "summary")
    capacity = next(row for row in rows if row["row_type"] == "capacity")
    trace = next(row for row in rows if row["row_type"] == "trace")
    dp2 = next(row for row in rows if row["row_type"] == "config" and row["scenario"] == phase424.SCENARIO)

    assert capacity["max_model_len"] == "131072"
    assert capacity["kv_cache_tokens"] == "131664"
    assert capacity["num_gpu_blocks"] == "8229"
    assert capacity["real_output_tok_s_gpu"] == "113.910186"
    assert dp2["phase422_error_ratio"] == "5.642576"
    assert dp2["phase424_error_ratio"] == "1.127078"
    assert dp2["classification"] == "reference_replaced_improved"
    assert trace["large_step_status"] == "true_64k_step_observed"
    assert trace["large_step_charge_status"] == "combined_non_attention_above_64k_lower_bound"
    assert summary["mechanism_verdict"] == "dp2_bt65536_reference_replaced_converged"
    assert summary["default_readiness"] == "No-Go"


def test_phase424_writer_rejects_default_claim(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    phase422_csv = tmp_path / "phase422.csv"
    after_csv = tmp_path / "after.csv"
    trace_csv = tmp_path / "trace.csv"
    _write_raw(raw)
    _phase422_csv(phase422_csv)
    _after_csv(after_csv)
    _trace_csv(trace_csv)
    rows = phase424.build_phase424_rows(
        raw_root=raw,
        phase422_csv=phase422_csv,
        after_csv=after_csv,
        trace_csv=trace_csv,
    )
    rows[0]["valid_for_default"] = "true"

    with pytest.raises(ValueError, match="valid_for_default"):
        phase424.write_phase424_csv(tmp_path / "bad.csv", rows)
