import csv

import pytest

from scripts import analyze_phase406_dp_lockstep_model as phase406


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_dp2_log(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "non-default args: {'data_parallel_size': 2, 'enable_prefix_caching': False, 'enable_chunked_prefill': True}",
                "Initializing a V1 LLM engine (v0.19.0) with config: enforce_eager=False, enable_chunked_prefill=True, data_parallel_size=2, compilation_config={'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>}",
                "(Worker_DP0_TP0_EP0 pid=1) ready",
                "(Worker_DP1_TP0_EP4 pid=2) ready",
                "Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): done",
                "Capturing CUDA graphs (decode, FULL): done",
            ]
        ),
        encoding="utf-8",
    )


def test_phase406_models_dp2_lockstep_and_keeps_tp8_unchanged(tmp_path):
    phase405_csv = tmp_path / "phase405.csv"
    _write_csv(
        phase405_csv,
        [
            {
                "source": "phase405_dp2_duty_cycle",
                "row_type": "dp2_duty_attribution",
                "scenario": "K2.5-tp4ep8dp2-8k2k",
                "tp": "4",
                "dp": "2",
                "ep": "8",
                "isl": "8000",
                "osl": "2000",
                "max_num_batched_tokens": "8000",
                "phase401_sim_output_tok_s_gpu": "270",
                "real_output_tok_s_gpu": "150",
                "observed_tput_ratio": "1.800000",
                "observed_direction": "sim_over_predicts_throughput",
                "sim_avg_decode_reqs_global": "96",
                "sim_peak_decode_reqs_global": "112",
                "real_running_global_mean": "64",
                "real_running_global_max": "104",
                "raw_batch_occupancy_ratio": "1.500000",
                "real_active_decode_ms_per_iter": "40",
                "sim_decode_ms_per_iter": "32",
                "active_iter_ratio": "1.250000",
                "duty_cycle_ratio": "1.400000",
                "reconstructed_tput_ratio": "1.750000",
                "phase406_target": "dp_prefill_occupancy_or_lockstep_duty_model",
            },
            {
                "source": "phase405_dp2_duty_cycle",
                "row_type": "dp2_duty_attribution",
                "scenario": "K2.5-tp4ep8dp2-32k3k",
                "tp": "4",
                "dp": "2",
                "ep": "8",
                "isl": "32000",
                "osl": "3000",
                "max_num_batched_tokens": "32000",
                "phase401_sim_output_tok_s_gpu": "90",
                "real_output_tok_s_gpu": "50",
                "observed_tput_ratio": "1.800000",
                "observed_direction": "sim_over_predicts_throughput",
                "sim_avg_decode_reqs_global": "18",
                "sim_peak_decode_reqs_global": "20",
                "real_running_global_mean": "15",
                "real_running_global_max": "20",
                "raw_batch_occupancy_ratio": "1.200000",
                "real_active_decode_ms_per_iter": "25",
                "sim_decode_ms_per_iter": "20",
                "active_iter_ratio": "1.250000",
                "duty_cycle_ratio": "1.440000",
                "reconstructed_tput_ratio": "1.800000",
                "phase406_target": "dp_prefill_occupancy_or_lockstep_duty_model",
            },
            {
                "source": "phase405_dp2_duty_cycle",
                "row_type": "tp8_control",
                "scenario": "K2.5-tp8ep8-32k3k",
                "tp": "8",
                "dp": "1",
                "ep": "8",
                "isl": "32000",
                "osl": "3000",
                "max_num_batched_tokens": "32000",
                "phase401_sim_output_tok_s_gpu": "39",
                "real_output_tok_s_gpu": "50",
                "observed_tput_ratio": "0.780000",
                "observed_direction": "sim_under_predicts_throughput",
                "sim_avg_decode_reqs_global": "13",
                "sim_peak_decode_reqs_global": "14",
                "real_running_global_mean": "13",
                "real_running_global_max": "14",
                "raw_batch_occupancy_ratio": "1.000000",
                "real_active_decode_ms_per_iter": "",
                "sim_decode_ms_per_iter": "34",
                "active_iter_ratio": "",
                "duty_cycle_ratio": "",
                "reconstructed_tput_ratio": "0.920000",
                "phase406_target": "phase406_dp_only_if_dp2_duty_gap_persists",
            },
        ],
    )
    raw_root = tmp_path / "phase403_dp2_stats"
    _write_dp2_log(raw_root / "K2.5-tp4ep8dp2-8k2k" / "serve.log")
    _write_dp2_log(raw_root / "K2.5-tp4ep8dp2-32k3k" / "serve.log")

    rows = phase406.build_phase406_rows(
        phase405_csv=phase405_csv,
        phase403_raw_root=raw_root,
    )
    by_scenario = {row["scenario"]: row for row in rows}

    dp32 = by_scenario["K2.5-tp4ep8dp2-32k3k"]
    assert dp32["row_type"] == "dp2_lockstep_paper_model"
    assert dp32["paper_model_rule"] == "two_replica_step_tokens_max"
    assert dp32["lockstep_penalty"] == "1.800000"
    assert dp32["coupled_sim_output_tok_s_gpu"] == "50.000000"
    assert dp32["coupled_ratio"] == "1.000000"
    assert float(dp32["coupled_error_pct"]) < 10.0
    assert dp32["serve_prereq_status"] == "passed"
    assert dp32["mechanism_verdict"] == "dp_pad_to_max_lockstep_model_matches_real"
    assert dp32["phase407_target"] == "runtime_dp_lockstep_coupling"

    tp8 = by_scenario["K2.5-tp8ep8-32k3k"]
    assert tp8["row_type"] == "tp8_early_exit_control"
    assert tp8["lockstep_penalty"] == "1.000000"
    assert tp8["coupled_sim_output_tok_s_gpu"] == "39.000000"
    assert tp8["mechanism_verdict"] == "tp8_dp1_no_lockstep_change"


def test_phase406_writer_rejects_runtime_or_default_upgrade(tmp_path):
    rows = [{field: "" for field in phase406.CSV_FIELDS}]
    rows[0].update(
        {
            "source": phase406.SOURCE,
            "row_type": "dp2_lockstep_paper_model",
            "scenario": "K2.5-tp4ep8dp2-32k3k",
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
        phase406.write_phase406_csv(tmp_path / "bad.csv", rows)
