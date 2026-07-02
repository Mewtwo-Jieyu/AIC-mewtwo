import csv

import pytest

from scripts import analyze_phase405_dp2_duty_cycle as phase405


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_phase405_decomposes_dp2_gap_into_duty_and_active_iter(tmp_path):
    phase401_csv = tmp_path / "phase401.csv"
    _write_csv(
        phase401_csv,
        [
            {
                "scenario": "K2.5-tp4ep8dp2-8k2k",
                "tp": "4",
                "dp": "2",
                "ep": "8",
                "isl": "8000",
                "osl": "2000",
                "max_num_batched_tokens": "8000",
                "phase401_sim_output_tok_s_gpu": "270",
                "phase401_direction": "sim_over_predicts_throughput",
                "sim_avg_prefill_reqs_per_iter": "0.1",
                "sim_avg_decode_reqs_per_iter": "24",
                "sim_peak_decode_reqs_per_iter": "28",
            },
            {
                "scenario": "K2.5-tp4ep8dp2-32k3k",
                "tp": "4",
                "dp": "2",
                "ep": "8",
                "isl": "32000",
                "osl": "3000",
                "max_num_batched_tokens": "32000",
                "phase401_sim_output_tok_s_gpu": "90",
                "phase401_direction": "sim_over_predicts_throughput",
                "sim_avg_prefill_reqs_per_iter": "0.01",
                "sim_avg_decode_reqs_per_iter": "9",
                "sim_peak_decode_reqs_per_iter": "10",
            },
            {
                "scenario": "K2.5-tp8ep8-32k3k",
                "tp": "8",
                "dp": "1",
                "ep": "8",
                "isl": "32000",
                "osl": "3000",
                "max_num_batched_tokens": "32000",
                "phase401_sim_output_tok_s_gpu": "39",
                "phase401_direction": "sim_under_predicts_throughput",
                "sim_avg_prefill_reqs_per_iter": "0.01",
                "sim_avg_decode_reqs_per_iter": "13",
                "sim_peak_decode_reqs_per_iter": "14",
            },
        ],
    )
    phase403_csv = tmp_path / "phase403.csv"
    _write_csv(
        phase403_csv,
        [
            {
                "scenario": "K2.5-tp4ep8dp2-8k2k",
                "phase403_output_tok_s_global": "1080",
                "phase403_output_tok_s_gpu": "135",
                "real_running_global_mean": "60",
                "real_running_global_p50": "64",
                "real_running_global_max": "104",
            },
            {
                "scenario": "K2.5-tp4ep8dp2-32k3k",
                "phase403_output_tok_s_global": "360",
                "phase403_output_tok_s_gpu": "45",
                "real_running_global_mean": "15",
                "real_running_global_p50": "18",
                "real_running_global_max": "20",
            },
        ],
    )
    phase404_csv = tmp_path / "phase404.csv"
    _write_csv(
        phase404_csv,
        [
            {
                "scenario": "K2.5-tp4ep8dp2-8k2k",
                "real_active_decode_ms_per_iter": "40",
                "sim_decode_ms_per_iter": "30",
                "active_decode_residual_ms": "10",
                "aggregate_implied_residual_ms": "50",
            },
            {
                "scenario": "K2.5-tp4ep8dp2-32k3k",
                "real_active_decode_ms_per_iter": "24",
                "sim_decode_ms_per_iter": "20",
                "active_decode_residual_ms": "4",
                "aggregate_implied_residual_ms": "30",
            },
            {
                "scenario": "K2.5-tp8ep8-32k3k",
                "real_active_decode_ms_per_iter": "",
                "sim_decode_ms_per_iter": "35",
                "active_decode_residual_ms": "",
                "aggregate_implied_residual_ms": "1",
            },
        ],
    )
    phase400_csv = tmp_path / "phase400.csv"
    _write_csv(
        phase400_csv,
        [
            {
                "scenario": "K2.5-tp8ep8-32k3k",
                "output_tok_s_global": "400",
                "output_tok_s_gpu": "50",
                "serve_running_reqs_mean": "13",
                "serve_running_reqs_max": "14",
            }
        ],
    )

    rows = phase405.build_phase405_rows(
        phase401_csv=phase401_csv,
        phase403_csv=phase403_csv,
        phase404_csv=phase404_csv,
        phase400_csv=phase400_csv,
        running_stats={
            "K2.5-tp4ep8dp2-8k2k": phase405.RunningStats(60, 64, 90, 104),
            "K2.5-tp4ep8dp2-32k3k": phase405.RunningStats(15, 18, 19, 20),
        },
    )
    by_scenario = {row["scenario"]: row for row in rows}

    dp32 = by_scenario["K2.5-tp4ep8dp2-32k3k"]
    assert dp32["row_type"] == "dp2_duty_attribution"
    assert dp32["observed_tput_ratio"] == "2.000000"
    assert dp32["active_iter_ratio"] == "1.200000"
    assert dp32["duty_cycle_ratio"] == "1.736111"
    assert float(dp32["reconstruction_error_pct"]) < 5.0
    assert dp32["attribution_verdict"] == "duty_cycle_plus_active_iter_reconstructs_dp2_gap"
    assert dp32["phase406_target"] == "dp_prefill_occupancy_or_lockstep_duty_model"

    tp8 = by_scenario["K2.5-tp8ep8-32k3k"]
    assert tp8["row_type"] == "tp8_control"
    assert tp8["attribution_verdict"] == "tp8_control_no_dp_duty_gap"


def test_phase405_writer_rejects_default_upgrade(tmp_path):
    rows = [{field: "" for field in phase405.CSV_FIELDS}]
    rows[0].update(
        {
            "source": phase405.SOURCE,
            "row_type": "dp2_duty_attribution",
            "scenario": "K2.5-tp4ep8dp2-32k3k",
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
        phase405.write_phase405_csv(tmp_path / "bad.csv", rows)
