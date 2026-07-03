import csv

import pytest

from scripts import analyze_phase407_dp_lockstep_joint_sim as phase407


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_phase407_gate_rows_require_independent_joint_penalty(tmp_path):
    phase401_csv = tmp_path / "phase401.csv"
    _write_csv(
        phase401_csv,
        [
            {
                "scenario": "K2.5-tp4ep8dp2-8k2k",
                "num_gpu_blocks": "28631",
                "phase401_sim_output_tok_s_gpu": "270",
            },
            {
                "scenario": "K2.5-tp4ep8dp2-32k3k",
                "num_gpu_blocks": "20050",
                "phase401_sim_output_tok_s_gpu": "90",
            },
            {
                "scenario": "K2.5-tp8ep8-32k3k",
                "num_gpu_blocks": "28825",
                "phase401_sim_output_tok_s_gpu": "39",
            },
        ],
    )
    real_csv = tmp_path / "phase403.csv"
    _write_csv(
        real_csv,
        [
            {
                "scenario": "K2.5-tp4ep8dp2-8k2k",
                "phase403_output_tok_s_gpu": "150",
                "real_running_global_mean": "64",
            },
            {
                "scenario": "K2.5-tp4ep8dp2-32k3k",
                "phase403_output_tok_s_gpu": "50",
                "real_running_global_mean": "15",
            },
        ],
    )
    phase400_csv = tmp_path / "phase400.csv"
    _write_csv(
        phase400_csv,
        [
            {
                "scenario": "K2.5-tp8ep8-32k3k",
                "output_tok_s_gpu": "50",
                "serve_running_reqs_mean": "13",
            }
        ],
    )

    results = {
        ("K2.5-tp4ep8dp2-8k2k", "uncoupled"): phase407.JointSimResult(
            output_tok_s_gpu=270.0,
            avg_decode_reqs_global=96.0,
            peak_decode_reqs_global=112,
            steady_state_iterations=100,
            steady_state_time_ms=10_000.0,
        ),
        ("K2.5-tp4ep8dp2-8k2k", "coupled"): phase407.JointSimResult(
            output_tok_s_gpu=150.0,
            avg_decode_reqs_global=65.0,
            peak_decode_reqs_global=104,
            steady_state_iterations=100,
            steady_state_time_ms=18_000.0,
        ),
        ("K2.5-tp4ep8dp2-32k3k", "uncoupled"): phase407.JointSimResult(
            output_tok_s_gpu=90.0,
            avg_decode_reqs_global=18.0,
            peak_decode_reqs_global=20,
            steady_state_iterations=100,
            steady_state_time_ms=10_000.0,
        ),
        ("K2.5-tp4ep8dp2-32k3k", "coupled"): phase407.JointSimResult(
            output_tok_s_gpu=50.0,
            avg_decode_reqs_global=15.2,
            peak_decode_reqs_global=20,
            steady_state_iterations=100,
            steady_state_time_ms=18_000.0,
        ),
        ("K2.5-tp8ep8-32k3k", "uncoupled"): phase407.JointSimResult(
            output_tok_s_gpu=39.0,
            avg_decode_reqs_global=13.0,
            peak_decode_reqs_global=14,
            steady_state_iterations=100,
            steady_state_time_ms=10_000.0,
        ),
        ("K2.5-tp8ep8-32k3k", "coupled"): phase407.JointSimResult(
            output_tok_s_gpu=39.0,
            avg_decode_reqs_global=13.0,
            peak_decode_reqs_global=14,
            steady_state_iterations=100,
            steady_state_time_ms=10_000.0,
        ),
    }

    rows = phase407.build_phase407_rows_from_results(
        simulation_results=results,
        phase401_csv=phase401_csv,
        phase403_csv=real_csv,
        phase400_csv=phase400_csv,
    )
    by_scenario = {row["scenario"]: row for row in rows}

    dp32 = by_scenario["K2.5-tp4ep8dp2-32k3k"]
    assert dp32["row_type"] == "dp2_joint_sim"
    assert dp32["phase405_penalty_read"] == "false"
    assert dp32["uncoupled_harness_gate"] == "passed"
    assert dp32["coupled_convergence_gate"] == "passed"
    assert dp32["joint_penalty"] == "1.800000"
    assert dp32["penalty_gate"] == "passed"
    assert dp32["occupancy_gate"] == "passed"
    assert dp32["mechanism_verdict"] == "independent_joint_sim_matches_dp2_real"

    tp8 = by_scenario["K2.5-tp8ep8-32k3k"]
    assert tp8["row_type"] == "tp8_early_exit_control"
    assert tp8["tp8_early_exit_gate"] == "passed"
    assert tp8["joint_penalty"] == "1.000000"
    assert tp8["mechanism_verdict"] == "tp8_dp1_joint_sim_no_change"


def test_phase407_writer_rejects_phase405_penalty_or_runtime_change(tmp_path):
    rows = [{field: "" for field in phase407.CSV_FIELDS}]
    rows[0].update(
        {
            "source": phase407.SOURCE,
            "row_type": "dp2_joint_sim",
            "scenario": "K2.5-tp4ep8dp2-32k3k",
            "phase405_penalty_read": "true",
            "gpu_allowed": "false",
            "ssh_allowed": "false",
            "runtime_modified": "false",
            "perf_database": "false",
            "valid_for_default": "false",
            "diagnostic_only": "true",
            "default_readiness": "No-Go",
        }
    )
    with pytest.raises(ValueError, match="phase405_penalty_read"):
        phase407.write_phase407_csv(tmp_path / "bad.csv", rows)
