import csv
import json

import pytest

from scripts import analyze_phase408_dp_replica_asymmetry as phase408


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _metrics_record(ts, engine0_prompt, engine1_prompt):
    body = "\n".join(
        [
            "# TYPE vllm:prompt_tokens_total counter",
            f'vllm:prompt_tokens_total{{engine="0",model_name="kimi-k2.5"}} {engine0_prompt}',
            f'vllm:prompt_tokens_total{{engine="1",model_name="kimi-k2.5"}} {engine1_prompt}',
            "# TYPE vllm:num_requests_waiting gauge",
            'vllm:num_requests_waiting{engine="0",model_name="kimi-k2.5"} 3',
            'vllm:num_requests_waiting{engine="1",model_name="kimi-k2.5"} 5',
            "# TYPE vllm:kv_cache_usage_perc gauge",
            'vllm:kv_cache_usage_perc{engine="0",model_name="kimi-k2.5"} 0.5',
            'vllm:kv_cache_usage_perc{engine="1",model_name="kimi-k2.5"} 0.6',
        ]
    )
    return {"ts": ts, "status": 200, "body": body}


def _write_alternating_metrics(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        _metrics_record("2026-07-03T00:00:00+00:00", 0, 0),
        _metrics_record("2026-07-03T00:00:01+00:00", 10, 10),
        _metrics_record("2026-07-03T00:00:02+00:00", 10, 10),
        _metrics_record("2026-07-03T00:00:03+00:00", 20, 20),
        _metrics_record("2026-07-03T00:00:04+00:00", 20, 20),
    ]
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def test_phase408_phase_offset_penalty_is_not_phase405_reuse():
    rates0 = [10.0, 0.0, 10.0, 0.0]
    rates1 = [10.0, 0.0, 10.0, 0.0]

    assert phase408.phase_offset_penalty(rates0, rates1, 0) == pytest.approx(1.0)
    assert phase408.phase_offset_penalty(rates0, rates1, 1) == pytest.approx(2.0)

    sweep = phase408.sweep_phase_offsets(
        rates0,
        rates1,
        target_penalty=2.0,
    )
    assert sweep["target_phi_steps"] == 1
    assert sweep["target_phi_penalty"] == pytest.approx(2.0)


def test_phase408_rows_use_phase407_uncoupled_target_and_metrics(tmp_path):
    phase407_csv = tmp_path / "phase407.csv"
    _write_csv(
        phase407_csv,
        [
            {
                "scenario": "K2.5-tp4ep8dp2-8k2k",
                "row_type": "dp2_joint_sim",
                "uncoupled_joint_output_tok_s_gpu": "200.0",
                "real_output_tok_s_gpu": "100.0",
            },
            {
                "scenario": "K2.5-tp4ep8dp2-32k3k",
                "row_type": "dp2_joint_sim",
                "uncoupled_joint_output_tok_s_gpu": "180.0",
                "real_output_tok_s_gpu": "90.0",
            },
            {
                "scenario": "K2.5-tp8ep8-32k3k",
                "row_type": "tp8_early_exit_control",
                "uncoupled_joint_output_tok_s_gpu": "50.0",
                "real_output_tok_s_gpu": "50.0",
            },
        ],
    )
    raw_root = tmp_path / "phase403_dp2_stats"
    for scenario in phase408.DP2_SCENARIOS:
        _write_alternating_metrics(raw_root / scenario / "metrics.jsonl")

    rows = phase408.build_phase408_rows(
        phase407_csv=phase407_csv,
        raw_root=raw_root,
    )
    by_scenario = {row["scenario"]: row for row in rows}

    row = by_scenario["K2.5-tp4ep8dp2-32k3k"]
    assert row["phase405_penalty_read"] == "false"
    assert row["needed_penalty_source"] == "phase407_uncoupled_div_real"
    assert row["needed_penalty"] == "2.000000"
    assert row["phi0_penalty"] == "1.000000"
    assert row["target_phi_penalty"] == "2.000000"
    assert row["phase_adjusted_ratio"] == "1.000000"
    assert row["diagnostic_knob"] == "observed_independent_completion_plus_circular_phase_offset"
    assert row["asymmetry_sufficient"] == "true"
    assert row["mechanism_verdict"] == "replica_asymmetry_sufficient_missing_variable"
    assert row["valid_for_default"] == "false"
    assert row["default_readiness"] == "No-Go"


def test_phase408_writer_rejects_phase405_or_default_claim(tmp_path):
    rows = [{field: "" for field in phase408.CSV_FIELDS}]
    rows[0].update(
        {
            "source": phase408.SOURCE,
            "row_type": "dp2_asymmetry_attribution",
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
        phase408.write_phase408_csv(tmp_path / "bad.csv", rows)

    rows[0]["phase405_penalty_read"] = "false"
    rows[0]["valid_for_default"] = "true"
    with pytest.raises(ValueError, match="valid_for_default"):
        phase408.write_phase408_csv(tmp_path / "bad.csv", rows)
