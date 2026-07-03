import csv
import json

import pytest

from scripts import analyze_phase411_dp_imbalance as phase411


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _metrics_record(ts, prompt0, prompt1, gen0, gen1):
    return json.dumps(
        {
            "ts": ts,
            "status": 200,
            "body": "\n".join(
                [
                    f'vllm:prompt_tokens_total{{engine="0",model_name="kimi"}} {prompt0}',
                    f'vllm:prompt_tokens_total{{engine="1",model_name="kimi"}} {prompt1}',
                    f'vllm:generation_tokens_total{{engine="0",model_name="kimi"}} {gen0}',
                    f'vllm:generation_tokens_total{{engine="1",model_name="kimi"}} {gen1}',
                    'vllm:request_success_total{engine="0",finished_reason="length",model_name="kimi"} 2',
                    'vllm:request_success_total{engine="1",finished_reason="length",model_name="kimi"} 1',
                ]
            ),
        }
    )


def test_phase411_decomposes_tail_and_balanced_penalty(tmp_path):
    raw_root = tmp_path / "phase409_iter_trace"
    scenario = phase411.DEFAULT_SCENARIO
    out_dir = raw_root / scenario
    out_dir.mkdir(parents=True)
    (out_dir / "serve.log").write_text(
        "\n".join(
            [
                "INFO EngineCore_DP0 Iteration(0): 1 context requests, 100 context tokens, 0 generation requests, 0 generation tokens, iteration elapsed time: 100.00 ms",
                "INFO EngineCore_DP1 Iteration(0): 1 context requests, 100 context tokens, 0 generation requests, 0 generation tokens, iteration elapsed time: 100.00 ms",
                "INFO EngineCore_DP0 Iteration(1): 0 context requests, 0 context tokens, 2 generation requests, 2 generation tokens, iteration elapsed time: 100.00 ms",
                "INFO EngineCore_DP1 Iteration(1): 0 context requests, 0 context tokens, 1 generation requests, 1 generation tokens, iteration elapsed time: 100.00 ms",
                "INFO EngineCore_DP0 Iteration(2): 0 context requests, 0 context tokens, 1 generation requests, 1 generation tokens, iteration elapsed time: 100.00 ms",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "metrics.jsonl").write_text(
        "\n".join(
            [
                _metrics_record("2026-07-03T00:00:00+00:00", 0, 0, 0, 0),
                _metrics_record("2026-07-03T00:00:01+00:00", 200, 100, 400, 200),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    phase407_csv = tmp_path / "phase407.csv"
    _write_csv(
        phase407_csv,
        [
            {
                "scenario": scenario,
                "isl": "100",
                "uncoupled_joint_output_tok_s_gpu": "3.0",
                "real_output_tok_s_gpu": "1.0",
            }
        ],
    )

    row = phase411.build_phase411_rows(
        raw_root=raw_root,
        phase407_csv=phase407_csv,
        scenario=scenario,
    )[0]

    assert row["tail_engine"] == "0"
    assert row["tail_wall_ms"] == "100.000000"
    assert row["tail_penalty"] == "1.500000"
    assert row["balanced_penalty"] == "2.000000"
    assert row["product_penalty"] == "3.000000"
    assert row["decomposition_gate"] == "passed"
    assert row["engine0_request_count_est"] == "2.000000"
    assert row["engine1_request_count_est"] == "1.000000"
    assert row["request_count_ratio"] == "2.000000"
    assert row["decode_length_ratio"] == "1.000000"
    assert row["dominant_imbalance"] == "request_count"
    assert row["valid_for_default"] == "false"
    assert row["default_readiness"] == "No-Go"


def test_phase411_writer_rejects_runtime_or_default_claim(tmp_path):
    rows = [{field: "" for field in phase411.CSV_FIELDS}]
    rows[0].update(
        {
            "source": phase411.SOURCE,
            "scenario": phase411.DEFAULT_SCENARIO,
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
        phase411.write_phase411_csv(tmp_path / "bad.csv", rows)

    rows[0]["runtime_modified"] = "false"
    rows[0]["valid_for_default"] = "true"
    with pytest.raises(ValueError, match="valid_for_default"):
        phase411.write_phase411_csv(tmp_path / "bad.csv", rows)
