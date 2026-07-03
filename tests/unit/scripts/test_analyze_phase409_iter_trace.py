import csv
import json

import pytest

from scripts import analyze_phase409_iter_trace as phase409


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_parse_iteration_lines_and_direct_lockstep_penalty(tmp_path):
    serve_log = tmp_path / "serve.log"
    serve_log.write_text(
        "\n".join(
            [
                "INFO EngineCore_DP0 Iteration(0): 1 context requests, 100 context tokens, 10 generation requests, 10 generation tokens, iteration elapsed time: 50.00 ms",
                "INFO EngineCore_DP1 Iteration(0): 0 context requests, 0 context tokens, 10 generation requests, 10 generation tokens, iteration elapsed time: 50.00 ms",
                "INFO EngineCore_DP0 Iteration(1): 0 context requests, 0 context tokens, 10 generation requests, 10 generation tokens, iteration elapsed time: 20.00 ms",
                "INFO EngineCore_DP1 Iteration(1): 1 context requests, 100 context tokens, 10 generation requests, 10 generation tokens, iteration elapsed time: 20.00 ms",
                "| Unpadded Tokens | Padded Tokens | Num Paddings | Runtime Mode | Count |",
                "| 110             | 110           | 0            | FULL         | 2     |",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    steps = phase409.parse_iteration_steps(serve_log)
    assert len(steps) == 4
    assert steps[0].engine == "0"
    assert steps[0].total_tokens == 110

    pairs = phase409.align_iteration_pairs(steps)
    summary = phase409.summarize_iteration_pairs(pairs)
    assert summary["aligned_steps"] == 2
    assert summary["total_unpadded_tokens"] == pytest.approx(240.0)
    assert summary["total_padded_tokens"] == pytest.approx(440.0)
    assert summary["direct_token_lockstep_penalty"] == pytest.approx(440.0 / 240.0)
    assert summary["co_prefill_decode_step_share"] == pytest.approx(1.0)

    cg_rows = phase409.parse_cudagraph_rows(serve_log)
    assert cg_rows[0]["unpadded_tokens"] == 110
    assert cg_rows[0]["count"] == 2


def test_phase409_rows_are_measurement_only_and_use_no_phase405(tmp_path):
    raw_root = tmp_path / "phase409_iter_trace"
    scenario = phase409.DEFAULT_SCENARIO
    out_dir = raw_root / scenario
    out_dir.mkdir(parents=True)
    (out_dir / "serve.log").write_text(
        "\n".join(
            [
                "INFO EngineCore_DP0 Iteration(0): 1 context requests, 100 context tokens, 10 generation requests, 10 generation tokens, iteration elapsed time: 50.00 ms",
                "INFO EngineCore_DP1 Iteration(0): 0 context requests, 0 context tokens, 10 generation requests, 10 generation tokens, iteration elapsed time: 50.00 ms",
                "INFO EngineCore_DP0 Iteration(1): 0 context requests, 0 generation tokens, 10 generation requests, 10 generation tokens, iteration elapsed time: 20.00 ms",
                "INFO EngineCore_DP1 Iteration(1): 1 context requests, 100 context tokens, 10 generation requests, 10 generation tokens, iteration elapsed time: 20.00 ms",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "bench_result.json").write_text(
        json.dumps({"output_tokens_per_second_per_gpu": 0.0714285714}),
        encoding="utf-8",
    )
    (out_dir / "metrics.jsonl").write_text("", encoding="utf-8")
    (out_dir / "meta.json").write_text(
        json.dumps({"scenario": scenario, "tp": 4, "dp": 2, "ep": 8}),
        encoding="utf-8",
    )
    phase407_csv = tmp_path / "phase407.csv"
    _write_csv(
        phase407_csv,
        [
            {
                "scenario": scenario,
                "uncoupled_joint_output_tok_s_gpu": "0.1349206349",
                "real_output_tok_s_gpu": "0.0714285714",
            }
        ],
    )

    rows = phase409.build_phase409_rows(
        raw_root=raw_root,
        phase407_csv=phase407_csv,
        scenario=scenario,
    )
    row = rows[0]
    assert row["phase405_penalty_read"] == "false"
    assert row["real_phase_direct_penalty_source"] == "serve_log_iteration_details"
    assert row["direct_token_lockstep_penalty"] == "1.833333"
    assert row["needed_penalty"] == "1.888889"
    assert row["penalty_gate"] == "passed"
    assert row["runtime_modified"] == "false"
    assert row["valid_for_default"] == "false"
    assert row["default_readiness"] == "No-Go"


def test_phase409_prefers_metrics_delta_penalty_when_available(tmp_path):
    raw_root = tmp_path / "phase409_iter_trace"
    scenario = phase409.DEFAULT_SCENARIO
    out_dir = raw_root / scenario
    out_dir.mkdir(parents=True)
    (out_dir / "serve.log").write_text(
        "\n".join(
            [
                "INFO EngineCore_DP0 Iteration(0): 1 context requests, 100 context tokens, 10 generation requests, 10 generation tokens, iteration elapsed time: 50.00 ms",
                "INFO EngineCore_DP1 Iteration(0): 0 context requests, 0 context tokens, 10 generation requests, 10 generation tokens, iteration elapsed time: 50.00 ms",
                "INFO EngineCore_DP0 Iteration(1): 0 context requests, 0 context tokens, 10 generation requests, 10 generation tokens, iteration elapsed time: 20.00 ms",
                "INFO EngineCore_DP1 Iteration(1): 1 context requests, 100 context tokens, 10 generation requests, 10 generation tokens, iteration elapsed time: 20.00 ms",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "metrics.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ts": "2026-07-03T00:00:00+00:00",
                        "status": 200,
                        "body": "\n".join(
                            [
                                'vllm:prompt_tokens_total{engine="0",model_name="kimi"} 0.0',
                                'vllm:prompt_tokens_total{engine="1",model_name="kimi"} 0.0',
                                'vllm:generation_tokens_total{engine="0",model_name="kimi"} 0.0',
                                'vllm:generation_tokens_total{engine="1",model_name="kimi"} 0.0',
                            ]
                        ),
                    }
                ),
                json.dumps(
                    {
                        "ts": "2026-07-03T00:00:01+00:00",
                        "status": 200,
                        "body": "\n".join(
                            [
                                'vllm:prompt_tokens_total{engine="0",model_name="kimi"} 100.0',
                                'vllm:prompt_tokens_total{engine="1",model_name="kimi"} 0.0',
                                'vllm:generation_tokens_total{engine="0",model_name="kimi"} 10.0',
                                'vllm:generation_tokens_total{engine="1",model_name="kimi"} 10.0',
                            ]
                        ),
                    }
                ),
                json.dumps(
                    {
                        "ts": "2026-07-03T00:00:02+00:00",
                        "status": 200,
                        "body": "\n".join(
                            [
                                'vllm:prompt_tokens_total{engine="0",model_name="kimi"} 100.0',
                                'vllm:prompt_tokens_total{engine="1",model_name="kimi"} 100.0',
                                'vllm:generation_tokens_total{engine="0",model_name="kimi"} 20.0',
                                'vllm:generation_tokens_total{engine="1",model_name="kimi"} 20.0',
                            ]
                        ),
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "bench_result.json").write_text(
        json.dumps({"output_tokens_per_second_per_gpu": 2.5}),
        encoding="utf-8",
    )
    (out_dir / "meta.json").write_text(
        json.dumps({"scenario": scenario, "tp": 4, "dp": 2, "ep": 8}),
        encoding="utf-8",
    )
    phase407_csv = tmp_path / "phase407.csv"
    _write_csv(
        phase407_csv,
        [
            {
                "scenario": scenario,
                "uncoupled_joint_output_tok_s_gpu": "4.722222",
                "real_output_tok_s_gpu": "2.5",
            }
        ],
    )

    row = phase409.build_phase409_rows(
        raw_root=raw_root,
        phase407_csv=phase407_csv,
        scenario=scenario,
    )[0]

    assert row["real_phase_direct_penalty_source"] == "metrics_prompt_generation_delta_lower_bound"
    assert row["direct_token_lockstep_penalty"] == "1.833333"
    assert row["trace_output_tok_s_gpu"] == "2.500000"
    assert row["trace_fidelity_gate"] == "passed"


def test_phase409_writer_rejects_runtime_or_default_claim(tmp_path):
    rows = [{field: "" for field in phase409.CSV_FIELDS}]
    rows[0].update(
        {
            "source": phase409.SOURCE,
            "scenario": phase409.DEFAULT_SCENARIO,
            "phase405_penalty_read": "false",
            "gpu_allowed": "true",
            "ssh_allowed": "true",
            "runtime_modified": "true",
            "perf_database": "false",
            "valid_for_default": "false",
            "diagnostic_only": "true",
            "default_readiness": "No-Go",
        }
    )
    with pytest.raises(ValueError, match="runtime_modified"):
        phase409.write_phase409_csv(tmp_path / "bad.csv", rows)

    rows[0]["runtime_modified"] = "false"
    rows[0]["valid_for_default"] = "true"
    with pytest.raises(ValueError, match="valid_for_default"):
        phase409.write_phase409_csv(tmp_path / "bad.csv", rows)
