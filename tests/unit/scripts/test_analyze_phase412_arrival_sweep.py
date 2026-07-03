import csv
import gzip
import json

import pytest

from scripts import analyze_phase412_arrival_sweep as phase412


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _metric_record(ts, prompt0, prompt1, gen0, gen1, success0, success1):
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
                    f'vllm:request_success_total{{engine="0",finished_reason="length",model_name="kimi"}} {success0}',
                    f'vllm:request_success_total{{engine="1",finished_reason="length",model_name="kimi"}} {success1}',
                ]
            ),
        }
    )


def _write_artifact(root, scenario, prompts, prompt0, prompt1, gen0, gen1, success0, success1, *, gzip_metrics=False):
    out_dir = root / scenario
    out_dir.mkdir(parents=True)
    (out_dir / "serve.log").write_text(
        "\n".join(
            [
                "INFO EngineCore_DP0 Iteration(0): 1 context requests, 32000 context tokens, 0 generation requests, 0 generation tokens, iteration elapsed time: 100.00 ms",
                "INFO EngineCore_DP1 Iteration(0): 1 context requests, 32000 context tokens, 0 generation requests, 0 generation tokens, iteration elapsed time: 100.00 ms",
                "INFO EngineCore_DP0 Iteration(1): 0 context requests, 0 context tokens, 2 generation requests, 2 generation tokens, iteration elapsed time: 20.00 ms",
                "INFO EngineCore_DP1 Iteration(1): 0 context requests, 0 context tokens, 2 generation requests, 2 generation tokens, iteration elapsed time: 20.00 ms",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    metrics_text = "\n".join(
        [
            _metric_record("2026-07-03T00:00:00+00:00", 0, 0, 0, 0, 0, 0),
            _metric_record("2026-07-03T00:00:10+00:00", prompt0, prompt1, gen0, gen1, success0, success1),
        ]
    ) + "\n"
    if gzip_metrics:
        with gzip.open(out_dir / "metrics.jsonl.gz", "wt", encoding="utf-8") as f:
            f.write(metrics_text)
    else:
        (out_dir / "metrics.jsonl").write_text(metrics_text, encoding="utf-8")
    (out_dir / "bench_result.json").write_text(
        json.dumps(
            {
                "num_prompts": prompts,
                "output_tokens_per_second_per_gpu": 10.0,
            }
        ),
        encoding="utf-8",
    )
    (out_dir / "meta.json").write_text(
        json.dumps({"phase": "phase412", "bench_num_prompts": prompts}),
        encoding="utf-8",
    )


def test_phase412_builds_monotonic_arrival_trend(tmp_path):
    baseline_root = tmp_path / "phase409_iter_trace"
    sweep_root = tmp_path / "phase412_arrival_sweep"
    scenario = phase412.DEFAULT_SCENARIO
    _write_artifact(
        baseline_root,
        scenario,
        128,
        prompt0=32000 * 75,
        prompt1=32000 * 53,
        gen0=3000 * 75,
        gen1=3000 * 53,
        success0=75,
        success1=53,
    )
    _write_artifact(
        sweep_root,
        scenario,
        512,
        prompt0=32000 * 258,
        prompt1=32000 * 254,
        gen0=3000 * 258,
        gen1=3000 * 254,
        success0=258,
        success1=254,
        gzip_metrics=True,
    )
    phase407_csv = tmp_path / "phase407.csv"
    _write_csv(
        phase407_csv,
        [
            {
                "scenario": scenario,
                "uncoupled_joint_output_tok_s_gpu": "18.0",
                "real_output_tok_s_gpu": "10.0",
            }
        ],
    )

    rows = phase412.build_phase412_rows(
        baseline_root=baseline_root,
        sweep_root=sweep_root,
        phase407_csv=phase407_csv,
        scenario=scenario,
    )

    assert [row["num_prompts"] for row in rows if row["row_type"] == "sample"] == ["128", "512"]
    sample512 = [row for row in rows if row["row_type"] == "sample" and row["num_prompts"] == "512"][0]
    trend = [row for row in rows if row["row_type"] == "trend"][0]
    assert sample512["request_count_ratio"] == "1.015748"
    assert sample512["overall_penalty"] == "1.800000"
    assert trend["request_imbalance_monotonic_down"] == "true"
    assert trend["penalty_monotonic_down"] == "true"
    assert trend["mechanism_verdict"] == "arrival_sweep_partial_collapse"
    assert trend["phase405_penalty_read"] == "false"
    assert trend["valid_for_default"] == "false"
    assert trend["default_readiness"] == "No-Go"


def test_phase412_writer_rejects_runtime_or_default_claim(tmp_path):
    rows = [{field: "" for field in phase412.CSV_FIELDS}]
    rows[0].update(
        {
            "source": phase412.SOURCE,
            "row_type": "trend",
            "scenario": phase412.DEFAULT_SCENARIO,
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
        phase412.write_phase412_csv(tmp_path / "bad.csv", rows)

    rows[0]["runtime_modified"] = "false"
    rows[0]["valid_for_default"] = "true"
    with pytest.raises(ValueError, match="valid_for_default"):
        phase412.write_phase412_csv(tmp_path / "bad.csv", rows)
