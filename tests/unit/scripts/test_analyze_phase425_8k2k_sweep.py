import csv
import gzip
import json

from scripts import analyze_phase425_8k2k_sweep as phase425


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


def _write_artifact(
    root,
    scenario,
    prompts,
    *,
    bench_tput,
    prompt0,
    prompt1,
    gen0,
    gen1,
    success0,
    success1,
    gzip_metrics=False,
):
    out_dir = root / scenario
    out_dir.mkdir(parents=True)
    metrics_text = "\n".join(
        [
            _metric_record("2026-07-06T00:00:00+00:00", 0, 0, 0, 0, 0, 0),
            _metric_record("2026-07-06T00:00:10+00:00", prompt0, prompt1, gen0, gen1, success0, success1),
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
                "max_concurrency": 128,
                "output_tokens_per_second_per_gpu": bench_tput,
            }
        ),
        encoding="utf-8",
    )
    (out_dir / "meta.json").write_text(
        json.dumps({"phase": "phase425", "bench_num_prompts": prompts}),
        encoding="utf-8",
    )


def test_phase425_builds_8k2k_arrival_artifact_split(tmp_path):
    scenario = phase425.DEFAULT_SCENARIO
    baseline_root = tmp_path / "phase403_dp2_stats"
    sweep_root = tmp_path / "phase425_8k2k_sweep"
    _write_artifact(
        baseline_root,
        scenario,
        128,
        bench_tput=10.0,
        prompt0=8000 * 96,
        prompt1=8000 * 32,
        gen0=2000 * 96,
        gen1=2000 * 32,
        success0=96,
        success1=32,
    )
    _write_artifact(
        sweep_root,
        scenario,
        512,
        bench_tput=12.5,
        prompt0=8000 * 256,
        prompt1=8000 * 256,
        gen0=800,
        gen1=800,
        success0=256,
        success1=256,
        gzip_metrics=True,
    )
    phase407_csv = tmp_path / "phase407.csv"
    _write_csv(
        phase407_csv,
        [
            {
                "scenario": scenario,
                "isl": "8000",
                "uncoupled_joint_output_tok_s_gpu": "25.53",
                "real_output_tok_s_gpu": "10.0",
            }
        ],
    )
    phase412_csv = tmp_path / "phase412.csv"
    _write_csv(
        phase412_csv,
        [
            {
                "source": "phase412_arrival_sweep",
                "row_type": "trend",
                "scenario": "K2.5-tp4ep8dp2-32k3k",
                "request_count_ratio": "1.000000",
                "overall_penalty": "1.740990",
                "steady_penalty": "1.357625",
                "mechanism_verdict": "arrival_sweep_partial_collapse",
            }
        ],
    )
    validation_csv = tmp_path / "phase424.csv"
    _write_csv(
        validation_csv,
        [
            {
                "row_type": "config",
                "scenario": scenario,
                "real_output_tok_s_gpu": "10.0",
                "phase424_sim_tok_s_gpu": "30.0",
            }
        ],
    )

    rows = phase425.build_phase425_rows(
        baseline_root=baseline_root,
        sweep_root=sweep_root,
        phase407_csv=phase407_csv,
        phase412_csv=phase412_csv,
        validation_csv=validation_csv,
        scenario=scenario,
    )

    assert [row["num_prompts"] for row in rows if row["row_type"] == "sample"] == ["128", "512"]
    trend = [row for row in rows if row["row_type"] == "trend"][0]
    assert trend["source"] == phase425.SOURCE
    assert trend["mechanism_verdict"] == "8k2k_burst_artifact_significant"
    assert trend["phase426_target"] == "decide_validation_arrival_mode_or_steady_gap_modeling"
    assert trend["request_imbalance_monotonic_down"] == "true"
    assert trend["baseline_overall_penalty"] == "2.553000"
    assert trend["last_steady_penalty"] == "1.276500"
    assert trend["burst_artifact_multiplier"] == "2.000000"
    assert trend["validation_baseline_penalty"] == "3.000000"
    assert trend["validation_last_overall_penalty"] == "2.400000"
    assert trend["validation_last_steady_penalty"] == "1.500000"
    assert trend["validation_burst_artifact_multiplier"] == "2.000000"
    assert trend["phase412_32k_steady_penalty"] == "1.357625"
    assert trend["valid_for_default"] == "false"
    assert trend["default_readiness"] == "No-Go"


def test_phase425_writer_rejects_runtime_or_default_claim(tmp_path):
    rows = [{field: "" for field in phase425.CSV_FIELDS}]
    rows[0].update(
        {
            "source": phase425.SOURCE,
            "row_type": "trend",
            "scenario": phase425.DEFAULT_SCENARIO,
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
    try:
        phase425.write_phase425_csv(tmp_path / "bad.csv", rows)
        raise AssertionError("runtime_modified=true must be rejected")
    except ValueError as exc:
        assert "runtime_modified" in str(exc)

    rows[0]["runtime_modified"] = "false"
    rows[0]["valid_for_default"] = "true"
    try:
        phase425.write_phase425_csv(tmp_path / "bad.csv", rows)
        raise AssertionError("valid_for_default=true must be rejected")
    except ValueError as exc:
        assert "valid_for_default" in str(exc)
