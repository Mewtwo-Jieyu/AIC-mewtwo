import csv
import gzip
import json

import pytest

from scripts import analyze_phase426_8k2k_steady_decompose as phase426


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _metric_record(ts, prompt0, prompt1, gen0, gen1):
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
                ]
            ),
        }
    )


def _write_phase426_artifact(root, scenario):
    artifact_dir = root / scenario
    artifact_dir.mkdir(parents=True)
    serve_log = "\n".join(
        [
            "INFO EngineCore_DP0 Iteration(0): 1 context requests, 100 context tokens, 0 generation requests, 0 generation tokens, iteration elapsed time: 5.00 ms",
            "INFO EngineCore_DP1 Iteration(0): 0 context requests, 0 context tokens, 4 generation requests, 4 generation tokens, iteration elapsed time: 5.00 ms",
            "INFO EngineCore_DP0 Iteration(1): 0 context requests, 0 context tokens, 4 generation requests, 4 generation tokens, iteration elapsed time: 20.00 ms",
            "INFO EngineCore_DP1 Iteration(1): 0 context requests, 0 context tokens, 4 generation requests, 4 generation tokens, iteration elapsed time: 20.00 ms",
            "INFO EngineCore_DP0 Iteration(2): 0 context requests, 0 context tokens, 8 generation requests, 8 generation tokens, iteration elapsed time: 24.00 ms",
            "INFO EngineCore_DP1 Iteration(2): 0 context requests, 0 context tokens, 8 generation requests, 8 generation tokens, iteration elapsed time: 24.00 ms",
        ]
    ) + "\n"
    with gzip.open(artifact_dir / "serve.log.gz", "wt", encoding="utf-8") as f:
        f.write(serve_log)

    metrics = "\n".join(
        [
            _metric_record("2026-07-03T00:00:00+00:00", 0, 0, 0, 0),
            _metric_record("2026-07-03T00:00:00.005000+00:00", 100, 0, 0, 4),
            _metric_record("2026-07-03T00:00:00.025000+00:00", 100, 0, 4, 8),
            _metric_record("2026-07-03T00:00:00.049000+00:00", 100, 0, 12, 16),
        ]
    ) + "\n"
    with gzip.open(artifact_dir / "metrics.jsonl.gz", "wt", encoding="utf-8") as f:
        f.write(metrics)
    (artifact_dir / "bench_result.json").write_text(
        json.dumps({"output_tokens_per_second_per_gpu": 10.0, "num_prompts": 512}),
        encoding="utf-8",
    )
    return artifact_dir


def test_phase426_identifies_fixed_decode_gap_and_ep_floor(tmp_path):
    scenario = phase426.DEFAULT_SCENARIO
    sweep_root = tmp_path / "phase425_8k2k_sweep"
    artifact_dir = _write_phase426_artifact(sweep_root, scenario)
    phase425_csv = tmp_path / "phase425.csv"
    _write_csv(
        phase425_csv,
        [
            {
                "source": "phase425_8k2k_sweep",
                "row_type": "trend",
                "scenario": scenario,
                "artifact_dir": str(artifact_dir),
                "uncoupled_output_tok_s_gpu": "107.142857",
                "steady_output_tok_s_gpu": "71.428571",
                "steady_penalty": "1.5",
                "validation_last_steady_penalty": "1.96",
                "validation_baseline_penalty": "2.55",
            }
        ],
    )
    phase413_csv = tmp_path / "phase413.csv"
    _write_csv(
        phase413_csv,
        [
            {
                "source": "phase413_steady_decompose",
                "scenario": "K2.5-tp4ep8dp2-32k3k",
                "steady_target_penalty": "1.357625",
                "prefill_attribution_share": "0.647177",
                "peer_stall_attribution_share": "0.000299",
                "decode_gap_attribution_share": "0.352524",
                "mechanism_verdict": "prefill_occupancy_dominates",
            }
        ],
    )

    row = phase426.build_phase426_rows(
        sweep_root=sweep_root,
        phase425_csv=phase425_csv,
        phase413_csv=phase413_csv,
        scenario=scenario,
        sim_decode_ms_by_batch={4: 13.0, 8: 17.0},
    )[0]

    assert row["source"] == phase426.SOURCE
    assert row["steady_consistency_gate"] == "passed"
    assert row["decode_gap_shape"] == "fixed_latency_floor"
    assert row["ep_floor_covers_decode_gap"] == "true"
    assert row["mechanism_verdict"] == "decode_fixed_gap_ep_latency_floor"
    assert row["phase413_32k_verdict"] == "prefill_occupancy_dominates"
    assert row["phase405_penalty_read"] == "false"
    assert row["gpu_allowed"] == "false"
    assert row["ssh_allowed"] == "false"
    assert row["runtime_modified"] == "false"
    assert row["valid_for_default"] == "false"
    assert row["default_readiness"] == "No-Go"


def test_phase426_writer_rejects_gpu_or_default_claim(tmp_path):
    rows = [{field: "" for field in phase426.CSV_FIELDS}]
    rows[0].update(
        {
            "source": phase426.SOURCE,
            "row_type": "summary",
            "scenario": phase426.DEFAULT_SCENARIO,
            "phase405_penalty_read": "false",
            "gpu_allowed": "true",
            "ssh_allowed": "false",
            "runtime_modified": "false",
            "perf_database": "false",
            "valid_for_default": "false",
            "diagnostic_only": "true",
            "default_readiness": "No-Go",
        }
    )
    with pytest.raises(ValueError, match="GPU/SSH"):
        phase426.write_phase426_csv(tmp_path / "bad.csv", rows)

    rows[0]["gpu_allowed"] = "false"
    rows[0]["valid_for_default"] = "true"
    with pytest.raises(ValueError, match="valid_for_default"):
        phase426.write_phase426_csv(tmp_path / "bad.csv", rows)
