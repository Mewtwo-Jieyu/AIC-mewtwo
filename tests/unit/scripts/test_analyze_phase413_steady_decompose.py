import csv
import gzip
import json

import pytest

from scripts import analyze_phase413_steady_decompose as phase413


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


def _write_phase413_artifact(root, scenario):
    artifact_dir = root / scenario
    artifact_dir.mkdir(parents=True)
    serve_log = "\n".join(
        [
            "INFO EngineCore_DP0 Iteration(0): 1 context requests, 100 context tokens, 0 generation requests, 0 generation tokens, iteration elapsed time: 100.00 ms",
            "INFO EngineCore_DP1 Iteration(0): 0 context requests, 0 context tokens, 1 generation requests, 1 generation tokens, iteration elapsed time: 150.00 ms",
            "INFO EngineCore_DP0 Iteration(1): 0 context requests, 0 context tokens, 2 generation requests, 2 generation tokens, iteration elapsed time: 10.00 ms",
            "INFO EngineCore_DP1 Iteration(1): 0 context requests, 0 context tokens, 2 generation requests, 2 generation tokens, iteration elapsed time: 10.00 ms",
            "INFO EngineCore_DP0 Iteration(2): 0 context requests, 0 context tokens, 4 generation requests, 4 generation tokens, iteration elapsed time: 20.00 ms",
            "INFO EngineCore_DP1 Iteration(2): 0 context requests, 0 context tokens, 4 generation requests, 4 generation tokens, iteration elapsed time: 20.00 ms",
        ]
    ) + "\n"
    with gzip.open(artifact_dir / "serve.log.gz", "wt", encoding="utf-8") as f:
        f.write(serve_log)
    metrics = "\n".join(
        [
            _metric_record("2026-07-03T00:00:00+00:00", 0, 0, 0, 0),
            _metric_record("2026-07-03T00:00:00.050000+00:00", 100, 0, 0, 1),
            _metric_record("2026-07-03T00:00:00.100000+00:00", 100, 0, 2, 3),
            _metric_record("2026-07-03T00:00:00.150000+00:00", 100, 0, 6, 7),
        ]
    ) + "\n"
    with gzip.open(artifact_dir / "metrics.jsonl.gz", "wt", encoding="utf-8") as f:
        f.write(metrics)
    (artifact_dir / "bench_result.json").write_text(
        json.dumps({"output_tokens_per_second_per_gpu": 10.0, "num_prompts": 512}),
        encoding="utf-8",
    )
    return artifact_dir


def test_phase413_decomposes_steady_window_from_gzip_trace(tmp_path):
    scenario = phase413.DEFAULT_SCENARIO
    sweep_root = tmp_path / "phase412_arrival_sweep"
    artifact_dir = _write_phase413_artifact(sweep_root, scenario)
    phase412_csv = tmp_path / "phase412.csv"
    _write_csv(
        phase412_csv,
        [
            {
                "source": "phase412_arrival_sweep",
                "row_type": "trend",
                "scenario": scenario,
                "artifact_dir": str(artifact_dir),
                "uncoupled_output_tok_s_gpu": "20.0",
                "steady_output_tok_s_gpu": "10.0",
                "steady_penalty": "2.0",
            }
        ],
    )

    row = phase413.build_phase413_rows(
        sweep_root=sweep_root,
        phase412_csv=phase412_csv,
        scenario=scenario,
        sim_decode_ms_by_batch={2: 8.0, 4: 16.0},
    )[0]

    assert row["source"] == phase413.SOURCE
    assert row["steady_target_penalty"] == "2.000000"
    assert float(row["steady_window_wall_ms"]) == pytest.approx(150.0, abs=0.001)
    assert float(row["prefill_wall_ms"]) == pytest.approx(100.0, abs=0.001)
    assert float(row["peer_stall_extra_ms"]) == pytest.approx(140.0, abs=0.001)
    assert float(row["clean_decode_wall_ms"]) == pytest.approx(50.0, abs=0.001)
    assert row["intrinsic_decode_ms"] == "10.000000"
    assert row["peer_stalled_decode_step_count"] == "1"
    assert row["decode_batch_mean"] == "2.333333"
    assert row["real_sim_decode_latency_ratio"] == "1.250000"
    assert row["mechanism_verdict"] == "dp_peer_stall_dominates"
    assert row["phase405_penalty_read"] == "false"
    assert row["gpu_allowed"] == "false"
    assert row["ssh_allowed"] == "false"
    assert row["valid_for_default"] == "false"
    assert row["default_readiness"] == "No-Go"


def test_phase413_writer_rejects_runtime_or_default_claim(tmp_path):
    rows = [{field: "" for field in phase413.CSV_FIELDS}]
    rows[0].update(
        {
            "source": phase413.SOURCE,
            "scenario": phase413.DEFAULT_SCENARIO,
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
        phase413.write_phase413_csv(tmp_path / "bad.csv", rows)

    rows[0]["runtime_modified"] = "false"
    rows[0]["valid_for_default"] = "true"
    with pytest.raises(ValueError, match="valid_for_default"):
        phase413.write_phase413_csv(tmp_path / "bad.csv", rows)
