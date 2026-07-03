import csv
import gzip
import json

import pytest

from scripts import analyze_phase414_prefill_decode_audit as phase414


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


def _write_artifact(root, scenario):
    artifact_dir = root / scenario
    artifact_dir.mkdir(parents=True)
    serve_log = "\n".join(
        [
            "INFO EngineCore_DP0 Iteration(0): 1 context requests, 32000 context tokens, 2 generation requests, 2 generation tokens, iteration elapsed time: 100.00 ms",
            "INFO EngineCore_DP1 Iteration(0): 1 context requests, 32000 context tokens, 2 generation requests, 2 generation tokens, iteration elapsed time: 100.00 ms",
            "INFO EngineCore_DP0 Iteration(1): 0 context requests, 0 context tokens, 2 generation requests, 2 generation tokens, iteration elapsed time: 10.00 ms",
            "INFO EngineCore_DP1 Iteration(1): 0 context requests, 0 context tokens, 2 generation requests, 2 generation tokens, iteration elapsed time: 10.00 ms",
            "INFO EngineCore_DP0 Iteration(2): 0 context requests, 0 context tokens, 2 generation requests, 2 generation tokens, iteration elapsed time: 20.00 ms",
            "INFO EngineCore_DP1 Iteration(2): 0 context requests, 0 context tokens, 2 generation requests, 2 generation tokens, iteration elapsed time: 20.00 ms",
        ]
    ) + "\n"
    with gzip.open(artifact_dir / "serve.log.gz", "wt", encoding="utf-8") as f:
        f.write(serve_log)
    metrics = "\n".join(
        [
            _metric_record("2026-07-03T00:00:00+00:00", 0, 0, 0, 0),
            _metric_record("2026-07-03T00:00:00.100000+00:00", 32000, 32000, 2, 2),
            _metric_record("2026-07-03T00:00:00.110000+00:00", 32000, 32000, 4, 4),
            _metric_record("2026-07-03T00:00:00.130000+00:00", 32000, 32000, 6, 6),
        ]
    ) + "\n"
    with gzip.open(artifact_dir / "metrics.jsonl.gz", "wt", encoding="utf-8") as f:
        f.write(metrics)
    (artifact_dir / "bench_result.json").write_text(
        json.dumps({"output_tokens_per_second_per_gpu": 10.0, "num_prompts": 512}),
        encoding="utf-8",
    )
    return artifact_dir


def test_phase414_audits_prefill_and_decode_gaps_from_gzip_trace(tmp_path):
    scenario = phase414.DEFAULT_SCENARIO
    sweep_root = tmp_path / "phase412_arrival_sweep"
    artifact_dir = _write_artifact(sweep_root, scenario)
    phase413_csv = tmp_path / "phase413.csv"
    _write_csv(
        phase413_csv,
        [
            {
                "scenario": scenario,
                "artifact_dir": str(artifact_dir),
                "steady_target_penalty": "1.5",
                "prefill_wall_ms": "60.0",
                "peer_stall_extra_ms": "0.0",
                "clean_decode_gap_ms": "40.0",
            }
        ],
    )

    rows = phase414.build_phase414_rows(
        sweep_root=sweep_root,
        phase413_csv=phase413_csv,
        scenario=scenario,
        sim_mixed_ms_func=lambda prefill_tokens, gen_reqs, kv_len: 70.0,
        sim_decode_ms_func=lambda batch, kv_len: 8.0 if kv_len < 33500 else 12.0,
    )
    row = rows[0]

    assert row["source"] == phase414.SOURCE
    assert row["prefill_step_count"] == "2"
    assert row["prefill_real_mean_ms"] == "100.000000"
    assert row["prefill_sim_mean_ms"] == "70.000000"
    assert row["prefill_real_sim_ratio"] == "1.428571"
    assert row["prefill_gap_ms"] == "60.000000"
    assert row["decode_real_first_ms"] == "10.000000"
    assert row["decode_real_last_ms"] == "20.000000"
    assert row["decode_sim_first_ms"] == "8.000000"
    assert row["decode_sim_last_ms"] == "12.000000"
    assert row["decode_gap_ms"] == "22.000000"
    assert row["target_missing_ms"] == "100.000000"
    assert row["reconstructed_missing_ms"] == "82.000000"
    assert row["mechanism_verdict"] == "prefill_charge_gap_dominates"
    assert row["phase405_penalty_read"] == "false"
    assert row["gpu_allowed"] == "false"
    assert row["ssh_allowed"] == "false"
    assert row["valid_for_default"] == "false"
    assert row["default_readiness"] == "No-Go"


def test_phase414_writer_rejects_runtime_or_default_claim(tmp_path):
    rows = [{field: "" for field in phase414.CSV_FIELDS}]
    rows[0].update(
        {
            "source": phase414.SOURCE,
            "scenario": phase414.DEFAULT_SCENARIO,
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
        phase414.write_phase414_csv(tmp_path / "bad.csv", rows)

    rows[0]["runtime_modified"] = "false"
    rows[0]["valid_for_default"] = "true"
    with pytest.raises(ValueError, match="valid_for_default"):
        phase414.write_phase414_csv(tmp_path / "bad.csv", rows)
