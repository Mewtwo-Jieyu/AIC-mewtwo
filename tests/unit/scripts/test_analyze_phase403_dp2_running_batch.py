import csv
import json
from pathlib import Path

import pytest

from scripts import analyze_phase403_dp2_running_batch as phase403


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _metrics_record(*, running0: int, running1: int, waiting0: int = 0, waiting1: int = 0, preempt0: int = 0, preempt1: int = 0) -> str:
    body = "\n".join(
        [
            f'vllm:num_requests_running{{engine="0",model_name="kimi-k2.5"}} {running0}',
            f'vllm:num_requests_running{{engine="1",model_name="kimi-k2.5"}} {running1}',
            f'vllm:num_requests_waiting{{engine="0",model_name="kimi-k2.5"}} {waiting0}',
            f'vllm:num_requests_waiting{{engine="1",model_name="kimi-k2.5"}} {waiting1}',
            f'vllm:num_preemptions_total{{engine="0",model_name="kimi-k2.5"}} {preempt0}',
            f'vllm:num_preemptions_total{{engine="1",model_name="kimi-k2.5"}} {preempt1}',
            'vllm:gpu_cache_usage_perc{engine="0",model_name="kimi-k2.5"} 0.91',
            'vllm:gpu_cache_usage_perc{engine="1",model_name="kimi-k2.5"} 0.88',
        ]
    )
    return json.dumps(
        {
            "ts": "2026-07-02T00:00:00+00:00",
            "status": 200,
            "body": body,
        }
    )


def _write_raw_scenario(raw_root: Path, name: str, output_tok_s: float, metrics: list[str]) -> None:
    scenario_dir = raw_root / name
    scenario_dir.mkdir(parents=True, exist_ok=True)
    (scenario_dir / "meta.json").write_text(
        json.dumps(
            {
                "name": name,
                "tp": 4,
                "dp": 2,
                "ep": 8,
                "isl": 32000 if "32k3k" in name else 8000,
                "osl": 3000 if "32k3k" in name else 2000,
                "max_num_batched_tokens": 32000 if "32k3k" in name else 8000,
                "batch_size": 128,
                "world_size": 8,
                "prefix_caching": False,
            }
        ),
        encoding="utf-8",
    )
    (scenario_dir / "bench_result.json").write_text(
        json.dumps({"ok_requests": 128, "failed_requests": 0, "output_tok_s": output_tok_s}),
        encoding="utf-8",
    )
    (scenario_dir / "serve.log").write_text(
        "\n".join(
            [
                "INFO args: --no-enable-prefix-caching",
                "(EngineCore_DP0 pid=1) INFO [kv_cache_utils.py:1319] GPU KV cache size: 320,800 tokens",
                "(EngineCore_DP1 pid=2) INFO [kv_cache_utils.py:1319] GPU KV cache size: 320,800 tokens",
            ]
        ),
        encoding="utf-8",
    )
    (scenario_dir / "metrics.jsonl").write_text("\n".join(metrics) + "\n", encoding="utf-8")


def test_phase403_classifies_dp2_gap_as_comm_or_sync_when_real_running_reaches_sim(tmp_path) -> None:
    raw_root = tmp_path / "phase403_dp2_stats"
    _write_raw_scenario(
        raw_root,
        "K2.5-tp4ep8dp2-8k2k",
        1112.0,
        [
            _metrics_record(running0=40, running1=45, waiting0=10, waiting1=12),
            _metrics_record(running0=50, running1=53, waiting0=0, waiting1=5, preempt0=2, preempt1=5),
        ],
    )
    _write_raw_scenario(
        raw_root,
        "K2.5-tp4ep8dp2-32k3k",
        400.0,
        [
            _metrics_record(running0=11, running1=12, waiting0=80, waiting1=82),
            _metrics_record(running0=14, running1=15, waiting0=10, waiting1=11, preempt0=1, preempt1=2),
        ],
    )
    phase401_csv = tmp_path / "phase401.csv"
    _write_csv(
        phase401_csv,
        [
            {
                "scenario": "K2.5-tp4ep8dp2-8k2k",
                "phase401_sim_output_tok_s_gpu": "271.958961",
                "phase401_error_ratio": "1.792956",
                "phase401_direction": "sim_over_predicts_throughput",
                "sim_peak_decode_reqs_per_iter": "56",
            },
            {
                "scenario": "K2.5-tp4ep8dp2-32k3k",
                "phase401_sim_output_tok_s_gpu": "92.590110",
                "phase401_error_ratio": "1.849433",
                "phase401_direction": "sim_over_predicts_throughput",
                "sim_peak_decode_reqs_per_iter": "10",
            },
        ],
    )

    rows = phase403.build_phase403_rows(raw_root=raw_root, phase401_csv=phase401_csv)
    by_scenario = {row["scenario"]: row for row in rows}

    row8 = by_scenario["K2.5-tp4ep8dp2-8k2k"]
    assert row8["phase403_output_tok_s_gpu"] == "139.000000"
    assert row8["real_running_global_max"] == "103.000000"
    assert row8["sim_peak_decode_reqs_global"] == "112.000000"
    assert row8["attribution_verdict"] == "dp_ep_comm_or_sync_under_modeled"

    row32 = by_scenario["K2.5-tp4ep8dp2-32k3k"]
    assert row32["real_running_global_max"] == "29.000000"
    assert row32["sim_peak_decode_reqs_global"] == "20.000000"
    assert row32["attribution_verdict"] == "dp_ep_comm_or_sync_under_modeled"
    assert row32["gpu_kv_cache_tokens_per_engine"] == "320800"
    assert row32["gpu_cache_usage_max_pct"] == "0.910000"


def test_phase403_writer_rejects_default_upgrade(tmp_path) -> None:
    rows = [
        {field: "" for field in phase403.CSV_FIELDS},
        {field: "" for field in phase403.CSV_FIELDS},
    ]
    for idx, row in enumerate(rows):
        row.update(
            {
                "source": phase403.SOURCE,
                "scenario": phase403.DP2_SCENARIOS[idx],
                "dp": "2",
                "failed_requests": "0",
                "metrics_samples": "1",
                "gpu_allowed": "true",
                "ssh_allowed": "true",
                "runtime_modified": "false",
                "perf_database": "false",
                "valid_for_default": "false",
                "diagnostic_only": "true",
                "default_readiness": "No-Go",
            }
        )
    rows[0]["valid_for_default"] = "true"

    with pytest.raises(ValueError, match="valid_for_default"):
        phase403.write_phase403_csv(tmp_path / "bad.csv", rows)
