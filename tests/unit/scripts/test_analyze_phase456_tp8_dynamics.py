import json

import scripts.validate_cb_simulator as validate
from scripts.analyze_phase456_tp8_dynamics import (
    SCENARIO,
    event_profile,
    parse_event_steps,
    parse_serve_profile,
)


def test_parse_phase456_serve_profile_extracts_capacity_and_watermark(tmp_path) -> None:
    serve_log = tmp_path / "serve.log"
    serve_log.write_text(
        "\n".join(
            [
                "INFO [kv_cache_utils.py:1319] GPU KV cache size: 546,160 tokens",
                "INFO [loggers.py:259] Engine 000: Avg prompt throughput: 0.0 tokens/s, "
                "Avg generation throughput: 100.0 tokens/s, Running: 52 reqs, Waiting: 74 reqs, "
                "GPU KV cache usage: 80.0%, Prefix cache hit rate: 0.0%",
                "INFO [loggers.py:259] Engine 000: Avg prompt throughput: 0.0 tokens/s, "
                "Avg generation throughput: 100.0 tokens/s, Running: 64 reqs, Waiting: 62 reqs, "
                "GPU KV cache usage: 99.0%, Prefix cache hit rate: 0.0%",
            ]
        ),
        encoding="utf-8",
    )

    profile = parse_serve_profile(serve_log)

    assert profile["kv_cache_tokens"] == 546_160
    assert profile["num_gpu_blocks"] == 34_135
    assert profile["running"]["p50"] == 58.0
    assert profile["kv_usage"]["p50"] == 89.5


def test_phase456_event_profile_reports_mixed_decode_batch(tmp_path) -> None:
    event_jsonl = tmp_path / "events.jsonl"
    rows = [
        {"ctx_tokens": 8000, "generation_requests": 0, "forward_busy_ms": 580.0, "num_tokens_padded": 8000},
        {"ctx_tokens": 7950, "generation_requests": 51, "forward_busy_ms": 608.0, "num_tokens_padded": 8000},
        {"ctx_tokens": 0, "generation_requests": 64, "forward_busy_ms": 30.0, "num_tokens_padded": 64},
    ]
    event_jsonl.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    profile = event_profile(parse_event_steps(event_jsonl))

    assert profile["mixed_share"]["mean"] == 1 / 3
    assert profile["mixed_decode_batch"]["p50"] == 51
    assert profile["decode_batch"]["p50"] == 64


def test_phase456_tp8_8k_validate_capacity_uses_phase454_true_blocks() -> None:
    point = next(pt for pt in validate.MULTI_CONFIG_DATA if pt.name == SCENARIO)
    capacity = validate._multi_config_kv_capacity(point)
    config = validate._make_multi_config_cb_config(
        point,
        overlap_factor=0.0,
        ep8_per_iteration_overhead_ms=0.0,
    )

    assert capacity.kv_cache_tokens == 546_160
    assert capacity.num_gpu_blocks == 34_135
    assert "phase454_gpu_batch_scope" in capacity.serve_log
    assert config.num_gpu_blocks == 34_135
