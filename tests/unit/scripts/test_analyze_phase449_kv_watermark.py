from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[3] / "scripts" / "analyze_phase449_kv_watermark.py"
    spec = importlib.util.spec_from_file_location("analyze_phase449_kv_watermark", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_phase449_parses_metrics_and_aligns_mixed_steps(tmp_path: Path) -> None:
    phase449 = _load_module()
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_rows = [
        {
            "ts": "2026-07-08T00:00:00+00:00",
            "body": "\n".join(
                [
                    'vllm:num_requests_running{engine="0",model_name="m"} 40.0',
                    'vllm:num_requests_waiting{engine="0",model_name="m"} 8.0',
                    'vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.91',
                    'vllm:num_preemptions_total{engine="0",model_name="m"} 0.0',
                ]
            ),
        },
        {
            "ts": "2026-07-08T00:00:02+00:00",
            "body": "\n".join(
                [
                    'vllm:num_requests_running{engine="0",model_name="m"} 44.0',
                    'vllm:num_requests_waiting{engine="0",model_name="m"} 12.0',
                    'vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.985',
                    'vllm:num_preemptions_total{engine="0",model_name="m"} 0.0',
                ]
            ),
        },
    ]
    metrics_path.write_text("\n".join(json.dumps(row) for row in metrics_rows) + "\n", encoding="utf-8")

    serve_path = tmp_path / "serve.log"
    serve_path.write_text(
        "\n".join(
            [
                "(EngineCore_DP0 pid=1) INFO 07-08 00:00:00 [kv_cache_utils.py:1319] GPU KV cache size: 458,128 tokens",
                "(EngineCore_DP0 pid=1) INFO 07-08 00:00:02 [core.py:359] Iteration(7): 2 context requests, 7957 context tokens, 43 generation requests, 43 generation tokens, iteration elapsed time: 1125.94 ms",
                "(EngineCore_DP0 pid=1) INFO 07-08 00:00:03 [core.py:359] Iteration(8): 0 context requests, 0 context tokens, 44 generation requests, 44 generation tokens, iteration elapsed time: 40.00 ms",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    snapshots = phase449.parse_metric_snapshots(metrics_path)
    kv_rows = phase449.parse_kv_capacity_rows(serve_path)
    mixed_steps = phase449.parse_iteration_steps(serve_path)
    aligned = phase449.align_mixed_steps_to_metrics(mixed_steps, snapshots, max_delta_s=3.0)

    assert kv_rows[0].kv_cache_tokens == 458128
    assert len(aligned) == 1
    assert aligned[0].running == 44.0
    assert aligned[0].kv_usage == 0.985


def test_phase449_capacity_diff_marks_stale_validate_capacity() -> None:
    phase449 = _load_module()
    row = phase449.capacity_diff_row(
        scenario="unit",
        validate_tokens=672128,
        validate_blocks=42008,
        real_tokens=458128,
        real_blocks=28633,
        source="serve.log:10",
    )

    assert row["section"] == "capacity_diff"
    assert row["metric"] == "validate_tokens_over_real_tokens"
    assert float(row["value"]) > 1.46
    assert row["passed"] is False


if __name__ == "__main__":
    test_phase449_parses_metrics_and_aligns_mixed_steps(Path("/tmp/phase449_test"))
    test_phase449_capacity_diff_marks_stale_validate_capacity()
