from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[3] / "scripts" / "analyze_phase451_preemption_ledger.py"
    spec = importlib.util.spec_from_file_location("analyze_phase451_preemption_ledger", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_counter_deltas_keep_recompute_separate_from_preemption(tmp_path: Path) -> None:
    phase451 = _load_module()
    tmp_path.mkdir(parents=True, exist_ok=True)
    metrics_path = tmp_path / "metrics.jsonl"
    first = "\n".join(
        [
            'vllm:num_preemptions_total{engine="0",model_name="m"} 2.0',
            'vllm:prompt_tokens_recomputed_total{engine="0",model_name="m"} 0.0',
            'vllm:request_success_total{engine="0",finished_reason="length",model_name="m"} 10.0',
        ]
    )
    last = "\n".join(
        [
            'vllm:num_preemptions_total{engine="0",model_name="m"} 7.0',
            'vllm:prompt_tokens_recomputed_total{engine="0",model_name="m"} 0.0',
            'vllm:request_success_total{engine="0",finished_reason="length",model_name="m"} 30.0',
        ]
    )
    metrics_path.write_text(
        "\n".join(
            [
                json.dumps({"ts": "2026-07-08T00:00:00+00:00", "body": first}),
                json.dumps({"ts": "2026-07-08T00:00:02+00:00", "body": last}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    deltas = phase451.collect_metric_deltas(metrics_path)

    assert deltas["num_preemptions_total"]["0"].delta == 5.0
    assert deltas["prompt_tokens_recomputed_total"]["0"].delta == 0.0
    assert deltas["request_success_length"]["0"].delta == 20.0


def test_preemption_driver_gate_requires_recomputed_tokens() -> None:
    phase451 = _load_module()

    verdict = phase451.preemption_driver_gate(
        mixed_steps=1380,
        preemptions=54,
        recomputed_tokens=0.0,
        tolerance=0.80,
    )

    assert verdict["status"] == "fail"
    assert verdict["passed"] is False
    assert verdict["preemption_to_mixed_ratio"] < 0.05


def test_sim_trace_summary_counts_mixed_share() -> None:
    phase451 = _load_module()
    trace = [
        {"replica_id": 0, "is_mixed": True, "prefill_tokens": 7999, "decode_reqs": 1},
        {"replica_id": 0, "is_mixed": False, "prefill_tokens": 0, "decode_reqs": 10},
        {"replica_id": 1, "is_mixed": True, "prefill_tokens": 7999, "decode_reqs": 1},
        {"replica_id": 1, "is_mixed": False, "prefill_tokens": 0, "decode_reqs": 10},
    ]

    summary = phase451.summarize_sim_trace(trace)

    assert summary["total_steps"] == 4
    assert summary["mixed_steps"] == 2
    assert summary["mixed_share"] == 0.5
    assert summary["mixed_steps_by_replica"] == {"0": 1, "1": 1}


if __name__ == "__main__":
    test_counter_deltas_keep_recompute_separate_from_preemption(Path("/tmp/phase451_test"))
    test_preemption_driver_gate_requires_recomputed_tokens()
    test_sim_trace_summary_counts_mixed_share()
