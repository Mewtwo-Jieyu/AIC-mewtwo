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


def test_real_event_summary_dedupes_tp_rows(tmp_path: Path) -> None:
    phase451 = _load_module()
    event_path = tmp_path / "event_timing.jsonl"
    mixed = {
        "schema": "phase446_graph_outer_event_v2",
        "dp_rank": "0",
        "ctx_tokens": 7999,
        "generation_requests": 1,
        "generation_tokens": 1,
        "num_tokens_unpadded": 8000,
        "num_tokens_padded": 8000,
        "forward_busy_ms": 10.0,
        "cudagraph_mode": "FULL",
    }
    decode = {
        "schema": "phase446_graph_outer_event_v2",
        "dp_rank": "0",
        "ctx_tokens": 0,
        "generation_requests": 42,
        "generation_tokens": 42,
        "num_tokens_unpadded": 42,
        "num_tokens_padded": 42,
        "forward_busy_ms": 2.0,
        "cudagraph_mode": "FULL",
    }
    rows = [mixed] * 4 + [decode] * 4
    event_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    summary = phase451.summarize_real_events(event_path, tp_width=4)

    assert summary["total_steps"] == 2
    assert summary["mixed_steps"] == 1
    assert summary["mixed_steps_by_engine"] == {"0": 1}


def test_real_iteration_summary_uses_enginecore_steps(tmp_path: Path) -> None:
    phase451 = _load_module()
    serve_log = tmp_path / "serve.log"
    serve_log.write_text(
        "\n".join(
            [
                "(EngineCore_DP0 pid=1) Iteration(0): 1 context requests, 8000 context tokens, 0 generation requests, 0 generation tokens, iteration elapsed time: 1.0 ms",
                "(EngineCore_DP0 pid=1) Iteration(1): 1 context requests, 7999 context tokens, 1 generation requests, 1 generation tokens, iteration elapsed time: 1.0 ms",
                "(EngineCore_DP1 pid=2) Iteration(0): 0 context requests, 0 context tokens, 42 generation requests, 42 generation tokens, iteration elapsed time: 1.0 ms",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = phase451.summarize_real_iterations(serve_log)

    assert summary["total_steps"] == 3
    assert summary["mixed_steps"] == 1
    assert summary["mixed_steps_by_engine"] == {"0": 1}


if __name__ == "__main__":
    test_counter_deltas_keep_recompute_separate_from_preemption(Path("/tmp/phase451_test"))
    test_preemption_driver_gate_requires_recomputed_tokens()
    test_sim_trace_summary_counts_mixed_share()
    test_real_event_summary_dedupes_tp_rows(Path("/tmp/phase451_test"))
    test_real_iteration_summary_uses_enginecore_steps(Path("/tmp/phase451_test"))
