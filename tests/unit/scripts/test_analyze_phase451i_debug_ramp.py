#!/usr/bin/env python3
"""Tests for Phase451-I DEBUG ramp analyzer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import json


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase451i_debug_ramp.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase451i_debug", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_received_counts_extracts_waiting_running_pairs() -> None:
    mod = _load_module()
    line = (
        "(ApiServer_0 pid=1) DEBUG 07-09 03:14:47 "
        "[v1/engine/core_client.py:1275] Received counts: [[0, 1], [63, 1]] "
        "(slice(0, 2, None))\n"
    )

    parsed = mod.parse_received_counts_line(line, index=7)

    assert parsed == {"index": 7, "w0": 0, "r0": 1, "w1": 63, "r1": 1}


def test_parse_iteration_line_extracts_engine_step_shape() -> None:
    mod = _load_module()
    line = (
        "(EngineCore_DP1 pid=1205) INFO 07-09 03:15:05 "
        "[v1/engine/core.py:359] Iteration(15): 2 context requests, "
        "7986 context tokens, 14 generation requests, 14 generation tokens, "
        "iteration elapsed time: 1115.64 ms\n"
    )

    parsed = mod.parse_iteration_line(line, line_no=22)

    assert parsed["dp_rank"] == 1
    assert parsed["iteration"] == 15
    assert parsed["ctx_tokens"] == 7986
    assert parsed["generation_requests"] == 14
    assert parsed["elapsed_ms"] == 1115.64


def test_summarize_debug_run_marks_burst_without_victim_identity() -> None:
    mod = _load_module()
    counts = [
        {"index": 0, "w0": 0, "r0": 0, "w1": 0, "r1": 0},
        {"index": 1, "w0": 0, "r0": 1, "w1": 63, "r1": 1},
        {"index": 2, "w0": 62, "r0": 2, "w1": 62, "r1": 2},
    ]
    iterations = [
        {
            "line_no": 10,
            "dp_rank": 0,
            "iteration": 0,
            "ctx_requests": 1,
            "ctx_tokens": 8000,
            "generation_requests": 0,
            "generation_tokens": 0,
            "elapsed_ms": 1200.0,
        },
        {
            "line_no": 11,
            "dp_rank": 1,
            "iteration": 1,
            "ctx_requests": 1,
            "ctx_tokens": 7999,
            "generation_requests": 1,
            "generation_tokens": 1,
            "elapsed_ms": 1100.0,
        },
    ]

    summary = mod.summarize_debug_run(counts, iterations, {"added": 0, "preempt": 0, "victim": 0})

    assert summary["visibility"] == "engine_burst_confirmed"
    assert summary["victim_observability"] == "not_available"
    assert summary["first_asymmetric_burst_counts"] == "[[0,1],[63,1]]"


def test_preempt_log_without_victim_does_not_expose_victim_identity() -> None:
    mod = _load_module()

    summary = mod.summarize_debug_run([], [], {"preempt": 1, "victim": 0})

    assert summary["victim_observability"] == "not_available"


def test_extract_final_metric_values_reads_last_engine_counters(tmp_path: Path) -> None:
    mod = _load_module()
    body = "\n".join(
        [
            'vllm:num_preemptions_total{engine="0",model_name="kimi"} 3.0',
            'vllm:num_preemptions_total{engine="1",model_name="kimi"} 4.0',
            'vllm:request_success_total{engine="0",finished_reason="length",model_name="kimi"} 64.0',
        ]
    )
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(json.dumps({"ts": "2026-07-09T00:00:00+00:00", "body": body}) + "\n", encoding="utf-8")

    values = mod.extract_final_metric_values(
        metrics,
        {"vllm:num_preemptions_total", "vllm:request_success_total"},
    )

    assert values["vllm:num_preemptions_total"]["0"] == 3.0
    assert values["vllm:num_preemptions_total"]["1"] == 4.0
    assert values["vllm:request_success_total"]["0:length"] == 64.0


if __name__ == "__main__":
    test_parse_received_counts_extracts_waiting_running_pairs()
    test_parse_iteration_line_extracts_engine_step_shape()
    test_summarize_debug_run_marks_burst_without_victim_identity()
    test_preempt_log_without_victim_does_not_expose_victim_identity()
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp:
        test_extract_final_metric_values_reads_last_engine_counters(Path(tmp))
