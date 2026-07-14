#!/usr/bin/env python3
"""Tests for Phase462 queue-order capture integrity and first divergence."""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts/analyze_phase462_queue_order_first_divergence.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase462_queue_divergence", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _state(order=(0, 1), computed=(32000, 32001), output=(0, 0)):
    return {
        "schedule_seq": 7,
        "visible_ordinals": list(order),
        "running_order": list(order),
        "waiting_order": [],
        "request_phase": {
            str(req): {
                "num_computed_tokens": token,
                "num_output_placeholders": placeholder,
                "block_counts": [2001],
            }
            for req, token, placeholder in zip(order, computed, output)
        },
        "scheduled_new": [],
        "scheduled_resumed": [],
        "scheduled_running": list(order),
        "preempted": [],
    }


def test_integrity_gate_requires_complete_id_map_schedule_sequence_and_preemption_links() -> None:
    mod = _load_module()
    rows = [
        {
            "kind": "arrival_map",
            "trace_ids": ["r0", "r1"],
            "arrival_ordinals": [0, 1],
            "batch_positions": [0, 1],
        },
        {"kind": "scheduler_input", "schedule_seq": 1},
        {
            "kind": "scheduler_output",
            "schedule_seq": 1,
            "preempted_request_ids": ["engine-r1"],
        },
        {
            "kind": "preempt_decision",
            "schedule_seq": 1,
            "trigger_request_id": "engine-r0",
            "victim_request_id": "engine-r1",
        },
        {"kind": "future_complete", "schedule_seq": 1},
        {"kind": "scheduler_input", "schedule_seq": 2},
        {"kind": "scheduler_output", "schedule_seq": 2, "preempted_request_ids": []},
        {"kind": "future_complete", "schedule_seq": 2},
        {"kind": "trace_footer", "component": "tokenizer", "flush_complete": True},
        {"kind": "trace_footer", "component": "engine", "flush_complete": True},
    ]

    verdict = mod.validate_capture(rows, expected_requests=2)

    assert verdict.passed is True
    assert verdict.mapped_requests == 2
    assert verdict.schedule_sequences == 2
    assert verdict.linked_preemptions == 1


def test_integrity_gate_fails_closed_on_schedule_hole() -> None:
    mod = _load_module()
    rows = [
        {"kind": "arrival_map", "trace_ids": ["r0"], "arrival_ordinals": [0]},
        {"kind": "scheduler_input", "schedule_seq": 1},
        {"kind": "scheduler_output", "schedule_seq": 1, "preempted_request_ids": []},
        {"kind": "future_complete", "schedule_seq": 1},
        {"kind": "scheduler_input", "schedule_seq": 3},
        {"kind": "scheduler_output", "schedule_seq": 3, "preempted_request_ids": []},
        {"kind": "future_complete", "schedule_seq": 3},
        {"kind": "trace_footer", "component": "tokenizer", "flush_complete": True},
        {"kind": "trace_footer", "component": "engine", "flush_complete": True},
    ]

    with pytest.raises(ValueError, match="schedule_seq_hole"):
        mod.validate_capture(rows, expected_requests=1)


def test_streaming_integrity_gate_reads_gzip_and_checks_footer_counts(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    tokenizer_rows = [
        {
            "kind": "arrival_map",
            "trace_ids": ["r0"],
            "arrival_ordinals": [0],
            "batch_positions": [0],
        },
        {
            "kind": "trace_footer",
            "component": "tokenizer",
            "event_count": 1,
            "flush_complete": True,
        },
    ]
    engine_rows = [
        {"kind": "scheduler_input", "schedule_seq": 1},
        {
            "kind": "scheduler_output",
            "schedule_seq": 1,
            "preempted_request_ids": [],
        },
        {"kind": "future_complete", "schedule_seq": 1},
        {
            "kind": "trace_footer",
            "component": "engine",
            "event_count": 3,
            "flush_complete": True,
        },
    ]
    paths = []
    for name, rows in (("tokenizer.jsonl.gz", tokenizer_rows), ("engine.jsonl.gz", engine_rows)):
        path = tmp_path / name
        with gzip.open(path, "wt", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row) + "\n")
        paths.append(path)

    verdict = mod.validate_capture_paths(paths, expected_requests=1)

    assert verdict.passed is True
    assert verdict.schedule_sequences == 1


def test_streaming_integrity_gate_rejects_incomplete_footer(tmp_path: Path) -> None:
    mod = _load_module()
    path = tmp_path / "tokenizer.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as output:
        output.write(
            json.dumps(
                {
                    "kind": "trace_footer",
                    "component": "tokenizer",
                    "event_count": 1,
                    "flush_complete": True,
                }
            )
            + "\n"
        )

    with pytest.raises(ValueError, match="trace_event_count_mismatch"):
        mod.validate_capture_paths([path], expected_requests=0)


def test_first_divergence_classifies_queue_order_before_phase() -> None:
    mod = _load_module()
    real = _state(order=(0, 1))
    sim = _state(order=(1, 0), computed=(32001, 32000))

    verdict = mod.classify_step(real, sim)

    assert verdict.kind == "queue_order_divergence"
    assert verdict.stage == "visible_set_and_queue_order"


def test_first_divergence_classifies_phase_when_order_matches() -> None:
    mod = _load_module()
    real = _state(computed=(32000, 32001))
    sim = _state(computed=(32000, 32002))

    verdict = mod.classify_step(real, sim)

    assert verdict.kind == "request_phase_divergence"
    assert verdict.stage == "per_request_phase"


def test_first_divergence_classifies_kv_capacity_before_scheduler_output() -> None:
    mod = _load_module()
    real = {**_state(), "free_blocks": 0}
    sim = {**_state(), "free_blocks": 1, "preempted": [1]}

    verdict = mod.classify_step(real, sim)

    assert verdict.kind == "kv_capacity_divergence"
    assert verdict.stage == "block_capacity"


def test_first_divergence_marks_coupled_completion_boundary() -> None:
    mod = _load_module()
    real = _state(order=(0, 1), computed=(32000, 32001))
    sim = _state(order=(1, 0), computed=(32000, 32002))

    verdict = mod.classify_step(real, sim)

    assert verdict.kind == "completion_boundary_coupled_divergence"
    assert verdict.stage == "queue_order_and_phase"


def test_first_divergence_falls_through_to_scheduler_output() -> None:
    mod = _load_module()
    real = _state()
    sim = _state()
    sim["preempted"] = [1]

    verdict = mod.classify_step(real, sim)

    assert verdict.kind == "scheduler_output_divergence"
    assert verdict.stage == "scheduler_output"


def test_schedule_boundary_judge_uses_input_queue_then_phase_then_output() -> None:
    mod = _load_module()
    real = {
        "schedule_seq": 4,
        "input": _state(order=(0, 1), computed=(32000, 32001)),
        "scheduled_new": [],
        "scheduled_resumed": [],
        "scheduled_running": [0, 1],
        "preempted": [],
    }
    sim = {
        "schedule_seq": 4,
        "input": _state(order=(1, 0), computed=(32000, 32002)),
        "scheduled_new": [],
        "scheduled_resumed": [],
        "scheduled_running": [1, 0],
        "preempted": [],
    }

    verdict = mod.classify_schedule_boundary(real, sim)

    assert verdict.kind == "completion_boundary_coupled_divergence"
    assert verdict.schedule_seq == 4


def test_canonical_real_step_maps_internal_ids_to_arrival_ordinals() -> None:
    mod = _load_module()
    input_row = {
        "kind": "scheduler_input",
        "schedule_seq": 3,
        "free_blocks": 824,
        "running_trace_order": ["trace-0"],
        "waiting_trace_order": ["trace-1"],
        "skipped_waiting_trace_order": [],
        "request_phase": [
            {
                "trace_id": "trace-0",
                "request_id": "engine-0",
                "num_computed_tokens": 32000,
                "num_output_placeholders": 1,
                "block_counts": [2000],
            },
            {
                "trace_id": "trace-1",
                "request_id": "engine-1",
                "num_computed_tokens": 0,
                "num_output_placeholders": 0,
                "block_counts": [0],
            },
        ],
    }
    output_row = {
        **input_row,
        "kind": "scheduler_output",
        "scheduled_new_request_ids": ["engine-1"],
        "scheduled_resumed_request_ids": [],
        "scheduled_running_request_ids": ["engine-0"],
        "preempted_request_ids": [],
    }

    step = mod.canonical_real_step(
        input_row,
        output_row,
        trace_to_ordinal={"trace-0": 0, "trace-1": 1},
    )

    assert step["input"]["running_order"] == [0]
    assert step["input"]["waiting_order"] == [1]
    assert step["input"]["free_blocks"] == 824
    assert step["scheduled_new"] == [1]
    assert step["scheduled_running"] == [0]


def test_streaming_judge_records_first_divergence_and_keeps_full_sim_signature() -> None:
    mod = _load_module()
    real_steps = [
        {
            "schedule_seq": 1,
            "input": _state(order=(0,), computed=(32000,), output=(0,)),
            "scheduled_new": [],
            "scheduled_resumed": [],
            "scheduled_running": [0],
            "preempted": [],
        },
        {
            "schedule_seq": 2,
            "input": _state(order=(0,), computed=(32001,), output=(1,)),
            "scheduled_new": [],
            "scheduled_resumed": [],
            "scheduled_running": [0],
            "preempted": [],
        },
    ]

    def run_sim(observer):
        observer(real_steps[0])
        divergent = dict(real_steps[1])
        divergent["input"] = _state(order=(0,), computed=(32002,), output=(1,))
        observer(divergent)
        return {
            "preemption": {
                "preemptions": 10,
                "unique_victims": 10,
                "repeat_victim_events": 0,
                "self_preemptions": 1,
            },
            "preemption_events": [{"local_step": 2}],
        }

    result = mod.judge_streaming_steps(iter(real_steps), run_sim=run_sim)

    assert result["aligned_steps"] == 2
    assert result["sim_steps"] == 2
    assert result["real_steps"] == 2
    assert result["first_divergence"]["kind"] == "request_phase_divergence"
    assert result["first_divergence"]["schedule_seq"] == 2
    assert result["first_divergence_by_stage"]["phase"]["schedule_seq"] == 2
    assert result["sim_preemption"]["preemptions"] == 10


def test_streaming_judge_fails_closed_on_schedule_count_mismatch() -> None:
    mod = _load_module()
    real_steps = [
        {
            "schedule_seq": 1,
            "input": _state(order=(0,), computed=(32000,), output=(0,)),
            "scheduled_new": [],
            "scheduled_resumed": [],
            "scheduled_running": [0],
            "preempted": [],
        }
    ]

    def run_sim(observer):
        observer(real_steps[0])
        observer({**real_steps[0], "schedule_seq": 2})
        return {"preemption": {}, "preemption_events": []}

    with pytest.raises(ValueError, match="sim_schedule_without_real_step"):
        mod.judge_streaming_steps(iter(real_steps), run_sim=run_sim)


def test_execution_latency_oracle_uses_serial_future_completion_boundaries() -> None:
    mod = _load_module()
    rows = [
        {"kind": "scheduler_input", "schedule_seq": 1, "ts_ns": 0},
        {"kind": "scheduler_input", "schedule_seq": 2, "ts_ns": 1_000_000},
        {"kind": "future_complete", "schedule_seq": 1, "ts_ns": 5_000_000},
        {"kind": "future_complete", "schedule_seq": 2, "ts_ns": 8_000_000},
    ]

    assert mod.derive_execution_latency_oracle(rows) == [5.0, 3.0]
