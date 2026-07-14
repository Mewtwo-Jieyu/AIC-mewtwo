#!/usr/bin/env python3
"""Tests for the Phase462 EngineCore batch-queue prototype."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase462_engine_loop_state_machine.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_phase462_engine_loop_state_machine", MODULE_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_deployment_spec_uses_runtime_async_and_pp_values() -> None:
    mod = _load_module()

    spec = mod.derive_deployment_queue_spec(
        executor="multiproc",
        pipeline_parallel_size=1,
        async_scheduling=True,
    )

    assert spec.queue_depth == 2
    assert spec.step_fn == "step_with_batch_queue"
    assert spec.source_rule == "2 if pp_size <= 1 and async_scheduling else pp_size"


def test_batch_queue_preschedules_to_depth_before_waiting() -> None:
    mod = _load_module()
    state = {"scheduled": 0}

    def schedule(now_ms: float):
        if state["scheduled"] >= 3:
            return None
        job = mod.PrototypeBatch(
            batch_id=state["scheduled"],
            launch_ms=now_ms,
            latency_ms=10.0,
        )
        state["scheduled"] += 1
        return job

    result = mod.run_batch_queue_machine(
        queue_depth=2,
        arrivals=[],
        on_drain=lambda _now, _items: None,
        schedule=schedule,
        on_complete=lambda _batch: None,
    )

    assert [row.launch_ms for row in result.launched] == [0.0, 0.0, 10.0]
    assert [row.complete_ms for row in result.completed] == [10.0, 20.0, 30.0]
    assert result.max_queue_depth == 2


def test_queue_prototype_exposes_diagnostic_schedule_boundaries() -> None:
    mod = _load_module()
    states = []

    result = mod._run_cb_queue_prototype(
        arrivals=[mod.TimedInput(0.0, 0), mod.TimedInput(0.0, 1)],
        measured_iteration_latencies_ms=None,
        queue_depth=2,
        request_limit=2,
        prototype_osl=2,
        schedule_observer=states.append,
    )

    assert result["completed_requests"] == 2
    assert states[0]["schedule_seq"] == 1
    assert states[0]["input"]["running_order"] == []
    assert states[0]["input"]["waiting_order"] == [0, 1]
    assert states[0]["output"]["running_order"] == [0]
    assert states[0]["scheduled_new"] == [0]
    assert set(states[0]["input"]["request_phase"]) == {"0", "1"}


def test_inputs_arriving_during_execution_wait_for_next_busy_loop_drain() -> None:
    mod = _load_module()
    drained: list[tuple[float, list[str]]] = []
    state = {"scheduled": 0}

    def schedule(now_ms: float):
        if state["scheduled"] >= 2:
            return None
        job = mod.PrototypeBatch(
            batch_id=state["scheduled"],
            launch_ms=now_ms,
            latency_ms=10.0,
        )
        state["scheduled"] += 1
        return job

    mod.run_batch_queue_machine(
        queue_depth=2,
        arrivals=[mod.TimedInput(5.0, "request-1")],
        on_drain=lambda now, items: drained.append((now, list(items))),
        schedule=schedule,
        on_complete=lambda _batch: None,
    )

    assert drained == [(10.0, ["request-1"])]


def test_schedule_callback_advances_state_before_second_preschedule() -> None:
    mod = _load_module()
    logical_step = 0

    def schedule(now_ms: float):
        nonlocal logical_step
        if logical_step >= 2:
            return None
        current = logical_step
        logical_step += 1
        return mod.PrototypeBatch(
            batch_id=current,
            launch_ms=now_ms,
            latency_ms=10.0,
            payload={"logical_step": current},
        )

    result = mod.run_batch_queue_machine(
        queue_depth=2,
        arrivals=[],
        on_drain=lambda _now, _items: None,
        schedule=schedule,
        on_complete=lambda _batch: None,
    )

    assert [row.payload["logical_step"] for row in result.launched] == [0, 1]


def test_gate_requires_all_four_signatures_and_keeps_runtime_locked() -> None:
    mod = _load_module()

    passed = mod.prototype_gate(
        drain_step_match=1.0,
        first_schedule_step_match=1.0,
        first_16_match=1.0,
        preemptions=10,
        self_preemptions=0,
        repeat_victim_events=0,
        target_preemptions=10,
        prediction_eligible=True,
    )
    blocked = mod.prototype_gate(
        drain_step_match=1.0,
        first_schedule_step_match=1.0,
        first_16_match=1.0,
        preemptions=10,
        self_preemptions=1,
        repeat_victim_events=0,
        target_preemptions=10,
        prediction_eligible=True,
    )
    replay_only = mod.prototype_gate(
        drain_step_match=1.0,
        first_schedule_step_match=1.0,
        first_16_match=1.0,
        preemptions=10,
        self_preemptions=0,
        repeat_victim_events=0,
        target_preemptions=10,
        prediction_eligible=False,
    )

    assert passed["passed"] is True
    assert passed["runtime_change_allowed"] is False
    assert blocked["passed"] is False
    assert replay_only["signatures_passed"] is True
    assert replay_only["passed"] is False


def test_build_predicted_inputs_keeps_tokenizer_batch_clustered() -> None:
    mod = _load_module()
    rows = [
        {
            "kind": "tokenizer_batch_enter",
            "batch_id": "b1",
            "batch_start_ns": 1_000_000,
            "trace_ids": ["phase462-32k3k-000000", "phase462-32k3k-000001"],
        },
        {
            "kind": "tokenizer_batch_complete",
            "batch_id": "b1",
            "prompt_token_lengths": [32_000, 32_000],
        },
    ]

    arrivals = mod.build_predicted_tokenizer_inputs(
        rows,
        predict_service_ms=lambda tokens, batch: tokens / 1_000 + batch,
        request_limit=128,
    )

    assert [item.payload for item in arrivals] == [0, 1]
    assert arrivals[0].arrival_ms == arrivals[1].arrival_ms == 67.0


def test_ground_truth_maps_receive_to_next_scheduler_drain_step() -> None:
    mod = _load_module()
    rows = [
        {
            "kind": "engine_receive",
            "trace_id": "phase462-32k3k-000000",
            "ts_ns": 100,
        },
        {"kind": "scheduler_step", "step": 1, "ts_ns": 120},
        {
            "kind": "engine_receive",
            "trace_id": "phase462-32k3k-000001",
            "ts_ns": 150,
        },
        {"kind": "scheduler_step", "step": 2, "ts_ns": 160},
    ]

    drain_steps = mod.map_real_drain_steps(rows, request_limit=128)

    assert drain_steps == {0: 1, 1: 2}


def test_signature_summary_counts_self_and_repeat_victims() -> None:
    mod = _load_module()

    summary = mod.summarize_preemption_signature(
        [
            {"trigger_req_id": 1, "victim_req_id": 1},
            {"trigger_req_id": 2, "victim_req_id": 1},
            {"trigger_req_id": 3, "victim_req_id": 3},
        ]
    )

    assert summary == {
        "preemptions": 3,
        "unique_victims": 2,
        "repeat_victim_events": 1,
        "self_preemptions": 2,
    }


def test_normalize_inputs_preserves_clusters_and_starts_at_zero() -> None:
    mod = _load_module()

    normalized = mod.normalize_timed_inputs(
        [
            mod.TimedInput(12.5, 2),
            mod.TimedInput(10.0, 0),
            mod.TimedInput(10.0, 1),
        ]
    )

    assert normalized == [
        mod.TimedInput(0.0, 0),
        mod.TimedInput(0.0, 1),
        mod.TimedInput(2.5, 2),
    ]


def test_exact_match_fraction_requires_same_request_keys() -> None:
    mod = _load_module()

    assert mod.exact_match_fraction({0: 1, 1: 3}, {0: 1, 1: 3}) == 1.0
    assert mod.exact_match_fraction({0: 1, 1: 3}, {0: 1}) == 0.5


def test_dp_merge_stays_blocked_without_validated_single_rank_machine() -> None:
    mod = _load_module()

    verdict = mod.dp_merge_verdict(
        single_rank_gate_passed=False,
        has_per_rank_queue_timestamps=True,
    )

    assert verdict == {
        "status": "blocked",
        "reason": "single_rank_state_machine_not_validated",
        "step3_inheritance_allowed": False,
    }


def test_dp_queue_evidence_scans_separate_event_rows_for_every_rank() -> None:
    mod = _load_module()

    evidence = mod.summarize_dp_queue_evidence(
        [
            {"dp_rank": 0, "kind": "engine_receive", "ts_ns": 1},
            {"dp_rank": 0, "kind": "scheduler_step", "ts_ns": 2},
            {"dp_rank": 1, "kind": "engine_receive", "ts_ns": 3},
            {"dp_rank": 1, "kind": "scheduler_step", "ts_ns": 4},
        ]
    )

    assert evidence["timestamped_queue_ranks"] == [0, 1]
    assert evidence["has_per_rank_queue_timestamps"] is True


def test_dp_queue_evidence_rejects_busy_rows_without_queue_events() -> None:
    mod = _load_module()

    evidence = mod.summarize_dp_queue_evidence(
        [
            {"dp_rank": 0, "forward_busy_ms": 1.0},
            {"dp_rank": 1, "forward_busy_ms": 2.0},
        ]
    )

    assert evidence["ranks"] == [0, 1]
    assert evidence["has_per_rank_queue_timestamps"] is False


def test_report_boundaries_never_promote_diagnostic_evidence() -> None:
    mod = _load_module()

    assert mod.REPORT_BOUNDARIES == {
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
        "default_aic": "No-Go",
    }


def test_async_output_state_separates_sampled_computed_and_placeholders() -> None:
    mod = _load_module()

    assert mod.can_schedule_async_decode(
        sampled_output_tokens=4,
        output_placeholders=1,
        osl=6,
    ) is True
    assert mod.can_schedule_async_decode(
        sampled_output_tokens=4,
        output_placeholders=2,
        osl=6,
    ) is False
    assert mod.recompute_tokens_after_preemption(
        isl=32_000,
        sampled_output_tokens=5,
    ) == 32_005
    assert mod.async_decode_kv_tokens(
        isl=32_000,
        computed_output_tokens=4,
        scheduled_tokens=1,
    ) == 32_005


def test_vllm_skips_waiting_admission_after_any_preemption() -> None:
    mod = _load_module()

    assert mod.waiting_admission_allowed(preemptions_this_step=0) is True
    assert mod.waiting_admission_allowed(preemptions_this_step=1) is False


def test_future_latencies_come_from_queue_depth_shifted_schedule_times() -> None:
    mod = _load_module()
    rows = [
        {"kind": "scheduler_step", "step": step, "ts_ns": ts_ms * 1_000_000}
        for step, ts_ms in enumerate([0, 1, 10, 13, 20], start=1)
    ]

    latencies = mod.derive_future_latencies_ms(rows, queue_depth=2)

    assert latencies == [10.0, 3.0, 7.0]


def test_runtime_deployment_is_parsed_from_log_not_defaults() -> None:
    mod = _load_module()

    runtime = mod.parse_runtime_deployment(
        [
            "INFO [vllm.py] Asynchronous scheduling is enabled.",
            "INFO config: pipeline_parallel_size=1, data_parallel_size=1",
            "INFO [multiproc_executor.py:134] DP group leader",
        ]
    )

    assert runtime == {
        "async_scheduling": True,
        "pipeline_parallel_size": 1,
        "executor": "multiproc",
    }
