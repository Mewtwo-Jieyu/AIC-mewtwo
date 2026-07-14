"""Tests for the production CB EngineCore loop and arrival layer."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from aiconfigurator.sdk.backends.cb_simulator.arrival import (
    TokenizerArrivalLayer,
    resolve_tokenizer_primitive,
)
from aiconfigurator.sdk.backends.cb_simulator.engine_loop import (
    AsyncCBScheduler,
    EngineLoopBatch,
    TimedInput,
    TimedInputSource,
    run_engine_loop,
)
from aiconfigurator.sdk.backends.cb_simulator.datatypes import (
    CBSimConfig,
    Request,
    RequestState,
)
from aiconfigurator.sdk.backends.cb_simulator.simulator import CBSimulator


class _ConstantLatency:
    def compute(
        self,
        prefill_tokens: int,
        prefill_batch_size: int,
        prefill_seq_len: int,
        decode_batch_size: int,
        decode_avg_kv_len: int,
    ) -> float:
        return 1.0

    def get_serving_state_query_audit(self) -> list[object]:
        return []


def test_engine_loop_preschedules_to_source_queue_depth() -> None:
    scheduled = 0

    def schedule(now_ms: float) -> EngineLoopBatch | None:
        nonlocal scheduled
        if scheduled >= 3:
            return None
        batch = EngineLoopBatch(
            batch_id=scheduled,
            launch_ms=now_ms,
            latency_ms=10.0,
        )
        scheduled += 1
        return batch

    result = run_engine_loop(
        queue_depth=2,
        input_source=TimedInputSource([]),
        on_drain=lambda _now, _items: None,
        schedule=schedule,
        on_complete=lambda _batch: None,
    )

    assert [batch.launch_ms for batch in result.launched] == [0.0, 0.0, 10.0]
    assert [batch.complete_ms for batch in result.completed] == [10.0, 20.0, 30.0]
    assert result.max_queue_depth == 2


def test_engine_loop_drains_inputs_at_busy_loop_not_mid_execution() -> None:
    drained: list[tuple[float, list[str]]] = []
    scheduled = 0

    def schedule(now_ms: float) -> EngineLoopBatch | None:
        nonlocal scheduled
        if scheduled >= 2:
            return None
        batch = EngineLoopBatch(
            batch_id=scheduled,
            launch_ms=now_ms,
            latency_ms=10.0,
        )
        scheduled += 1
        return batch

    run_engine_loop(
        queue_depth=2,
        input_source=TimedInputSource([TimedInput(5.0, "request-1")]),
        on_drain=lambda now, items: drained.append((now, list(items))),
        schedule=schedule,
        on_complete=lambda _batch: None,
    )

    assert drained == [(10.0, ["request-1"])]


def test_tokenizer_arrival_layer_batches_full_cohort_and_waits_for_tail() -> None:
    primitive = resolve_tokenizer_primitive(
        model_path="moonshotai/Kimi-K2.5",
        system="h200_sxm",
        backend="vllm",
        version="0.19.0",
        prompt_tokens=8_000,
    )
    assert primitive is not None
    layer = TokenizerArrivalLayer(primitive)

    layer.submit_many(
        [(request_id, 8_000) for request_id in range(33)],
        now_ms=0.0,
    )
    first_complete = primitive.predict_ms(32 * 8_000, 32)
    first = layer.drain(first_complete)
    tail_complete = first_complete + primitive.wait_timeout_ms + primitive.predict_ms(8_000, 1)

    assert first == list(range(32))
    assert layer.next_event_ms() == pytest.approx(tail_complete)
    assert layer.drain(tail_complete) == [32]


def test_tokenizer_primitive_is_exact_scoped_and_fail_closed() -> None:
    assert resolve_tokenizer_primitive(
        model_path="another/model",
        system="h200_sxm",
        backend="vllm",
        version="0.19.0",
        prompt_tokens=8_000,
    ) is None

    with pytest.raises(ValueError, match="unsupported prompt length"):
        resolve_tokenizer_primitive(
            model_path="moonshotai/Kimi-K2.5",
            system="h200_sxm",
            backend="vllm",
            version="0.19.0",
            prompt_tokens=16_000,
        )


def test_engine_loop_completion_callback_can_submit_future_input() -> None:
    source = TimedInputSource([])
    drained: list[object] = []
    scheduled = 0

    def schedule(now_ms: float) -> EngineLoopBatch | None:
        nonlocal scheduled
        if scheduled >= 1:
            return None
        scheduled += 1
        return EngineLoopBatch(batch_id=1, launch_ms=now_ms, latency_ms=4.0)

    def complete(batch: EngineLoopBatch) -> None:
        source.add(replace(TimedInput(0.0, "replacement"), arrival_ms=batch.complete_ms + 1.0))

    run_engine_loop(
        queue_depth=2,
        input_source=source,
        on_drain=lambda _now, items: drained.extend(items),
        schedule=schedule,
        on_complete=complete,
    )

    assert drained == ["replacement"]


def test_request_tracks_sampled_computed_and_placeholder_states() -> None:
    request = Request(request_id=1, isl=8_000, osl=100, arrival_time_ms=0.0)

    request.generated_tokens = 7
    request.output_placeholders = 2
    request.sampled_output_tokens += 1

    assert request.generated_tokens == 8
    assert request.sampled_output_tokens == 8
    assert request.computed_output_tokens == 7
    assert request.output_placeholders == 2


def test_async_scheduler_uses_computed_kv_and_resets_it_on_preemption() -> None:
    config = CBSimConfig(num_gpu_blocks=1, block_size=16)
    scheduler = AsyncCBScheduler(config)
    request = Request(request_id=1, isl=16, osl=100, arrival_time_ms=0.0)
    request.state = RequestState.DECODING
    request.prefill_tokens_remaining = 0
    request.sampled_output_tokens = 20
    request.computed_output_tokens = 3

    assert scheduler._blocks_needed(request) == 2

    waiting: list[Request] = []
    running = [request]
    scheduler._preempt(request, waiting, running, scheduler.empty_result(), set())

    assert request.prefill_tokens_remaining == 36
    assert request.computed_output_tokens == 0


def test_gpu_block_config_separates_physical_and_allocatable_capacity() -> None:
    config = CBSimConfig(num_gpu_blocks=28_825)

    assert config.num_gpu_blocks == 28_825
    assert config.num_allocatable_gpu_blocks == 28_824
    assert CBSimConfig(num_gpu_blocks=0).num_allocatable_gpu_blocks == 0


def test_null_block_capacity_preempts_peer_before_tail_self_preemption() -> None:
    """Replay the N128 seq=944 boundary from the Phase462 capture."""
    config = CBSimConfig(
        max_num_batched_tokens=32_000,
        max_num_seqs=128,
        num_gpu_blocks=28_825,
        block_size=16,
    )
    scheduler = AsyncCBScheduler(config)
    computed_tokens = [
        32_942,
        32_939,
        32_938,
        32_937,
        32_936,
        32_935,
        32_934,
        32_933,
        32_932,
        32_931,
        32_930,
        32_929,
        32_928,
        32_927,
    ]
    running: list[Request] = []
    for request_id, num_computed_tokens in enumerate(computed_tokens):
        request = Request(
            request_id=request_id,
            isl=32_000,
            osl=1_200,
            arrival_time_ms=0.0,
        )
        request.state = RequestState.DECODING
        request.prefill_tokens_remaining = 0
        request.computed_output_tokens = num_computed_tokens - request.isl
        request.sampled_output_tokens = request.computed_output_tokens + 1
        request.output_placeholders = 1
        running.append(request)

    waiting = [
        Request(
            request_id=request_id,
            isl=32_000,
            osl=1_200,
            arrival_time_ms=0.0,
        )
        for request_id in range(14, 128)
    ]

    first = scheduler.schedule(waiting, running)

    assert [event["trigger_request_id"] for event in scheduler.preemption_events] == [12]
    assert [event["victim_request_id"] for event in scheduler.preemption_events] == [13]
    assert [request.request_id for request in first.decode_reqs] == list(range(13))

    for request in first.decode_reqs:
        request.computed_output_tokens += 1
        request.output_placeholders = 0

    scheduler.schedule(waiting, running)

    assert all(
        event["trigger_request_id"] != event["victim_request_id"]
        for event in scheduler.preemption_events
    )


def _make_exact_engine_loop_sim(config: CBSimConfig) -> CBSimulator:
    sim = CBSimulator(
        backend=MagicMock(),
        model=SimpleNamespace(model_path="moonshotai/Kimi-K2.5"),
        database=SimpleNamespace(
            system="h200_sxm",
            backend="vllm",
            version="0.19.0",
        ),
        config=config,
    )
    sim._create_latency_calc = lambda _prefix: _ConstantLatency()
    return sim


def test_exact_deployment_uses_engine_loop_and_disables_decode_skip() -> None:
    sim = _make_exact_engine_loop_sim(
        CBSimConfig(
            max_num_batched_tokens=8_000,
            num_requests=8,
            warmup_requests=2,
            engine_loop_enabled=True,
        )
    )

    result = sim.run(isl=8_000, osl=8, concurrency=2)
    audit = sim.get_last_engine_loop_audit()

    assert result.throughput_tok_s > 0
    assert audit["path"] == "engine_loop"
    assert audit["queue_depth"] == 2
    assert audit["max_queue_depth"] == 2
    assert audit["decode_skip_enabled"] is False
    assert audit["sim_runtime_seconds"] >= 0
    assert audit["steady_window_mode"] != "unavailable"
    assert isinstance(sim.get_last_preemption_events(), list)


def test_exact_deployment_rejects_unsupported_prompt_length() -> None:
    sim = _make_exact_engine_loop_sim(CBSimConfig(engine_loop_enabled=True))

    with pytest.raises(ValueError, match="unsupported prompt length"):
        sim.run(isl=16_000, osl=8, concurrency=2)


def test_engine_loop_enabled_rejects_unscoped_deployment() -> None:
    sim = CBSimulator(
        backend=MagicMock(),
        model=SimpleNamespace(model_path="another/model"),
        database=SimpleNamespace(
            system="h200_sxm",
            backend="vllm",
            version="0.19.0",
        ),
        config=CBSimConfig(engine_loop_enabled=True),
    )

    with pytest.raises(ValueError, match="no exact tokenizer primitive"):
        sim.run(isl=8_000, osl=8, concurrency=2)


def test_engine_loop_enabled_rejects_unimplemented_dp_path() -> None:
    sim = _make_exact_engine_loop_sim(CBSimConfig(engine_loop_enabled=True))

    with pytest.raises(NotImplementedError, match="multi-replica engine loop"):
        sim.run_multi_replica(
            isl=8_000,
            osl=8,
            concurrency=2,
            data_parallel_size=2,
        )


def test_engine_loop_stays_out_of_default_path_before_gate_passes() -> None:
    sim = _make_exact_engine_loop_sim(
        CBSimConfig(num_requests=8, warmup_requests=2)
    )

    sim.run(isl=8_000, osl=8, concurrency=2)

    assert sim.get_last_engine_loop_audit() == {
        "path": "legacy",
        "decode_skip_enabled": True,
        "engine_loop_enabled": False,
    }
