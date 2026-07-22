"""Phase468 forward workload descriptor contract tests."""
from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from aiconfigurator.sdk.backends.cb_simulator.datatypes import (
    CBSimConfig,
    Request,
    RequestState,
    ScheduleResult,
)
from aiconfigurator.sdk.backends.cb_simulator.forward_descriptor import (
    ForwardWorkloadDescriptor,
    build_forward_workload_descriptor,
)
from aiconfigurator.sdk.backends.cb_simulator.simulator import CBSimulator
from aiconfigurator.sdk.config import RuntimeConfig
from aiconfigurator.sdk.inference_summary import InferenceSummary


class _FakeLatencyCalc:
    def compute(self, **_kwargs) -> float:
        return 1.0

    def get_performance_source_map(self) -> dict:
        return {"context": {}, "generation": {}}

    def get_charge_ledger(self) -> list:
        return []


def _request(
    request_id: int,
    *,
    isl: int = 100,
    remaining: int = 100,
    generated: int = 0,
    state: RequestState = RequestState.WAITING,
) -> Request:
    req = Request(request_id=request_id, isl=isl, osl=8, arrival_time_ms=0.0)
    req.prefill_tokens_remaining = remaining
    req.generated_tokens = generated
    req.state = state
    return req


def _identity() -> dict[str, str | int | bool]:
    return {
        "scenario": "tp8-bt65536",
        "engine_step_id": 7,
        "dp_rank": 0,
        "execution_mode": "tp_single_replica",
        "model_path": "moonshotai/Kimi-K2.5",
        "model_config_sha256": "a" * 64,
        "hardware": "H200_SXM",
        "backend": "vllm",
        "backend_version": "0.19.0",
        "tp": 8,
        "pp": 1,
        "dp": 1,
        "moe_tp": 1,
        "moe_ep": 8,
        "cp": 1,
        "topology": "tp8pp1dp1moetp1ep8cp1",
        "quant_runtime": "CompressedTensorsWNA16MarlinMoEMethod",
    }


def _capture_config(**overrides) -> CBSimConfig:
    values = {
        "max_num_batched_tokens": 100,
        "num_requests": 8,
        "warmup_requests": 2,
        "capture_forward_workloads": True,
        "forward_workload_scenario": "phase468-test",
        "forward_workload_model_config_sha256": "a" * 64,
        "forward_workload_cp_size": 1,
        "forward_workload_quant_runtime": (
            "CompressedTensorsWNA16MarlinMoEMethod"
        ),
    }
    values.update(overrides)
    return CBSimConfig(**values)


def _sim(config: CBSimConfig, *, dp: int = 1) -> CBSimulator:
    model = SimpleNamespace(
        model_path="moonshotai/Kimi-K2.5",
        config=SimpleNamespace(
            tp_size=8 // dp,
            pp_size=1,
            attention_dp_size=dp,
            moe_tp_size=1,
            moe_ep_size=8,
        )
    )
    sim = CBSimulator(
        backend=MagicMock(),
        model=model,
        database=SimpleNamespace(
            backend="vllm",
            version="0.19.0",
            system="H200_SXM",
        ),
        config=config,
    )
    sim._create_latency_calc = lambda _prefix: _FakeLatencyCalc()
    return sim


def test_builder_records_exact_preupdate_workload_without_mutating_requests() -> None:
    prefill = _request(
        1,
        isl=100,
        remaining=40,
        generated=3,
        state=RequestState.PREFILLING,
    )
    decode = _request(
        2,
        isl=80,
        remaining=0,
        generated=11,
        state=RequestState.DECODING,
    )
    schedule = ScheduleResult(
        prefill_reqs=[prefill],
        prefill_tokens={1: 25},
        decode_reqs=[decode],
    )
    before = (asdict(prefill), asdict(decode))

    descriptor = build_forward_workload_descriptor(schedule, active=True, **_identity())

    assert descriptor.num_prefill_requests == 1
    assert descriptor.sum_prefill_tokens == 25
    assert descriptor.sum_prefill_kv_tokens == 63
    assert descriptor.num_decode_requests == 1
    assert descriptor.sum_decode_kv_tokens == 91
    assert descriptor.diagnostic_only is True
    assert descriptor.valid_for_default is False
    assert descriptor.perf_database is False
    assert descriptor.model_path == "moonshotai/Kimi-K2.5"
    assert descriptor.model_config_sha256 == "a" * 64
    assert (descriptor.tp, descriptor.pp, descriptor.dp) == (8, 1, 1)
    assert (descriptor.moe_tp, descriptor.moe_ep, descriptor.cp) == (1, 8, 1)
    assert descriptor.topology == "tp8pp1dp1moetp1ep8cp1"
    assert (asdict(prefill), asdict(decode)) == before


def test_builder_keeps_chunked_prefill_existing_kv_separate_from_new_tokens() -> None:
    req = _request(
        1,
        isl=1000,
        remaining=600,
        state=RequestState.PREFILLING,
    )
    schedule = ScheduleResult(prefill_reqs=[req], prefill_tokens={1: 128})

    descriptor = build_forward_workload_descriptor(schedule, active=True, **_identity())

    assert descriptor.sum_prefill_tokens == 128
    assert descriptor.sum_prefill_kv_tokens == 400


def test_builder_records_recompute_from_actual_reset_state() -> None:
    req = _request(
        1,
        isl=1000,
        remaining=1000,
        generated=0,
        state=RequestState.PREEMPTED,
    )
    schedule = ScheduleResult(prefill_reqs=[req], prefill_tokens={1: 256})

    descriptor = build_forward_workload_descriptor(schedule, active=True, **_identity())

    assert descriptor.sum_prefill_tokens == 256
    assert descriptor.sum_prefill_kv_tokens == 0


def test_idle_descriptor_requires_zero_workload() -> None:
    descriptor = build_forward_workload_descriptor(None, active=False, **_identity())

    assert isinstance(descriptor, ForwardWorkloadDescriptor)
    assert descriptor.active is False
    assert descriptor.num_prefill_requests == 0
    assert descriptor.sum_prefill_tokens == 0
    assert descriptor.sum_prefill_kv_tokens == 0
    assert descriptor.num_decode_requests == 0
    assert descriptor.sum_decode_kv_tokens == 0


def test_builder_rejects_prefill_map_that_does_not_match_requests() -> None:
    req = _request(1)
    schedule = ScheduleResult(prefill_reqs=[req], prefill_tokens={2: 10})

    with pytest.raises(ValueError, match="prefill token map"):
        build_forward_workload_descriptor(schedule, active=True, **_identity())


def test_capture_disabled_produces_no_descriptors() -> None:
    sim = _sim(CBSimConfig(num_requests=8, warmup_requests=2))

    sim.run(isl=100, osl=8, concurrency=2)

    assert sim.get_last_forward_workload_descriptors() == []


def test_single_replica_capture_is_deterministic_and_rank_zero() -> None:
    config = _capture_config()
    first = _sim(config)
    second = _sim(config)

    first.run(isl=100, osl=8, concurrency=2)
    second.run(isl=100, osl=8, concurrency=2)
    first_rows = first.get_last_forward_workload_descriptors()
    second_rows = second.get_last_forward_workload_descriptors()

    assert first_rows
    assert first_rows == second_rows
    assert {row.dp_rank for row in first_rows} == {0}
    assert {row.execution_mode for row in first_rows} == {"tp_single_replica"}
    step_ids = [row.engine_step_id for row in first_rows]
    assert step_ids == sorted(set(step_ids))


def test_dp_legacy_records_each_actual_rank_event() -> None:
    sim = _sim(_capture_config(), dp=2)

    sim.run_multi_replica(
        isl=100,
        osl=8,
        concurrency=4,
        data_parallel_size=2,
        lockstep=False,
    )
    rows = sim.get_last_forward_workload_descriptors()

    assert {row.dp_rank for row in rows} == {0, 1}
    assert {row.execution_mode for row in rows} == {"dp_legacy"}
    assert all(row.active for row in rows)


def test_dp_lockstep_records_both_ranks_for_every_global_step() -> None:
    sim = _sim(_capture_config(), dp=2)

    sim.run_multi_replica(
        isl=100,
        osl=8,
        concurrency=3,
        data_parallel_size=2,
        lockstep=True,
    )
    rows = sim.get_last_forward_workload_descriptors()
    step_ids = sorted({row.engine_step_id for row in rows})

    assert step_ids
    for step_id in step_ids:
        step_rows = [row for row in rows if row.engine_step_id == step_id]
        assert [row.dp_rank for row in step_rows] == [0, 1]
        assert {row.execution_mode for row in step_rows} == {"dp_lockstep"}
    assert any(not row.active for row in rows)


def test_engine_loop_capture_is_explicitly_unsupported() -> None:
    sim = _sim(_capture_config(engine_loop_enabled=True))

    with pytest.raises(
        NotImplementedError,
        match="forward workload capture does not support engine loop",
    ):
        sim.run(isl=100, osl=8, concurrency=2)


def test_capture_requires_exact_model_config_identity() -> None:
    sim = _sim(
        _capture_config(forward_workload_model_config_sha256=""),
    )

    with pytest.raises(ValueError, match="model config SHA256"):
        sim.run(isl=100, osl=8, concurrency=2)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"pp": 2, "topology": "tp8pp2dp1moetp1ep8cp1"}, "PP=1"),
        ({"cp": 2, "topology": "tp8pp1dp1moetp1ep8cp2"}, "CP=1"),
        ({"tp": 4, "topology": "tp4pp1dp1moetp1ep8cp1"}, "8 GPUs"),
        ({"topology": "tp8dp1ep8"}, "topology"),
    ],
)
def test_descriptor_rejects_incomplete_or_unsupported_topology(
    override,
    message,
) -> None:
    identity = _identity()
    identity.update(override)

    with pytest.raises(ValueError, match=message):
        build_forward_workload_descriptor(None, active=False, **identity)


def test_descriptor_rejects_rank_outside_data_parallel_topology() -> None:
    identity = _identity()
    identity["dp_rank"] = 1

    with pytest.raises(ValueError, match="dp_rank"):
        build_forward_workload_descriptor(None, active=False, **identity)


def test_descriptor_rejects_moe_world_mismatch() -> None:
    identity = _identity()
    identity.update(
        moe_tp=2,
        topology="tp8pp1dp1moetp2ep8cp1",
    )

    with pytest.raises(ValueError, match="MoE world"):
        build_forward_workload_descriptor(None, active=False, **identity)


def test_inference_summary_preserves_forward_workload_descriptors() -> None:
    summary = InferenceSummary(RuntimeConfig(isl=100, osl=8))
    descriptor = build_forward_workload_descriptor(None, active=False, **_identity())

    summary.set_forward_workload_descriptors([descriptor])

    assert summary.get_forward_workload_descriptors() == [descriptor]
