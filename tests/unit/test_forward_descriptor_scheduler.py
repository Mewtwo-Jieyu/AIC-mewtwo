from __future__ import annotations

from dataclasses import asdict

import pytest

from aiconfigurator.sdk.backends.cb_simulator.forward_descriptor import (
    compare_scheduler_runtime_descriptors,
    scheduler_runtime_descriptor_from_scheduled,
)


def _base_kwargs() -> dict[str, object]:
    return {
        "source": "cb_sim",
        "scenario": "10k2k_b32_bt8192",
        "iteration": 42,
        "phase": "mixed",
        "scheduled_context_tokens": 225,
        "scheduled_decode_tokens": 16,
        "scheduled_context_reqs": 15,
        "scheduled_decode_reqs": 1,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 32,
        "forward_token_count": 248,
        "cudagraph_runtime_mode": "NONE",
        "topology_key": "tp4dp2moetp1ep8",
        "tp": 4,
        "dp": 2,
        "moe_tp": 1,
        "moe_ep": 8,
    }


def _descriptor(**overrides: object):
    kwargs = _base_kwargs()
    kwargs.update(overrides)
    return scheduler_runtime_descriptor_from_scheduled(**kwargs)


def test_scheduler_descriptor_contains_only_mechanism_fields() -> None:
    descriptor = _descriptor()

    values = asdict(descriptor)
    assert values == {
        "source": "cb_sim",
        "scenario": "10k2k_b32_bt8192",
        "iteration": 42,
        "phase": "mixed",
        "scheduled_context_tokens": 225,
        "scheduled_decode_tokens": 16,
        "scheduled_total_tokens": 241,
        "scheduled_context_reqs": 15,
        "scheduled_decode_reqs": 1,
        "scheduled_total_reqs": 16,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 32,
        "forward_token_count": 248,
        "forward_regime": "NONE:248",
        "cudagraph_runtime_mode": "NONE",
        "topology_key": "tp4dp2moetp1ep8",
        "tp": 4,
        "dp": 2,
        "moe_tp": 1,
        "moe_ep": 8,
        "valid_for_default": False,
        "perf_database": False,
        "diagnostic_only": True,
    }
    for forbidden in (
        "latency_ms",
        "duration_ms",
        "residual_ms",
        "profiled_cuda_time_ms",
        "nccl_trace_line_count",
        "throughput_ratio",
    ):
        assert forbidden not in values


def test_scheduler_descriptor_requires_request_split() -> None:
    kwargs = _base_kwargs()
    del kwargs["scheduled_context_reqs"]

    with pytest.raises(TypeError, match="scheduled_context_reqs"):
        scheduler_runtime_descriptor_from_scheduled(**kwargs)


def test_scheduler_descriptor_fail_fast_validation() -> None:
    with pytest.raises(ValueError, match="phase"):
        _descriptor(phase="decode")

    with pytest.raises(ValueError, match="scheduled_decode_reqs"):
        _descriptor(scheduled_decode_reqs=-1)

    with pytest.raises(ValueError, match="forward_token_count"):
        _descriptor(forward_token_count=240)

    with pytest.raises(ValueError, match="topology_key"):
        _descriptor(topology_key="tp4dp2moetp4ep8")

    with pytest.raises(ValueError, match="tp"):
        _descriptor(tp=0)


def test_compare_scheduler_runtime_descriptors_is_ordinal_diagnostic() -> None:
    cb_row = _descriptor(
        source="cb_sim",
        iteration=10,
        forward_token_count=241,
        cudagraph_runtime_mode="AIC_UNSET",
    )
    vllm_row = _descriptor(
        source="vllm_runtime",
        iteration=20,
        forward_token_count=248,
        cudagraph_runtime_mode="NONE",
    )

    rows = compare_scheduler_runtime_descriptors([cb_row], [vllm_row])

    assert len(rows) == 1
    row = rows[0]
    assert row.phase == "mixed"
    assert row.ordinal_in_phase == 1
    assert row.cb_iter_index == 10
    assert row.vllm_iter_index == 20
    assert row.scheduled_total_token_delta == 0
    assert row.scheduled_total_req_delta == 0
    assert row.forward_token_delta == 7
    assert row.same_scheduled_total_tokens == 1
    assert row.same_scheduled_total_reqs == 1
    assert row.same_forward_token_count == 0
    assert row.same_forward_regime == 0
