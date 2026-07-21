# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from aiconfigurator.sdk.backends.cb_simulator.iteration_latency import IterationLatencyCalculator
from aiconfigurator.sdk.inference_summary import InferenceSummary
from aiconfigurator.sdk.perf_source import PerfSourceRecord, validate_performance_result_sources
from aiconfigurator.sdk.performance_result import PerformanceResult


def _source(source_id: str) -> PerfSourceRecord:
    return PerfSourceRecord.structural(
        source_id=source_id,
        formula_id=f"{source_id}_v1",
        formula_inputs={"input": 1},
    )


class _Model:
    model_path = "moonshotai/Kimi-K2.5"
    config = SimpleNamespace(
        tp_size=4,
        attention_dp_size=2,
        moe_tp_size=1,
        moe_ep_size=8,
    )


class _Backend:
    def __init__(self) -> None:
        self.calls = 0

    def run_static(self, model, database, runtime_config, mode, **kwargs):
        self.calls += 1
        summary = InferenceSummary(runtime_config)
        if mode == "static_ctx":
            summary.set_context_latency_dict(
                {"context_moe": 10.0, "context_dense": 5.0, "context_attention": 7.0}
            )
            summary.set_context_source_map(
                {
                    "context_moe": (_source("base_moe"),),
                    "context_dense": (_source("base_dense"),),
                    "context_attention": (_source("base_context_attention"),),
                }
            )
        else:
            summary.set_generation_latency_dict(
                {"generation_moe": 11.0, "generation_attention": 3.0}
            )
            summary.set_generation_source_map(
                {
                    "generation_moe": (_source("base_generation_moe"),),
                    "generation_attention": (_source("base_generation_attention"),),
                }
            )
        return summary


class _BackendWithUnknownSource(_Backend):
    def run_static(self, model, database, runtime_config, mode, **kwargs):
        summary = super().run_static(model, database, runtime_config, mode, **kwargs)
        if mode == "static_ctx":
            source_map = summary.get_context_source_map()
            source_map["context_dense"] = ({"unknown": True},)
        return summary


class _Database:
    backend = "vllm"
    system = "h200_sxm"
    version = "0.19.0"
    _vllm_serving_state_data = None

    def __init__(self, *, forward_total: bool) -> None:
        self.forward_total = forward_total

    def query_vllm_serving_state(self, **kwargs):
        row_kind = kwargs["row_kind"]
        if row_kind == "forward_total":
            if self.forward_total:
                return PerformanceResult(40.0, sources=(_source("forward_total"),))
            return None
        if row_kind == "category" and kwargs["category"] == "moe_gemm_or_aux":
            return PerformanceResult(20.0, sources=(_source("serving_moe"),))
        return None


def _compute(calc: IterationLatencyCalculator) -> None:
    calc.compute(
        prefill_tokens=8,
        prefill_batch_size=1,
        prefill_seq_len=8,
        decode_batch_size=2,
        decode_avg_kv_len=16,
    )


def test_forward_total_records_only_the_charged_serving_state_source() -> None:
    backend = _Backend()
    calc = IterationLatencyCalculator(backend, _Model(), _Database(forward_total=True))

    _compute(calc)

    assert backend.calls == 0
    assert calc.get_performance_source_map() == {
        "context": {"serving_state:mixed_prefill:forward_total": (_source("forward_total"),)},
        "generation": {},
    }


def test_forward_total_with_overhead_exposes_unapproved_charge_source() -> None:
    calc = IterationLatencyCalculator(
        _Backend(),
        _Model(),
        _Database(forward_total=True),
        per_iteration_overhead_ms=0.5,
    )

    _compute(calc)

    ledger = calc.get_charge_ledger()
    assert len(ledger) == 1
    assert ledger[0].reconciled
    assert ledger[0].charge_sum_ms == pytest.approx(40.5)
    overhead = next(charge for charge in ledger[0].charges if charge.charge_id == "per_iteration_overhead")
    assert overhead.latency_ms == pytest.approx(0.5)
    assert overhead.sources[0].formula_id == "unapproved_per_iteration_overhead"
    assert overhead.sources[0].approved is False


def test_category_override_replaces_only_the_charged_category_sources() -> None:
    calc = IterationLatencyCalculator(_Backend(), _Model(), _Database(forward_total=False))

    _compute(calc)

    source_map = calc.get_performance_source_map()
    assert "context_moe" not in source_map["context"]
    assert set(source_map["context"]) == {
        "serving_state:mixed_prefill:moe_gemm_or_aux",
        "context_dense",
        "context_attention",
    }
    assert set(source_map["generation"]) == {"generation_attention"}
    for phase in source_map.values():
        for sources in phase.values():
            validate_performance_result_sources(PerformanceResult(1.0, sources=sources))


def test_charge_ledger_preserves_unhashable_unknown_source_for_audit() -> None:
    calc = IterationLatencyCalculator(
        _BackendWithUnknownSource(),
        _Model(),
        _Database(forward_total=False),
    )

    _compute(calc)

    assert any(
        isinstance(source, dict)
        for charge in calc.get_charge_ledger()[0].charges
        for source in charge.sources
    )
