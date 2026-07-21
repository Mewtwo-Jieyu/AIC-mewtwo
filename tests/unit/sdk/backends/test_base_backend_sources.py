from __future__ import annotations

from types import SimpleNamespace

from aiconfigurator.sdk.backends.base_backend import BaseBackend
from aiconfigurator.sdk.config import RuntimeConfig
from aiconfigurator.sdk.perf_source import PerfSourceRecord
from aiconfigurator.sdk.performance_result import PerformanceResult


class _SourceBackend(BaseBackend):
    def run_agg(self, model, database, runtime_config, **kwargs):  # noqa: ANN001, ANN003
        raise NotImplementedError

    def find_best_agg_result_under_constraints(
        self, model, database, runtime_config, **kwargs  # noqa: ANN001, ANN003
    ):
        raise NotImplementedError

    def _get_memory_usage(
        self,
        model,
        database,
        batch_size,
        beam_width,
        isl,
        osl,
        num_tokens=0,
        **kwargs,
    ):
        return {"total": 0.0}


class _SourceOperation:
    _name = "source_op"

    def query(self, database, **kwargs):  # noqa: ANN001, ANN003
        return PerformanceResult(
            2.0,
            energy=4.0,
            sources=(
                PerfSourceRecord.structural(
                    formula_id="test_source_op",
                    formula_inputs={"x": kwargs["x"]},
                ),
            ),
        )


def _model() -> SimpleNamespace:
    config = SimpleNamespace(
        tp_size=1,
        pp_size=1,
        attention_dp_size=1,
        moe_tp_size=1,
        moe_ep_size=1,
        gemm_quant_mode=SimpleNamespace(name="float16"),
        kvcache_quant_mode=SimpleNamespace(name="float16"),
        fmha_quant_mode=SimpleNamespace(name="float16"),
        moe_quant_mode=SimpleNamespace(name="float16"),
        comm_quant_mode=SimpleNamespace(name="half"),
    )
    return SimpleNamespace(
        model_name="test/model",
        model_path="test/model",
        config=config,
        context_ops=[_SourceOperation()],
        generation_ops=[_SourceOperation()],
        _nextn=0,
    )


def test_run_static_collects_sources_before_float_conversion() -> None:
    database = SimpleNamespace(
        backend="vllm",
        version="test",
        system="test",
        system_spec={"gpu": {"mem_capacity": 1 << 30}},
    )

    summary = _SourceBackend().run_static(
        _model(),
        database,
        RuntimeConfig(batch_size=1, beam_width=1, isl=8, osl=3),
        mode="static",
        stride=2,
        latency_correction_scale=0.5,
    )

    context = summary.get_context_source_map()["source_op"]
    generation = summary.get_generation_source_map()["source_op"]
    assert context[0].formula_id == "test_source_op"
    assert [transform.factor for transform in context[0].transforms] == [0.5]
    assert [transform.factor for transform in generation[0].transforms] == [2.0, 0.5]
