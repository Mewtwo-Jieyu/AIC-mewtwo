"""Tests for CB simulator scheduler and simulator."""
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from aiconfigurator.sdk import common
from aiconfigurator.sdk.backends.cb_simulator.datatypes import (
    CBSimConfig, Request, RequestState,
)
from aiconfigurator.sdk.backends.cb_simulator.iteration_latency import IterationLatencyCalculator
from aiconfigurator.sdk.backends.cb_simulator.scheduler import CBScheduler
from aiconfigurator.sdk.backends.cb_simulator.simulator import CBSimulator
from aiconfigurator.sdk.backends.vllm_backend import VLLMBackend
from aiconfigurator.sdk.config import RuntimeConfig
from aiconfigurator.sdk.inference_summary import InferenceSummary
from aiconfigurator.sdk.operations import MoEDispatch
from aiconfigurator.sdk.operations import MoE
from aiconfigurator.sdk.performance_result import PerformanceResult


def _load_diagnose_cb_iter_latency_module():
    script_path = (
        Path(__file__).resolve().parents[4]
        / "scripts"
        / "diagnose_cb_iter_latency.py"
    )
    spec = importlib.util.spec_from_file_location(
        "diagnose_cb_iter_latency",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_validate_cb_simulator_module():
    script_path = (
        Path(__file__).resolve().parents[4]
        / "scripts"
        / "validate_cb_simulator.py"
    )
    spec = importlib.util.spec_from_file_location(
        "validate_cb_simulator",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_req(rid: int, isl: int = 1000, osl: int = 100) -> Request:
    return Request(request_id=rid, isl=isl, osl=osl, arrival_time_ms=0.0)


def _make_decoding(rid: int, isl: int = 1000, osl: int = 200, gen: int = 10) -> Request:
    r = _make_req(rid, isl, osl)
    r.state = RequestState.DECODING
    r.prefill_tokens_remaining = 0
    r.generated_tokens = gen
    return r


def _make_prefilling(rid: int, isl: int = 10000, remaining: int = 5000) -> Request:
    r = _make_req(rid, isl, osl=200)
    r.state = RequestState.PREFILLING
    r.prefill_tokens_remaining = remaining
    return r


class TestCBScheduler:
    def test_single_request_full_prefill(self) -> None:
        cfg = CBSimConfig(max_num_batched_tokens=8192)
        sched = CBScheduler(cfg)
        result = sched.schedule(
            waiting=[_make_req(0, isl=4000)],
            running=[],
        )
        assert len(result.prefill_reqs) == 1
        assert result.prefill_tokens[0] == 4000

    def test_chunked_prefill_caps_at_budget(self) -> None:
        cfg = CBSimConfig(max_num_batched_tokens=4096)
        sched = CBScheduler(cfg)
        result = sched.schedule(
            waiting=[_make_req(0, isl=10000)],
            running=[],
        )
        assert result.prefill_tokens[0] == 4096

    def test_decode_reserves_budget_first(self) -> None:
        cfg = CBSimConfig(max_num_batched_tokens=100)
        sched = CBScheduler(cfg)
        running = [_make_decoding(i) for i in range(80)]
        result = sched.schedule(
            waiting=[_make_req(100, isl=50)],
            running=running,
        )
        assert len(result.decode_reqs) == 80
        assert result.prefill_tokens[100] == 20  # 100 - 80

    def test_running_order_interleaves_prefill_before_later_decodes(self) -> None:
        cfg = CBSimConfig(max_num_batched_tokens=4)
        sched = CBScheduler(cfg)
        early_decode = [_make_decoding(0), _make_decoding(1)]
        partial = _make_prefilling(2, isl=10, remaining=3)
        late_decode = [_make_decoding(3), _make_decoding(4)]

        result = sched.schedule(
            waiting=[],
            running=[*early_decode, partial, *late_decode],
        )

        assert result.decode_reqs == early_decode
        assert result.prefill_tokens[2] == 2
        assert late_decode[0] not in result.decode_reqs
        assert late_decode[1] not in result.decode_reqs

    def test_max_seqs_blocks_new_prefill(self) -> None:
        cfg = CBSimConfig(max_num_batched_tokens=8192, max_num_seqs=2)
        sched = CBScheduler(cfg)
        running = [_make_decoding(i) for i in range(2)]
        result = sched.schedule(
            waiting=[_make_req(10, isl=100)],
            running=running,
        )
        assert len(result.prefill_reqs) == 0
        assert len(result.decode_reqs) == 2

    def test_continue_partial_prefill_in_running(self) -> None:
        """Partially prefilled request in running set continues."""
        cfg = CBSimConfig(max_num_batched_tokens=8192)
        sched = CBScheduler(cfg)
        partial = _make_prefilling(0, isl=10000, remaining=3000)
        result = sched.schedule(
            waiting=[],
            running=[partial],
        )
        assert result.prefill_tokens[0] == 3000

    def test_multiple_full_prefills_allowed(self) -> None:
        """Multiple small prefills can fit in one iteration."""
        cfg = CBSimConfig(max_num_batched_tokens=8192, max_num_seqs=256)
        sched = CBScheduler(cfg)
        result = sched.schedule(
            waiting=[_make_req(i, isl=2000) for i in range(4)],
            running=[],
        )
        # 4 x 2000 = 8000 <= 8192, all should be admitted
        assert len(result.prefill_reqs) == 4
        assert result.total_prefill_tokens == 8000

    def test_stops_after_first_partial_chunk(self) -> None:
        """Stops admitting after first partial prefill."""
        cfg = CBSimConfig(max_num_batched_tokens=5000, max_num_seqs=256)
        sched = CBScheduler(cfg)
        result = sched.schedule(
            waiting=[_make_req(0, isl=3000), _make_req(1, isl=3000)],
            running=[],
        )
        # First: 3000 full. Second: 2000 partial -> stop.
        assert len(result.prefill_reqs) == 2
        assert result.prefill_tokens[0] == 3000
        assert result.prefill_tokens[1] == 2000

    def test_empty_schedule(self) -> None:
        cfg = CBSimConfig()
        sched = CBScheduler(cfg)
        result = sched.schedule([], [])
        assert result.is_empty

    def test_decode_overflow_handled(self) -> None:
        """When decode requests exceed budget, decode scheduling is capped."""
        cfg = CBSimConfig(max_num_batched_tokens=10)
        sched = CBScheduler(cfg)
        running = [_make_decoding(i) for i in range(20)]
        result = sched.schedule(waiting=[], running=running)
        assert len(result.decode_reqs) == 10
        assert len(result.prefill_reqs) == 0

    def test_long_prefill_threshold_caps_chunk(self) -> None:
        cfg = CBSimConfig(
            max_num_batched_tokens=4096,
            long_prefill_token_threshold=1024,
        )
        sched = CBScheduler(cfg)
        result = sched.schedule(
            waiting=[_make_req(0, isl=10000)],
            running=[],
        )
        assert result.prefill_tokens[0] == 1024

    def test_long_prefill_threshold_zero_disabled(self) -> None:
        cfg = CBSimConfig(
            max_num_batched_tokens=4096,
            long_prefill_token_threshold=0,
        )
        sched = CBScheduler(cfg)
        result = sched.schedule(
            waiting=[_make_req(0, isl=10000)],
            running=[],
        )
        assert result.prefill_tokens[0] == 4096

    def test_preemption_frees_and_requeues(self) -> None:
        cfg = CBSimConfig(
            max_num_batched_tokens=32,
            num_gpu_blocks=4,
            block_size=16,
        )
        sched = CBScheduler(cfg)
        keep = _make_prefilling(0, isl=48, remaining=16)
        victim = _make_prefilling(1, isl=48, remaining=16)
        waiting = [_make_req(2, isl=32)]
        running = [keep, victim]

        result = sched.schedule(waiting=waiting, running=running)

        assert victim.state == RequestState.PREEMPTED
        assert victim.num_preemptions == 1
        assert waiting[0] is victim
        assert running == [keep]
        assert result.prefill_tokens[0] == 16
        assert result.prefill_tokens[2] == 16

    def test_waiting_admission_does_not_preempt_running(self) -> None:
        cfg = CBSimConfig(
            max_num_batched_tokens=32,
            num_gpu_blocks=4,
            block_size=16,
        )
        sched = CBScheduler(cfg)
        running_req = _make_decoding(0, isl=32, gen=0)
        waiting_req = _make_req(1, isl=32)
        running = [running_req]
        waiting = [waiting_req]

        result = sched.schedule(waiting=waiting, running=running)

        assert running == [running_req]
        assert waiting == [waiting_req]
        assert running_req.num_preemptions == 0
        assert waiting_req.state == RequestState.WAITING
        assert result.decode_reqs == [running_req]
        assert result.prefill_reqs == []
        assert result.prefill_tokens == {}

    def test_no_preemption_with_unlimited_blocks(self) -> None:
        cfg = CBSimConfig(
            max_num_batched_tokens=32,
            num_gpu_blocks=0,
            block_size=16,
        )
        sched = CBScheduler(cfg)
        running = [_make_prefilling(0, isl=48, remaining=16)]
        waiting = [_make_req(1, isl=32)]

        result = sched.schedule(waiting=waiting, running=running)

        assert running[0].state == RequestState.PREFILLING
        assert running[0].num_preemptions == 0
        assert waiting[0].state == RequestState.WAITING
        assert result.prefill_tokens[0] == 16
        assert result.prefill_tokens[1] == 16


# --- Simulator tests with mocked latency calculator ---


class _FakeLatencyCalc:
    """Deterministic fake for iteration latency.

    Models sub-linear scaling: fixed overhead + sqrt(batch) for decode,
    so higher concurrency yields better throughput (batching benefit).
    """

    def compute(
        self, prefill_tokens: int, prefill_batch_size: int,
        prefill_seq_len: int, decode_batch_size: int,
        decode_avg_kv_len: int,
    ) -> float:
        import math
        # Fixed 1ms base + 0.001ms per prefill token + sqrt scaling for decode
        base = 1.0
        prefill_cost = prefill_tokens * 0.001
        decode_cost = math.sqrt(max(decode_batch_size, 0)) * 0.5
        return base + prefill_cost + decode_cost


class _MixedVsPureDecodeLatencyCalc:
    """Fake calc where a mixed/prefill iteration is far more expensive than
    a pure decode iteration, and pure-decode latency grows with KV length.

    Used to prove the skip trapezoid seeds from pure-decode latency, not from
    the triggering (mixed) iteration.
    """

    MIXED_MS = 10_000.0

    def compute(
        self, prefill_tokens: int, prefill_batch_size: int,
        prefill_seq_len: int, decode_batch_size: int,
        decode_avg_kv_len: int,
    ) -> float:
        if prefill_tokens > 0:
            return self.MIXED_MS
        return 5.0 + decode_avg_kv_len * 0.001


class _KVGrowthLatencyCalc(_FakeLatencyCalc):
    def compute(
        self, prefill_tokens: int, prefill_batch_size: int,
        prefill_seq_len: int, decode_batch_size: int,
        decode_avg_kv_len: int,
    ) -> float:
        base = super().compute(
            prefill_tokens,
            prefill_batch_size,
            prefill_seq_len,
            decode_batch_size,
            decode_avg_kv_len,
        )
        return base + decode_avg_kv_len * 0.001


def _make_testable_sim(config: CBSimConfig | None = None) -> CBSimulator:
    """Create CBSimulator with injected fake latency calc."""
    cfg = config or CBSimConfig(num_requests=20, warmup_requests=5)
    sim = CBSimulator(
        backend=MagicMock(), model=MagicMock(),
        database=MagicMock(), config=cfg,
    )
    # Override the factory hook (R2-4)
    sim._create_latency_calc = lambda prefix: _FakeLatencyCalc()
    return sim


class TestCBSimulatorUnit:
    """Unit tests with mocked latency calculator."""

    def test_produces_positive_metrics(self) -> None:
        """Smoke test: simulator produces positive TTFT/TPOT/throughput."""
        sim = _make_testable_sim()
        result = sim.run(isl=1000, osl=50, concurrency=4, num_gpus=1)
        assert result.mean_ttft_ms > 0
        assert result.mean_tpot_ms > 0
        assert result.throughput_tok_s > 0
        assert result.steady_state_requests > 0

    def test_throughput_increases_with_concurrency(self) -> None:
        sim = _make_testable_sim()
        r1 = sim.run(isl=1000, osl=50, concurrency=1, num_gpus=1)
        r4 = sim.run(isl=1000, osl=50, concurrency=4, num_gpus=1)
        assert r4.throughput_tok_s > r1.throughput_tok_s

    def test_ttft_increases_with_concurrency(self) -> None:
        sim = _make_testable_sim()
        r1 = sim.run(isl=1000, osl=50, concurrency=1, num_gpus=1)
        r8 = sim.run(isl=1000, osl=50, concurrency=8, num_gpus=1)
        assert r8.mean_ttft_ms > r1.mean_ttft_ms

    def test_iteration_stats_populated(self) -> None:
        sim = _make_testable_sim()
        result = sim.run(isl=1000, osl=50, concurrency=4, num_gpus=1)
        assert result.avg_decode_reqs_per_iter > 0
        assert result.avg_tokens_per_iter > 0
        assert result.steady_state_iterations > 0
        assert result.steady_state_time_ms > 0
        assert result.peak_tokens_per_iter >= result.avg_tokens_per_iter

    def test_request_state_transitions(self) -> None:
        r = Request(request_id=0, isl=100, osl=10, arrival_time_ms=0.0)
        assert r.state == RequestState.WAITING
        r.state = RequestState.PREFILLING
        r.prefill_tokens_remaining = 0
        r.state = RequestState.DECODING
        r.generated_tokens = 9
        r.state = RequestState.DONE
        assert r.state == RequestState.DONE

    def test_kv_cache_len_tracks_progress(self) -> None:
        r = Request(request_id=0, isl=1000, osl=100, arrival_time_ms=0.0)
        assert r.kv_cache_len == 0
        r.prefill_tokens_remaining = 500
        assert r.kv_cache_len == 500
        r.prefill_tokens_remaining = 0
        r.generated_tokens = 50
        assert r.kv_cache_len == 1050

    def test_decode_skip_accounts_for_kv_growth(self) -> None:
        sim = _make_testable_sim(CBSimConfig(num_requests=20, warmup_requests=5))

        skip_lat = sim._estimate_decode_skip_latency(
            latency_calc=_KVGrowthLatencyCalc(),
            decode_batch_size=8,
            start_avg_kv_len=1024,
            skip_iters=10,
        )

        assert skip_lat > 20.0

    def test_decode_skip_seed_uses_pure_decode_not_triggering_iter(self) -> None:
        # Phase397d: the skip trapezoid must ramp between two PURE-DECODE
        # latencies (at start_avg_kv_len and start_avg_kv_len + skip_iters),
        # NOT reuse the triggering iteration's (possibly mixed/prefill) latency.
        sim = _make_testable_sim(CBSimConfig(num_requests=20, warmup_requests=5))
        calc = _MixedVsPureDecodeLatencyCalc()

        start_kv = 2048
        skip = 12
        bs = 8
        skip_lat = sim._estimate_decode_skip_latency(
            latency_calc=calc,
            decode_batch_size=bs,
            start_avg_kv_len=start_kv,
            skip_iters=skip,
        )

        first = calc.compute(
            prefill_tokens=0, prefill_batch_size=0, prefill_seq_len=1,
            decode_batch_size=bs, decode_avg_kv_len=start_kv,
        )
        end = calc.compute(
            prefill_tokens=0, prefill_batch_size=0, prefill_seq_len=1,
            decode_batch_size=bs, decode_avg_kv_len=start_kv + skip,
        )
        expected = (first + end) * skip / 2.0

        assert skip_lat == pytest.approx(expected)
        # No prefill/mixed compute() call may leak into the seed: the pure-decode
        # endpoints are tiny relative to the mixed latency, so a contaminated
        # seed would inflate skip_lat by orders of magnitude.
        assert skip_lat < calc.MIXED_MS

    def test_preempted_request_reprefills(self) -> None:
        cfg = CBSimConfig(
            max_num_batched_tokens=32,
            num_requests=2,
            warmup_requests=0,
            num_gpu_blocks=6,
            block_size=16,
        )
        sim = _make_testable_sim(cfg)
        captured: dict[str, list[Request]] = {}
        original_collect = sim._collect_metrics

        def _capture_collect(*args, **kwargs):
            captured["completed"] = list(args[0])
            return original_collect(*args, **kwargs)

        sim._collect_metrics = _capture_collect
        sim.run(isl=48, osl=2, concurrency=2, num_gpus=1)

        preempted = [r for r in captured["completed"] if r.num_preemptions > 0]
        assert len(preempted) == 1
        assert preempted[0].state == RequestState.DONE
        assert preempted[0].prefill_tokens_remaining == 0
        assert preempted[0].first_token_ms > 0


class _FakeBackendForIteration:
    def run_static(self, model, database, runtime_config: RuntimeConfig, mode: str, **kwargs):
        summary = InferenceSummary(runtime_config)
        if mode == "static_ctx":
            summary.set_context_latency_dict(
                {
                    "gemm": 10.0,
                    "moe": 5.0,
                    "context_attention": 8.0,
                }
            )
        elif mode == "static_gen":
            summary.set_generation_latency_dict(
                {
                    "generation_attention": 7.0,
                    "generation_moe": 11.0,
                    "generation_dispatch": 13.0,
                }
            )
        else:
            raise AssertionError(f"unexpected mode: {mode}")
        return summary


class _TokenScaledBackendForIteration:
    """static_ctx non-attention scales with isl; attention is fixed.

    Lets tests observe that a mixed iteration charges the token-parallel
    non-attention ops at the MERGED token count (prefill chunk + decode batch),
    not the prefill chunk alone.
    """

    def run_static(self, model, database, runtime_config: RuntimeConfig, mode: str, **kwargs):
        summary = InferenceSummary(runtime_config)
        if mode == "static_ctx":
            isl = runtime_config.isl
            summary.set_context_latency_dict(
                {
                    "gemm": 0.01 * isl,
                    "moe": 0.005 * isl,
                    "context_attention": 8.0,
                }
            )
        elif mode == "static_gen":
            summary.set_generation_latency_dict(
                {
                    "generation_attention": 7.0,
                    "generation_moe": 11.0,
                    "generation_dispatch": 13.0,
                }
            )
        else:
            raise AssertionError(f"unexpected mode: {mode}")
        return summary


class _FakeModelForIteration:
    class config:
        tp_size = 4


class _FakeKimiDP2ModelForServingState:
    class config:
        tp_size = 4
        attention_dp_size = 2
        moe_tp_size = 1
        moe_ep_size = 8


class _FakeKimiTP8ModelForServingState:
    class config:
        tp_size = 8
        attention_dp_size = 1
        moe_tp_size = 1
        moe_ep_size = 8


class _ServingStateBackendForIteration:
    def run_static(self, model, database, runtime_config: RuntimeConfig, mode: str, **kwargs):
        summary = InferenceSummary(runtime_config)
        if mode == "static_ctx":
            summary.set_context_latency_dict(
                {
                    "context_moe": 10.0,
                    "context_moe_pre_dispatch": 12.0,
                    "context_moe_post_dispatch": 8.0,
                    "context_dense": 5.0,
                    "context_attention": 7.0,
                }
            )
        elif mode == "static_gen":
            summary.set_generation_latency_dict(
                {
                    "generation_attention": 3.0,
                    "generation_moe": 10.0,
                    "generation_moe_pre_dispatch": 12.0,
                    "generation_moe_post_dispatch": 8.0,
                }
            )
        else:
            raise AssertionError(f"unexpected mode: {mode}")
        return summary


class _ServingStateDB:
    system = "h200_sxm"
    backend = "vllm"
    version = "0.19.0"

    def __init__(self) -> None:
        self.calls = []

    def query_vllm_serving_state(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("row_kind") == "non_attn_total":
            return None
        if kwargs["phase"] == "mixed_prefill" and kwargs["category"] == "moe_gemm_or_aux":
            return PerformanceResult(50.0, energy=0.0)
        if kwargs["phase"] == "mixed_prefill" and kwargs["category"] == "ep_a2a":
            return PerformanceResult(100.0, energy=0.0)
        return None


class _NonAttnTotalServingStateDB(_ServingStateDB):
    def query_vllm_serving_state(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("row_kind") == "non_attn_total":
            return PerformanceResult(900.0, energy=0.0)
        return None


class _ForwardTotalServingStateDB(_ServingStateDB):
    def query_vllm_serving_state(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("row_kind") == "forward_total" and kwargs["phase"] == "mixed_prefill":
            return PerformanceResult(1100.0, energy=0.0)
        if kwargs.get("row_kind") == "forward_total" and kwargs["phase"] == "decode":
            return PerformanceResult(40.0, energy=0.0)
        raise AssertionError(f"forward_total should short-circuit before category queries: {kwargs}")


class _BoundedServingStateDB(_ServingStateDB):
    def __init__(self) -> None:
        super().__init__()
        scope = (
            "kimi-k2.5",
            "tp4dp2ep8",
            "mixed_prefill",
            "moe_gemm_or_aux",
            7168,
            8,
            8,
            "CompressedTensorsWNA16MarlinMoEMethod",
        )
        decode_scope = (
            "kimi-k2.5",
            "tp4dp2ep8",
            "decode",
            "moe_gemm_or_aux",
            7168,
            8,
            8,
            "CompressedTensorsWNA16MarlinMoEMethod",
        )
        self._vllm_serving_state_data = {
            scope: {
                8000: {
                    1: {"latency": 50.0, "energy": 0.0},
                    34: {"latency": 50.0, "energy": 0.0},
                }
            },
            decode_scope: {
                8: {8: {"latency": 1.0, "energy": 0.0}},
                36: {36: {"latency": 2.0, "energy": 0.0}},
                52: {52: {"latency": 3.0, "energy": 0.0}},
            },
        }

    def query_vllm_serving_state(self, **kwargs):
        self.calls.append(kwargs)
        key = (
            kwargs["model"],
            kwargs["topology"],
            kwargs["phase"],
            kwargs["category"],
            kwargs["hidden_size"],
            kwargs["topk"],
            kwargs["moe_ep_size"],
            kwargs["quant_runtime"],
        )
        table = self._vllm_serving_state_data.get(key)
        if table is None:
            return None
        batch_table = table.get(kwargs["bucket_tokens"])
        if batch_table is None:
            return None
        result = batch_table.get(kwargs["decode_batch"])
        return None if result is None else PerformanceResult(result["latency"], energy=0.0)


class TestIterationLatencyCalculator:
    def test_mixed_folds_decode_non_attention_into_merged_pass(self) -> None:
        # In a mixed iteration the token-parallel non-attention ops are charged
        # once over the merged batch (Pass 1). Pass 3 contributes only the
        # decode attention term; decode non-attention is NOT double-counted.
        calc = IterationLatencyCalculator(
            backend=_FakeBackendForIteration(),
            model=MagicMock(),
            database=MagicMock(),
        )
        total = calc.compute(
            prefill_tokens=1024,
            prefill_batch_size=1,
            prefill_seq_len=1024,
            decode_batch_size=4,
            decode_avg_kv_len=2048,
        )
        breakdown = calc.get_last_breakdown()
        assert breakdown is not None
        # Mixed iterations charge the physical serial sum of merged non-attn,
        # context attention, and decode attention.
        assert total == pytest.approx(30.0)
        assert breakdown.context_non_attention_ms == pytest.approx(15.0)
        assert breakdown.context_attention_ms == pytest.approx(8.0)
        assert breakdown.generation_non_attention_ms == pytest.approx(0.0)
        assert breakdown.generation_attention_ms == pytest.approx(7.0)

    def test_serving_state_replaces_mixed_prefill_categories_for_dp2_only(self) -> None:
        db = _ServingStateDB()
        calc = IterationLatencyCalculator(
            backend=_ServingStateBackendForIteration(),
            model=_FakeKimiDP2ModelForServingState(),
            database=db,
        )

        total = calc.compute(
            prefill_tokens=8000,
            prefill_batch_size=1,
            prefill_seq_len=8000,
            decode_batch_size=8,
            decode_avg_kv_len=8000,
        )

        assert total == pytest.approx(165.0)
        assert {call["category"] for call in db.calls} >= {"moe_gemm_or_aux", "ep_a2a"}

    def test_serving_state_non_attn_total_replaces_whole_block_when_categories_miss(self) -> None:
        db = _NonAttnTotalServingStateDB()
        calc = IterationLatencyCalculator(
            backend=_ServingStateBackendForIteration(),
            model=_FakeKimiDP2ModelForServingState(),
            database=db,
        )

        total = calc.compute(
            prefill_tokens=8000,
            prefill_batch_size=1,
            prefill_seq_len=8000,
            decode_batch_size=64,
            decode_avg_kv_len=8000,
        )

        assert total == pytest.approx(910.0)
        assert any(call.get("row_kind") == "non_attn_total" for call in db.calls)

    def test_serving_state_forward_total_replaces_mixed_step_before_category_queries(self) -> None:
        db = _ForwardTotalServingStateDB()
        calc = IterationLatencyCalculator(
            backend=_ServingStateBackendForIteration(),
            model=_FakeKimiDP2ModelForServingState(),
            database=db,
        )

        total = calc.compute(
            prefill_tokens=8000,
            prefill_batch_size=1,
            prefill_seq_len=8000,
            decode_batch_size=64,
            decode_avg_kv_len=8000,
        )

        assert total == pytest.approx(1100.0)
        assert [call["row_kind"] for call in db.calls] == ["forward_total"]

    def test_serving_state_forward_total_replaces_decode_step_before_category_queries(self) -> None:
        db = _ForwardTotalServingStateDB()
        calc = IterationLatencyCalculator(
            backend=_ServingStateBackendForIteration(),
            model=_FakeKimiDP2ModelForServingState(),
            database=db,
        )

        total = calc.compute(
            prefill_tokens=0,
            prefill_batch_size=0,
            prefill_seq_len=1,
            decode_batch_size=64,
            decode_avg_kv_len=8000,
        )

        assert total == pytest.approx(40.0)
        assert [call["row_kind"] for call in db.calls] == ["forward_total"]

    def test_serving_state_does_not_apply_to_tp8(self) -> None:
        db = _ServingStateDB()
        calc = IterationLatencyCalculator(
            backend=_ServingStateBackendForIteration(),
            model=_FakeKimiTP8ModelForServingState(),
            database=db,
        )

        total = calc.compute(
            prefill_tokens=8000,
            prefill_batch_size=1,
            prefill_seq_len=8000,
            decode_batch_size=8,
            decode_avg_kv_len=8000,
        )

        assert total == pytest.approx(45.0)
        assert db.calls == []

    def test_serving_state_records_out_of_grid_misses(self) -> None:
        db = _BoundedServingStateDB()
        calc = IterationLatencyCalculator(
            backend=_ServingStateBackendForIteration(),
            model=_FakeKimiDP2ModelForServingState(),
            database=db,
        )

        calc.compute(
            prefill_tokens=8000,
            prefill_batch_size=1,
            prefill_seq_len=8000,
            decode_batch_size=64,
            decode_avg_kv_len=8000,
        )
        calc.compute(
            prefill_tokens=0,
            prefill_batch_size=0,
            prefill_seq_len=1,
            decode_batch_size=64,
            decode_avg_kv_len=8000,
        )

        audit = [record.as_dict() for record in calc.get_serving_state_query_audit()]
        assert {
            (row["phase"], row["category"], row["miss_reason"])
            for row in audit
            if row["category"] == "moe_gemm_or_aux"
        } >= {
            ("mixed_prefill", "moe_gemm_or_aux", "decode_batch_above_range"),
            ("decode", "moe_gemm_or_aux", "bucket_above_range"),
        }

    def test_mixed_non_attention_uses_merged_total_tokens(self) -> None:
        # Mixed iterations must charge the token-parallel non-attention ops at
        # the MERGED token count (prefill + decode), matching the single real
        # fused-MoE forward -- not at the prefill chunk alone (old split).
        calc = IterationLatencyCalculator(
            backend=_TokenScaledBackendForIteration(),
            model=MagicMock(),
            database=MagicMock(),
        )
        prefill_tokens = 1000
        decode_bs = 24

        calc.compute(
            prefill_tokens=prefill_tokens,
            prefill_batch_size=1,
            prefill_seq_len=prefill_tokens,
            decode_batch_size=decode_bs,
            decode_avg_kv_len=2048,
        )
        mixed = calc.get_last_breakdown()

        # Reference: a pure-prefill iteration over the MERGED token count.
        calc.compute(
            prefill_tokens=prefill_tokens + decode_bs,
            prefill_batch_size=1,
            prefill_seq_len=prefill_tokens + decode_bs,
            decode_batch_size=0,
            decode_avg_kv_len=0,
        )
        merged_ref = calc.get_last_breakdown()

        # Reference: a pure-prefill iteration over ONLY the prefill chunk
        # (the latency the old split granularity would have charged).
        calc.compute(
            prefill_tokens=prefill_tokens,
            prefill_batch_size=1,
            prefill_seq_len=prefill_tokens,
            decode_batch_size=0,
            decode_avg_kv_len=0,
        )
        split_ref = calc.get_last_breakdown()

        assert mixed is not None
        assert merged_ref is not None
        assert split_ref is not None
        assert mixed.context_non_attention_ms == pytest.approx(
            merged_ref.context_non_attention_ms
        )
        assert mixed.context_non_attention_ms > split_ref.context_non_attention_ms
        # Decode non-attention is folded into the merged pass, not double-counted.
        assert mixed.generation_non_attention_ms == pytest.approx(0.0)

    def test_breakdown_is_cached(self) -> None:
        calc = IterationLatencyCalculator(
            backend=_FakeBackendForIteration(),
            model=MagicMock(),
            database=MagicMock(),
        )
        total1 = calc.compute(
            prefill_tokens=512,
            prefill_batch_size=1,
            prefill_seq_len=1024,
            decode_batch_size=2,
            decode_avg_kv_len=1536,
        )
        total2 = calc.compute(
            prefill_tokens=512,
            prefill_batch_size=1,
            prefill_seq_len=1024,
            decode_batch_size=2,
            decode_avg_kv_len=1536,
        )
        assert total1 == pytest.approx(total2)
        assert calc.get_last_breakdown() is not None

    def test_generation_moe_terms_are_not_tp_scaled_per_rank(self) -> None:
        # phase397e/f: generation_moe(+dispatch) latencies are already PER-RANK
        # (TP is encoded in the moe_tp_size lookup key), so the calculator must
        # NOT divide them by tp_size again. Even with a tp_size=4 model the raw
        # per-rank non-attention (generation_moe 11.0 + generation_dispatch 13.0
        # = 24.0) is charged unchanged, and pure decode sums it serially with
        # attention (24.0 + 7.0 = 31.0).
        calc = IterationLatencyCalculator(
            backend=_FakeBackendForIteration(),
            model=_FakeModelForIteration(),
            database=MagicMock(),
        )
        total = calc.compute(
            prefill_tokens=0,
            prefill_batch_size=0,
            prefill_seq_len=1,
            decode_batch_size=4,
            decode_avg_kv_len=2048,
        )
        breakdown = calc.get_last_breakdown()
        assert breakdown is not None
        assert breakdown.generation_non_attention_ms == pytest.approx(24.0)
        assert breakdown.generation_attention_ms == pytest.approx(7.0)
        assert total == pytest.approx(31.0)

    def test_pure_decode_latency_includes_dispatch(self) -> None:
        calc = IterationLatencyCalculator(
            backend=_FakeBackendForIteration(),
            model=MagicMock(),
            database=MagicMock(),
        )
        total = calc.compute(
            prefill_tokens=0,
            prefill_batch_size=0,
            prefill_seq_len=1,
            decode_batch_size=4,
            decode_avg_kv_len=2048,
        )
        breakdown = calc.get_last_breakdown()

        assert breakdown is not None
        assert breakdown.generation_non_attention_ms == pytest.approx(24.0)
        assert breakdown.generation_attention_ms == pytest.approx(7.0)
        assert total == pytest.approx(31.0)

    def test_per_iteration_overhead_is_added_to_total(self) -> None:
        calc = IterationLatencyCalculator(
            backend=_FakeBackendForIteration(),
            model=MagicMock(),
            database=MagicMock(),
            per_iteration_overhead_ms=2.5,
        )
        total = calc.compute(
            prefill_tokens=0,
            prefill_batch_size=0,
            prefill_seq_len=1,
            decode_batch_size=4,
            decode_avg_kv_len=2048,
        )
        breakdown = calc.get_last_breakdown()

        assert breakdown is not None
        assert breakdown.iteration_overhead_ms == pytest.approx(2.5)
        assert total == pytest.approx(33.5)

    def test_per_iteration_overhead_only_applies_with_decode(self) -> None:
        calc = IterationLatencyCalculator(
            backend=_FakeBackendForIteration(),
            model=MagicMock(),
            database=MagicMock(),
            per_iteration_overhead_ms=2.5,
        )
        total = calc.compute(
            prefill_tokens=1024,
            prefill_batch_size=1,
            prefill_seq_len=1024,
            decode_batch_size=0,
            decode_avg_kv_len=0,
        )
        breakdown = calc.get_last_breakdown()

        assert breakdown is not None
        assert breakdown.iteration_overhead_ms == pytest.approx(0.0)
        assert total == pytest.approx(23.0)

    def test_mixed_serial_sum_is_independent_of_overlap_factor(self) -> None:
        calc_no_overlap = IterationLatencyCalculator(
            backend=_FakeBackendForIteration(),
            model=MagicMock(),
            database=MagicMock(),
            overlap_factor=0.0,
        )
        calc_full_overlap = IterationLatencyCalculator(
            backend=_FakeBackendForIteration(),
            model=MagicMock(),
            database=MagicMock(),
            overlap_factor=1.0,
        )

        args = dict(
            prefill_tokens=1024,
            prefill_batch_size=1,
            prefill_seq_len=1024,
            decode_batch_size=4,
            decode_avg_kv_len=2048,
        )
        total_no_overlap = calc_no_overlap.compute(**args)
        total_full_overlap = calc_full_overlap.compute(**args)

        assert total_no_overlap == pytest.approx(30.0)
        assert total_full_overlap == pytest.approx(30.0)

    def test_pure_prefill_serial_sum_is_independent_of_overlap_factor(self) -> None:
        calc_no_overlap = IterationLatencyCalculator(
            backend=_FakeBackendForIteration(),
            model=MagicMock(),
            database=MagicMock(),
            overlap_factor=0.0,
        )
        calc_full_overlap = IterationLatencyCalculator(
            backend=_FakeBackendForIteration(),
            model=MagicMock(),
            database=MagicMock(),
            overlap_factor=1.0,
        )

        args = dict(
            prefill_tokens=1024,
            prefill_batch_size=1,
            prefill_seq_len=1024,
            decode_batch_size=0,
            decode_avg_kv_len=0,
        )
        total_no_overlap = calc_no_overlap.compute(**args)
        total_full_overlap = calc_full_overlap.compute(**args)

        assert total_no_overlap == pytest.approx(23.0)
        assert total_full_overlap == pytest.approx(23.0)

    def test_overlap_factor_validates_range(self) -> None:
        with pytest.raises(ValueError, match="overlap_factor"):
            IterationLatencyCalculator(
                backend=_FakeBackendForIteration(),
                model=MagicMock(),
                database=MagicMock(),
                overlap_factor=1.1,
            )


class _FakePerfDbForMoEDispatch:
    backend = common.BackendName.vllm.value
    system = "h200_sxm"
    version = "0.19.0"
    system_spec = {
        "gpu": {"sm_version": 90},
        "node": {"num_gpus_per_node": 8, "intra_node_bw": 450000000000},
    }

    def __init__(self) -> None:
        self.custom_allreduce_volumes = []
        self.nccl_volumes = []
        self.vllm_module_calls = []

    def query_custom_allreduce(self, quant_mode, num_gpus, volume):
        self.custom_allreduce_volumes.append(volume)
        return 1.0

    def query_nccl(self, quant_mode, num_gpus, op, volume):
        self.nccl_volumes.append((op, volume))
        return 2.0

    def query_vllm_module(
        self,
        model,
        hardware,
        vllm_version,
        topology,
        bucket_tokens,
        module_boundary,
        quant_runtime,
    ):
        self.vllm_module_calls.append(
            (
                model,
                hardware,
                vllm_version,
                topology,
                bucket_tokens,
                module_boundary,
                quant_runtime,
            )
        )
        return PerformanceResult(3.0, energy=0.0)


class TestMoEDispatchScaling:
    def test_vllm_dispatch_uses_module_perf_exact_bucket(self) -> None:
        op = MoEDispatch(
            "dispatch",
            1.0,
            hidden_size=1024,
            topk=8,
            num_experts=256,
            moe_tp_size=1,
            moe_ep_size=1,
            attention_dp_size=4,
            pre_dispatch=True,
            scale_num_tokens=1,
        )
        db = _FakePerfDbForMoEDispatch()

        latency = op.query(
            db,
            x=8192,
            model_name="moonshotai/Kimi-K2.5",
            vllm_module_topology="tp4dp2ep8",
        )

        assert float(latency) == pytest.approx(3.0)
        assert db.custom_allreduce_volumes == []
        assert db.nccl_volumes == []
        assert db.vllm_module_calls == [
            (
                "kimi-k2.5",
                "h200_sxm",
                "0.19.0",
                "tp4dp2ep8",
                8192,
                "ep8_comm_dispatch_combine",
                "CompressedTensorsWNA16MarlinMoEMethod",
            )
        ]

    def test_vllm_dispatch_falls_back_for_non_exact_bucket(self) -> None:
        op = MoEDispatch(
            "dispatch",
            1.0,
            hidden_size=1024,
            topk=8,
            num_experts=256,
            moe_tp_size=1,
            moe_ep_size=1,
            attention_dp_size=4,
            pre_dispatch=True,
            scale_num_tokens=1,
        )
        db = _FakePerfDbForMoEDispatch()

        latency = op.query(
            db,
            x=128,
            model_name="moonshotai/Kimi-K2.5",
            vllm_module_topology="tp4dp2ep8",
        )

        assert float(latency) == pytest.approx(2.0)
        assert db.vllm_module_calls == []
        assert db.nccl_volumes == [("all_gather", 128 * 1024 * 4)]

    def test_vllm_ep8_dispatch_non_exact_bucket_uses_combined_alltoall_model(self) -> None:
        op = MoEDispatch(
            "dispatch",
            60.0,
            hidden_size=7168,
            topk=8,
            num_experts=384,
            moe_tp_size=1,
            moe_ep_size=8,
            attention_dp_size=2,
            pre_dispatch=True,
            scale_num_tokens=4,
        )
        db = _FakePerfDbForMoEDispatch()

        latency = op.query(
            db,
            x=32006,
            context_prefill_tokens=32000,
            model_name="moonshotai/Kimi-K2.5",
            vllm_module_topology="tp4dp2ep8",
        )

        assert float(latency) >= 489.3
        assert db.vllm_module_calls == []
        assert db.custom_allreduce_volumes == []
        assert db.nccl_volumes == []

    def test_vllm_dispatch_post_scope_is_zero_to_avoid_double_count(self) -> None:
        op = MoEDispatch(
            "dispatch",
            1.0,
            hidden_size=1024,
            topk=8,
            num_experts=256,
            moe_tp_size=1,
            moe_ep_size=1,
            attention_dp_size=4,
            pre_dispatch=False,
            scale_num_tokens=1,
        )
        db = _FakePerfDbForMoEDispatch()

        latency = op.query(
            db,
            x=8192,
            model_name="moonshotai/Kimi-K2.5",
            vllm_module_topology="tp4dp2ep8",
        )

        assert float(latency) == pytest.approx(0.0)
        assert db.custom_allreduce_volumes == []
        assert db.nccl_volumes == []
        assert db.vllm_module_calls == []

    def test_vllm_dispatch_post_scope_falls_back_for_non_exact_bucket(self) -> None:
        op = MoEDispatch(
            "dispatch",
            1.0,
            hidden_size=1024,
            topk=8,
            num_experts=256,
            moe_tp_size=1,
            moe_ep_size=1,
            attention_dp_size=4,
            pre_dispatch=False,
            scale_num_tokens=1,
        )
        db = _FakePerfDbForMoEDispatch()

        latency = op.query(
            db,
            x=128,
            model_name="moonshotai/Kimi-K2.5",
            vllm_module_topology="tp4dp2ep8",
        )

        assert float(latency) == pytest.approx(2.0)
        assert db.vllm_module_calls == []
        assert db.nccl_volumes == [("reduce_scatter", 128 * 1024 * 4)]

    def test_vllm_dispatch_non_topology_scope_keeps_existing_comm_path(self) -> None:
        op = MoEDispatch(
            "dispatch",
            1.0,
            hidden_size=1024,
            topk=8,
            num_experts=256,
            moe_tp_size=1,
            moe_ep_size=1,
            attention_dp_size=4,
            pre_dispatch=True,
            scale_num_tokens=8,
        )
        db = _FakePerfDbForMoEDispatch()

        latency = op.query(
            db,
            x=8192,
            model_name="moonshotai/Kimi-K2.5",
            vllm_module_topology="tp8dp1ep8",
        )

        assert float(latency) == pytest.approx(2.0)
        assert db.custom_allreduce_volumes == []
        assert db.nccl_volumes == [("all_gather", 1024 * 1024 * 4)]
        assert db.vllm_module_calls == []

    def test_vllm_dispatch_non_kimi_scope_keeps_existing_comm_path(self) -> None:
        op = MoEDispatch(
            "dispatch",
            1.0,
            hidden_size=1024,
            topk=8,
            num_experts=256,
            moe_tp_size=1,
            moe_ep_size=1,
            attention_dp_size=4,
            pre_dispatch=True,
            scale_num_tokens=8,
        )
        db = _FakePerfDbForMoEDispatch()

        latency = op.query(db, x=8192, model_name="meta-llama/Llama-3.1-70B")

        assert float(latency) == pytest.approx(2.0)
        assert db.custom_allreduce_volumes == []
        assert db.nccl_volumes == [("all_gather", 1024 * 1024 * 4)]
        assert db.vllm_module_calls == []

    def test_vllm_dispatch_legacy_scope_keeps_existing_comm_path(self) -> None:
        op = MoEDispatch(
            "dispatch",
            1.0,
            hidden_size=1024,
            topk=8,
            num_experts=256,
            moe_tp_size=1,
            moe_ep_size=1,
            attention_dp_size=4,
            pre_dispatch=True,
            scale_num_tokens=8,
        )
        db = _FakePerfDbForMoEDispatch()
        db.version = "0.12.0"

        latency = op.query(db, x=8192, model_name="moonshotai/Kimi-K2.5")

        assert float(latency) == pytest.approx(2.0)
        assert db.custom_allreduce_volumes == []
        assert db.nccl_volumes == [("all_gather", 1024 * 1024 * 4)]
        assert db.vllm_module_calls == []


class _FakePerfDbForMoE:
    def __init__(self) -> None:
        self.calls = []

    def query_moe(
        self,
        *,
        num_tokens,
        hidden_size,
        inter_size,
        topk,
        num_experts,
        moe_tp_size,
        moe_ep_size,
        quant_mode,
        workload_distribution,
        is_context,
        moe_backend,
        is_gated,
        enable_eplb,
    ):
        self.calls.append(num_tokens)
        return type("PerfResult", (), {"energy": 0.0, "__float__": lambda self: 1.0})()


class _FakeVLLMModulePerfDbForMoE:
    backend = common.BackendName.vllm.value
    system = "h200_sxm"
    version = "0.19.0"

    def __init__(self) -> None:
        self.query_moe_calls = []
        self.vllm_module_calls = []

    def query_moe(self, **kwargs):
        self.query_moe_calls.append(kwargs)
        return PerformanceResult(99.0, energy=0.0)

    def query_vllm_module(
        self,
        model,
        hardware,
        vllm_version,
        topology,
        bucket_tokens,
        module_boundary,
        quant_runtime,
    ):
        self.vllm_module_calls.append(
            (
                model,
                hardware,
                vllm_version,
                topology,
                bucket_tokens,
                module_boundary,
                quant_runtime,
            )
        )
        return PerformanceResult(4.0, energy=0.0)


class TestMoEScaling:
    def test_context_moe_scales_num_tokens(self) -> None:
        op = MoE(
            "moe",
            1.0,
            hidden_size=1024,
            inter_size=2048,
            topk=8,
            num_experts=256,
            moe_tp_size=16,
            moe_ep_size=1,
            quant_mode=common.MoEQuantMode.float16,
            workload_distribution="uniform",
            attention_dp_size=1,
            is_context=True,
            scale_num_tokens=16,
        )
        db = _FakePerfDbForMoE()

        latency = op.query(db, x=480)

        assert float(latency) == pytest.approx(1.0)
        assert db.calls == [30]

    def test_context_moe_uses_explicit_prefill_tokens_for_mixed_steps(self) -> None:
        op = MoE(
            "moe",
            1.0,
            hidden_size=1024,
            inter_size=2048,
            topk=8,
            num_experts=256,
            moe_tp_size=1,
            moe_ep_size=8,
            quant_mode=common.MoEQuantMode.float16,
            workload_distribution="uniform",
            attention_dp_size=2,
            is_context=True,
            scale_num_tokens=4,
        )
        db = _FakePerfDbForMoE()

        latency = op.query(db, x=32006, context_prefill_tokens=32000)

        assert float(latency) == pytest.approx(1.0)
        assert db.calls == [32000]

    def test_vllm_moe_uses_module_perf_exact_bucket(self) -> None:
        op = MoE(
            "moe",
            2.0,
            hidden_size=1024,
            inter_size=2048,
            topk=8,
            num_experts=256,
            moe_tp_size=1,
            moe_ep_size=8,
            quant_mode=common.MoEQuantMode.float16,
            workload_distribution="uniform",
            attention_dp_size=1,
            is_context=True,
            scale_num_tokens=1,
        )
        db = _FakeVLLMModulePerfDbForMoE()

        latency = op.query(
            db,
            x=241,
            model_name="moonshotai/Kimi-K2.5",
            vllm_module_topology="tp4dp2ep8",
        )

        assert float(latency) == pytest.approx(8.0)
        assert db.query_moe_calls == []
        assert db.vllm_module_calls == [
            (
                "kimi-k2.5",
                "h200_sxm",
                "0.19.0",
                "tp4dp2ep8",
                241,
                "fusedmoe_runner_compute",
                "CompressedTensorsWNA16MarlinMoEMethod",
            )
        ]

    def test_vllm_moe_uses_module_perf_paired_bucket(self) -> None:
        op = MoE(
            "moe",
            1.0,
            hidden_size=1024,
            inter_size=2048,
            topk=8,
            num_experts=256,
            moe_tp_size=1,
            moe_ep_size=8,
            quant_mode=common.MoEQuantMode.float16,
            workload_distribution="uniform",
            attention_dp_size=1,
            is_context=True,
            scale_num_tokens=1,
        )
        db = _FakeVLLMModulePerfDbForMoE()

        latency = op.query(
            db,
            x=30,
            model_name="moonshotai/Kimi-K2.5",
            vllm_module_topology="tp4dp2ep8",
        )

        assert float(latency) == pytest.approx(4.0)
        assert db.query_moe_calls == []
        assert db.vllm_module_calls == [
            (
                "kimi-k2.5",
                "h200_sxm",
                "0.19.0",
                "tp4dp2ep8",
                30,
                "fusedmoe_runner_compute",
                "CompressedTensorsWNA16MarlinMoEMethod",
            )
        ]

    def test_vllm_moe_falls_back_for_non_exact_bucket(self) -> None:
        op = MoE(
            "moe",
            1.0,
            hidden_size=1024,
            inter_size=2048,
            topk=8,
            num_experts=256,
            moe_tp_size=1,
            moe_ep_size=8,
            quant_mode=common.MoEQuantMode.float16,
            workload_distribution="uniform",
            attention_dp_size=1,
            is_context=True,
            scale_num_tokens=1,
        )
        db = _FakeVLLMModulePerfDbForMoE()

        latency = op.query(
            db,
            x=128,
            model_name="moonshotai/Kimi-K2.5",
            vllm_module_topology="tp4dp2ep8",
        )

        assert float(latency) == pytest.approx(99.0)
        assert db.vllm_module_calls == []
        assert db.query_moe_calls
        assert db.query_moe_calls[0]["num_tokens"] == 128

    def test_vllm_moe_non_topology_scope_keeps_existing_query_moe_path(self) -> None:
        op = MoE(
            "moe",
            1.0,
            hidden_size=1024,
            inter_size=2048,
            topk=8,
            num_experts=256,
            moe_tp_size=1,
            moe_ep_size=8,
            quant_mode=common.MoEQuantMode.float16,
            workload_distribution="uniform",
            attention_dp_size=1,
            is_context=True,
            scale_num_tokens=1,
        )
        db = _FakeVLLMModulePerfDbForMoE()

        latency = op.query(
            db,
            x=241,
            model_name="moonshotai/Kimi-K2.5",
            vllm_module_topology="tp8dp1ep8",
        )

        assert float(latency) == pytest.approx(99.0)
        assert db.query_moe_calls[0]["num_tokens"] == 241
        assert db.vllm_module_calls == []

    def test_vllm_moe_non_kimi_scope_keeps_existing_query_moe_path(self) -> None:
        op = MoE(
            "moe",
            1.0,
            hidden_size=1024,
            inter_size=2048,
            topk=8,
            num_experts=256,
            moe_tp_size=1,
            moe_ep_size=8,
            quant_mode=common.MoEQuantMode.float16,
            workload_distribution="uniform",
            attention_dp_size=1,
            is_context=True,
            scale_num_tokens=1,
        )
        db = _FakeVLLMModulePerfDbForMoE()

        latency = op.query(db, x=241, model_name="meta-llama/Llama-3.1-70B")

        assert float(latency) == pytest.approx(99.0)
        assert db.query_moe_calls[0]["num_tokens"] == 241
        assert db.vllm_module_calls == []

    def test_vllm_moe_legacy_scope_keeps_existing_query_moe_path(self) -> None:
        op = MoE(
            "moe",
            1.0,
            hidden_size=1024,
            inter_size=2048,
            topk=8,
            num_experts=256,
            moe_tp_size=1,
            moe_ep_size=8,
            quant_mode=common.MoEQuantMode.float16,
            workload_distribution="uniform",
            attention_dp_size=1,
            is_context=True,
            scale_num_tokens=1,
        )
        db = _FakeVLLMModulePerfDbForMoE()
        db.version = "0.12.0"

        latency = op.query(db, x=241, model_name="moonshotai/Kimi-K2.5")

        assert float(latency) == pytest.approx(99.0)
        assert db.query_moe_calls[0]["num_tokens"] == 241
        assert db.vllm_module_calls == []


class _CaptureModelNameOp:
    _name = "capture_model_name"

    def __init__(self) -> None:
        self.model_names = []
        self.topologies = []

    def query(self, database, **kwargs):
        self.model_names.append(kwargs["model_name"])
        self.topologies.append(kwargs["vllm_module_topology"])
        return PerformanceResult(0.0, energy=0.0)


class TestBaseBackendModelNamePropagation:
    def test_run_static_passes_model_path_and_topology_when_model_name_is_absent(self, monkeypatch) -> None:
        context_op = _CaptureModelNameOp()
        generation_op = _CaptureModelNameOp()
        model = SimpleNamespace(
            model_path="moonshotai/Kimi-K2.5",
            _nextn=0,
            context_ops=[context_op],
            generation_ops=[generation_op],
            config=SimpleNamespace(
                attention_dp_size=2,
                pp_size=1,
                tp_size=4,
                moe_tp_size=1,
                moe_ep_size=8,
                gemm_quant_mode=SimpleNamespace(name="fp16"),
                kvcache_quant_mode=SimpleNamespace(name="fp16"),
                fmha_quant_mode=SimpleNamespace(name="fp16"),
                moe_quant_mode=SimpleNamespace(name="fp16"),
                comm_quant_mode=SimpleNamespace(name="fp16"),
            ),
        )
        database = SimpleNamespace(
            backend="vllm",
            version="0.19.0",
            system="h200_sxm",
            system_spec={"gpu": {"mem_capacity": 80 << 30}},
        )
        backend = VLLMBackend()
        monkeypatch.setattr(backend, "_get_memory_usage", lambda *args, **kwargs: {"total": 0.0})

        backend.run_static(
            model,
            database,
            RuntimeConfig(batch_size=1, isl=2, osl=2),
            mode="static",
        )

        assert context_op.model_names == ["moonshotai/Kimi-K2.5"]
        assert context_op.topologies == ["tp4dp2ep8"]
        assert generation_op.model_names == ["moonshotai/Kimi-K2.5"]
        assert generation_op.topologies == ["tp4dp2ep8"]


class _FakeCBResult:
    mean_ttft_ms = 111.0
    mean_tpot_ms = 2.0
    throughput_tok_s = 1234.0
    throughput_tok_s_gpu = 1234.0 / 16.0
    peak_tokens_per_iter = 700
    avg_prefill_reqs_per_iter = 1.5
    avg_decode_reqs_per_iter = 14.0
    avg_tokens_per_iter = 600.0
    peak_prefill_reqs_per_iter = 2
    peak_decode_reqs_per_iter = 16
    steady_state_iterations = 100
    steady_state_time_ms = 5000.0
    total_iterations = 120


class _FakeCBSim:
    def __init__(self, *args, **kwargs):
        pass

    def run(self, **kwargs):
        assert kwargs["num_gpus"] == 16
        return _FakeCBResult()


class TestVLLMCBSimBoundary:
    def test_run_agg_cb_sim_uses_cb_sim_throughput(self, monkeypatch) -> None:
        backend = VLLMBackend()
        model = MagicMock()
        model.model_path = "fake-model"
        model.config.tp_size = 16
        model.config.pp_size = 1
        model.config.attention_dp_size = 1
        model.config.moe_tp_size = 16
        model.config.moe_ep_size = 1
        model.config.gemm_quant_mode.name = "fp16"
        model.config.kvcache_quant_mode.name = "fp16"
        model.config.fmha_quant_mode.name = "fp16"
        model.config.moe_quant_mode.name = "fp16"
        model.config.comm_quant_mode.name = "fp16"

        database = MagicMock()
        database.backend = "vllm"
        database.version = "0.12.0"
        database.system = "h200_sxm"
        database.system_spec = {"gpu": {"mem_capacity": 80 << 30}}

        monkeypatch.setattr(
            "aiconfigurator.sdk.backends.cb_simulator.CBSimulator",
            _FakeCBSim,
        )
        monkeypatch.setattr(
            backend,
            "_get_memory_usage",
            lambda *args, **kwargs: {"total": 1.0},
        )

        summary = backend._run_agg_cb_sim(
            model=model,
            database=database,
            runtime_config=RuntimeConfig(batch_size=16, isl=8000, osl=2000),
            ctx_tokens=8000,
        )
        result_dict = summary.get_result_dict()
        per_ops = summary.get_per_ops_data()

        assert result_dict["ttft"] == pytest.approx(111.0)
        assert result_dict["tpot"] == pytest.approx(2.0)
        assert result_dict["tokens/s"] == pytest.approx(1234.0)
        assert result_dict["tokens/s/gpu"] == pytest.approx(1234.0 / 16.0)
        assert per_ops["cb_sim_scheduling"]["overlap_factor"] == pytest.approx(0.0)
        assert per_ops["cb_sim_scheduling"]["per_iteration_overhead_ms"] == pytest.approx(
            0.0
        )
        assert per_ops["cb_sim_boundary"]["throughput_source"] == "cb_sim"
        assert per_ops["cb_sim_boundary"]["cb_sim_tokens_s"] == pytest.approx(1234.0)

    def test_run_agg_cb_sim_cache_distinguishes_parallel_config(self, monkeypatch) -> None:
        calls = []

        class _CountingCBSim:
            def __init__(self, *args, **kwargs):
                pass

            def run(self, **kwargs):
                calls.append(kwargs["num_gpus"])
                return SimpleNamespace(
                    mean_ttft_ms=111.0,
                    mean_tpot_ms=2.0,
                    throughput_tok_s=1000.0 * len(calls),
                    peak_tokens_per_iter=700,
                    avg_prefill_reqs_per_iter=1.5,
                    avg_decode_reqs_per_iter=14.0,
                    avg_tokens_per_iter=600.0,
                    peak_prefill_reqs_per_iter=2,
                    peak_decode_reqs_per_iter=16,
                    steady_state_iterations=100,
                    steady_state_time_ms=5000.0,
                    total_iterations=120,
                )

        def make_model(tp, dp, moe_tp, moe_ep):
            model = MagicMock()
            model.model_path = "fake-model"
            model.config.tp_size = tp
            model.config.pp_size = 1
            model.config.attention_dp_size = dp
            model.config.moe_tp_size = moe_tp
            model.config.moe_ep_size = moe_ep
            model.config.gemm_quant_mode.name = "fp16"
            model.config.kvcache_quant_mode.name = "fp16"
            model.config.fmha_quant_mode.name = "fp16"
            model.config.moe_quant_mode.name = "fp16"
            model.config.comm_quant_mode.name = "fp16"
            return model

        backend = VLLMBackend()
        database = MagicMock()
        database.backend = "vllm"
        database.version = "0.12.0"
        database.system = "h200_sxm"
        database.system_spec = {"gpu": {"mem_capacity": 80 << 30}}

        monkeypatch.setattr(
            "aiconfigurator.sdk.backends.cb_simulator.CBSimulator",
            _CountingCBSim,
        )
        monkeypatch.setattr(
            backend,
            "_get_memory_usage",
            lambda *args, **kwargs: {"total": 1.0},
        )

        runtime_config = RuntimeConfig(batch_size=128, isl=8000, osl=2000)
        cb_config = CBSimConfig(max_num_batched_tokens=8000)

        first = backend.run_agg(
            make_model(8, 1, 1, 8),
            database,
            runtime_config,
            ctx_tokens=8000,
            method="cb_sim",
            cb_config=cb_config,
        )
        second = backend.run_agg(
            make_model(4, 2, 1, 8),
            database,
            runtime_config,
            ctx_tokens=8000,
            method="cb_sim",
            cb_config=cb_config,
        )

        assert calls == [8, 4]
        assert first.get_result_dict()["tokens/s"] == pytest.approx(1000.0)
        assert second.get_result_dict()["tokens/s"] == pytest.approx(4000.0)

    def test_run_agg_cb_sim_does_not_apply_tp16_throughput_calibration(self, monkeypatch) -> None:
        backend = VLLMBackend()
        model = MagicMock()
        model.model_path = "moonshotai/Kimi-K2.5"
        model.config.tp_size = 16
        model.config.pp_size = 1
        model.config.attention_dp_size = 1
        model.config.moe_tp_size = 16
        model.config.moe_ep_size = 1
        model.config.gemm_quant_mode.name = "fp16"
        model.config.kvcache_quant_mode.name = "fp16"
        model.config.fmha_quant_mode.name = "fp16"
        model.config.moe_quant_mode.name = "fp16"
        model.config.comm_quant_mode.name = "fp16"

        database = MagicMock()
        database.backend = "vllm"
        database.version = "0.12.0"
        database.system = "h200_sxm"
        database.system_spec = {"gpu": {"mem_capacity": 80 << 30}}

        monkeypatch.setattr(
            "aiconfigurator.sdk.backends.cb_simulator.CBSimulator",
            _FakeCBSim,
        )
        monkeypatch.setattr(
            backend,
            "_get_memory_usage",
            lambda *args, **kwargs: {"total": 1.0},
        )

        summary = backend._run_agg_cb_sim(
            model=model,
            database=database,
            runtime_config=RuntimeConfig(batch_size=128, isl=10000, osl=3000),
            ctx_tokens=10000,
        )
        result_dict = summary.get_result_dict()
        per_ops = summary.get_per_ops_data()

        assert result_dict["tokens/s"] == pytest.approx(1234.0)
        assert result_dict["tokens/s/gpu"] == pytest.approx(1234.0 / 16.0)
        assert result_dict["tpot"] == pytest.approx(2.0)
        assert per_ops["cb_sim_scheduling"]["per_iteration_overhead_ms"] == pytest.approx(
            0.0
        )
        assert per_ops["cb_sim_boundary"]["throughput_source"] == "cb_sim"
        assert "cb_sim_calibration_applied" not in per_ops["cb_sim_scheduling"]

    def test_run_agg_cb_sim_does_not_apply_ep8_throughput_calibration(self, monkeypatch) -> None:
        class _FakeEP8CBSim:
            def __init__(self, *args, **kwargs):
                pass

            def run(self, **kwargs):
                assert kwargs["num_gpus"] == 8
                return _FakeCBResult()

        backend = VLLMBackend()
        model = MagicMock()
        model.model_path = "moonshotai/Kimi-K2.5"
        model.config.tp_size = 8
        model.config.pp_size = 1
        model.config.attention_dp_size = 1
        model.config.moe_tp_size = 1
        model.config.moe_ep_size = 8
        model.config.gemm_quant_mode.name = "fp16"
        model.config.kvcache_quant_mode.name = "fp16"
        model.config.fmha_quant_mode.name = "fp16"
        model.config.moe_quant_mode.name = "fp16"
        model.config.comm_quant_mode.name = "fp16"

        database = MagicMock()
        database.backend = "vllm"
        database.version = "0.12.0"
        database.system = "h200_sxm"
        database.system_spec = {"gpu": {"mem_capacity": 80 << 30}}

        monkeypatch.setattr(
            "aiconfigurator.sdk.backends.cb_simulator.CBSimulator",
            _FakeEP8CBSim,
        )
        monkeypatch.setattr(
            backend,
            "_get_memory_usage",
            lambda *args, **kwargs: {"total": 1.0},
        )

        summary = backend._run_agg_cb_sim(
            model=model,
            database=database,
            runtime_config=RuntimeConfig(batch_size=128, isl=8000, osl=2000),
            ctx_tokens=8000,
        )
        result_dict = summary.get_result_dict()
        per_ops = summary.get_per_ops_data()

        assert result_dict["tokens/s"] == pytest.approx(1234.0)
        assert result_dict["tokens/s/gpu"] == pytest.approx(1234.0 / 8.0)
        assert result_dict["tpot"] == pytest.approx(2.0)
        assert per_ops["cb_sim_scheduling"]["per_iteration_overhead_ms"] == pytest.approx(
            0.0
        )
        assert (
            per_ops["cb_sim_scheduling"]["per_iteration_overhead_source_key"]
            == "none"
        )
        assert (
            per_ops["cb_sim_scheduling"]["per_iteration_overhead_topology_key"]
            == "tp8dp1moetp1ep8"
        )
        assert (
            per_ops["cb_sim_scheduling"]["per_iteration_overhead_shape_key"]
            == "decode_present:isl8000:osl2000:bs128:ctx8000:max_bt8000:max_seqs256"
        )
        assert per_ops["cb_sim_scheduling"]["per_iteration_overhead_diagnostic_only"] is True
        assert per_ops["cb_sim_scheduling"]["per_iteration_overhead_valid_for_default"] is False
        assert per_ops["cb_sim_scheduling"]["per_iteration_overhead_perf_database"] is False
        assert per_ops["cb_sim_boundary"]["throughput_source"] == "cb_sim"
        assert "cb_sim_calibration_applied" not in per_ops["cb_sim_scheduling"]


class TestVLLMCalibrationBoundaries:
    def test_default_method_does_not_apply_cb_calibration(self) -> None:
        backend = VLLMBackend()
        model = MagicMock()
        model.model_path = "moonshotai/Kimi-K2.5"
        model.config.tp_size = 16
        model.config.pp_size = 1
        model.config.attention_dp_size = 1
        database = MagicMock()
        database.system = "h200_sxm"

        runtime_config = RuntimeConfig(batch_size=32, isl=16000, osl=3000)

        assert backend._should_apply_cb_calibration(model, database, runtime_config)
        assert backend._DEFAULT_METHOD == "batch_sync"
        assert backend._CALIBRATED_METHOD == "batch_sync_calibrated"

    def test_cb_calibration_is_rejected_outside_calibrated_regime(self) -> None:
        backend = VLLMBackend()
        model = MagicMock()
        model.model_path = "meta-llama/Llama-3.1-70B"
        model.config.tp_size = 16
        model.config.pp_size = 1
        model.config.attention_dp_size = 1
        database = MagicMock()
        database.system = "h200_sxm"

        runtime_config = RuntimeConfig(batch_size=32, isl=16000, osl=3000)

        assert not backend._should_apply_cb_calibration(model, database, runtime_config)


class TestVLLMMemoryEstimation:
    def test_memory_cap_requires_chunked_prefill_flag(self, monkeypatch) -> None:
        backend = VLLMBackend()
        captured = {}

        class _FakeTRTLLMBackend:
            def _get_memory_usage(self, model, database, batch_size, beam_width, isl, osl, num_tokens):
                captured["num_tokens"] = num_tokens
                return {"total": 1.0}

        monkeypatch.setattr(
            "aiconfigurator.sdk.backends.trtllm_backend.TRTLLMBackend",
            _FakeTRTLLMBackend,
        )

        backend._get_memory_usage(
            model=MagicMock(),
            database=MagicMock(),
            batch_size=2,
            beam_width=1,
            isl=120000,
            osl=4000,
            num_tokens=240000,
            prefix=128,
            enable_chunked_prefill=False,
        )
        assert captured["num_tokens"] == 240000

        backend._get_memory_usage(
            model=MagicMock(),
            database=MagicMock(),
            batch_size=2,
            beam_width=1,
            isl=120000,
            osl=4000,
            num_tokens=240000,
            prefix=128,
            enable_chunked_prefill=True,
            chunked_prefill_tokens=4096,
        )
        assert captured["num_tokens"] == 4096


class TestDiagnoseCBIterLatencyScript:
    def test_parse_vllm_iteration_log(self, tmp_path: Path) -> None:
        module = _load_diagnose_cb_iter_latency_module()
        log_path = tmp_path / "serve_iter_log.txt"
        log_path.write_text(
            "\n".join(
                [
                    "INFO Iteration(42): 2 context requests, 8192 context tokens, "
                    "126 generation requests, 126 generation tokens, "
                    "iteration elapsed time: 12.34 ms",
                    "INFO Iteration(43): 0 context requests, 0 context tokens, "
                    "128 generation requests, 128 generation tokens, "
                    "iteration elapsed time: 4.50 ms",
                ]
            )
        )

        rows = module.parse_vllm_iteration_log(log_path)

        assert len(rows) == 2
        assert rows[0].iter_index == 42
        assert rows[0].phase_type == "mixed"
        assert rows[0].context_tokens == 8192
        assert rows[0].iter_lat_ms == pytest.approx(12.34)
        assert rows[1].phase_type == "pure_decode"
        assert rows[1].generation_requests == 128

    def test_parse_metrics_delta(self, tmp_path: Path) -> None:
        module = _load_diagnose_cb_iter_latency_module()
        before_path = tmp_path / "metrics_before.txt"
        after_path = tmp_path / "metrics_after.txt"
        before_path.write_text(
            "\n".join(
                [
                    'vllm:iteration_tokens_total_count{engine="0"} 10',
                    'vllm:iteration_tokens_total_sum{engine="0"} 1000',
                    'vllm:request_prefill_time_seconds_count{engine="0"} 4',
                    'vllm:request_prefill_time_seconds_sum{engine="0"} 2.0',
                    'vllm:request_decode_time_seconds_count{engine="0"} 6',
                    'vllm:request_decode_time_seconds_sum{engine="0"} 0.6',
                    'vllm:prompt_tokens_total{engine="0"} 300',
                    'vllm:generation_tokens_total{engine="0"} 700',
                    'vllm:request_success_total{engine="0"} 2',
                ]
            )
        )
        after_path.write_text(
            "\n".join(
                [
                    'vllm:iteration_tokens_total_count{engine="0"} 20',
                    'vllm:iteration_tokens_total_count{engine="1"} 5',
                    'vllm:iteration_tokens_total_sum{engine="0"} 2200',
                    'vllm:iteration_tokens_total_sum{engine="1"} 300',
                    'vllm:request_prefill_time_seconds_count{engine="0"} 8',
                    'vllm:request_prefill_time_seconds_sum{engine="0"} 4.4',
                    'vllm:request_decode_time_seconds_count{engine="0"} 16',
                    'vllm:request_decode_time_seconds_sum{engine="0"} 1.6',
                    'vllm:prompt_tokens_total{engine="0"} 900',
                    'vllm:generation_tokens_total{engine="0"} 1600',
                    'vllm:request_success_total{engine="0"} 5',
                ]
            )
        )

        before = module.parse_metrics_snapshot(before_path)
        after = module.parse_metrics_snapshot(after_path)
        delta = module.compare_metrics_snapshots(
            before,
            after,
            benchmark_wall_ms=150.0,
        )

        assert delta.iter_count_delta == pytest.approx(15.0)
        assert delta.iteration_tokens_delta == pytest.approx(1500.0)
        assert delta.avg_tokens_per_iter == pytest.approx(100.0)
        assert delta.avg_iter_lat_ms == pytest.approx(10.0)
        assert delta.avg_prefill_time_ms == pytest.approx(600.0)
        assert delta.avg_decode_time_ms == pytest.approx(100.0)
        assert delta.prompt_tokens_delta == pytest.approx(600.0)
        assert delta.generation_tokens_delta == pytest.approx(900.0)
        assert delta.request_success_delta == pytest.approx(3.0)

    def test_compare_phasewise_uses_phase_local_ordinals(self) -> None:
        module = _load_diagnose_cb_iter_latency_module()
        cb_rows = [
            module.CBIterationTraceRow(
                iter_index=1,
                phase_type="mixed",
                trace_type="scheduled",
                prefill_requests=1,
                prefill_tokens=4096,
                decode_batch_size=64,
                decode_avg_kv_len=3000,
                iter_lat_ms=10.0,
                ctx_non_attn_ms=3.0,
                ctx_attn_ms=2.0,
                gen_non_attn_ms=3.0,
                gen_attn_ms=2.0,
                clock_ms=10.0,
                completed_requests=0,
                running_requests=64,
                waiting_requests=64,
                in_steady_state=0,
            ),
            module.CBIterationTraceRow(
                iter_index=2,
                phase_type="mixed",
                trace_type="scheduled",
                prefill_requests=1,
                prefill_tokens=2048,
                decode_batch_size=64,
                decode_avg_kv_len=3500,
                iter_lat_ms=12.0,
                ctx_non_attn_ms=3.0,
                ctx_attn_ms=3.0,
                gen_non_attn_ms=4.0,
                gen_attn_ms=2.0,
                clock_ms=22.0,
                completed_requests=0,
                running_requests=64,
                waiting_requests=60,
                in_steady_state=0,
            ),
        ]
        vllm_rows = [
            module.VLLMIterationTraceRow(
                iter_index=101,
                phase_type="mixed",
                context_requests=2,
                context_tokens=8192,
                generation_requests=126,
                generation_tokens=126,
                iter_lat_ms=13.5,
            ),
            module.VLLMIterationTraceRow(
                iter_index=102,
                phase_type="mixed",
                context_requests=1,
                context_tokens=4096,
                generation_requests=127,
                generation_tokens=127,
                iter_lat_ms=14.5,
            ),
        ]

        compare_rows, compare_summaries = module.compare_phasewise(
            cb_rows,
            vllm_rows,
        )

        assert len(compare_rows) == 2
        assert compare_rows[0].ordinal_in_phase == 1
        assert compare_rows[0].overhead_ms == pytest.approx(3.5)
        assert compare_rows[1].ordinal_in_phase == 2
        assert compare_rows[1].overhead_ms == pytest.approx(2.5)
        mixed_summary = next(
            row for row in compare_summaries if row.phase_type == "mixed"
        )
        assert mixed_summary.aligned_pairs == 2
        assert mixed_summary.overhead_mean_ms == pytest.approx(3.0)

    def test_alpha_overhead_sweep_uses_active_validation_gates(self, monkeypatch) -> None:
        module = _load_diagnose_cb_iter_latency_module()

        fake_result = SimpleNamespace(
            throughput_max=9.00,
            throughput_mean=1.10,
            multi_config_max=1.51,
            multi_config_mean=1.20,
            ttft_max=9.00,
            ttft_mean=1.30,
        )
        fake_validate = SimpleNamespace(
            THROUGHPUT_MAX_ACCEPTANCE=1.50,
            MULTI_CONFIG_MAX_ACCEPTANCE=1.50,
            TTFT_MAX_ACCEPTANCE=1.79,
            THROUGHPUT_GATE_STATUS="legacy_skip",
            TTFT_GATE_STATUS="legacy_skip",
            run_validation=lambda **kwargs: fake_result,
        )
        monkeypatch.setattr(module, "_load_validate_module", lambda: fake_validate)

        rows = module.run_alpha_overhead_sweep(
            SimpleNamespace(
                alpha_values="0.0",
                overhead_values="0.0",
                ep8_per_iteration_overhead_ms=0.0,
            )
        )

        assert rows[0].passed == 0
        assert rows[0].acceptance_score == pytest.approx(1.51 / 1.50)


class TestValidateCBSimulatorBudgetAwareTopK:
    def test_phase397w_validate_gate_is_019_multi_config_only(self) -> None:
        module = _load_validate_cb_simulator_module()

        assert module.VALIDATION_DB_VERSION == "0.19.0"
        assert module.DEFAULT_EP8_PER_ITERATION_OVERHEAD_MS == pytest.approx(0.0)
        assert module.THROUGHPUT_GATE_STATUS == "legacy_skip"
        assert module.TTFT_GATE_STATUS == "legacy_skip"
        assert module.MULTI_CONFIG_MAX_ACCEPTANCE == pytest.approx(1.5)
        assert {point.name for point in module.MULTI_CONFIG_DATA} == {
            "K2.5-tp8ep8-8k2k",
            "K2.5-tp8ep8-32k3k",
            "K2.5-tp4ep8dp2-8k2k",
            "K2.5-tp4ep8dp2-32k3k",
            "K2.5-tp8ep8-8k2k-bt65536",
            "K2.5-tp4ep8dp2-8k2k-bt65536",
        }

    def test_multi_config_points_define_budget_explicitly(self) -> None:
        module = _load_validate_cb_simulator_module()

        by_name = {point.name: point for point in module.MULTI_CONFIG_DATA}

        assert by_name["K2.5-tp8ep8-8k2k"].max_num_batched_tokens == 8000
        assert by_name["K2.5-tp4ep8dp2-32k3k"].max_num_batched_tokens == 32000
        assert by_name["K2.5-tp8ep8-8k2k-bt65536"].max_num_batched_tokens == 65536
        assert by_name["K2.5-tp4ep8dp2-8k2k-bt65536"].max_num_batched_tokens == 65536

    def test_diagnostic_topk_uses_budget_and_keeps_baseline_gates(self, monkeypatch) -> None:
        module = _load_validate_cb_simulator_module()
        captured: list[tuple[int, int]] = []

        class _FakeSummary:
            def __init__(self, sim_gpu: float) -> None:
                self._sim_gpu = sim_gpu

            def get_result_dict(self):
                return {"tokens/s/gpu": self._sim_gpu}

            def get_per_ops_data(self):
                return {"cb_sim_boundary": {"throughput_source": "cb_sim"}}

        class _FakeBackend:
            def run_agg(self, model, db, runtime_config, **kwargs):
                cb_config = kwargs["cb_config"]
                captured.append((kwargs["ctx_tokens"], cb_config.max_num_batched_tokens))
                sim_gpu = 100.0 + cb_config.max_num_batched_tokens / 1024.0
                return _FakeSummary(sim_gpu)

        def fake_load_model_and_db(*, tp, dp, moe_tp, moe_ep):
            model = SimpleNamespace(model_path="fake", config=SimpleNamespace())
            db = SimpleNamespace(system="h200_sxm", backend="vllm")
            return model, db, None

        monkeypatch.setattr(module, "VLLMBackend", _FakeBackend)
        monkeypatch.setattr(module, "_load_model_and_db", fake_load_model_and_db)

        before_gates = (
            module.THROUGHPUT_MAX_ACCEPTANCE,
            module.MULTI_CONFIG_MAX_ACCEPTANCE,
            module.TTFT_MAX_ACCEPTANCE,
        )
        rows = module.run_diagnostic_multi_config_topk(verbose=False)
        after_gates = (
            module.THROUGHPUT_MAX_ACCEPTANCE,
            module.MULTI_CONFIG_MAX_ACCEPTANCE,
            module.TTFT_MAX_ACCEPTANCE,
        )
        by_name = {row.name: row for row in rows}

        assert by_name["K2.5-tp8ep8-8k2k-bt65536"].max_bt == 65536
        assert by_name["K2.5-tp4ep8dp2-8k2k-bt65536"].max_bt == 65536
        assert by_name["K2.5-tp8ep8-8k2k"].max_bt == 8000
        assert captured.count((65536, 65536)) == 2
        assert all(row.diagnostic_only for row in rows)
        assert not any(row.valid_for_default for row in rows)
        assert not any(row.perf_database for row in rows)
        assert before_gates == after_gates


class TestValidateCBSimulatorBudgetBreakdown:
    def test_budget_breakdown_captures_scheduling_fields_and_pairs_baseline(
        self,
        monkeypatch,
    ) -> None:
        module = _load_validate_cb_simulator_module()
        captured: list[tuple[int, int]] = []

        class _FakeSummary:
            def __init__(self, max_bt: int) -> None:
                self._max_bt = max_bt

            def get_result_dict(self):
                return {"tokens/s/gpu": 100.0 + self._max_bt / 1024.0}

            def get_per_ops_data(self):
                return {
                    "cb_sim_scheduling": {
                        "avg_prefill_reqs_per_iter": self._max_bt / 8192.0,
                        "avg_decode_reqs_per_iter": 128.0,
                        "avg_tokens_per_iter": float(self._max_bt),
                        "peak_prefill_reqs_per_iter": self._max_bt / 4096.0,
                        "peak_decode_reqs_per_iter": 128.0,
                        "peak_tokens_per_iter": float(self._max_bt + 128),
                        "steady_state_iterations": self._max_bt // 1024,
                        "steady_state_time_ms": float(self._max_bt * 2),
                    },
                    "cb_sim_boundary": {"throughput_source": "cb_sim"},
                }

        class _FakeBackend:
            def run_agg(self, model, db, runtime_config, **kwargs):
                cb_config = kwargs["cb_config"]
                captured.append((kwargs["ctx_tokens"], cb_config.max_num_batched_tokens))
                return _FakeSummary(cb_config.max_num_batched_tokens)

        def fake_load_model_and_db(*, tp, dp, moe_tp, moe_ep):
            model = SimpleNamespace(model_path="fake", config=SimpleNamespace())
            db = SimpleNamespace(system="h200_sxm", backend="vllm")
            return model, db, None

        monkeypatch.setattr(module, "VLLMBackend", _FakeBackend)
        monkeypatch.setattr(module, "_load_model_and_db", fake_load_model_and_db)

        before_gates = (
            module.THROUGHPUT_MAX_ACCEPTANCE,
            module.MULTI_CONFIG_MAX_ACCEPTANCE,
            module.TTFT_MAX_ACCEPTANCE,
        )
        rows = module.run_diagnostic_multi_config_budget_breakdown(verbose=False)
        after_gates = (
            module.THROUGHPUT_MAX_ACCEPTANCE,
            module.MULTI_CONFIG_MAX_ACCEPTANCE,
            module.TTFT_MAX_ACCEPTANCE,
        )
        by_name = {row.name: row for row in rows}
        bt_row = by_name["K2.5-tp8ep8-8k2k-bt65536"]

        assert captured.count((65536, 65536)) == 2
        assert bt_row.max_bt == 65536
        assert bt_row.avg_prefill_reqs_per_iter == pytest.approx(8.0)
        assert bt_row.avg_decode_reqs_per_iter == pytest.approx(128.0)
        assert bt_row.avg_tokens_per_iter == pytest.approx(65536.0)
        assert bt_row.peak_prefill_reqs_per_iter == pytest.approx(16.0)
        assert bt_row.peak_decode_reqs_per_iter == pytest.approx(128.0)
        assert bt_row.peak_tokens_per_iter == pytest.approx(65664.0)
        assert bt_row.steady_state_iterations == 64
        assert bt_row.steady_state_time_ms == pytest.approx(131072.0)
        assert bt_row.paired_baseline_name == "K2.5-tp8ep8-8k2k"
        assert bt_row.paired_baseline_max_bt == 8000
        assert bt_row.max_bt_vs_paired_baseline == pytest.approx(8.192)
        assert bt_row.diagnostic_only is True
        assert bt_row.valid_for_default is False
        assert bt_row.perf_database is False
        assert before_gates == after_gates
