"""Tests for CB simulator scheduler and simulator."""
from unittest.mock import MagicMock

import pytest
from aiconfigurator.sdk.backends.cb_simulator.datatypes import (
    CBSimConfig, Request, RequestState,
)
from aiconfigurator.sdk.backends.cb_simulator.scheduler import CBScheduler
from aiconfigurator.sdk.backends.cb_simulator.simulator import CBSimulator


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
        """When decode requests exceed budget, no prefill and all decodes scheduled."""
        cfg = CBSimConfig(max_num_batched_tokens=10)
        sched = CBScheduler(cfg)
        running = [_make_decoding(i) for i in range(20)]
        result = sched.schedule(waiting=[], running=running)
        assert len(result.decode_reqs) == 20
        assert len(result.prefill_reqs) == 0


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
