"""Discrete-event CB simulation engine.

Closed-loop: maintains target concurrency by replacing completed
requests immediately. Request lifecycle:
WAITING/PREEMPTED -> PREFILLING -> DECODING -> DONE.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .datatypes import CBSimConfig, CBSimResult, Request, RequestState
from .iteration_latency import IterationLatencyCalculator
from .scheduler import CBScheduler

if TYPE_CHECKING:
    from aiconfigurator.sdk.backends.base_backend import BaseBackend
    from aiconfigurator.sdk.models import BaseModel
    from aiconfigurator.sdk.perf_database import PerfDatabase

logger = logging.getLogger(__name__)


class CBSimulator:
    """Lightweight vLLM continuous batching simulator."""

    def __init__(
        self,
        backend: BaseBackend,
        model: BaseModel,
        database: PerfDatabase,
        config: CBSimConfig | None = None,
    ) -> None:
        self._config = config or CBSimConfig()
        self._scheduler = CBScheduler(self._config)
        self._backend = backend
        self._model = model
        self._database = database

    def _create_latency_calc(self, prefix: int) -> IterationLatencyCalculator:
        """Factory hook for latency calculator. Override in tests."""
        return IterationLatencyCalculator(
            self._backend,
            self._model,
            self._database,
            prefix=prefix,
            overlap_factor=self._config.overlap_factor,
            per_iteration_overhead_ms=self._config.per_iteration_overhead_ms,
        )

    def _estimate_decode_skip_latency(
        self,
        latency_calc: IterationLatencyCalculator,
        decode_batch_size: int,
        start_avg_kv_len: int,
        skip_iters: int,
    ) -> float:
        """Approximate skipped pure-decode time with a rising-latency ramp.

        Decode-only iteration latency grows with KV length. Using a constant
        `iter_lat * skip` systematically overestimates throughput for long
        decode segments, especially at high batch size. A trapezoid estimate
        keeps the skip optimization but lets the end-of-segment KV growth
        increase the skipped wall time.

        Both endpoints are computed as PURE-DECODE iterations (at
        ``start_avg_kv_len`` and ``start_avg_kv_len + skip_iters``). The
        triggering iteration that fired this skip is often a mixed/prefill
        iteration whose latency includes the prefill chunk; seeding the ramp
        with it would inflate every skipped pure-decode iteration by the
        prefill cost. See docs/iter_gap_investigation/phase397c/d.
        """
        if skip_iters <= 0:
            return 0.0

        first_iter_latency_ms = latency_calc.compute(
            prefill_tokens=0,
            prefill_batch_size=0,
            prefill_seq_len=1,
            decode_batch_size=decode_batch_size,
            decode_avg_kv_len=start_avg_kv_len,
        )
        end_iter_latency_ms = latency_calc.compute(
            prefill_tokens=0,
            prefill_batch_size=0,
            prefill_seq_len=1,
            decode_batch_size=decode_batch_size,
            decode_avg_kv_len=start_avg_kv_len + skip_iters,
        )
        return (first_iter_latency_ms + end_iter_latency_ms) * skip_iters / 2.0

    def run(
        self,
        isl: int,
        osl: int,
        concurrency: int,
        prefix: int = 0,
        num_gpus: int = 1,
    ) -> CBSimResult:
        """Run closed-loop simulation.

        Args:
            isl: Input sequence length.
            osl: Output sequence length.
            concurrency: Target number of concurrent requests.
            prefix: Shared prefix length.
            num_gpus: Number of GPUs (for throughput scaling).

        Returns:
            CBSimResult with TTFT, TPOT, throughput metrics.
        """
        latency_calc = self._create_latency_calc(prefix)

        waiting: list[Request] = []
        running: list[Request] = []
        completed: list[Request] = []
        clock_ms = 0.0
        next_id = 0
        total_iters = 0
        # Iteration stats accumulators (for R2-3)
        sum_prefill_reqs = 0
        sum_decode_reqs = 0
        sum_tokens = 0
        steady_iters = 0
        steady_time_ms = 0.0
        steady_output_tokens = 0
        peak_prefill_reqs = 0
        peak_decode_reqs = 0
        peak_tokens = 0

        # Seed initial requests
        for _ in range(min(concurrency, self._config.num_requests)):
            waiting.append(Request(
                request_id=next_id, isl=isl, osl=osl, arrival_time_ms=0.0,
            ))
            next_id += 1

        max_iters = self._config.num_requests * (
            osl + isl // self._config.max_num_batched_tokens + 10
        )

        while len(completed) < self._config.num_requests and total_iters < max_iters:
            has_external_supply = bool(waiting) or next_id < self._config.num_requests
            # Schedule
            schedule = self._scheduler.schedule(waiting, running)
            if schedule.is_empty:
                break

            # Compute avg KV length for decode requests
            avg_kv = int(np.mean([r.kv_cache_len for r in schedule.decode_reqs])) \
                if schedule.decode_reqs else 0

            iter_lat = latency_calc.compute(
                prefill_tokens=schedule.total_prefill_tokens,
                prefill_batch_size=len(schedule.prefill_reqs),
                prefill_seq_len=isl,
                decode_batch_size=len(schedule.decode_reqs),
                decode_avg_kv_len=avg_kv,
            )

            in_steady_state = (
                len(completed) >= self._config.warmup_requests and has_external_supply
            )
            clock_ms += iter_lat
            total_iters += 1
            sum_prefill_reqs += len(schedule.prefill_reqs)
            sum_decode_reqs += len(schedule.decode_reqs)
            sum_tokens += schedule.total_tokens
            peak_prefill_reqs = max(peak_prefill_reqs, len(schedule.prefill_reqs))
            peak_decode_reqs = max(peak_decode_reqs, len(schedule.decode_reqs))
            peak_tokens = max(peak_tokens, schedule.total_tokens)
            if in_steady_state:
                steady_iters += 1
                steady_time_ms += iter_lat

            # --- Update prefill progress ---
            for req in schedule.prefill_reqs:
                tokens = schedule.prefill_tokens[req.request_id]
                # WAITING/PREEMPTED -> PREFILLING: move from waiting to running
                if req.state in {RequestState.WAITING, RequestState.PREEMPTED}:
                    req.state = RequestState.PREFILLING
                    if req.prefill_start_ms < 0:
                        req.prefill_start_ms = clock_ms - iter_lat
                    if req in waiting:
                        waiting.remove(req)
                    running.append(req)

                req.prefill_tokens_remaining -= tokens

                # PREFILLING -> DECODING: prefill complete
                if req.prefill_tokens_remaining <= 0:
                    req.prefill_tokens_remaining = 0
                    if req.first_token_ms < 0:
                        req.first_token_ms = clock_ms
                    req.state = RequestState.DECODING

            # --- Update decode progress ---
            newly_done: list[Request] = []
            for req in schedule.decode_reqs:
                req.generated_tokens += 1
                if in_steady_state:
                    steady_output_tokens += 1
                if req.generated_tokens >= req.osl - 1:
                    req.state = RequestState.DONE
                    req.finish_ms = clock_ms
                    newly_done.append(req)

            # Remove done, enqueue replacements
            for req in newly_done:
                running.remove(req)
                completed.append(req)
                if next_id < self._config.num_requests:
                    waiting.append(Request(
                        request_id=next_id, isl=isl, osl=osl,
                        arrival_time_ms=clock_ms,
                    ))
                    next_id += 1

            # --- Batch-skip optimization for pure decode phases ---
            if (not waiting
                    and all(r.state == RequestState.DECODING for r in running)
                    and running):
                min_remaining = min(
                    r.osl - 1 - r.generated_tokens for r in running
                )
                skip = max(0, min_remaining - 1)
                if skip > 0:
                    skip_lat = self._estimate_decode_skip_latency(
                        latency_calc=latency_calc,
                        decode_batch_size=len(running),
                        start_avg_kv_len=avg_kv + 1,
                        skip_iters=skip,
                    )
                    clock_ms += skip_lat
                    total_iters += skip
                    sum_decode_reqs += len(running) * skip
                    sum_tokens += len(running) * skip
                    peak_decode_reqs = max(peak_decode_reqs, len(running))
                    peak_tokens = max(peak_tokens, len(running))
                    if len(completed) >= self._config.warmup_requests:
                        steady_iters += skip
                        steady_time_ms += skip_lat
                        steady_output_tokens += len(running) * skip
                    for req in running:
                        req.generated_tokens += skip

        return self._collect_metrics(
            completed, num_gpus, total_iters,
            sum_prefill_reqs, sum_decode_reqs, sum_tokens,
            steady_iters, steady_time_ms, steady_output_tokens,
            peak_prefill_reqs, peak_decode_reqs, peak_tokens,
        )

    def _collect_metrics(
        self,
        completed: list[Request],
        num_gpus: int,
        total_iters: int,
        sum_prefill_reqs: int = 0,
        sum_decode_reqs: int = 0,
        sum_tokens: int = 0,
        steady_iters: int = 0,
        steady_time_ms: float = 0.0,
        steady_output_tokens: int = 0,
        peak_prefill_reqs: int = 0,
        peak_decode_reqs: int = 0,
        peak_tokens: int = 0,
    ) -> CBSimResult:
        """Collect TTFT, TPOT, throughput from completed requests."""
        warmup = self._config.warmup_requests
        steady = completed[warmup:]
        if len(steady) < 2 or steady_iters <= 0 or steady_time_ms <= 0:
            logger.warning(
                "Insufficient steady-state data: requests=%d iterations=%d time_ms=%.3f",
                len(steady), steady_iters, steady_time_ms,
            )
            return CBSimResult(
                mean_ttft_ms=float("inf"), p50_ttft_ms=float("inf"),
                p99_ttft_ms=float("inf"), mean_tpot_ms=float("inf"),
                throughput_tok_s=0.0, throughput_tok_s_gpu=0.0,
                num_gpus=num_gpus, total_iterations=total_iters,
                steady_state_requests=len(steady),
                steady_state_iterations=steady_iters,
                steady_state_time_ms=steady_time_ms,
            )

        ttfts = [r.first_token_ms - r.arrival_time_ms for r in steady]
        tpots = [
            (r.finish_ms - r.first_token_ms) / max(r.osl - 1, 1)
            for r in steady
        ]

        wall_s = steady_time_ms / 1000.0

        n = max(total_iters, 1)
        return CBSimResult(
            mean_ttft_ms=float(np.mean(ttfts)),
            p50_ttft_ms=float(np.median(ttfts)),
            p99_ttft_ms=float(np.percentile(ttfts, 99)),
            mean_tpot_ms=float(np.mean(tpots)),
            throughput_tok_s=steady_output_tokens / wall_s if wall_s > 0 else 0.0,
            throughput_tok_s_gpu=(
                steady_output_tokens / wall_s / max(num_gpus, 1) if wall_s > 0 else 0.0
            ),
            num_gpus=num_gpus,
            total_iterations=total_iters,
            steady_state_requests=len(steady),
            steady_state_iterations=steady_iters,
            steady_state_time_ms=steady_time_ms,
            avg_prefill_reqs_per_iter=sum_prefill_reqs / n,
            avg_decode_reqs_per_iter=sum_decode_reqs / n,
            avg_tokens_per_iter=sum_tokens / n,
            peak_prefill_reqs_per_iter=peak_prefill_reqs,
            peak_decode_reqs_per_iter=peak_decode_reqs,
            peak_tokens_per_iter=peak_tokens,
        )
