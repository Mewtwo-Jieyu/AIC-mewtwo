"""Discrete-event CB simulation engine.

Closed-loop: maintains target concurrency by replacing completed
requests immediately. Request lifecycle: WAITING -> PREFILLING -> DECODING -> DONE.
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
            self._backend, self._model, self._database, prefix=prefix,
        )

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

            clock_ms += iter_lat
            total_iters += 1
            sum_prefill_reqs += len(schedule.prefill_reqs)
            sum_decode_reqs += len(schedule.decode_reqs)
            sum_tokens += schedule.total_tokens

            # --- Update prefill progress ---
            for req in schedule.prefill_reqs:
                tokens = schedule.prefill_tokens[req.request_id]
                # WAITING -> PREFILLING: move from waiting to running
                if req.state == RequestState.WAITING:
                    req.state = RequestState.PREFILLING
                    req.prefill_start_ms = clock_ms - iter_lat
                    if req in waiting:
                        waiting.remove(req)
                    running.append(req)

                req.prefill_tokens_remaining -= tokens

                # PREFILLING -> DECODING: prefill complete
                if req.prefill_tokens_remaining <= 0:
                    req.prefill_tokens_remaining = 0
                    req.first_token_ms = clock_ms
                    req.state = RequestState.DECODING

            # --- Update decode progress ---
            newly_done: list[Request] = []
            for req in schedule.decode_reqs:
                req.generated_tokens += 1
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
                    skip_lat = iter_lat * skip  # approx: same decode composition
                    clock_ms += skip_lat
                    total_iters += skip
                    sum_decode_reqs += len(running) * skip
                    sum_tokens += len(running) * skip
                    for req in running:
                        req.generated_tokens += skip

        return self._collect_metrics(
            completed, num_gpus, total_iters,
            sum_prefill_reqs, sum_decode_reqs, sum_tokens,
        )

    def _collect_metrics(
        self,
        completed: list[Request],
        num_gpus: int,
        total_iters: int,
        sum_prefill_reqs: int = 0,
        sum_decode_reqs: int = 0,
        sum_tokens: int = 0,
    ) -> CBSimResult:
        """Collect TTFT, TPOT, throughput from completed requests."""
        warmup = self._config.warmup_requests
        steady = completed[warmup:]
        if len(steady) < 2:
            logger.warning("Too few steady-state requests: %d", len(steady))
            return CBSimResult(
                mean_ttft_ms=float("inf"), p50_ttft_ms=float("inf"),
                p99_ttft_ms=float("inf"), mean_tpot_ms=float("inf"),
                throughput_tok_s=0.0, throughput_tok_s_gpu=0.0,
                num_gpus=num_gpus, total_iterations=total_iters,
                steady_state_requests=len(steady),
            )

        ttfts = [r.first_token_ms - r.arrival_time_ms for r in steady]
        tpots = [
            (r.finish_ms - r.first_token_ms) / max(r.osl - 1, 1)
            for r in steady
        ]

        window_start = steady[0].arrival_time_ms
        window_end = steady[-1].finish_ms
        total_out_tokens = sum(r.osl - 1 for r in steady)
        wall_s = (window_end - window_start) / 1000.0

        n = max(total_iters, 1)
        return CBSimResult(
            mean_ttft_ms=float(np.mean(ttfts)),
            p50_ttft_ms=float(np.median(ttfts)),
            p99_ttft_ms=float(np.percentile(ttfts, 99)),
            mean_tpot_ms=float(np.mean(tpots)),
            throughput_tok_s=total_out_tokens / wall_s if wall_s > 0 else 0.0,
            throughput_tok_s_gpu=(
                total_out_tokens / wall_s / max(num_gpus, 1) if wall_s > 0 else 0.0
            ),
            num_gpus=num_gpus,
            total_iterations=total_iters,
            steady_state_requests=len(steady),
            avg_prefill_reqs_per_iter=sum_prefill_reqs / n,
            avg_decode_reqs_per_iter=sum_decode_reqs / n,
            avg_tokens_per_iter=sum_tokens / n,
        )
