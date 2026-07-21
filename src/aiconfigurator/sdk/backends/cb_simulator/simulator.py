"""Discrete-event CB simulation engine.

Closed-loop: maintains target concurrency by replacing completed
requests immediately. Request lifecycle:
WAITING/PREEMPTED -> PREFILLING -> DECODING -> DONE.
"""
from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

from .arrival import TokenizerArrivalLayer, TokenizerPrimitive, resolve_tokenizer_primitive
from .backend_semantic_profile import resolve_backend_semantic_profile
from .datatypes import CBSimConfig, CBSimResult, Request, RequestState
from .dp_admission import DPAdmissionRouter, DPReplicaCounts
from .engine_loop import AsyncCBScheduler, EngineLoopBatch, run_engine_loop
from .iteration_latency import IterationLatencyCalculator, ServingStateQueryAudit
from .scheduler import CBScheduler

if TYPE_CHECKING:
    from aiconfigurator.sdk.backends.base_backend import BaseBackend
    from aiconfigurator.sdk.models import BaseModel
    from aiconfigurator.sdk.perf_database import PerfDatabase

logger = logging.getLogger(__name__)


@dataclass
class _ReplicaState:
    """Mutable state owned by one DP replica."""

    replica_id: int
    scheduler: CBScheduler
    waiting: list[Request] = field(default_factory=list)
    running: list[Request] = field(default_factory=list)
    clock_ms: float = 0.0
    total_iters: int = 0


class CBSimulator:
    """Lightweight vLLM continuous batching simulator."""

    def __init__(
        self,
        backend: BaseBackend,
        model: BaseModel,
        database: PerfDatabase,
        config: CBSimConfig | None = None,
    ) -> None:
        base_config = config or CBSimConfig()
        database_backend = getattr(database, "backend", None)
        database_version = getattr(database, "version", None)
        if not isinstance(database_backend, str) or not isinstance(
            database_version, str
        ):
            raise ValueError(
                "CB simulator requires exact database backend and version metadata"
            )
        profile = resolve_backend_semantic_profile(
            backend=database_backend,
            version=database_version,
        )
        if (
            base_config.semantic_profile is not None
            and base_config.semantic_profile != profile
        ):
            raise ValueError(
                "explicit backend semantic profile does not match database profile"
            )
        self._semantic_profile = profile
        self._config = replace(base_config, semantic_profile=profile)
        self._engine_loop_enabled = profile.engine_loop_default_enabled
        if base_config.engine_loop_enabled:
            if not profile.engine_loop_diagnostic_available:
                raise ValueError("engine loop diagnostics are disabled by profile")
            self._engine_loop_enabled = True
        self._scheduler = CBScheduler(self._config)
        self._backend = backend
        self._model = model
        self._database = database
        self._last_latency_calc: IterationLatencyCalculator | None = None
        self._last_schedule_trace: list[dict[str, float | int | bool]] = []
        self._last_engine_loop_audit: dict[str, float | int | bool | str] = {
            "path": "not_run"
        }
        self._last_preemption_events: list[dict[str, int | bool]] = []

    def _resolve_engine_loop_primitive(self, isl: int) -> TokenizerPrimitive | None:
        model_path = getattr(self._model, "model_path", None)
        system = getattr(self._database, "system", None)
        backend = getattr(self._database, "backend", None)
        version = getattr(self._database, "version", None)
        if not all(
            isinstance(value, str)
            for value in (model_path, system, backend, version)
        ):
            return None
        return resolve_tokenizer_primitive(
            model_path=model_path,
            system=system,
            backend=backend,
            version=version,
            prompt_tokens=isl,
        )

    def _create_latency_calc(self, prefix: int) -> IterationLatencyCalculator:
        """Factory hook for latency calculator. Override in tests."""
        return IterationLatencyCalculator(
            self._backend,
            self._model,
            self._database,
            prefix=prefix,
            overlap_factor=self._config.overlap_factor,
            per_iteration_overhead_ms=self._config.per_iteration_overhead_ms,
            serving_state_max_num_batched_tokens=self._config.max_num_batched_tokens,
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
        if self._engine_loop_enabled:
            primitive = self._resolve_engine_loop_primitive(isl)
            if primitive is None:
                raise ValueError(
                    "engine loop enabled but no exact tokenizer primitive "
                    "matches this deployment"
                )
            return self._run_single_engine_loop(
                isl=isl,
                osl=osl,
                concurrency=concurrency,
                prefix=prefix,
                num_gpus=num_gpus,
                primitive=primitive,
            )

        self._last_engine_loop_audit = {
            "path": "legacy",
            "decode_skip_enabled": True,
            "engine_loop_enabled": self._engine_loop_enabled,
        }
        self._last_preemption_events = []
        latency_calc = self._create_latency_calc(prefix)
        self._last_latency_calc = latency_calc
        self._last_schedule_trace = []

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

    def _run_single_engine_loop(
        self,
        *,
        isl: int,
        osl: int,
        concurrency: int,
        prefix: int,
        num_gpus: int,
        primitive: TokenizerPrimitive,
    ) -> CBSimResult:
        """Run one replica with vLLM's non-blocking EngineCore batch queue."""
        latency_calc = self._create_latency_calc(prefix)
        self._last_latency_calc = latency_calc
        self._last_schedule_trace = []

        scheduler = AsyncCBScheduler(self._config)
        arrival = TokenizerArrivalLayer(primitive)
        waiting: list[Request] = []
        running: list[Request] = []
        completed: list[Request] = []
        first_prefill_completion_steps: dict[int, int] = {}
        request_completion_steps: dict[int, int] = {}
        next_id = 0
        stats = {
            "total_iters": 0,
            "sum_prefill_reqs": 0,
            "sum_decode_reqs": 0,
            "sum_tokens": 0,
            "steady_iters": 0,
            "steady_time_ms": 0.0,
            "steady_output_tokens": 0,
            "peak_prefill_reqs": 0,
            "peak_decode_reqs": 0,
            "peak_tokens": 0,
        }

        initial = min(concurrency, self._config.num_requests)
        arrival.submit_many(
            [((request_id, 0.0), isl) for request_id in range(initial)],
            now_ms=0.0,
        )
        next_id = initial
        max_iters = self._config.num_requests * (
            osl + isl // self._config.max_num_batched_tokens + 10
        )

        def on_drain(now_ms: float, items: list[object]) -> None:
            for item in items:
                request_id, submitted_ms = item
                waiting.append(
                    Request(
                        request_id=int(request_id),
                        isl=isl,
                        osl=osl,
                        arrival_time_ms=float(submitted_ms),
                    )
                )

        def schedule(now_ms: float) -> EngineLoopBatch | None:
            if (
                len(completed) >= self._config.num_requests
                or stats["total_iters"] >= max_iters
            ):
                return None

            guarded: list[Request] = []
            for req in running:
                if (
                    req.state == RequestState.DECODING
                    and req.sampled_output_tokens + req.output_placeholders >= req.osl
                ):
                    req.state = RequestState.DONE
                    guarded.append(req)

            has_external_supply = (
                bool(waiting)
                or next_id < self._config.num_requests
                or arrival.next_event_ms() is not None
            )
            in_steady_state = (
                len(completed) >= self._config.warmup_requests
                and has_external_supply
            )
            scheduler.in_steady_state = in_steady_state
            result = scheduler.schedule(waiting, running)
            for req in guarded:
                if req.state == RequestState.DONE:
                    req.state = RequestState.DECODING
            if result.is_empty:
                return None

            avg_kv = (
                int(np.mean([
                    req.isl + req.computed_output_tokens
                    for req in result.decode_reqs
                ]))
                if result.decode_reqs
                else 0
            )
            iter_lat = latency_calc.compute(
                prefill_tokens=result.total_prefill_tokens,
                prefill_batch_size=len(result.prefill_reqs),
                prefill_seq_len=isl,
                decode_batch_size=len(result.decode_reqs),
                decode_avg_kv_len=avg_kv,
            )

            completed_prefill_ids: list[int] = []
            for req in result.prefill_reqs:
                tokens = result.prefill_tokens[req.request_id]
                if req.state in {RequestState.WAITING, RequestState.PREEMPTED}:
                    req.state = RequestState.PREFILLING
                    if req.prefill_start_ms < 0:
                        req.prefill_start_ms = now_ms
                    if req in waiting:
                        waiting.remove(req)
                    if req not in running:
                        running.append(req)
                req.prefill_tokens_remaining -= tokens
                if req.prefill_tokens_remaining <= 0:
                    req.prefill_tokens_remaining = 0
                    req.state = RequestState.DECODING
                    req.computed_output_tokens = req.sampled_output_tokens
                    req.output_placeholders += 1
                    completed_prefill_ids.append(req.request_id)

            for req in result.decode_reqs:
                req.computed_output_tokens += 1
                req.output_placeholders += 1

            stats["total_iters"] += 1
            stats["sum_prefill_reqs"] += len(result.prefill_reqs)
            stats["sum_decode_reqs"] += len(result.decode_reqs)
            stats["sum_tokens"] += result.total_tokens
            stats["peak_prefill_reqs"] = max(
                stats["peak_prefill_reqs"], len(result.prefill_reqs)
            )
            stats["peak_decode_reqs"] = max(
                stats["peak_decode_reqs"], len(result.decode_reqs)
            )
            stats["peak_tokens"] = max(
                stats["peak_tokens"], result.total_tokens
            )
            if in_steady_state:
                stats["steady_iters"] += 1
                stats["steady_time_ms"] += iter_lat

            return EngineLoopBatch(
                batch_id=int(stats["total_iters"]),
                launch_ms=now_ms,
                latency_ms=iter_lat,
                payload={
                    "prefill_reqs": len(result.prefill_reqs),
                    "prefill_tokens": result.total_prefill_tokens,
                    "decode_reqs": len(result.decode_reqs),
                    "total_tokens": result.total_tokens,
                    "completed_prefill_ids": completed_prefill_ids,
                    "decode_ids": [req.request_id for req in result.decode_reqs],
                    "sampled_output_ids": completed_prefill_ids
                    + [req.request_id for req in result.decode_reqs],
                    "in_steady_state": in_steady_state,
                },
            )

        def find_request(request_id: int) -> Request | None:
            return next(
                (
                    req
                    for req in [*running, *waiting]
                    if req.request_id == request_id
                ),
                None,
            )

        def on_complete(batch: EngineLoopBatch) -> None:
            nonlocal next_id
            prefill_ids = set(batch.payload["completed_prefill_ids"])
            decode_ids = set(batch.payload["decode_ids"])
            newly_done: list[Request] = []
            for request_id in batch.payload["sampled_output_ids"]:
                req = find_request(int(request_id))
                if req is None:
                    continue
                req.output_placeholders -= 1
                if req.output_placeholders < 0:
                    raise RuntimeError("negative output placeholder count")
                req.sampled_output_tokens += 1
                if request_id in prefill_ids and req.first_token_ms < 0:
                    req.first_token_ms = batch.complete_ms
                    first_prefill_completion_steps[req.request_id] = batch.batch_id
                if req.state in {
                    RequestState.WAITING,
                    RequestState.PREEMPTED,
                    RequestState.PREFILLING,
                }:
                    req.prefill_tokens_remaining += 1
                if req.sampled_output_tokens >= req.osl:
                    req.state = RequestState.DONE
                    req.finish_ms = batch.complete_ms
                    request_completion_steps[req.request_id] = batch.batch_id
                    newly_done.append(req)

            if batch.payload["in_steady_state"]:
                stats["steady_output_tokens"] += len(decode_ids)

            self._last_schedule_trace.append(
                {
                    "replica_id": 0,
                    "local_iter": batch.batch_id,
                    "start_ms": batch.launch_ms,
                    "end_ms": batch.complete_ms,
                    "prefill_reqs": batch.payload["prefill_reqs"],
                    "prefill_tokens": batch.payload["prefill_tokens"],
                    "decode_reqs": batch.payload["decode_reqs"],
                    "total_tokens": batch.payload["total_tokens"],
                    "is_mixed": bool(
                        batch.payload["prefill_reqs"]
                        and batch.payload["decode_reqs"]
                    ),
                }
            )

            for req in newly_done:
                if req in running:
                    running.remove(req)
                if req in waiting:
                    waiting.remove(req)
                completed.append(req)
                if next_id < self._config.num_requests:
                    submitted_ms = batch.complete_ms
                    arrival.submit_many(
                        [((next_id, submitted_ms), isl)],
                        now_ms=submitted_ms,
                    )
                    next_id += 1

        runtime_start = time.perf_counter()
        machine = run_engine_loop(
            queue_depth=self._semantic_profile.batch_queue_depth,
            input_source=arrival,
            on_drain=on_drain,
            schedule=schedule,
            on_complete=on_complete,
        )
        runtime_seconds = time.perf_counter() - runtime_start
        completed.sort(key=lambda req: req.finish_ms)

        all_events = scheduler.preemption_events
        self._last_preemption_events = [dict(event) for event in all_events]
        if first_prefill_completion_steps and request_completion_steps:
            final_first_prefill = max(first_prefill_completion_steps.values())
            first_completion = min(request_completion_steps.values())
            if final_first_prefill < first_completion:
                steady_start_step = final_first_prefill + 1
                steady_end_step = first_completion
                steady_window_mode = "fully_admitted_before_drain"
            else:
                steady_start_step = first_completion + 1
                steady_end_step = final_first_prefill + 1
                steady_window_mode = "replacement_plateau_while_waiting_nonempty"
        else:
            steady_start_step = 0
            steady_end_step = 0
            steady_window_mode = "unavailable"
        steady_events = [
            event
            for event in all_events
            if steady_start_step <= int(event["step"]) < steady_end_step
        ]

        def preemption_signature(
            events: list[dict[str, int | bool]],
        ) -> tuple[int, int, int]:
            victims = Counter(int(event["victim_request_id"]) for event in events)
            self_preemptions = sum(
                event["trigger_request_id"] == event["victim_request_id"]
                for event in events
            )
            repeats = sum(count - 1 for count in victims.values())
            return len(events), self_preemptions, repeats

        preemptions, self_preemptions, repeats = preemption_signature(all_events)
        steady_preemptions, steady_self_preemptions, steady_repeats = (
            preemption_signature(steady_events)
        )
        self._last_engine_loop_audit = {
            "path": "engine_loop",
            "queue_depth": self._semantic_profile.batch_queue_depth,
            "max_queue_depth": machine.max_queue_depth,
            "decode_skip_enabled": False,
            "sim_runtime_seconds": runtime_seconds,
            "simulated_wall_ms": machine.final_clock_ms,
            "preemptions": preemptions,
            "self_preemptions": self_preemptions,
            "repeat_victim_events": repeats,
            "steady_preemptions": steady_preemptions,
            "steady_self_preemptions": steady_self_preemptions,
            "steady_repeat_victim_events": steady_repeats,
            "steady_start_step": steady_start_step,
            "steady_end_step": steady_end_step,
            "steady_window_mode": steady_window_mode,
        }

        return self._collect_metrics(
            completed,
            num_gpus,
            int(stats["total_iters"]),
            int(stats["sum_prefill_reqs"]),
            int(stats["sum_decode_reqs"]),
            int(stats["sum_tokens"]),
            int(stats["steady_iters"]),
            float(stats["steady_time_ms"]),
            int(stats["steady_output_tokens"]),
            int(stats["peak_prefill_reqs"]),
            int(stats["peak_decode_reqs"]),
            int(stats["peak_tokens"]),
        )

    def run_multi_replica(
        self,
        isl: int,
        osl: int,
        concurrency: int,
        data_parallel_size: int,
        prefix: int = 0,
        num_gpus: int = 1,
        lockstep: bool = False,
    ) -> CBSimResult:
        """Run global closed-loop simulation across DP replicas.

        ``run`` intentionally remains the single-replica primitive.  This method
        adds only the architecture missing from the DP path: independent
        schedulers/clocks/KV states plus a global admission router.
        """
        if data_parallel_size > 1 and self._engine_loop_enabled:
            raise NotImplementedError(
                "multi-replica engine loop is deferred to Phase462 Step 3"
            )
        if data_parallel_size <= 1:
            return self.run(
                isl=isl,
                osl=osl,
                concurrency=concurrency,
                prefix=prefix,
                num_gpus=num_gpus,
            )
        if lockstep:
            return self._run_multi_replica_lockstep(
                isl=isl,
                osl=osl,
                concurrency=concurrency,
                data_parallel_size=data_parallel_size,
                prefix=prefix,
                num_gpus=num_gpus,
            )

        latency_calc = self._create_latency_calc(prefix)
        self._last_latency_calc = latency_calc
        self._last_schedule_trace = []

        replicas = [
            _ReplicaState(
                replica_id=idx,
                scheduler=CBScheduler(self._config),
            )
            for idx in range(data_parallel_size)
        ]
        router = DPAdmissionRouter(data_parallel_size)
        completed: list[Request] = []
        next_id = 0

        def actual_counts() -> list[DPReplicaCounts]:
            return [
                DPReplicaCounts(waiting=len(replica.waiting), running=len(replica.running))
                for replica in replicas
            ]

        def route_request(now_ms: float) -> None:
            nonlocal next_id
            replica_idx = router.route(
                now_ms=now_ms,
                actual_counts=actual_counts(),
            )
            replicas[replica_idx].waiting.append(
                Request(
                    request_id=next_id,
                    isl=isl,
                    osl=osl,
                    arrival_time_ms=now_ms,
                )
            )
            next_id += 1

        for _ in range(min(concurrency, self._config.num_requests)):
            route_request(0.0)

        max_iters = self._config.num_requests * data_parallel_size * (
            osl + isl // self._config.max_num_batched_tokens + 10
        )
        total_iters = 0
        sum_prefill_reqs = 0
        sum_decode_reqs = 0
        sum_tokens = 0
        steady_iters = 0
        steady_start_ms: float | None = None
        steady_end_ms = 0.0
        steady_output_tokens = 0
        peak_prefill_reqs = 0
        peak_decode_reqs = 0
        peak_tokens = 0

        while len(completed) < self._config.num_requests and total_iters < max_iters:
            active = [
                replica for replica in replicas
                if replica.waiting or replica.running
            ]
            if not active:
                break
            replica = min(active, key=lambda item: (item.clock_ms, item.replica_id))
            has_external_supply = next_id < self._config.num_requests
            schedule = replica.scheduler.schedule(replica.waiting, replica.running)
            if schedule.is_empty:
                break

            avg_kv = int(np.mean([r.kv_cache_len for r in schedule.decode_reqs])) \
                if schedule.decode_reqs else 0
            iter_lat = latency_calc.compute(
                prefill_tokens=schedule.total_prefill_tokens,
                prefill_batch_size=len(schedule.prefill_reqs),
                prefill_seq_len=isl,
                decode_batch_size=len(schedule.decode_reqs),
                decode_avg_kv_len=avg_kv,
            )
            step_start_ms = replica.clock_ms
            step_end_ms = step_start_ms + iter_lat
            in_steady_state = (
                len(completed) >= self._config.warmup_requests and has_external_supply
            )

            replica.clock_ms = step_end_ms
            replica.total_iters += 1
            total_iters += 1
            sum_prefill_reqs += len(schedule.prefill_reqs)
            sum_decode_reqs += len(schedule.decode_reqs)
            sum_tokens += schedule.total_tokens
            peak_prefill_reqs = max(peak_prefill_reqs, len(schedule.prefill_reqs))
            peak_decode_reqs = max(peak_decode_reqs, len(schedule.decode_reqs))
            peak_tokens = max(peak_tokens, schedule.total_tokens)
            if in_steady_state:
                steady_iters += 1
                if steady_start_ms is None:
                    steady_start_ms = step_start_ms
                steady_end_ms = max(steady_end_ms, step_end_ms)

            self._last_schedule_trace.append(
                {
                    "replica_id": replica.replica_id,
                    "local_iter": replica.total_iters,
                    "start_ms": step_start_ms,
                    "end_ms": step_end_ms,
                    "prefill_reqs": len(schedule.prefill_reqs),
                    "prefill_tokens": schedule.total_prefill_tokens,
                    "decode_reqs": len(schedule.decode_reqs),
                    "total_tokens": schedule.total_tokens,
                    "is_mixed": bool(schedule.prefill_reqs and schedule.decode_reqs),
                }
            )

            for req in schedule.prefill_reqs:
                tokens = schedule.prefill_tokens[req.request_id]
                if req.state in {RequestState.WAITING, RequestState.PREEMPTED}:
                    req.state = RequestState.PREFILLING
                    if req.prefill_start_ms < 0:
                        req.prefill_start_ms = step_start_ms
                    if req in replica.waiting:
                        replica.waiting.remove(req)
                    replica.running.append(req)

                req.prefill_tokens_remaining -= tokens
                if req.prefill_tokens_remaining <= 0:
                    req.prefill_tokens_remaining = 0
                    if req.first_token_ms < 0:
                        req.first_token_ms = step_end_ms
                    req.state = RequestState.DECODING

            newly_done: list[Request] = []
            for req in schedule.decode_reqs:
                req.generated_tokens += 1
                if in_steady_state:
                    steady_output_tokens += 1
                if req.generated_tokens >= req.osl - 1:
                    req.state = RequestState.DONE
                    req.finish_ms = step_end_ms
                    newly_done.append(req)

            for req in newly_done:
                replica.running.remove(req)
                completed.append(req)
                if next_id < self._config.num_requests:
                    route_request(step_end_ms)

        completed.sort(key=lambda req: req.finish_ms)
        steady_time_ms = (
            steady_end_ms - steady_start_ms
            if steady_start_ms is not None and steady_end_ms > steady_start_ms
            else 0.0
        )
        return self._collect_metrics(
            completed,
            num_gpus,
            total_iters,
            sum_prefill_reqs,
            sum_decode_reqs,
            sum_tokens,
            steady_iters,
            steady_time_ms,
            steady_output_tokens,
            peak_prefill_reqs,
            peak_decode_reqs,
            peak_tokens,
        )

    def _run_multi_replica_lockstep(
        self,
        isl: int,
        osl: int,
        concurrency: int,
        data_parallel_size: int,
        prefix: int,
        num_gpus: int,
    ) -> CBSimResult:
        """Run DP replicas with vLLM-style pad-to-max iteration lockstep."""
        latency_calc = self._create_latency_calc(prefix)
        self._last_latency_calc = latency_calc
        self._last_schedule_trace = []

        replicas = [
            _ReplicaState(
                replica_id=idx,
                scheduler=CBScheduler(self._config),
            )
            for idx in range(data_parallel_size)
        ]
        router = DPAdmissionRouter(data_parallel_size)
        completed: list[Request] = []
        next_id = 0
        global_clock_ms = 0.0

        def actual_counts() -> list[DPReplicaCounts]:
            return [
                DPReplicaCounts(waiting=len(replica.waiting), running=len(replica.running))
                for replica in replicas
            ]

        def route_request(now_ms: float) -> None:
            nonlocal next_id
            replica_idx = router.route(
                now_ms=now_ms,
                actual_counts=actual_counts(),
            )
            replicas[replica_idx].waiting.append(
                Request(
                    request_id=next_id,
                    isl=isl,
                    osl=osl,
                    arrival_time_ms=now_ms,
                )
            )
            next_id += 1

        for _ in range(min(concurrency, self._config.num_requests)):
            route_request(0.0)

        max_iters = self._config.num_requests * data_parallel_size * (
            osl + isl // self._config.max_num_batched_tokens + 10
        )
        total_iters = 0
        sum_prefill_reqs = 0
        sum_decode_reqs = 0
        sum_tokens = 0
        steady_iters = 0
        steady_start_ms: float | None = None
        steady_end_ms = 0.0
        steady_output_tokens = 0
        peak_prefill_reqs = 0
        peak_decode_reqs = 0
        peak_tokens = 0

        while len(completed) < self._config.num_requests and total_iters < max_iters:
            has_external_supply = next_id < self._config.num_requests
            cycle: list[tuple[_ReplicaState, object, float]] = []
            for replica in replicas:
                if not replica.waiting and not replica.running:
                    continue
                schedule = replica.scheduler.schedule(replica.waiting, replica.running)
                if schedule.is_empty:
                    continue
                avg_kv = int(np.mean([r.kv_cache_len for r in schedule.decode_reqs])) \
                    if schedule.decode_reqs else 0
                iter_lat = latency_calc.compute(
                    prefill_tokens=schedule.total_prefill_tokens,
                    prefill_batch_size=len(schedule.prefill_reqs),
                    prefill_seq_len=isl,
                    decode_batch_size=len(schedule.decode_reqs),
                    decode_avg_kv_len=avg_kv,
                )
                cycle.append((replica, schedule, iter_lat))
            if not cycle:
                break

            step_start_ms = global_clock_ms
            step_ms = max(iter_lat for _, _, iter_lat in cycle)
            step_end_ms = step_start_ms + step_ms
            in_steady_state = (
                len(completed) >= self._config.warmup_requests and has_external_supply
            )
            global_clock_ms = step_end_ms

            if in_steady_state:
                if steady_start_ms is None:
                    steady_start_ms = step_start_ms
                steady_end_ms = step_end_ms

            for replica, schedule, _iter_lat in cycle:
                replica.clock_ms = step_end_ms
                replica.total_iters += 1
                total_iters += 1
                sum_prefill_reqs += len(schedule.prefill_reqs)
                sum_decode_reqs += len(schedule.decode_reqs)
                sum_tokens += schedule.total_tokens
                peak_prefill_reqs = max(peak_prefill_reqs, len(schedule.prefill_reqs))
                peak_decode_reqs = max(peak_decode_reqs, len(schedule.decode_reqs))
                peak_tokens = max(peak_tokens, schedule.total_tokens)
                if in_steady_state:
                    steady_iters += 1

                self._last_schedule_trace.append(
                    {
                        "replica_id": replica.replica_id,
                        "local_iter": replica.total_iters,
                        "start_ms": step_start_ms,
                        "end_ms": step_end_ms,
                        "prefill_reqs": len(schedule.prefill_reqs),
                        "prefill_tokens": schedule.total_prefill_tokens,
                        "decode_reqs": len(schedule.decode_reqs),
                        "total_tokens": schedule.total_tokens,
                        "is_mixed": bool(schedule.prefill_reqs and schedule.decode_reqs),
                    }
                )

            for replica, schedule, _iter_lat in cycle:
                for req in schedule.prefill_reqs:
                    tokens = schedule.prefill_tokens[req.request_id]
                    if req.state in {RequestState.WAITING, RequestState.PREEMPTED}:
                        req.state = RequestState.PREFILLING
                        if req.prefill_start_ms < 0:
                            req.prefill_start_ms = step_start_ms
                        if req in replica.waiting:
                            replica.waiting.remove(req)
                        replica.running.append(req)

                    req.prefill_tokens_remaining -= tokens
                    if req.prefill_tokens_remaining <= 0:
                        req.prefill_tokens_remaining = 0
                        if req.first_token_ms < 0:
                            req.first_token_ms = step_end_ms
                        req.state = RequestState.DECODING

                newly_done: list[Request] = []
                for req in schedule.decode_reqs:
                    req.generated_tokens += 1
                    if in_steady_state:
                        steady_output_tokens += 1
                    if req.generated_tokens >= req.osl - 1:
                        req.state = RequestState.DONE
                        req.finish_ms = step_end_ms
                        newly_done.append(req)

                for req in newly_done:
                    replica.running.remove(req)
                    completed.append(req)

            while next_id < self._config.num_requests and sum(
                len(replica.waiting) + len(replica.running)
                for replica in replicas
            ) < concurrency:
                route_request(step_end_ms)

        completed.sort(key=lambda req: req.finish_ms)
        steady_time_ms = (
            steady_end_ms - steady_start_ms
            if steady_start_ms is not None and steady_end_ms > steady_start_ms
            else 0.0
        )
        return self._collect_metrics(
            completed,
            num_gpus,
            total_iters,
            sum_prefill_reqs,
            sum_decode_reqs,
            sum_tokens,
            steady_iters,
            steady_time_ms,
            steady_output_tokens,
            peak_prefill_reqs,
            peak_decode_reqs,
            peak_tokens,
        )

    def get_last_serving_state_query_audit(self) -> list[ServingStateQueryAudit]:
        """Return serving-state query audit records from the most recent run."""
        if self._last_latency_calc is None:
            return []
        return self._last_latency_calc.get_serving_state_query_audit()

    def get_last_performance_source_map(self) -> dict:
        """Return charged performance sources from the most recent run."""
        if self._last_latency_calc is None:
            return {"context": {}, "generation": {}}
        return self._last_latency_calc.get_performance_source_map()

    def get_last_charge_ledger(self) -> list:
        """Return additive iteration charges from the most recent run."""
        if self._last_latency_calc is None:
            return []
        return self._last_latency_calc.get_charge_ledger()

    def get_last_schedule_trace(self) -> list[dict[str, float | int | bool]]:
        """Return per-replica schedule trace from the most recent multi run."""
        return list(self._last_schedule_trace)

    def get_last_engine_loop_audit(self) -> dict[str, float | int | bool | str]:
        """Return path, timing, and preemption counters from the latest run."""
        return dict(self._last_engine_loop_audit)

    def get_last_preemption_events(self) -> list[dict[str, int | bool]]:
        """Return source-aligned preemption decision packets from the latest run."""
        return [dict(event) for event in self._last_preemption_events]

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
