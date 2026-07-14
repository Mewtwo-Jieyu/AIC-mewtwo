"""Source-aligned EngineCore batch-queue state machine."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from typing import Callable, Protocol, Sequence

from .datatypes import CBSimConfig, Request, RequestState, ScheduleResult
from .scheduler import CBScheduler


@dataclass(frozen=True)
class TimedInput:
    arrival_ms: float
    payload: object


class EngineLoopInputSource(Protocol):
    def drain(self, now_ms: float) -> list[object]: ...

    def next_event_ms(self) -> float | None: ...


class TimedInputSource:
    """Mutable deterministic input source used by the engine busy-loop."""

    def __init__(self, arrivals: Sequence[TimedInput]) -> None:
        self._pending = sorted(arrivals, key=lambda item: item.arrival_ms)

    def add(self, item: TimedInput) -> None:
        self._pending.append(item)
        self._pending.sort(key=lambda pending: pending.arrival_ms)

    def drain(self, now_ms: float) -> list[object]:
        split = 0
        while split < len(self._pending) and self._pending[split].arrival_ms <= now_ms:
            split += 1
        ready = [item.payload for item in self._pending[:split]]
        del self._pending[:split]
        return ready

    def next_event_ms(self) -> float | None:
        if not self._pending:
            return None
        return self._pending[0].arrival_ms


@dataclass(frozen=True)
class EngineLoopBatch:
    batch_id: int
    launch_ms: float
    latency_ms: float
    complete_ms: float = -1.0
    payload: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EngineLoopResult:
    launched: list[EngineLoopBatch]
    completed: list[EngineLoopBatch]
    max_queue_depth: int
    final_clock_ms: float


class AsyncCBScheduler(CBScheduler):
    """CBScheduler adapter for vLLM's sampled/computed/placeholder states."""

    def __init__(self, config: CBSimConfig) -> None:
        super().__init__(config)
        self.active_trigger = -1
        self.preemptions_this_step = 0
        self.step_index = 0
        self.in_steady_state = False
        self.preemption_events: list[dict[str, int | bool]] = []

    @staticmethod
    def empty_result() -> ScheduleResult:
        return ScheduleResult()

    def schedule(
        self,
        waiting: list[Request],
        running: list[Request],
    ) -> ScheduleResult:
        self.preemptions_this_step = 0
        self.step_index += 1
        return super().schedule(waiting, running)

    def _next_waiting_candidate(
        self,
        waiting: list[Request],
        admitted_ids: set[int],
        preempted_ids: set[int],
    ) -> Request | None:
        if self.preemptions_this_step:
            return None
        return super()._next_waiting_candidate(
            waiting,
            admitted_ids,
            preempted_ids,
        )

    def _blocks_needed(self, req: Request, scheduled_tokens: int = 0) -> int:
        if req.state == RequestState.DECODING:
            total_tokens = req.isl + req.computed_output_tokens + scheduled_tokens
            if total_tokens <= 0 or self._config.block_size <= 0:
                return 0
            return (
                total_tokens + self._config.block_size - 1
            ) // self._config.block_size
        return super()._blocks_needed(req, scheduled_tokens)

    def _ensure_block_capacity(
        self,
        current_req: Request,
        waiting: list[Request],
        running: list[Request],
        result: ScheduleResult,
        preempted_ids: set[int],
    ) -> tuple[bool, int]:
        self.active_trigger = current_req.request_id
        try:
            return super()._ensure_block_capacity(
                current_req,
                waiting,
                running,
                result,
                preempted_ids,
            )
        finally:
            self.active_trigger = -1

    def _preempt(
        self,
        victim: Request,
        waiting: list[Request],
        running: list[Request],
        result: ScheduleResult,
        preempted_ids: set[int],
    ) -> int:
        self.preemptions_this_step += 1
        self.preemption_events.append(
            {
                "step": self.step_index,
                "trigger_request_id": self.active_trigger,
                "victim_request_id": victim.request_id,
                "victim_preemptions_before": victim.num_preemptions,
                "victim_sampled_output_tokens": victim.sampled_output_tokens,
                "victim_computed_output_tokens": victim.computed_output_tokens,
                "victim_output_placeholders": victim.output_placeholders,
                "recompute_tokens": victim.isl + victim.sampled_output_tokens,
                "metrics_steady_state": self.in_steady_state,
            }
        )
        released = super()._preempt(
            victim,
            waiting,
            running,
            result,
            preempted_ids,
        )
        victim.prefill_tokens_remaining = (
            victim.isl + victim.sampled_output_tokens
        )
        victim.computed_output_tokens = 0
        return released


def run_engine_loop(
    *,
    queue_depth: int,
    input_source: EngineLoopInputSource,
    on_drain: Callable[[float, list[object]], None],
    schedule: Callable[[float], EngineLoopBatch | None],
    on_complete: Callable[[EngineLoopBatch], None],
) -> EngineLoopResult:
    """Run the vLLM EngineCore busy-loop with serialized model execution."""
    if queue_depth < 1:
        raise ValueError("queue_depth must be positive")

    batch_queue: deque[EngineLoopBatch] = deque()
    launched: list[EngineLoopBatch] = []
    completed: list[EngineLoopBatch] = []
    now_ms = 0.0
    executor_available_ms = 0.0
    max_depth = 0

    while True:
        drained = input_source.drain(now_ms)
        if drained:
            on_drain(now_ms, drained)

        scheduled = None
        if len(batch_queue) < queue_depth:
            scheduled = schedule(now_ms)
        if scheduled is not None:
            if scheduled.latency_ms < 0:
                raise ValueError("batch latency must be non-negative")
            complete_ms = max(now_ms, executor_available_ms) + scheduled.latency_ms
            executor_available_ms = complete_ms
            scheduled = replace(
                scheduled,
                launch_ms=now_ms,
                complete_ms=complete_ms,
            )
            batch_queue.append(scheduled)
            launched.append(scheduled)
            max_depth = max(max_depth, len(batch_queue))
            if len(batch_queue) < queue_depth and batch_queue[0].complete_ms > now_ms:
                continue

        if batch_queue:
            finished = batch_queue.popleft()
            now_ms = max(now_ms, finished.complete_ms)
            completed.append(finished)
            on_complete(finished)
            continue

        next_event_ms = input_source.next_event_ms()
        if next_event_ms is not None:
            if next_event_ms <= now_ms:
                raise RuntimeError("input source did not drain its ready event")
            now_ms = next_event_ms
            continue
        if scheduled is None:
            break

    return EngineLoopResult(
        launched=launched,
        completed=completed,
        max_queue_depth=max_depth,
        final_clock_ms=now_ms,
    )
