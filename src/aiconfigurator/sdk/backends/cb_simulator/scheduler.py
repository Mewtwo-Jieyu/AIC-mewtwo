"""vLLM-style chunked prefill scheduler."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .datatypes import RequestState, ScheduleResult

if TYPE_CHECKING:
    from .datatypes import CBSimConfig, Request

logger = logging.getLogger(__name__)


class CBScheduler:
    """Simplified vLLM continuous batching scheduler."""

    def __init__(self, config: CBSimConfig) -> None:
        self._config = config

    def _cap_prefill_chunk(self, remaining: int, budget: int) -> int:
        """Apply long-prefill chunk capping before consuming token budget."""
        chunk = min(remaining, budget)
        threshold = self._config.long_prefill_token_threshold
        if threshold > 0:
            chunk = min(chunk, threshold)
        return chunk

    def _blocks_needed(self, req: Request, scheduled_tokens: int = 0) -> int:
        """Return total KV blocks needed after this iteration's new tokens land."""
        total_tokens = req.kv_cache_len + scheduled_tokens
        if total_tokens <= 0 or self._config.block_size <= 0:
            return 0
        return (total_tokens + self._config.block_size - 1) // self._config.block_size

    def _scheduled_token_deltas(self, result: ScheduleResult) -> dict[int, int]:
        deltas = dict(result.prefill_tokens)
        for req in result.decode_reqs:
            deltas[req.request_id] = deltas.get(req.request_id, 0) + 1
        return deltas

    def _total_blocks(
        self,
        running: list[Request],
        result: ScheduleResult,
    ) -> int:
        reqs: dict[int, Request] = {}
        for req in running:
            reqs[req.request_id] = req
        for req in result.prefill_reqs:
            reqs[req.request_id] = req
        for req in result.decode_reqs:
            reqs[req.request_id] = req
        scheduled_deltas = self._scheduled_token_deltas(result)
        return sum(
            self._blocks_needed(req, scheduled_deltas.get(req.request_id, 0))
            for req in reqs.values()
        )

    def _remove_from_result(
        self,
        req: Request,
        result: ScheduleResult,
    ) -> int:
        released_tokens = 0
        if req in result.decode_reqs:
            result.decode_reqs.remove(req)
            released_tokens += 1
        prefill_tokens = result.prefill_tokens.pop(req.request_id, 0)
        if prefill_tokens > 0:
            released_tokens += prefill_tokens
        if req in result.prefill_reqs:
            result.prefill_reqs.remove(req)
        return released_tokens

    def _preempt(
        self,
        victim: Request,
        waiting: list[Request],
        running: list[Request],
        result: ScheduleResult,
        preempted_ids: set[int],
    ) -> int:
        released_tokens = self._remove_from_result(victim, result)
        if victim in running:
            running.remove(victim)
        if victim in waiting:
            waiting.remove(victim)
        victim.state = RequestState.PREEMPTED
        victim.prefill_tokens_remaining = victim.isl + victim.generated_tokens
        victim.num_preemptions += 1
        waiting.insert(0, victim)
        preempted_ids.add(victim.request_id)
        return released_tokens

    def _ensure_block_capacity(
        self,
        current_req: Request,
        waiting: list[Request],
        running: list[Request],
        result: ScheduleResult,
        preempted_ids: set[int],
    ) -> tuple[bool, int]:
        """Preempt running-tail requests until block usage fits the limit."""
        if self._config.num_gpu_blocks <= 0:
            return True, 0

        released_tokens = 0
        while self._total_blocks(running, result) > self._config.num_gpu_blocks:
            if running:
                victim = running[-1]
                released_tokens += self._preempt(
                    victim, waiting, running, result, preempted_ids,
                )
                if victim is current_req:
                    return False, released_tokens
                continue

            released_tokens += self._remove_from_result(current_req, result)
            return False, released_tokens

        return True, released_tokens

    def _fits_block_capacity(
        self,
        running: list[Request],
        result: ScheduleResult,
    ) -> bool:
        if self._config.num_gpu_blocks <= 0:
            return True
        return self._total_blocks(running, result) <= self._config.num_gpu_blocks

    def _next_waiting_candidate(
        self,
        waiting: list[Request],
        admitted_ids: set[int],
        preempted_ids: set[int],
    ) -> Request | None:
        for req in waiting:
            if req.request_id in admitted_ids or req.request_id in preempted_ids:
                continue
            if req.state in {RequestState.WAITING, RequestState.PREEMPTED}:
                return req
        return None

    def schedule(
        self,
        waiting: list[Request],
        running: list[Request],
    ) -> ScheduleResult:
        """Schedule one iteration.

        Args:
            waiting: Queue of WAITING requests (FCFS order). May be mutated.
            running: PREFILLING or DECODING requests.

        Returns:
            ScheduleResult with prefill and decode assignments.
        """
        budget = self._config.max_num_batched_tokens
        result = ScheduleResult()
        preempted_ids: set[int] = set()

        # Step 1: Schedule RUNNING requests in queue order. vLLM v1 does not
        # have a separate decode-first phase; each running request consumes the
        # next tokens it needs until the per-step token budget is exhausted.
        for req in list(running):
            if req not in running:
                continue
            if budget <= 0:
                break
            if req.state == RequestState.DECODING:
                result.decode_reqs.append(req)
                budget -= 1
                fits, released = self._ensure_block_capacity(
                    req, waiting, running, result, preempted_ids,
                )
                budget += released
                if not fits:
                    return result
            elif req.state == RequestState.PREFILLING:
                chunk = self._cap_prefill_chunk(
                    req.prefill_tokens_remaining, max(budget, 0),
                )
                if chunk <= 0:
                    continue
                result.prefill_reqs.append(req)
                result.prefill_tokens[req.request_id] = chunk
                budget -= chunk
                fits, released = self._ensure_block_capacity(
                    req, waiting, running, result, preempted_ids,
                )
                budget += released
                if not fits:
                    return result

        # Step 2: Admit new requests from waiting queue.
        while budget > 0:
            num_seqs = len(running) + sum(
                1 for req in result.prefill_reqs if req not in running
            )
            if num_seqs >= self._config.max_num_seqs:
                break
            admitted_ids = {req.request_id for req in result.prefill_reqs}
            req = self._next_waiting_candidate(waiting, admitted_ids, preempted_ids)
            if req is None:
                break
            chunk = self._cap_prefill_chunk(req.prefill_tokens_remaining, budget)
            if chunk <= 0:
                break

            result.prefill_reqs.append(req)
            result.prefill_tokens[req.request_id] = chunk
            budget -= chunk
            if not self._fits_block_capacity(running, result):
                budget += self._remove_from_result(req, result)
                break

            # If this was a partial prefill, stop admitting more.
            if chunk < req.prefill_tokens_remaining:
                break

        return result
