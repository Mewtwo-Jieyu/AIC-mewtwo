"""vLLM-style chunked prefill scheduler.

Models vLLM's core scheduling behavior:
1. Reserve 1 token per DECODING request in running set.
2. Continue any PREFILLING request in running set (partial prefill).
3. Admit new waiting requests: allow multiple full prefills, stop after
   the first partial chunk.
4. Respect max_num_seqs limit.

Simplifications vs real vLLM: no preemption, no swap, no LoRA, no priority.
"""
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

        # Step 1: Reserve 1 token per DECODING request.
        for req in running:
            if req.state == RequestState.DECODING:
                result.decode_reqs.append(req)
                budget -= 1
        # Guard: budget can go negative if decode demand exceeds token limit.
        # In this case, no prefill is possible.

        # Step 2: Continue PREFILLING requests already in running set.
        if budget > 0:
            for req in running:
                if req.state == RequestState.PREFILLING:
                    chunk = min(req.prefill_tokens_remaining, max(budget, 0))
                    if chunk > 0:
                        result.prefill_reqs.append(req)
                        result.prefill_tokens[req.request_id] = chunk
                        budget -= chunk
                    break  # at most one continuing partial prefill

        # Step 3: Admit new requests from waiting queue.
        num_seqs = len(running)
        admit_idx = 0
        while (budget > 0
               and admit_idx < len(waiting)
               and num_seqs < self._config.max_num_seqs):
            req = waiting[admit_idx]
            chunk = min(req.prefill_tokens_remaining, budget)
            if chunk <= 0:
                break

            result.prefill_reqs.append(req)
            result.prefill_tokens[req.request_id] = chunk
            budget -= chunk
            num_seqs += 1
            admit_idx += 1

            # If this was a partial prefill, stop admitting more.
            if chunk < req.prefill_tokens_remaining:
                break

        return result
