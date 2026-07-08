"""DP admission helpers for CB simulator.

The vLLM V1 data-parallel frontend routes new requests to the engine with the
lowest stale load score:

    score = waiting * 4 + running

The frontend refreshes engine counts periodically and then applies a local
optimistic waiting increment for every routed request.  This module keeps that
logic isolated from the per-replica scheduler so the simulator can model a
global closed-loop client without baking DP behavior into ``CBScheduler``.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DPReplicaCounts:
    """Request counts visible to the DP load balancer."""

    waiting: int
    running: int

    @property
    def score(self) -> int:
        return self.waiting * 4 + self.running


class DPAdmissionRouter:
    """vLLM-style stale score router for DP replicas."""

    def __init__(
        self,
        num_replicas: int,
        *,
        stats_update_interval_ms: float = 100.0,
        client_count: int = 1,
    ) -> None:
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if stats_update_interval_ms <= 0:
            raise ValueError("stats_update_interval_ms must be positive")
        if client_count <= 0:
            raise ValueError("client_count must be positive")
        self._num_replicas = num_replicas
        self._stats_update_interval_ms = stats_update_interval_ms
        self._client_count = client_count
        self._cached_counts = [DPReplicaCounts(0, 0) for _ in range(num_replicas)]
        self._last_update_ms: float | None = None

    def route(
        self,
        *,
        now_ms: float,
        actual_counts: list[DPReplicaCounts],
    ) -> int:
        """Return the replica index for a new request."""
        if len(actual_counts) != self._num_replicas:
            raise ValueError("actual_counts length must match num_replicas")
        if self._should_refresh(now_ms):
            self._cached_counts = list(actual_counts)
            self._last_update_ms = now_ms

        replica = min(
            range(self._num_replicas),
            key=lambda idx: (self._cached_counts[idx].score, idx),
        )
        count = self._cached_counts[replica]
        self._cached_counts[replica] = DPReplicaCounts(
            waiting=count.waiting + self._client_count,
            running=count.running,
        )
        return replica

    def _should_refresh(self, now_ms: float) -> bool:
        if self._last_update_ms is None:
            return True
        return now_ms - self._last_update_ms >= self._stats_update_interval_ms
