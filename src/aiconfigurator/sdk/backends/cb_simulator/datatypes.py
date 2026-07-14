"""Data types for the vLLM continuous batching simulator."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto

logger = logging.getLogger(__name__)

VLLM_NULL_BLOCKS_PER_POOL = 1


class RequestState(Enum):
    """Explicit request lifecycle state."""
    WAITING = auto()     # in queue, not yet admitted
    PREFILLING = auto()  # admitted, prefill in progress
    DECODING = auto()    # prefill done, generating tokens
    PREEMPTED = auto()   # evicted from KV cache, must recompute
    DONE = auto()        # generation complete


@dataclass(frozen=True)
class CBSimConfig:
    """Configuration for the CB simulator."""
    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 256
    num_requests: int = 200
    warmup_requests: int = 50
    long_prefill_token_threshold: int = 0
    scheduler_reserve_full_isl: bool = True
    num_gpu_blocks: int = 0  # Physical BlockPool size; 0 disables capacity.
    block_size: int = 16
    overlap_factor: float = 1.0
    per_iteration_overhead_ms: float = 0.0
    engine_loop_enabled: bool = False

    @property
    def num_allocatable_gpu_blocks(self) -> int:
        """Blocks available to the scheduler after vLLM's null block reserve."""
        if self.num_gpu_blocks <= 0:
            return 0
        return self.num_gpu_blocks - VLLM_NULL_BLOCKS_PER_POOL


@dataclass
class Request:
    """A single inference request tracked through the simulation."""
    request_id: int
    isl: int
    osl: int
    arrival_time_ms: float

    # Mutable state
    state: RequestState = RequestState.WAITING
    prefill_tokens_remaining: int = -1
    sampled_output_tokens: int = 0
    computed_output_tokens: int = 0
    output_placeholders: int = 0
    prefill_start_ms: float = -1.0
    first_token_ms: float = -1.0
    finish_ms: float = -1.0
    num_preemptions: int = 0

    def __post_init__(self) -> None:
        if self.prefill_tokens_remaining < 0:
            self.prefill_tokens_remaining = self.isl

    @property
    def kv_cache_len(self) -> int:
        """Current KV cache length = prefilled tokens + generated tokens."""
        return (self.isl - self.prefill_tokens_remaining) + self.sampled_output_tokens

    @property
    def generated_tokens(self) -> int:
        """Legacy alias for callers that do not model asynchronous execution."""
        return self.sampled_output_tokens

    @generated_tokens.setter
    def generated_tokens(self, value: int) -> None:
        self.sampled_output_tokens = value
        self.computed_output_tokens = value
        self.output_placeholders = 0


@dataclass
class ScheduleResult:
    """Output of one scheduler iteration."""
    prefill_reqs: list[Request] = field(default_factory=list)
    prefill_tokens: dict[int, int] = field(default_factory=dict)
    decode_reqs: list[Request] = field(default_factory=list)

    @property
    def total_prefill_tokens(self) -> int:
        return sum(self.prefill_tokens.values())

    @property
    def total_tokens(self) -> int:
        return self.total_prefill_tokens + len(self.decode_reqs)

    @property
    def is_empty(self) -> bool:
        return not self.prefill_reqs and not self.decode_reqs


@dataclass
class CBSimResult:
    """Simulation output metrics."""
    mean_ttft_ms: float
    p50_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    throughput_tok_s: float
    throughput_tok_s_gpu: float
    num_gpus: int
    total_iterations: int
    steady_state_requests: int
    steady_state_iterations: int = 0
    steady_state_time_ms: float = 0.0
    # Iteration-averaged scheduling counters (for ColumnsAgg fields)
    avg_prefill_reqs_per_iter: float = 0.0
    avg_decode_reqs_per_iter: float = 0.0
    avg_tokens_per_iter: float = 0.0
    peak_prefill_reqs_per_iter: int = 0
    peak_decode_reqs_per_iter: int = 0
    peak_tokens_per_iter: int = 0
