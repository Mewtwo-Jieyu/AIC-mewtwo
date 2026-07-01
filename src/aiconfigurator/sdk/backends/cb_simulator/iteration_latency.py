"""Mixed-iteration latency calculator for the CB simulator.

Reuses BaseBackend.run_static() following the EXACT 3-pass decomposition
from vllm_backend.py _get_mix_step_latency (lines 149-230):

    Pass 1: Non-attention ops -- run_static(bs=1, isl=total_tokens, mode="static_ctx")
            Extract all ops EXCEPT "context_attention".
    Pass 2: Context attention -- run_static(bs=prefill_bs, isl=seq_len, mode="static_ctx")
            Extract ONLY "context_attention", scaled by chunk ratio.
    Pass 3: Generation phase -- run_static(bs=decode_bs, isl=kv_len, mode="static_gen")
            Split into generation_attention and generation_non_attention.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from aiconfigurator.sdk.backends.base_backend import BaseBackend
    from aiconfigurator.sdk.models import BaseModel
    from aiconfigurator.sdk.perf_database import PerfDatabase

from aiconfigurator.sdk.config import RuntimeConfig

logger = logging.getLogger(__name__)

_KV_LEN_BUCKET = 512


def _bucket(value: int, size: int) -> int:
    """Round up to nearest bucket size, minimum 1."""
    return max(1, ((value + size - 1) // size) * size)


@dataclass(frozen=True)
class IterationLatencyBreakdown:
    """Detailed latency decomposition for one simulated iteration."""
    total_ms: float
    context_non_attention_ms: float
    context_attention_ms: float
    generation_non_attention_ms: float
    generation_attention_ms: float
    iteration_overhead_ms: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "total_ms": self.total_ms,
            "context_non_attention_ms": self.context_non_attention_ms,
            "context_attention_ms": self.context_attention_ms,
            "generation_non_attention_ms": self.generation_non_attention_ms,
            "generation_attention_ms": self.generation_attention_ms,
            "iteration_overhead_ms": self.iteration_overhead_ms,
        }


class IterationLatencyCalculator:
    """Compute per-iteration latency for mixed prefill+decode iterations."""

    def __init__(
        self,
        backend: BaseBackend,
        model: BaseModel,
        database: PerfDatabase,
        prefix: int = 0,
        overlap_factor: float = 1.0,
        per_iteration_overhead_ms: float = 0.0,
    ) -> None:
        if not 0.0 <= overlap_factor <= 1.0:
            raise ValueError("overlap_factor must be between 0.0 and 1.0")
        if per_iteration_overhead_ms < 0:
            raise ValueError("per_iteration_overhead_ms must be non-negative")
        self._backend = backend
        self._model = model
        self._database = database
        self._prefix = prefix
        self._overlap_factor = overlap_factor
        self._per_iteration_overhead_ms = per_iteration_overhead_ms
        self._cache: dict[tuple, IterationLatencyBreakdown] = {}
        self._last_breakdown: IterationLatencyBreakdown | None = None

    def _combine_with_overlap(self, a_ms: float, b_ms: float) -> float:
        return max(a_ms, b_ms) + self._overlap_factor * min(a_ms, b_ms)

    def _split_context_non_attention(self, ctx_dict: dict[str, float]) -> tuple[float, float]:
        compute_ms = 0.0
        dispatch_ms = 0.0
        for name, lat in ctx_dict.items():
            if name == "context_attention":
                continue
            if "dispatch" in name:
                dispatch_ms += lat
            else:
                compute_ms += lat
        return compute_ms, dispatch_ms

    def _split_generation_non_attention(self, gen_dict: dict[str, float]) -> tuple[float, float]:
        # No /tp scaling here: generation_moe(+dispatch) latencies from the perf
        # database are already PER-RANK (TP is encoded in the moe_tp_size/topology
        # lookup key; the collector benchmarks a single sharded GPU). Dividing by
        # tp_size again is a double-count (phase397e). Charge the raw per-rank
        # latency for every non-attention op.
        compute_ms = 0.0
        dispatch_ms = 0.0
        for name, lat in gen_dict.items():
            if name == "generation_attention":
                continue
            if "dispatch" in name:
                dispatch_ms += lat
            else:
                compute_ms += lat
        return compute_ms, dispatch_ms

    def compute(
        self,
        prefill_tokens: int,
        prefill_batch_size: int,
        prefill_seq_len: int,
        decode_batch_size: int,
        decode_avg_kv_len: int,
    ) -> float:
        """Compute latency (ms) for one mixed iteration.

        Args:
            prefill_tokens: Total prefill tokens in this iteration.
            prefill_batch_size: Number of prefilling requests.
            prefill_seq_len: Original ISL (for context attention shape).
            decode_batch_size: Number of decoding requests.
            decode_avg_kv_len: Average KV cache length of decode requests.

        Returns:
            Iteration latency in milliseconds.
        """
        total_tokens = prefill_tokens + decode_batch_size
        if total_tokens <= 0:
            return 0.0

        kv_bucket = _bucket(decode_avg_kv_len, _KV_LEN_BUCKET) if decode_avg_kv_len > 0 else 0

        cache_key = (total_tokens, prefill_batch_size, prefill_seq_len,
                     decode_batch_size, kv_bucket)
        if cache_key in self._cache:
            self._last_breakdown = self._cache[cache_key]
            return self._cache[cache_key].total_ms

        breakdown = self._compute_3pass(
            total_tokens, prefill_tokens, prefill_batch_size,
            prefill_seq_len, decode_batch_size, kv_bucket,
        )
        self._cache[cache_key] = breakdown
        self._last_breakdown = breakdown
        return breakdown.total_ms

    def get_last_breakdown(self) -> IterationLatencyBreakdown | None:
        """Return the most recent iteration breakdown."""
        return self._last_breakdown

    def _compute_3pass(
        self, total_tokens: int, prefill_tokens: int, prefill_bs: int,
        prefill_seq_len: int, decode_bs: int, decode_kv_len: int,
    ) -> IterationLatencyBreakdown:
        """3-pass latency decomposition matching _get_mix_step_latency.

        See vllm_backend.py lines 149-230 for the original pattern.
        """
        is_mixed = prefill_tokens > 0 and decode_bs > 0

        # --- Pass 1: Non-attention ops (GEMM, MoE, EP8 comm) ---
        # These ops are token-parallel: a decode token costs the same as a
        # prefill token. In a MIXED iteration vLLM runs them ONCE over the
        # merged batch (prefill chunk + decode tokens), matching the real
        # fused-MoE forward (phase124 tokens_actual == max_num_batched_tokens).
        # We therefore charge them at total_tokens for mixed iterations and let
        # Pass 3 contribute only the decode ATTENTION term. For pure prefill
        # total_tokens == prefill_tokens, so this is unchanged there; pure
        # decode skips this pass entirely.
        context_non_attn_ms = 0.0
        if prefill_tokens > 0 and prefill_bs > 0:
            non_attn_tokens = total_tokens if is_mixed else prefill_tokens
            prefix_adj = int(
                self._prefix * np.floor(prefill_tokens / max(prefill_seq_len, 1))
            )
            summary = self._backend.run_static(
                self._model, self._database,
                RuntimeConfig(
                    batch_size=1,
                    beam_width=1,
                    isl=non_attn_tokens,
                    osl=1,
                    prefix=prefix_adj,
                ),
                mode="static_ctx",
            )
            ctx_dict = summary.get_context_latency_dict()
            context_compute_ms, context_dispatch_ms = self._split_context_non_attention(ctx_dict)
            context_non_attn_ms = context_compute_ms + context_dispatch_ms
        else:
            context_compute_ms = 0.0
            context_dispatch_ms = 0.0

        # --- Pass 2: Context attention (prefilling requests only) ---
        # Matches vllm_backend.py:188-201
        context_attn_ms = 0.0
        if prefill_tokens > 0 and prefill_bs > 0:
            ctx_bs = int(np.ceil(prefill_tokens / max(prefill_seq_len, 1)))
            summary = self._backend.run_static(
                self._model, self._database,
                RuntimeConfig(
                    batch_size=max(ctx_bs, 1), beam_width=1,
                    isl=prefill_seq_len, osl=1, prefix=self._prefix,
                ),
                mode="static_ctx",
            )
            ctx_dict = summary.get_context_latency_dict()
            # Scale: only processing prefill_tokens out of full (ctx_bs * seq_len)
            scale = prefill_tokens / max(ctx_bs * prefill_seq_len, 1)
            context_attn_ms = ctx_dict.get("context_attention", 0.0) * scale

        # --- Pass 3: Generation phase (decoding requests) ---
        # Reuse static_gen, but split attention from the rest. This is the
        # main place where mixed iterations differ from the old batch-sync
        # approximation: decode-side non-attention ops still exist in mixed
        # iterations and must not be dropped.
        generation_non_attn_ms = 0.0
        gen_attn_ms = 0.0
        if decode_bs > 0 and decode_kv_len > 0:
            summary = self._backend.run_static(
                self._model, self._database,
                RuntimeConfig(
                    batch_size=decode_bs, beam_width=1,
                    isl=decode_kv_len, osl=2,
                ),
                mode="static_gen",
            )
            gen_dict = summary.get_generation_latency_dict()
            gen_attn_ms = gen_dict.get("generation_attention", 0.0)
            if is_mixed:
                # Decode non-attention is token-parallel and already charged
                # once in the merged Pass 1; do not double-count it here.
                generation_compute_ms = 0.0
                generation_dispatch_ms = 0.0
            else:
                generation_compute_ms, generation_dispatch_ms = self._split_generation_non_attention(gen_dict)
                generation_non_attn_ms = generation_compute_ms + generation_dispatch_ms
        else:
            generation_compute_ms = 0.0
            generation_dispatch_ms = 0.0

        # Non-attention and attention phases share fixed GPU setup cost and can
        # overlap via overlap_factor.
        overhead_ms = self._per_iteration_overhead_ms if decode_bs > 0 else 0.0
        if is_mixed:
            # One merged non-attention forward (Pass 1, total_tokens) overlaps
            # with the prefill + decode attention kernels of the same forward.
            total_ms = self._combine_with_overlap(
                context_non_attn_ms,
                context_attn_ms + gen_attn_ms,
            )
        elif prefill_tokens > 0:
            # Pure prefill iteration: no decode lane to overlap with.
            total_ms = self._combine_with_overlap(
                context_non_attn_ms,
                context_attn_ms,
            )
        else:
            # Pure decode iteration: within each transformer layer the attention
            # and the MoE/FFN non-attention ops run SERIALLY, so they add rather
            # than overlap. overlap_factor=0 (max) wrongly dropped the smaller of
            # the two terms; combined with the /tp double-count (removed above)
            # this under-counted every decode iteration ~3x (phase397e). Charge
            # the physically-serial sum here; overlap_factor still governs the
            # mixed and pure-prefill branches.
            total_ms = generation_non_attn_ms + gen_attn_ms
        total_ms += overhead_ms
        breakdown = IterationLatencyBreakdown(
            total_ms=total_ms,
            context_non_attention_ms=context_non_attn_ms,
            context_attention_ms=context_attn_ms,
            generation_non_attention_ms=generation_non_attn_ms,
            generation_attention_ms=gen_attn_ms,
            iteration_overhead_ms=overhead_ms,
        )
        logger.debug(
            "iter_lat=%.3fms (ctx_non_attn=%.3f ctx_attn=%.3f gen_non_attn=%.3f gen_attn=%.3f) "
            "tokens=%d (pfx=%d dec=%d)",
            total_ms,
            context_non_attn_ms,
            context_attn_ms,
            generation_non_attn_ms,
            gen_attn_ms,
            total_tokens, prefill_tokens, decode_bs,
        )
        return breakdown
