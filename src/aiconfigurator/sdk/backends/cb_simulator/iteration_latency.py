"""Mixed-iteration latency calculator for the CB simulator.

Reuses BaseBackend.run_static() following the EXACT 3-pass decomposition
from vllm_backend.py _get_mix_step_latency (lines 149-230):

    Pass 1: Non-attention ops -- run_static(bs=1, isl=total_tokens, mode="static_ctx")
            Extract all ops EXCEPT "context_attention".
    Pass 2: Context attention -- run_static(bs=prefill_bs, isl=seq_len, mode="static_ctx")
            Extract ONLY "context_attention", scaled by chunk ratio.
    Pass 3: Generation attention -- run_static(bs=decode_bs, isl=kv_len, mode="static_gen")
            Extract ONLY "generation_attention" (NOT sum of all gen ops).
"""
from __future__ import annotations

import logging
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


class IterationLatencyCalculator:
    """Compute per-iteration latency for mixed prefill+decode iterations."""

    def __init__(
        self,
        backend: BaseBackend,
        model: BaseModel,
        database: PerfDatabase,
        prefix: int = 0,
    ) -> None:
        self._backend = backend
        self._model = model
        self._database = database
        self._prefix = prefix
        self._cache: dict[tuple, float] = {}

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
            return self._cache[cache_key]

        latency = self._compute_3pass(
            total_tokens, prefill_tokens, prefill_batch_size,
            prefill_seq_len, decode_batch_size, kv_bucket,
        )
        self._cache[cache_key] = latency
        return latency

    def _compute_3pass(
        self, total_tokens: int, prefill_tokens: int, prefill_bs: int,
        prefill_seq_len: int, decode_bs: int, decode_kv_len: int,
    ) -> float:
        """3-pass latency decomposition matching _get_mix_step_latency.

        See vllm_backend.py lines 149-230 for the original pattern.
        """
        # --- Pass 1: Non-attention ops (GEMM, MoE, etc.) ---
        # Matches vllm_backend.py:166-184
        prefix_adj = int(
            self._prefix * np.floor(prefill_tokens / max(prefill_seq_len, 1))
        ) if prefill_tokens > 0 else 0
        summary = self._backend.run_static(
            self._model, self._database,
            RuntimeConfig(
                batch_size=1, beam_width=1, isl=total_tokens, osl=1,
                prefix=prefix_adj,
            ),
            mode="static_ctx",
        )
        ctx_dict = summary.get_context_latency_dict()
        non_attn_ms = sum(
            lat for name, lat in ctx_dict.items()
            if name != "context_attention"
        )

        # --- Pass 2: Context attention (prefilling requests only) ---
        # Matches vllm_backend.py:188-201
        ctx_attn_ms = 0.0
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
            ctx_attn_ms = ctx_dict.get("context_attention", 0.0) * scale

        # --- Pass 3: Generation attention ONLY (decoding requests) ---
        # Matches vllm_backend.py:206-217
        # CRITICAL: Extract ONLY "generation_attention", NOT sum of all gen ops.
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

        total_ms = non_attn_ms + ctx_attn_ms + gen_attn_ms
        logger.debug(
            "iter_lat=%.3fms (non_attn=%.3f ctx_attn=%.3f gen_attn=%.3f) "
            "tokens=%d (pfx=%d dec=%d)",
            total_ms, non_attn_ms, ctx_attn_ms, gen_attn_ms,
            total_tokens, prefill_tokens, decode_bs,
        )
        return total_ms
