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
_SERVING_STATE_PERFDB_MODEL = "kimi-k2.5"
_SERVING_STATE_TOPOLOGY = "tp4dp2ep8"
_SERVING_STATE_HARDWARE = "h200_sxm"
_SERVING_STATE_VERSION = "0.19.0"
_SERVING_STATE_QUANT_RUNTIME = "CompressedTensorsWNA16MarlinMoEMethod"
_SERVING_STATE_HIDDEN_SIZE = 7168
_SERVING_STATE_TOPK = 8
_SERVING_STATE_MOE_EP_SIZE = 8
_SERVING_STATE_CATEGORIES = ("ep_a2a", "moe_gemm_or_aux", "other_cuda", "collective_other")


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


@dataclass(frozen=True)
class ServingStateQueryAudit:
    """One serving-state PerfDB query attempt."""

    phase: str
    category: str
    bucket_tokens: int
    decode_batch: int
    hit: bool
    miss_reason: str
    bucket_min: int | None = None
    bucket_max: int | None = None
    decode_batch_min: int | None = None
    decode_batch_max: int | None = None

    def as_dict(self) -> dict[str, int | str | bool | None]:
        return {
            "phase": self.phase,
            "category": self.category,
            "bucket_tokens": self.bucket_tokens,
            "decode_batch": self.decode_batch,
            "hit": self.hit,
            "miss_reason": self.miss_reason,
            "bucket_min": self.bucket_min,
            "bucket_max": self.bucket_max,
            "decode_batch_min": self.decode_batch_min,
            "decode_batch_max": self.decode_batch_max,
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
        self._serving_state_query_audit: list[ServingStateQueryAudit] = []

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

    def _serving_state_scope_enabled(self) -> bool:
        config = getattr(self._model, "config", None)
        return (
            getattr(self._database, "backend", None) == "vllm"
            and getattr(self._database, "system", None) == _SERVING_STATE_HARDWARE
            and getattr(self._database, "version", None) == _SERVING_STATE_VERSION
            and getattr(config, "tp_size", None) == 4
            and getattr(config, "attention_dp_size", None) == 2
            and getattr(config, "moe_tp_size", None) == 1
            and getattr(config, "moe_ep_size", None) == 8
            and hasattr(self._database, "query_vllm_serving_state")
        )

    def _serving_state_category(self, op_name: str) -> str | None:
        name = op_name.lower()
        if "attention" in name:
            return None
        if "dispatch" in name or "alltoall" in name or "all2all" in name:
            return "ep_a2a"
        if "moe" in name:
            return "moe_gemm_or_aux"
        if "allreduce" in name or "all_reduce" in name or "_ar_" in name or name.endswith("_ar"):
            return "collective_other"
        if "other_cuda" in name or "memcpy" in name or "memset" in name:
            return "other_cuda"
        return None

    def _query_serving_state_category(
        self,
        *,
        phase: str,
        category: str,
        bucket_tokens: int,
        decode_batch: int,
    ) -> float | None:
        if not self._serving_state_scope_enabled():
            self._record_serving_state_audit(
                phase=phase,
                category=category,
                bucket_tokens=bucket_tokens,
                decode_batch=decode_batch,
                hit=False,
                miss_reason="scope_disabled",
                row_kind="non_attn_total",
            )
            return None
        result = self._database.query_vllm_serving_state(
            model=_SERVING_STATE_PERFDB_MODEL,
            topology=_SERVING_STATE_TOPOLOGY,
            phase=phase,
            row_kind="category",
            category=category,
            bucket_tokens=bucket_tokens,
            decode_batch=decode_batch,
            hidden_size=_SERVING_STATE_HIDDEN_SIZE,
            topk=_SERVING_STATE_TOPK,
            moe_ep_size=_SERVING_STATE_MOE_EP_SIZE,
            quant_runtime=_SERVING_STATE_QUANT_RUNTIME,
        )
        self._record_serving_state_audit(
            phase=phase,
            category=category,
            bucket_tokens=bucket_tokens,
            decode_batch=decode_batch,
            hit=result is not None,
            miss_reason="hit" if result is not None else self._serving_state_miss_reason(
                phase=phase,
                category=category,
                bucket_tokens=bucket_tokens,
                decode_batch=decode_batch,
            ),
        )
        return None if result is None else float(result)

    def _query_serving_state_non_attn_total(
        self,
        *,
        phase: str,
        bucket_tokens: int,
        decode_batch: int,
    ) -> float | None:
        category = "non_attn_total"
        if not self._serving_state_scope_enabled():
            self._record_serving_state_audit(
                phase=phase,
                category=category,
                bucket_tokens=bucket_tokens,
                decode_batch=decode_batch,
                hit=False,
                miss_reason="scope_disabled",
            )
            return None
        result = self._database.query_vllm_serving_state(
            model=_SERVING_STATE_PERFDB_MODEL,
            topology=_SERVING_STATE_TOPOLOGY,
            phase=phase,
            row_kind="non_attn_total",
            category=category,
            bucket_tokens=bucket_tokens,
            decode_batch=decode_batch,
            hidden_size=_SERVING_STATE_HIDDEN_SIZE,
            topk=_SERVING_STATE_TOPK,
            moe_ep_size=_SERVING_STATE_MOE_EP_SIZE,
            quant_runtime=_SERVING_STATE_QUANT_RUNTIME,
        )
        self._record_serving_state_audit(
            phase=phase,
            category=category,
            bucket_tokens=bucket_tokens,
            decode_batch=decode_batch,
            hit=result is not None,
            miss_reason="hit" if result is not None else self._serving_state_miss_reason(
                phase=phase,
                category=category,
                bucket_tokens=bucket_tokens,
                decode_batch=decode_batch,
                row_kind="non_attn_total",
            ),
            row_kind="non_attn_total",
        )
        return None if result is None else float(result)

    def _serving_state_table_for(
        self,
        *,
        phase: str,
        category: str,
        row_kind: str = "category",
    ) -> dict | None:
        data = getattr(self._database, "_vllm_serving_state_data", None)
        if data is None:
            return None
        key = (
            _SERVING_STATE_PERFDB_MODEL,
            _SERVING_STATE_TOPOLOGY,
            phase,
            row_kind,
            category,
            _SERVING_STATE_HIDDEN_SIZE,
            _SERVING_STATE_TOPK,
            _SERVING_STATE_MOE_EP_SIZE,
            _SERVING_STATE_QUANT_RUNTIME,
        )
        table = data.get(key)
        if table is None and row_kind == "category":
            # Pre-Phase440 tests and ad-hoc fakes used the legacy key without
            # row_kind. Keep read compatibility; real PerfDB rows use row_kind.
            legacy_key = (
                _SERVING_STATE_PERFDB_MODEL,
                _SERVING_STATE_TOPOLOGY,
                phase,
                category,
                _SERVING_STATE_HIDDEN_SIZE,
                _SERVING_STATE_TOPK,
                _SERVING_STATE_MOE_EP_SIZE,
                _SERVING_STATE_QUANT_RUNTIME,
            )
            table = data.get(legacy_key)
        return table if table else None

    def _serving_state_miss_reason(
        self,
        *,
        phase: str,
        category: str,
        bucket_tokens: int,
        decode_batch: int,
        row_kind: str = "category",
    ) -> str:
        table = self._serving_state_table_for(phase=phase, category=category, row_kind=row_kind)
        if table is None:
            return "table_missing"

        token_values = sorted(int(token) for token in table)
        if not token_values:
            return "table_empty"
        if bucket_tokens < token_values[0]:
            return "bucket_below_range"
        if bucket_tokens > token_values[-1]:
            return "bucket_above_range"

        candidate_tokens = [token for token in token_values if token == bucket_tokens]
        if not candidate_tokens:
            left = [token for token in token_values if token < bucket_tokens]
            right = [token for token in token_values if token > bucket_tokens]
            if not left or not right:
                return "bucket_bracket_missing"
            candidate_tokens = [max(left), min(right)]

        batch_values: list[int] = []
        for token in candidate_tokens:
            batch_values.extend(int(batch) for batch in table[token])
        if not batch_values:
            return "decode_batch_table_empty"
        if decode_batch < min(batch_values):
            return "decode_batch_below_range"
        if decode_batch > max(batch_values):
            return "decode_batch_above_range"
        return "interpolation_gap"

    def _record_serving_state_audit(
        self,
        *,
        phase: str,
        category: str,
        bucket_tokens: int,
        decode_batch: int,
        hit: bool,
        miss_reason: str,
        row_kind: str = "category",
    ) -> None:
        table = self._serving_state_table_for(phase=phase, category=category, row_kind=row_kind)
        bucket_min = bucket_max = decode_batch_min = decode_batch_max = None
        if table:
            token_values = sorted(int(token) for token in table)
            if token_values:
                bucket_min = token_values[0]
                bucket_max = token_values[-1]
                batch_values: list[int] = []
                for batch_table in table.values():
                    batch_values.extend(int(batch) for batch in batch_table)
                if batch_values:
                    decode_batch_min = min(batch_values)
                    decode_batch_max = max(batch_values)
        self._serving_state_query_audit.append(
            ServingStateQueryAudit(
                phase=phase,
                category=category,
                bucket_tokens=bucket_tokens,
                decode_batch=decode_batch,
                hit=hit,
                miss_reason=miss_reason,
                bucket_min=bucket_min,
                bucket_max=bucket_max,
                decode_batch_min=decode_batch_min,
                decode_batch_max=decode_batch_max,
            )
        )

    def _serving_state_adjusted_non_attention(
        self,
        latency_dict: dict[str, float],
        *,
        phase: str,
        bucket_tokens: int,
        decode_batch: int,
    ) -> float | None:
        if not self._serving_state_scope_enabled():
            return None

        category_sums = {category: 0.0 for category in _SERVING_STATE_CATEGORIES}
        passthrough_ms = 0.0
        for op_name, latency in latency_dict.items():
            if "attention" in op_name.lower():
                continue
            category = self._serving_state_category(op_name)
            if category is None:
                passthrough_ms += float(latency)
                continue
            if category in category_sums:
                category_sums[category] += float(latency)
            else:
                passthrough_ms += float(latency)

        total_ms = passthrough_ms
        used_any = False
        for category in _SERVING_STATE_CATEGORIES:
            serving_ms = self._query_serving_state_category(
                phase=phase,
                category=category,
                bucket_tokens=bucket_tokens,
                decode_batch=decode_batch,
            )
            if serving_ms is None:
                total_ms += category_sums[category]
            else:
                total_ms += serving_ms
                used_any = True
        if used_any:
            return total_ms

        non_attn_total_ms = self._query_serving_state_non_attn_total(
            phase=phase,
            bucket_tokens=bucket_tokens,
            decode_batch=decode_batch,
        )
        if non_attn_total_ms is not None:
            return non_attn_total_ms
        return None

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

    def get_serving_state_query_audit(self) -> list[ServingStateQueryAudit]:
        """Return serving-state query hit/miss records from this calculator."""
        return list(self._serving_state_query_audit)

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
                op_query_overrides={"context_prefill_tokens": prefill_tokens} if is_mixed else None,
            )
            ctx_dict = summary.get_context_latency_dict()
            context_compute_ms, context_dispatch_ms = self._split_context_non_attention(ctx_dict)
            serving_state_ms = self._serving_state_adjusted_non_attention(
                ctx_dict,
                phase="mixed_prefill" if is_mixed else "prefill",
                bucket_tokens=non_attn_tokens,
                decode_batch=decode_bs,
            )
            context_non_attn_ms = (
                serving_state_ms if serving_state_ms is not None else context_compute_ms + context_dispatch_ms
            )
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
                serving_state_ms = self._serving_state_adjusted_non_attention(
                    gen_dict,
                    phase="decode",
                    bucket_tokens=decode_bs,
                    decode_batch=decode_bs,
                )
                generation_non_attn_ms = (
                    serving_state_ms if serving_state_ms is not None else generation_compute_ms + generation_dispatch_ms
                )
        else:
            generation_compute_ms = 0.0
            generation_dispatch_ms = 0.0

        # Non-attention and attention phases share fixed GPU setup cost and can
        # overlap via overlap_factor.
        overhead_ms = self._per_iteration_overhead_ms if decode_bs > 0 else 0.0
        if is_mixed:
            # Mixed iteration: the merged non-attention pass, context attention,
            # and decode attention execute serially within the fused forward.
            total_ms = context_non_attn_ms + context_attn_ms + gen_attn_ms
        elif prefill_tokens > 0:
            # Pure prefill executes the same forward stream as mixed/pure decode:
            # token-parallel non-attention work and attention both consume wall time.
            total_ms = context_non_attn_ms + context_attn_ms
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
