"""vLLM forward descriptor schema for experimental cb_sim diagnostics."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


def make_topology_key(tp: int, dp: int, moe_tp: int, moe_ep: int) -> str:
    return f"tp{tp}dp{dp}moetp{moe_tp}ep{moe_ep}"


@dataclass(frozen=True)
class VLLMForwardDescriptor:
    """Runtime forward shape descriptor.

    This is an input-shape record only. It must not change latency by itself.
    """

    source: str
    scenario: str
    iteration: int
    phase: str
    scheduled_context_tokens: int
    scheduled_decode_tokens: int
    forward_token_count: int
    batch_desc_num_tokens: int
    cudagraph_runtime_mode: str
    forward_regime: str
    topology_key: str
    tp: int
    dp: int
    moe_tp: int
    moe_ep: int
    max_num_batched_tokens: int
    max_num_seqs: int
    decode_avg_kv_len: int = 0


@dataclass(frozen=True)
class ForwardDescriptorCompareRow:
    scenario: str
    phase: str
    ordinal_in_phase: int
    topology_key: str
    cb_iter_index: int
    cb_context_tokens: int
    cb_decode_tokens: int
    cb_forward_token_count: int
    cb_forward_regime: str
    vllm_iter_index: int
    vllm_context_tokens: int
    vllm_decode_tokens: int
    vllm_forward_token_count: int
    vllm_batch_desc_num_tokens: int
    vllm_cudagraph_runtime_mode: str
    vllm_forward_regime: str
    scheduled_total_delta: int
    forward_token_delta: int
    same_forward_token_count: int
    same_forward_regime: int


@dataclass(frozen=True)
class VLLMRuntimeShapeKey:
    """Mechanism-shape key for experimental vLLM diagnostics.

    This is not a latency record. It must not contain residual or throughput data.
    """

    source: str
    scenario: str
    iteration: int
    phase: str
    topology_key: str
    tp: int
    dp: int
    moe_tp: int
    moe_ep: int
    cudagraph_runtime_mode: str
    forward_token_count: int
    attention_actual_tokens: int
    attention_max_query_len: int
    slot_mapping_tokens: int
    block_table_shape: str
    forward_context_tokens: int


@dataclass(frozen=True)
class RuntimeShapeKeyCompareRow:
    scenario: str
    phase: str
    ordinal_in_phase: int
    topology_key: str
    cb_iter_index: int
    cb_cudagraph_runtime_mode: str
    cb_forward_token_count: int
    cb_attention_actual_tokens: int
    cb_attention_max_query_len: int
    cb_slot_mapping_tokens: int
    cb_block_table_shape: str
    cb_forward_context_tokens: int
    vllm_iter_index: int
    vllm_cudagraph_runtime_mode: str
    vllm_forward_token_count: int
    vllm_attention_actual_tokens: int
    vllm_attention_max_query_len: int
    vllm_slot_mapping_tokens: int
    vllm_block_table_shape: str
    vllm_forward_context_tokens: int
    same_cudagraph_runtime_mode: int
    same_forward_token_count: int
    same_attention_actual_tokens: int
    same_attention_max_query_len: int
    same_slot_mapping_tokens: int
    same_block_table_shape: int
    same_forward_context_tokens: int


@dataclass(frozen=True)
class AttentionRuntimeShapeKey:
    phase: str
    topology_key: str
    cudagraph_runtime_mode: str
    attention_actual_tokens: int
    attention_max_query_len: int


@dataclass(frozen=True)
class KVRuntimeShapeKey:
    phase: str
    topology_key: str
    slot_mapping_tokens: int
    block_table_shape: str


@dataclass(frozen=True)
class ForwardWrapperShapeKey:
    phase: str
    topology_key: str
    cudagraph_runtime_mode: str
    forward_context_tokens: int
    forward_token_count: int


@dataclass(frozen=True)
class RuntimeShapeSubkeyDistributionRow:
    key_type: str
    scenario: str
    phase: str
    topology_key: str
    rank_rows: int
    iterations: int
    ranks: str
    cudagraph_runtime_mode: str
    attention_actual_tokens: str
    attention_max_query_len: str
    slot_mapping_tokens: str
    block_table_shape: str
    forward_context_tokens: str
    forward_token_count: str


@dataclass(frozen=True)
class VLLMCompiledBodyRuntimeKey:
    """Compiled-body mechanism key for experimental vLLM diagnostics.

    This is not a latency record. It must not contain profiler, trace count,
    sync wait, or residual values.
    """

    source: str
    scenario: str
    phase: str
    topology_key: str
    tp: int
    dp: int
    ep: int
    world_size: int
    forward_regime: str
    tokens_padded: int
    tokens_actual: int
    cudagraph_runtime_mode: str
    compiled_body: bool
    tp_comm_candidate: bool
    ep_or_global_comm_candidate: bool
    unknown_comm_present: bool
    moe_module: str
    moe_kernel: str
    moe_hidden: int
    moe_intermediate: int
    moe_experts: int
    moe_topk: int
    moe_dtype: str
    tuning_config_loaded: bool
    fallback: bool
    valid_for_default: bool
    perf_database: bool
    diagnostic_only: bool


def make_forward_regime(cudagraph_runtime_mode: str, forward_token_count: int) -> str:
    if not cudagraph_runtime_mode:
        raise ValueError("cudagraph_runtime_mode must be non-empty")
    if forward_token_count < 0:
        raise ValueError("forward_token_count must be non-negative")
    return f"{cudagraph_runtime_mode}:{forward_token_count}"


def descriptor_from_scheduled(
    *,
    source: str,
    scenario: str,
    iteration: int,
    phase: str,
    scheduled_context_tokens: int,
    scheduled_decode_tokens: int,
    topology_key: str,
    tp: int,
    dp: int,
    moe_tp: int,
    moe_ep: int,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    decode_avg_kv_len: int = 0,
) -> VLLMForwardDescriptor:
    """Build AIC's current descriptor view from scheduled tokens only."""
    forward_token_count = scheduled_context_tokens + scheduled_decode_tokens
    mode = "AIC_UNSET"
    return VLLMForwardDescriptor(
        source=source,
        scenario=scenario,
        iteration=iteration,
        phase=phase,
        scheduled_context_tokens=scheduled_context_tokens,
        scheduled_decode_tokens=scheduled_decode_tokens,
        forward_token_count=forward_token_count,
        batch_desc_num_tokens=forward_token_count,
        cudagraph_runtime_mode=mode,
        forward_regime=make_forward_regime(mode, forward_token_count),
        topology_key=topology_key,
        tp=tp,
        dp=dp,
        moe_tp=moe_tp,
        moe_ep=moe_ep,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        decode_avg_kv_len=decode_avg_kv_len,
    )


def _required_int(raw: dict[str, str], field: str) -> int:
    value = raw.get(field, "")
    if value == "":
        raise ValueError(f"missing {field}")
    return int(float(value))


def _single_value(raw: dict[str, str], field: str) -> str:
    value = raw.get(field, "")
    if value == "":
        raise ValueError(f"missing {field}")
    values = [item.strip() for item in value.split(",") if item.strip()]
    if len(values) != 1:
        raise ValueError(f"{field} must contain exactly one value, got {value!r}")
    return values[0]


def _required_non_empty(raw: dict[str, str], field: str) -> str:
    value = raw.get(field, "")
    if value == "":
        raise ValueError(f"missing {field}")
    return value


def _required_false(raw: dict[str, str], field: str) -> None:
    value = _required_non_empty(raw, field).strip().lower()
    if value not in {"false", "0"}:
        raise ValueError(f"{field} must be false")


def _parse_shape_tokens(raw_shape: str, field: str) -> int:
    if raw_shape == "":
        raise ValueError(f"missing {field}")
    values = []
    for item in raw_shape.split("+"):
        if ":" not in item:
            raise ValueError(f"{field} shape item is invalid: {item!r}")
        shape = item.split(":", 1)[1]
        first_dim = shape.split("x", 1)[0]
        values.append(int(first_dim))
    distinct = set(values)
    if len(distinct) != 1:
        raise ValueError(f"{field} has non-uniform token shapes: {raw_shape!r}")
    return values[0]


def compiled_body_runtime_key_from_nccl_summary_rows(
    rows: list[dict[str, str]],
    *,
    source: str,
    scenario: str,
    phase: str,
    topology_key: str,
    tp: int,
    dp: int,
    ep: int,
    world_size: int,
    forward_regime: str,
    tokens_padded: int,
    tokens_actual: int,
    cudagraph_runtime_mode: str,
    moe_module: str,
    moe_kernel: str,
    moe_hidden: int,
    moe_intermediate: int,
    moe_experts: int,
    moe_topk: int,
    moe_dtype: str,
    tuning_config_loaded: bool,
    fallback: bool,
) -> VLLMCompiledBodyRuntimeKey:
    """Build a compiled-body key from Phase 39 NCCL summary rows.

    The summary rows only decide communication candidate booleans. They must
    not contribute timing, profiler, or residual values to the key.
    """
    if not rows:
        raise ValueError("no NCCL summary rows")

    allowed_candidates = {
        "tp_comm_candidate",
        "ep_or_global_comm_candidate",
        "compiled_comm_envelope_unknown",
    }
    candidates: set[str] = set()
    for row in rows:
        _required_false(row, "valid_for_default")
        _required_false(row, "perf_database")
        candidate = _required_non_empty(row, "comm_candidate")
        if candidate not in allowed_candidates:
            raise ValueError(f"unknown comm_candidate: {candidate}")
        candidates.add(candidate)

    return VLLMCompiledBodyRuntimeKey(
        source=source,
        scenario=scenario,
        phase=phase,
        topology_key=topology_key,
        tp=tp,
        dp=dp,
        ep=ep,
        world_size=world_size,
        forward_regime=forward_regime,
        tokens_padded=tokens_padded,
        tokens_actual=tokens_actual,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        compiled_body=True,
        tp_comm_candidate="tp_comm_candidate" in candidates,
        ep_or_global_comm_candidate="ep_or_global_comm_candidate" in candidates,
        unknown_comm_present="compiled_comm_envelope_unknown" in candidates,
        moe_module=moe_module,
        moe_kernel=moe_kernel,
        moe_hidden=moe_hidden,
        moe_intermediate=moe_intermediate,
        moe_experts=moe_experts,
        moe_topk=moe_topk,
        moe_dtype=moe_dtype,
        tuning_config_loaded=tuning_config_loaded,
        fallback=fallback,
        valid_for_default=False,
        perf_database=False,
        diagnostic_only=True,
    )


def compiled_body_runtime_key_from_nccl_summary_csv(
    path: Path,
    **kwargs: object,
) -> VLLMCompiledBodyRuntimeKey:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return compiled_body_runtime_key_from_nccl_summary_rows(rows, **kwargs)


def descriptor_from_runtime_shape_summary(
    raw: dict[str, str],
    *,
    source: str,
    scenario: str,
    topology_key: str,
    tp: int,
    dp: int,
    moe_tp: int,
    moe_ep: int,
    max_num_batched_tokens: int,
    max_num_seqs: int,
) -> VLLMForwardDescriptor:
    """Build descriptor from Phase 7 runtime-shape iteration summary CSV."""
    forward_min = _required_int(raw, "forward_token_count_min")
    forward_max = _required_int(raw, "forward_token_count_max")
    if forward_min != forward_max:
        raise ValueError(
            "forward_token_count differs across ranks for iteration "
            f"{raw.get('iteration', '')}: {forward_min} != {forward_max}"
        )
    batch_desc_min = _required_int(raw, "batch_desc_num_tokens_min")
    batch_desc_max = _required_int(raw, "batch_desc_num_tokens_max")
    if batch_desc_min != batch_desc_max:
        raise ValueError(
            "batch_desc_num_tokens differs across ranks for iteration "
            f"{raw.get('iteration', '')}: {batch_desc_min} != {batch_desc_max}"
        )
    mode = _single_value(raw, "cudagraph_runtime_modes")
    regime = _single_value(raw, "forward_regimes")
    expected_regime = make_forward_regime(mode, forward_max)
    if regime != expected_regime:
        raise ValueError(f"forward_regime mismatch: {regime!r} != {expected_regime!r}")
    return VLLMForwardDescriptor(
        source=source,
        scenario=scenario,
        iteration=_required_int(raw, "iteration"),
        phase=raw["phase"],
        scheduled_context_tokens=_required_int(raw, "scheduled_context_tokens"),
        scheduled_decode_tokens=_required_int(raw, "scheduled_decode_tokens"),
        forward_token_count=forward_max,
        batch_desc_num_tokens=batch_desc_max,
        cudagraph_runtime_mode=mode,
        forward_regime=regime,
        topology_key=topology_key,
        tp=tp,
        dp=dp,
        moe_tp=moe_tp,
        moe_ep=moe_ep,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        decode_avg_kv_len=0,
    )


def runtime_shape_key_from_scheduled(
    *,
    source: str,
    scenario: str,
    iteration: int,
    phase: str,
    scheduled_context_tokens: int,
    scheduled_decode_tokens: int,
    topology_key: str,
    tp: int,
    dp: int,
    moe_tp: int,
    moe_ep: int,
) -> VLLMRuntimeShapeKey:
    """Build AIC's scheduled-token view of the runtime shape key."""
    forward_token_count = scheduled_context_tokens + scheduled_decode_tokens
    if scheduled_context_tokens > 0:
        attention_max_query_len = scheduled_context_tokens
    elif scheduled_decode_tokens > 0:
        attention_max_query_len = 1
    else:
        attention_max_query_len = 0
    return VLLMRuntimeShapeKey(
        source=source,
        scenario=scenario,
        iteration=iteration,
        phase=phase,
        topology_key=topology_key,
        tp=tp,
        dp=dp,
        moe_tp=moe_tp,
        moe_ep=moe_ep,
        cudagraph_runtime_mode="AIC_UNSET",
        forward_token_count=forward_token_count,
        attention_actual_tokens=forward_token_count,
        attention_max_query_len=attention_max_query_len,
        slot_mapping_tokens=forward_token_count,
        block_table_shape="AIC_UNSET",
        forward_context_tokens=forward_token_count,
    )


def runtime_shape_key_from_rank_row(
    raw: dict[str, str],
    *,
    source: str,
    scenario: str,
    topology_key: str,
    tp: int,
    dp: int,
    moe_tp: int,
    moe_ep: int,
) -> VLLMRuntimeShapeKey:
    """Build a mechanism key from Phase 10 rank-level runtime shape rows."""
    return VLLMRuntimeShapeKey(
        source=source,
        scenario=scenario,
        iteration=_required_int(raw, "iteration"),
        phase=raw["phase"],
        topology_key=topology_key,
        tp=tp,
        dp=dp,
        moe_tp=moe_tp,
        moe_ep=moe_ep,
        cudagraph_runtime_mode=_single_value(raw, "cudagraph_runtime_mode"),
        forward_token_count=_required_int(raw, "forward_token_count"),
        attention_actual_tokens=_required_int(raw, "attention_num_tokens"),
        attention_max_query_len=_required_int(raw, "attention_max_query_len"),
        slot_mapping_tokens=_parse_shape_tokens(
            _single_value(raw, "slot_mapping_shape"),
            "slot_mapping_shape",
        ),
        block_table_shape=_single_value(raw, "block_table_shape"),
        forward_context_tokens=_required_int(raw, "forward_context_num_tokens"),
    )


def attention_subkey_from_runtime_shape_key(
    key: VLLMRuntimeShapeKey,
) -> AttentionRuntimeShapeKey:
    return AttentionRuntimeShapeKey(
        phase=key.phase,
        topology_key=key.topology_key,
        cudagraph_runtime_mode=key.cudagraph_runtime_mode,
        attention_actual_tokens=key.attention_actual_tokens,
        attention_max_query_len=key.attention_max_query_len,
    )


def kv_subkey_from_runtime_shape_key(key: VLLMRuntimeShapeKey) -> KVRuntimeShapeKey:
    return KVRuntimeShapeKey(
        phase=key.phase,
        topology_key=key.topology_key,
        slot_mapping_tokens=key.slot_mapping_tokens,
        block_table_shape=key.block_table_shape,
    )


def forward_wrapper_subkey_from_runtime_shape_key(
    key: VLLMRuntimeShapeKey,
) -> ForwardWrapperShapeKey:
    return ForwardWrapperShapeKey(
        phase=key.phase,
        topology_key=key.topology_key,
        cudagraph_runtime_mode=key.cudagraph_runtime_mode,
        forward_context_tokens=key.forward_context_tokens,
        forward_token_count=key.forward_token_count,
    )


def compare_forward_descriptors(
    cb_rows: list[VLLMForwardDescriptor],
    vllm_rows: list[VLLMForwardDescriptor],
) -> list[ForwardDescriptorCompareRow]:
    compare_rows: list[ForwardDescriptorCompareRow] = []
    for phase in ("prefill", "mixed", "pure_decode"):
        cb_phase = [row for row in cb_rows if row.phase == phase]
        vllm_phase = [row for row in vllm_rows if row.phase == phase]
        for ordinal, (cb_row, vllm_row) in enumerate(
            zip(cb_phase, vllm_phase, strict=False),
            start=1,
        ):
            cb_total = cb_row.scheduled_context_tokens + cb_row.scheduled_decode_tokens
            vllm_total = (
                vllm_row.scheduled_context_tokens + vllm_row.scheduled_decode_tokens
            )
            compare_rows.append(
                ForwardDescriptorCompareRow(
                    scenario=cb_row.scenario,
                    phase=phase,
                    ordinal_in_phase=ordinal,
                    topology_key=cb_row.topology_key,
                    cb_iter_index=cb_row.iteration,
                    cb_context_tokens=cb_row.scheduled_context_tokens,
                    cb_decode_tokens=cb_row.scheduled_decode_tokens,
                    cb_forward_token_count=cb_row.forward_token_count,
                    cb_forward_regime=cb_row.forward_regime,
                    vllm_iter_index=vllm_row.iteration,
                    vllm_context_tokens=vllm_row.scheduled_context_tokens,
                    vllm_decode_tokens=vllm_row.scheduled_decode_tokens,
                    vllm_forward_token_count=vllm_row.forward_token_count,
                    vllm_batch_desc_num_tokens=vllm_row.batch_desc_num_tokens,
                    vllm_cudagraph_runtime_mode=vllm_row.cudagraph_runtime_mode,
                    vllm_forward_regime=vllm_row.forward_regime,
                    scheduled_total_delta=vllm_total - cb_total,
                    forward_token_delta=(
                        vllm_row.forward_token_count - cb_row.forward_token_count
                    ),
                    same_forward_token_count=int(
                        vllm_row.forward_token_count == cb_row.forward_token_count
                    ),
                    same_forward_regime=int(
                        vllm_row.forward_regime == cb_row.forward_regime
                    ),
                )
            )
    return compare_rows


def compare_runtime_shape_keys(
    cb_rows: list[VLLMRuntimeShapeKey],
    vllm_rows: list[VLLMRuntimeShapeKey],
) -> list[RuntimeShapeKeyCompareRow]:
    compare_rows: list[RuntimeShapeKeyCompareRow] = []
    for phase in ("prefill", "mixed", "pure_decode"):
        cb_phase = [row for row in cb_rows if row.phase == phase]
        vllm_phase = [row for row in vllm_rows if row.phase == phase]
        for ordinal, (cb_row, vllm_row) in enumerate(
            zip(cb_phase, vllm_phase, strict=False),
            start=1,
        ):
            compare_rows.append(
                RuntimeShapeKeyCompareRow(
                    scenario=cb_row.scenario,
                    phase=phase,
                    ordinal_in_phase=ordinal,
                    topology_key=cb_row.topology_key,
                    cb_iter_index=cb_row.iteration,
                    cb_cudagraph_runtime_mode=cb_row.cudagraph_runtime_mode,
                    cb_forward_token_count=cb_row.forward_token_count,
                    cb_attention_actual_tokens=cb_row.attention_actual_tokens,
                    cb_attention_max_query_len=cb_row.attention_max_query_len,
                    cb_slot_mapping_tokens=cb_row.slot_mapping_tokens,
                    cb_block_table_shape=cb_row.block_table_shape,
                    cb_forward_context_tokens=cb_row.forward_context_tokens,
                    vllm_iter_index=vllm_row.iteration,
                    vllm_cudagraph_runtime_mode=vllm_row.cudagraph_runtime_mode,
                    vllm_forward_token_count=vllm_row.forward_token_count,
                    vllm_attention_actual_tokens=vllm_row.attention_actual_tokens,
                    vllm_attention_max_query_len=vllm_row.attention_max_query_len,
                    vllm_slot_mapping_tokens=vllm_row.slot_mapping_tokens,
                    vllm_block_table_shape=vllm_row.block_table_shape,
                    vllm_forward_context_tokens=vllm_row.forward_context_tokens,
                    same_cudagraph_runtime_mode=int(
                        cb_row.cudagraph_runtime_mode
                        == vllm_row.cudagraph_runtime_mode
                    ),
                    same_forward_token_count=int(
                        cb_row.forward_token_count == vllm_row.forward_token_count
                    ),
                    same_attention_actual_tokens=int(
                        cb_row.attention_actual_tokens
                        == vllm_row.attention_actual_tokens
                    ),
                    same_attention_max_query_len=int(
                        cb_row.attention_max_query_len
                        == vllm_row.attention_max_query_len
                    ),
                    same_slot_mapping_tokens=int(
                        cb_row.slot_mapping_tokens == vllm_row.slot_mapping_tokens
                    ),
                    same_block_table_shape=int(
                        cb_row.block_table_shape == vllm_row.block_table_shape
                    ),
                    same_forward_context_tokens=int(
                        cb_row.forward_context_tokens
                        == vllm_row.forward_context_tokens
                    ),
                )
            )
    return compare_rows
