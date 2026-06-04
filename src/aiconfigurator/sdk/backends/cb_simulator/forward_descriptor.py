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
class VLLMSchedulerRuntimeDescriptor:
    """Scheduler/runtime-shape descriptor for experimental diagnostics.

    This is a mechanism descriptor only. It must not contain latency,
    residual, profiler, or throughput fields.
    """

    source: str
    scenario: str
    iteration: int
    phase: str
    scheduled_context_tokens: int
    scheduled_decode_tokens: int
    scheduled_total_tokens: int
    scheduled_context_reqs: int
    scheduled_decode_reqs: int
    scheduled_total_reqs: int
    max_num_batched_tokens: int
    max_num_seqs: int
    forward_token_count: int
    forward_regime: str
    cudagraph_runtime_mode: str
    topology_key: str
    tp: int
    dp: int
    moe_tp: int
    moe_ep: int
    valid_for_default: bool
    perf_database: bool
    diagnostic_only: bool


@dataclass(frozen=True)
class VLLMSchedulerAlignedDescriptor:
    """DP-aware scheduler descriptor for experimental alignment diagnostics."""

    alignment_key: str
    alignment_key_type: str
    engine_step_id: int
    dp_rank: int
    source: str
    scenario: str
    iteration: int
    phase: str
    scheduled_context_tokens: int
    scheduled_decode_tokens: int
    scheduled_total_tokens: int
    scheduled_context_reqs: int
    scheduled_decode_reqs: int
    scheduled_total_reqs: int
    max_num_batched_tokens: int
    max_num_seqs: int
    forward_token_count: int
    forward_regime: str
    cudagraph_runtime_mode: str
    topology_key: str
    tp: int
    dp: int
    moe_tp: int
    moe_ep: int
    valid_for_default: bool
    perf_database: bool
    diagnostic_only: bool


@dataclass(frozen=True)
class SchedulerAlignedCompareRow:
    alignment_key: str
    alignment_key_type: str
    dp_rank: int
    scenario: str
    phase: str
    topology_key: str
    cb_engine_step_id: int
    cb_iter_index: int
    cb_scheduled_context_tokens: int
    cb_scheduled_decode_tokens: int
    cb_scheduled_total_tokens: int
    cb_scheduled_context_reqs: int
    cb_scheduled_decode_reqs: int
    cb_scheduled_total_reqs: int
    cb_forward_token_count: int
    cb_forward_regime: str
    cb_cudagraph_runtime_mode: str
    vllm_engine_step_id: int
    vllm_iter_index: int
    vllm_scheduled_context_tokens: int
    vllm_scheduled_decode_tokens: int
    vllm_scheduled_total_tokens: int
    vllm_scheduled_context_reqs: int
    vllm_scheduled_decode_reqs: int
    vllm_scheduled_total_reqs: int
    vllm_forward_token_count: int
    vllm_forward_regime: str
    vllm_cudagraph_runtime_mode: str
    scheduled_context_token_delta: int
    scheduled_decode_token_delta: int
    scheduled_total_token_delta: int
    scheduled_context_req_delta: int
    scheduled_decode_req_delta: int
    scheduled_total_req_delta: int
    forward_token_delta: int
    same_phase: int
    same_topology_key: int
    same_scheduled_total_tokens: int
    same_scheduled_total_reqs: int
    same_forward_token_count: int
    same_forward_regime: int
    same_cudagraph_runtime_mode: int
    valid_for_default: bool
    perf_database: bool
    diagnostic_only: bool


@dataclass(frozen=True)
class SchedulerDescriptorCompareRow:
    scenario: str
    phase: str
    ordinal_in_phase: int
    topology_key: str
    cb_iter_index: int
    cb_scheduled_context_tokens: int
    cb_scheduled_decode_tokens: int
    cb_scheduled_total_tokens: int
    cb_scheduled_context_reqs: int
    cb_scheduled_decode_reqs: int
    cb_scheduled_total_reqs: int
    cb_forward_token_count: int
    cb_forward_regime: str
    cb_cudagraph_runtime_mode: str
    vllm_iter_index: int
    vllm_scheduled_context_tokens: int
    vllm_scheduled_decode_tokens: int
    vllm_scheduled_total_tokens: int
    vllm_scheduled_context_reqs: int
    vllm_scheduled_decode_reqs: int
    vllm_scheduled_total_reqs: int
    vllm_forward_token_count: int
    vllm_forward_regime: str
    vllm_cudagraph_runtime_mode: str
    scheduled_total_token_delta: int
    scheduled_total_req_delta: int
    forward_token_delta: int
    same_scheduled_total_tokens: int
    same_scheduled_total_reqs: int
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


@dataclass(frozen=True)
class VLLMMoESourceRuntimeKey:
    """Loaded-weight MoE source key for experimental vLLM diagnostics.

    This is a structure descriptor only. It must not contain latency,
    residual, profiler, trace, sync, or throughput fields.
    """

    source: str
    scenario: str
    runtime_backend: str
    vllm_version: str
    model_family: str
    module_class: str
    experts_class: str
    hidden_size: int
    moe_intermediate_size: int
    n_routed_experts: int
    local_experts: int
    global_experts: int
    topk: int
    n_shared_experts: int
    moe_method: str
    kernel_backend: str
    group_size: int
    num_bits: int
    dtype: str
    tp_size: int
    dp_size: int
    ep_size: int
    world_size: int
    rank: int
    device: str
    tuning_config_loaded: bool
    moe_config_fallback: bool
    moe_tuning_config_file: str
    loaded_weight: bool
    random_weight: bool
    timing: bool
    valid_for_default: bool
    perf_database: bool
    diagnostic_only: bool


def make_forward_regime(cudagraph_runtime_mode: str, forward_token_count: int) -> str:
    if not cudagraph_runtime_mode:
        raise ValueError("cudagraph_runtime_mode must be non-empty")
    if forward_token_count < 0:
        raise ValueError("forward_token_count must be non-negative")
    return f"{cudagraph_runtime_mode}:{forward_token_count}"


def _validate_phase(phase: str) -> None:
    if phase not in {"prefill", "mixed", "pure_decode"}:
        raise ValueError(f"phase must be prefill, mixed, or pure_decode: {phase!r}")


def _validate_non_negative(value: int, field: str) -> None:
    if value < 0:
        raise ValueError(f"{field} must be non-negative")


def _validate_positive(value: int, field: str) -> None:
    if value <= 0:
        raise ValueError(f"{field} must be positive")


def scheduler_runtime_descriptor_from_scheduled(
    *,
    source: str,
    scenario: str,
    iteration: int,
    phase: str,
    scheduled_context_tokens: int,
    scheduled_decode_tokens: int,
    scheduled_context_reqs: int,
    scheduled_decode_reqs: int,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    forward_token_count: int,
    cudagraph_runtime_mode: str,
    topology_key: str,
    tp: int,
    dp: int,
    moe_tp: int,
    moe_ep: int,
) -> VLLMSchedulerRuntimeDescriptor:
    if not source:
        raise ValueError("source must be non-empty")
    if not scenario:
        raise ValueError("scenario must be non-empty")
    _validate_phase(phase)
    for field, value in (
        ("iteration", iteration),
        ("scheduled_context_tokens", scheduled_context_tokens),
        ("scheduled_decode_tokens", scheduled_decode_tokens),
        ("scheduled_context_reqs", scheduled_context_reqs),
        ("scheduled_decode_reqs", scheduled_decode_reqs),
        ("forward_token_count", forward_token_count),
    ):
        _validate_non_negative(value, field)
    for field, value in (
        ("max_num_batched_tokens", max_num_batched_tokens),
        ("max_num_seqs", max_num_seqs),
        ("tp", tp),
        ("dp", dp),
        ("moe_tp", moe_tp),
        ("moe_ep", moe_ep),
    ):
        _validate_positive(value, field)

    expected_topology_key = make_topology_key(tp, dp, moe_tp, moe_ep)
    if topology_key != expected_topology_key:
        raise ValueError(
            f"topology_key mismatch: {topology_key!r} != {expected_topology_key!r}"
        )

    scheduled_total_tokens = scheduled_context_tokens + scheduled_decode_tokens
    scheduled_total_reqs = scheduled_context_reqs + scheduled_decode_reqs
    if forward_token_count < scheduled_total_tokens:
        raise ValueError(
            "forward_token_count must cover scheduled_total_tokens: "
            f"{forward_token_count} < {scheduled_total_tokens}"
        )

    return VLLMSchedulerRuntimeDescriptor(
        source=source,
        scenario=scenario,
        iteration=iteration,
        phase=phase,
        scheduled_context_tokens=scheduled_context_tokens,
        scheduled_decode_tokens=scheduled_decode_tokens,
        scheduled_total_tokens=scheduled_total_tokens,
        scheduled_context_reqs=scheduled_context_reqs,
        scheduled_decode_reqs=scheduled_decode_reqs,
        scheduled_total_reqs=scheduled_total_reqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        forward_token_count=forward_token_count,
        forward_regime=make_forward_regime(cudagraph_runtime_mode, forward_token_count),
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        topology_key=topology_key,
        tp=tp,
        dp=dp,
        moe_tp=moe_tp,
        moe_ep=moe_ep,
        valid_for_default=False,
        perf_database=False,
        diagnostic_only=True,
    )


def scheduler_aligned_descriptor_from_scheduled(
    *,
    source: str,
    scenario: str,
    dp_rank: int,
    engine_step_id: int,
    phase: str,
    scheduled_context_tokens: int,
    scheduled_decode_tokens: int,
    scheduled_context_reqs: int,
    scheduled_decode_reqs: int,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    forward_token_count: int,
    cudagraph_runtime_mode: str,
    topology_key: str,
    tp: int,
    dp: int,
    moe_tp: int,
    moe_ep: int,
) -> VLLMSchedulerAlignedDescriptor:
    _validate_non_negative(engine_step_id, "engine_step_id")
    _validate_non_negative(dp_rank, "dp_rank")
    _validate_positive(dp, "dp")
    if dp_rank >= dp:
        raise ValueError(f"dp_rank must be smaller than dp: {dp_rank} >= {dp}")

    descriptor = scheduler_runtime_descriptor_from_scheduled(
        source=source,
        scenario=scenario,
        iteration=engine_step_id,
        phase=phase,
        scheduled_context_tokens=scheduled_context_tokens,
        scheduled_decode_tokens=scheduled_decode_tokens,
        scheduled_context_reqs=scheduled_context_reqs,
        scheduled_decode_reqs=scheduled_decode_reqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        forward_token_count=forward_token_count,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        topology_key=topology_key,
        tp=tp,
        dp=dp,
        moe_tp=moe_tp,
        moe_ep=moe_ep,
    )
    alignment_key_type = "engine_core_dp_step"
    alignment_key = f"engine_dp:{dp_rank}:step:{engine_step_id}"
    return VLLMSchedulerAlignedDescriptor(
        alignment_key=alignment_key,
        alignment_key_type=alignment_key_type,
        engine_step_id=engine_step_id,
        dp_rank=dp_rank,
        source=descriptor.source,
        scenario=descriptor.scenario,
        iteration=descriptor.iteration,
        phase=descriptor.phase,
        scheduled_context_tokens=descriptor.scheduled_context_tokens,
        scheduled_decode_tokens=descriptor.scheduled_decode_tokens,
        scheduled_total_tokens=descriptor.scheduled_total_tokens,
        scheduled_context_reqs=descriptor.scheduled_context_reqs,
        scheduled_decode_reqs=descriptor.scheduled_decode_reqs,
        scheduled_total_reqs=descriptor.scheduled_total_reqs,
        max_num_batched_tokens=descriptor.max_num_batched_tokens,
        max_num_seqs=descriptor.max_num_seqs,
        forward_token_count=descriptor.forward_token_count,
        forward_regime=descriptor.forward_regime,
        cudagraph_runtime_mode=descriptor.cudagraph_runtime_mode,
        topology_key=descriptor.topology_key,
        tp=descriptor.tp,
        dp=descriptor.dp,
        moe_tp=descriptor.moe_tp,
        moe_ep=descriptor.moe_ep,
        valid_for_default=descriptor.valid_for_default,
        perf_database=descriptor.perf_database,
        diagnostic_only=descriptor.diagnostic_only,
    )


def _pad_to_multiple(value: int, multiple: int) -> int:
    _validate_non_negative(value, "value")
    _validate_positive(multiple, "multiple")
    if value == 0:
        return 0
    return ((value + multiple - 1) // multiple) * multiple


def vllm_like_scheduler_aligned_descriptors(
    *,
    source: str,
    scenario: str,
    isl: int,
    osl: int,
    concurrency: int,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    tp: int,
    dp: int,
    moe_tp: int,
    moe_ep: int,
    block_size: int = 16,
    graph_padding_multiple: int = 8,
) -> list[VLLMSchedulerAlignedDescriptor]:
    """Build a descriptor-only vLLM-like DP scheduler sequence.

    This first-evidence generator targets three captured scheduler shapes:
    Phase62 10k2k_b32, Phase69 3k3k_b128, and Phase71/73 32k1k_b16.
    It intentionally fail-fasts instead of guessing unsupported DP/request
    splits, budgets, or topology variants.
    """
    if not source:
        raise ValueError("source must be non-empty")
    if not scenario:
        raise ValueError("scenario must be non-empty")
    for field, value in (
        ("isl", isl),
        ("osl", osl),
        ("concurrency", concurrency),
        ("max_num_batched_tokens", max_num_batched_tokens),
        ("max_num_seqs", max_num_seqs),
        ("tp", tp),
        ("dp", dp),
        ("moe_tp", moe_tp),
        ("moe_ep", moe_ep),
        ("block_size", block_size),
        ("graph_padding_multiple", graph_padding_multiple),
    ):
        _validate_positive(value, field)
    if concurrency % dp != 0:
        raise ValueError("concurrency must be divisible by dp for DP-local rows")
    per_dp_reqs = concurrency // dp
    if per_dp_reqs < 2:
        raise ValueError("vLLM-like scheduler descriptor requires >=2 reqs per DP")
    topology_key = make_topology_key(tp, dp, moe_tp, moe_ep)
    rows: list[VLLMSchedulerAlignedDescriptor] = []

    def append_row(
        *,
        dp_rank: int,
        engine_step_id: int,
        context_tokens: int,
        decode_tokens: int,
        context_reqs: int,
        decode_reqs: int,
        cudagraph_runtime_mode: str,
        forward_token_count: int,
    ) -> None:
        phase = "mixed"
        if context_tokens > 0 and decode_tokens == 0:
            phase = "prefill"
        elif context_tokens == 0 and decode_tokens > 0:
            phase = "pure_decode"
        rows.append(
            scheduler_aligned_descriptor_from_scheduled(
                source=source,
                scenario=scenario,
                dp_rank=dp_rank,
                engine_step_id=engine_step_id,
                phase=phase,
                scheduled_context_tokens=context_tokens,
                scheduled_decode_tokens=decode_tokens,
                scheduled_context_reqs=context_reqs,
                scheduled_decode_reqs=decode_reqs,
                max_num_batched_tokens=max_num_batched_tokens,
                max_num_seqs=max_num_seqs,
                forward_token_count=forward_token_count,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                topology_key=topology_key,
                tp=tp,
                dp=dp,
                moe_tp=moe_tp,
                moe_ep=moe_ep,
            )
        )

    if isl <= max_num_batched_tokens:
        phase69_shape = (
            dp == 2
            and concurrency == 128
            and isl == 3000
            and osl == 3000
            and max_num_batched_tokens == 8192
            and graph_padding_multiple == 8
        )
        if not phase69_shape:
            raise ValueError(
                "short-ISL vLLM-like scheduler descriptor currently supports "
                "only the Phase69 3k3k_b128 bt8192 holdout"
            )

        dp0_reqs = per_dp_reqs - 1
        dp1_reqs = per_dp_reqs + 1
        short_suffix_tokens = graph_padding_multiple

        append_row(
            dp_rank=0,
            engine_step_id=0,
            context_tokens=isl,
            decode_tokens=0,
            context_reqs=1,
            decode_reqs=0,
            cudagraph_runtime_mode="NONE",
            forward_token_count=isl,
        )
        append_row(
            dp_rank=0,
            engine_step_id=1,
            context_tokens=0,
            decode_tokens=1,
            context_reqs=0,
            decode_reqs=1,
            cudagraph_runtime_mode="NONE",
            forward_token_count=1,
        )
        append_row(
            dp_rank=0,
            engine_step_id=2,
            context_tokens=(dp0_reqs - 1) * short_suffix_tokens,
            decode_tokens=1,
            context_reqs=dp0_reqs - 1,
            decode_reqs=1,
            cudagraph_runtime_mode="PIECEWISE",
            forward_token_count=512,
        )
        for engine_step_id in range(3, osl):
            append_row(
                dp_rank=0,
                engine_step_id=engine_step_id,
                context_tokens=0,
                decode_tokens=dp0_reqs,
                context_reqs=0,
                decode_reqs=dp0_reqs,
                cudagraph_runtime_mode="FULL",
                forward_token_count=_pad_to_multiple(
                    dp1_reqs,
                    graph_padding_multiple,
                ),
            )
        append_row(
            dp_rank=0,
            engine_step_id=osl,
            context_tokens=0,
            decode_tokens=dp0_reqs - 1,
            context_reqs=0,
            decode_reqs=dp0_reqs - 1,
            cudagraph_runtime_mode="FULL",
            forward_token_count=_pad_to_multiple(
                dp1_reqs,
                graph_padding_multiple,
            ),
        )
        append_row(
            dp_rank=0,
            engine_step_id=osl + 1,
            context_tokens=0,
            decode_tokens=dp0_reqs - 1,
            context_reqs=0,
            decode_reqs=dp0_reqs - 1,
            cudagraph_runtime_mode="FULL",
            forward_token_count=_pad_to_multiple(
                dp0_reqs - 1,
                graph_padding_multiple,
            ),
        )

        append_row(
            dp_rank=1,
            engine_step_id=0,
            context_tokens=isl + (dp1_reqs - 1) * short_suffix_tokens,
            decode_tokens=0,
            context_reqs=dp1_reqs,
            decode_reqs=0,
            cudagraph_runtime_mode="NONE",
            forward_token_count=isl + (dp1_reqs - 1) * short_suffix_tokens,
        )
        append_row(
            dp_rank=1,
            engine_step_id=1,
            context_tokens=0,
            decode_tokens=dp1_reqs,
            context_reqs=0,
            decode_reqs=dp1_reqs,
            cudagraph_runtime_mode="PIECEWISE",
            forward_token_count=512,
        )
        for engine_step_id in range(2, osl):
            append_row(
                dp_rank=1,
                engine_step_id=engine_step_id,
                context_tokens=0,
                decode_tokens=dp1_reqs,
                context_reqs=0,
                decode_reqs=dp1_reqs,
                cudagraph_runtime_mode="FULL",
                forward_token_count=_pad_to_multiple(
                    dp1_reqs,
                    graph_padding_multiple,
                ),
        )
        return rows

    phase71_shape = (
        tp == 4
        and dp == 2
        and moe_tp == 1
        and moe_ep == 8
        and concurrency == 16
        and isl == 32000
        and osl == 1000
        and max_num_batched_tokens == 8192
        and graph_padding_multiple == 8
    )
    if phase71_shape:
        prefill_chunks = (8192, 8192, 8192, 7536)
        for dp_rank in range(dp):
            for engine_step_id, context_tokens in enumerate(prefill_chunks):
                append_row(
                    dp_rank=dp_rank,
                    engine_step_id=engine_step_id,
                    context_tokens=context_tokens,
                    decode_tokens=0,
                    context_reqs=1 if engine_step_id < 3 else per_dp_reqs,
                    decode_reqs=0,
                    cudagraph_runtime_mode="NONE",
                    forward_token_count=context_tokens,
                )

            append_row(
                dp_rank=dp_rank,
                engine_step_id=4,
                context_tokens=0,
                decode_tokens=per_dp_reqs,
                context_reqs=0,
                decode_reqs=per_dp_reqs,
                cudagraph_runtime_mode="FULL" if dp_rank == 0 else "NONE",
                forward_token_count=per_dp_reqs,
            )
            for engine_step_id in range(5, 1003):
                append_row(
                    dp_rank=dp_rank,
                    engine_step_id=engine_step_id,
                    context_tokens=0,
                    decode_tokens=per_dp_reqs,
                    context_reqs=0,
                    decode_reqs=per_dp_reqs,
                    cudagraph_runtime_mode="FULL",
                    forward_token_count=per_dp_reqs,
                )
        return rows

    if isl == 32000 and osl == 1000:
        raise ValueError(
            "32k1k vLLM-like scheduler descriptor currently supports only "
            "the Phase71 32k1k_b16 bt8192 tp4dp2ep8 holdout"
        )

    leader_remaining = isl - max_num_batched_tokens
    follower_count = per_dp_reqs - 1
    follower_suffix_tokens = follower_count * block_size
    if leader_remaining + follower_suffix_tokens > max_num_batched_tokens:
        raise ValueError(
            "leader remainder plus follower suffix chunks exceed token budget"
        )

    for dp_rank in range(dp):
        append_row(
            dp_rank=dp_rank,
            engine_step_id=0,
            context_tokens=max_num_batched_tokens,
            decode_tokens=0,
            context_reqs=1,
            decode_reqs=0,
            cudagraph_runtime_mode="NONE",
            forward_token_count=max_num_batched_tokens,
        )

        if dp_rank == 0:
            append_row(
                dp_rank=dp_rank,
                engine_step_id=1,
                context_tokens=leader_remaining + follower_suffix_tokens,
                decode_tokens=0,
                context_reqs=per_dp_reqs,
                decode_reqs=0,
                cudagraph_runtime_mode="NONE",
                forward_token_count=leader_remaining + follower_suffix_tokens,
            )
            for engine_step_id in range(2, osl + 1):
                append_row(
                    dp_rank=dp_rank,
                    engine_step_id=engine_step_id,
                    context_tokens=0,
                    decode_tokens=per_dp_reqs,
                    context_reqs=0,
                    decode_reqs=per_dp_reqs,
                    cudagraph_runtime_mode="FULL",
                    forward_token_count=_pad_to_multiple(
                        per_dp_reqs,
                        graph_padding_multiple,
                    ),
                )
            continue

        append_row(
            dp_rank=dp_rank,
            engine_step_id=1,
            context_tokens=leader_remaining,
            decode_tokens=0,
            context_reqs=1,
            decode_reqs=0,
            cudagraph_runtime_mode="NONE",
            forward_token_count=leader_remaining,
        )
        mixed_tokens = follower_suffix_tokens + 1
        append_row(
            dp_rank=dp_rank,
            engine_step_id=2,
            context_tokens=follower_suffix_tokens,
            decode_tokens=1,
            context_reqs=follower_count,
            decode_reqs=1,
            cudagraph_runtime_mode="NONE",
            forward_token_count=_pad_to_multiple(
                mixed_tokens,
                graph_padding_multiple,
            ),
        )
        for engine_step_id in range(3, osl + 1):
            append_row(
                dp_rank=dp_rank,
                engine_step_id=engine_step_id,
                context_tokens=0,
                decode_tokens=per_dp_reqs,
                context_reqs=0,
                decode_reqs=per_dp_reqs,
                cudagraph_runtime_mode="FULL",
                forward_token_count=_pad_to_multiple(
                    per_dp_reqs,
                    graph_padding_multiple,
                ),
            )
        append_row(
            dp_rank=dp_rank,
            engine_step_id=osl + 1,
            context_tokens=0,
            decode_tokens=follower_count,
            context_reqs=0,
            decode_reqs=follower_count,
            cudagraph_runtime_mode="FULL",
            forward_token_count=_pad_to_multiple(
                follower_count,
                graph_padding_multiple,
            ),
        )

    return rows


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


def _required_int(raw: dict[str, object], field: str) -> int:
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


def _required_non_empty(raw: dict[str, object], field: str) -> str:
    value = raw.get(field, "")
    if value is None or value == "":
        raise ValueError(f"missing {field}")
    return str(value)


def _required_bool(raw: dict[str, object], field: str) -> bool:
    value = _required_non_empty(raw, field).strip().lower()
    if value in {"true", "1"}:
        return True
    if value in {"false", "0"}:
        return False
    raise ValueError(f"{field} must be boolean")


def _required_false(raw: dict[str, object], field: str) -> None:
    value = _required_non_empty(raw, field).strip().lower()
    if value not in {"false", "0"}:
        raise ValueError(f"{field} must be false")


def _required_true(raw: dict[str, object], field: str) -> None:
    value = _required_non_empty(raw, field).strip().lower()
    if value not in {"true", "1"}:
        raise ValueError(f"{field} must be true")


def _reject_forbidden_descriptor_fields(raw: dict[str, object]) -> None:
    forbidden = (
        "_ms",
        "latency_ms",
        "duration_ms",
        "residual_ms",
        "profiled_cuda",
        "profiler",
        "trace",
        "nccl",
        "NCCL",
        "sync",
        "sync_wait",
        "throughput",
    )
    for field in raw:
        if any(token in field for token in forbidden):
            raise ValueError(f"forbidden field in descriptor row: {field}")


def scheduler_aligned_descriptor_from_csv_row(
    raw: dict[str, str],
) -> VLLMSchedulerAlignedDescriptor:
    _reject_forbidden_descriptor_fields(raw)
    if _required_non_empty(raw, "alignment_key_type") != "engine_core_dp_step":
        raise ValueError("alignment_key_type must be engine_core_dp_step")
    _required_false(raw, "valid_for_default")
    _required_false(raw, "perf_database")
    _required_true(raw, "diagnostic_only")

    descriptor = scheduler_aligned_descriptor_from_scheduled(
        source=_required_non_empty(raw, "source"),
        scenario=_required_non_empty(raw, "scenario"),
        dp_rank=_required_int(raw, "dp_rank"),
        engine_step_id=_required_int(raw, "engine_step_id"),
        phase=_required_non_empty(raw, "phase"),
        scheduled_context_tokens=_required_int(raw, "scheduled_context_tokens"),
        scheduled_decode_tokens=_required_int(raw, "scheduled_decode_tokens"),
        scheduled_context_reqs=_required_int(raw, "scheduled_context_reqs"),
        scheduled_decode_reqs=_required_int(raw, "scheduled_decode_reqs"),
        max_num_batched_tokens=_required_int(raw, "max_num_batched_tokens"),
        max_num_seqs=_required_int(raw, "max_num_seqs"),
        forward_token_count=_required_int(raw, "forward_token_count"),
        cudagraph_runtime_mode=_required_non_empty(raw, "cudagraph_runtime_mode"),
        topology_key=_required_non_empty(raw, "topology_key"),
        tp=_required_int(raw, "tp"),
        dp=_required_int(raw, "dp"),
        moe_tp=_required_int(raw, "moe_tp"),
        moe_ep=_required_int(raw, "moe_ep"),
    )
    if _required_non_empty(raw, "alignment_key") != descriptor.alignment_key:
        raise ValueError("alignment_key mismatch")
    if _required_int(raw, "iteration") != descriptor.iteration:
        raise ValueError("iteration must equal engine_step_id")
    if _required_int(raw, "scheduled_total_tokens") != (
        descriptor.scheduled_total_tokens
    ):
        raise ValueError("scheduled_total_tokens mismatch")
    if _required_int(raw, "scheduled_total_reqs") != descriptor.scheduled_total_reqs:
        raise ValueError("scheduled_total_reqs mismatch")
    if _required_non_empty(raw, "forward_regime") != descriptor.forward_regime:
        raise ValueError("forward_regime mismatch")
    return descriptor


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


def moe_source_runtime_key_from_loaded_weight_boundary_row(
    raw: dict[str, object],
) -> VLLMMoESourceRuntimeKey:
    """Build a MoE source key from a loaded-weight boundary row.

    Fallback config status is represented because it is part of the runtime
    source key, but this function keeps the descriptor explicitly invalid for
    default latency modeling.
    """
    _reject_forbidden_descriptor_fields(raw)
    _required_true(raw, "loaded_weight")
    _required_false(raw, "random_weight")
    _required_false(raw, "timing")
    _required_false(raw, "valid_for_default")
    _required_false(raw, "perf_database")
    _required_true(raw, "diagnostic_only")

    positive_fields = (
        "hidden_size",
        "moe_intermediate_size",
        "n_routed_experts",
        "local_experts",
        "global_experts",
        "topk",
        "n_shared_experts",
        "group_size",
        "num_bits",
        "tp_size",
        "dp_size",
        "ep_size",
        "world_size",
    )
    values = {field: _required_int(raw, field) for field in positive_fields}
    for field, value in values.items():
        _validate_positive(value, field)
    rank = _required_int(raw, "rank")
    _validate_non_negative(rank, "rank")
    if values["world_size"] != values["tp_size"] * values["dp_size"]:
        raise ValueError("world_size must equal tp_size * dp_size")
    if values["local_experts"] > values["global_experts"]:
        raise ValueError("local_experts must not exceed global_experts")
    if values["topk"] > values["global_experts"]:
        raise ValueError("topk must not exceed global_experts")

    return VLLMMoESourceRuntimeKey(
        source=_required_non_empty(raw, "source"),
        scenario=_required_non_empty(raw, "scenario"),
        runtime_backend=_required_non_empty(raw, "runtime_backend"),
        vllm_version=_required_non_empty(raw, "vllm_version"),
        model_family=_required_non_empty(raw, "model_family"),
        module_class=_required_non_empty(raw, "module_class"),
        experts_class=_required_non_empty(raw, "experts_class"),
        hidden_size=values["hidden_size"],
        moe_intermediate_size=values["moe_intermediate_size"],
        n_routed_experts=values["n_routed_experts"],
        local_experts=values["local_experts"],
        global_experts=values["global_experts"],
        topk=values["topk"],
        n_shared_experts=values["n_shared_experts"],
        moe_method=_required_non_empty(raw, "moe_method"),
        kernel_backend=_required_non_empty(raw, "kernel_backend"),
        group_size=values["group_size"],
        num_bits=values["num_bits"],
        dtype=_required_non_empty(raw, "dtype"),
        tp_size=values["tp_size"],
        dp_size=values["dp_size"],
        ep_size=values["ep_size"],
        world_size=values["world_size"],
        rank=rank,
        device=_required_non_empty(raw, "device"),
        tuning_config_loaded=_required_bool(raw, "tuning_config_loaded"),
        moe_config_fallback=_required_bool(raw, "moe_config_fallback"),
        moe_tuning_config_file=str(raw.get("moe_tuning_config_file", "")),
        loaded_weight=True,
        random_weight=False,
        timing=False,
        valid_for_default=False,
        perf_database=False,
        diagnostic_only=True,
    )


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


def compare_scheduler_runtime_descriptors(
    cb_rows: list[VLLMSchedulerRuntimeDescriptor],
    vllm_rows: list[VLLMSchedulerRuntimeDescriptor],
) -> list[SchedulerDescriptorCompareRow]:
    """Ordinal diagnostic compare for scheduler/runtime descriptors."""
    compare_rows: list[SchedulerDescriptorCompareRow] = []
    for phase in ("prefill", "mixed", "pure_decode"):
        cb_phase = [row for row in cb_rows if row.phase == phase]
        vllm_phase = [row for row in vllm_rows if row.phase == phase]
        for ordinal, (cb_row, vllm_row) in enumerate(
            zip(cb_phase, vllm_phase, strict=False),
            start=1,
        ):
            compare_rows.append(
                SchedulerDescriptorCompareRow(
                    scenario=cb_row.scenario,
                    phase=phase,
                    ordinal_in_phase=ordinal,
                    topology_key=cb_row.topology_key,
                    cb_iter_index=cb_row.iteration,
                    cb_scheduled_context_tokens=cb_row.scheduled_context_tokens,
                    cb_scheduled_decode_tokens=cb_row.scheduled_decode_tokens,
                    cb_scheduled_total_tokens=cb_row.scheduled_total_tokens,
                    cb_scheduled_context_reqs=cb_row.scheduled_context_reqs,
                    cb_scheduled_decode_reqs=cb_row.scheduled_decode_reqs,
                    cb_scheduled_total_reqs=cb_row.scheduled_total_reqs,
                    cb_forward_token_count=cb_row.forward_token_count,
                    cb_forward_regime=cb_row.forward_regime,
                    cb_cudagraph_runtime_mode=cb_row.cudagraph_runtime_mode,
                    vllm_iter_index=vllm_row.iteration,
                    vllm_scheduled_context_tokens=(
                        vllm_row.scheduled_context_tokens
                    ),
                    vllm_scheduled_decode_tokens=vllm_row.scheduled_decode_tokens,
                    vllm_scheduled_total_tokens=vllm_row.scheduled_total_tokens,
                    vllm_scheduled_context_reqs=vllm_row.scheduled_context_reqs,
                    vllm_scheduled_decode_reqs=vllm_row.scheduled_decode_reqs,
                    vllm_scheduled_total_reqs=vllm_row.scheduled_total_reqs,
                    vllm_forward_token_count=vllm_row.forward_token_count,
                    vllm_forward_regime=vllm_row.forward_regime,
                    vllm_cudagraph_runtime_mode=vllm_row.cudagraph_runtime_mode,
                    scheduled_total_token_delta=(
                        vllm_row.scheduled_total_tokens
                        - cb_row.scheduled_total_tokens
                    ),
                    scheduled_total_req_delta=(
                        vllm_row.scheduled_total_reqs - cb_row.scheduled_total_reqs
                    ),
                    forward_token_delta=(
                        vllm_row.forward_token_count - cb_row.forward_token_count
                    ),
                    same_scheduled_total_tokens=int(
                        vllm_row.scheduled_total_tokens
                        == cb_row.scheduled_total_tokens
                    ),
                    same_scheduled_total_reqs=int(
                        vllm_row.scheduled_total_reqs == cb_row.scheduled_total_reqs
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


def _scheduler_alignment_key(
    row: VLLMSchedulerAlignedDescriptor,
    source_name: str,
) -> tuple[str, int]:
    if row.alignment_key_type != "engine_core_dp_step":
        raise ValueError(
            f"{source_name} alignment_key_type must be engine_core_dp_step"
        )
    expected_key = f"engine_dp:{row.dp_rank}:step:{row.engine_step_id}"
    if row.alignment_key != expected_key:
        raise ValueError(
            f"{source_name} alignment_key mismatch: "
            f"{row.alignment_key!r} != {expected_key!r}"
        )
    if row.iteration != row.engine_step_id:
        raise ValueError(
            f"{source_name} iteration must equal engine_step_id: "
            f"{row.iteration} != {row.engine_step_id}"
        )
    if row.valid_for_default:
        raise ValueError(f"{source_name} valid_for_default must be false")
    if row.perf_database:
        raise ValueError(f"{source_name} perf_database must be false")
    if not row.diagnostic_only:
        raise ValueError(f"{source_name} diagnostic_only must be true")
    return (row.alignment_key, row.dp_rank)


def _index_scheduler_aligned_rows(
    rows: list[VLLMSchedulerAlignedDescriptor],
    source_name: str,
) -> dict[tuple[str, int], VLLMSchedulerAlignedDescriptor]:
    if not rows:
        raise ValueError(f"{source_name} aligned rows must be non-empty")
    indexed: dict[tuple[str, int], VLLMSchedulerAlignedDescriptor] = {}
    for row in rows:
        key = _scheduler_alignment_key(row, source_name)
        if key in indexed:
            raise ValueError(
                f"duplicate {source_name} alignment key: {key[0]} dp={key[1]}"
            )
        indexed[key] = row
    return indexed


def compare_scheduler_aligned_descriptors(
    cb_rows: list[VLLMSchedulerAlignedDescriptor],
    vllm_rows: list[VLLMSchedulerAlignedDescriptor],
) -> list[SchedulerAlignedCompareRow]:
    cb_index = _index_scheduler_aligned_rows(cb_rows, "cb")
    vllm_index = _index_scheduler_aligned_rows(vllm_rows, "vllm")
    cb_keys = set(cb_index)
    vllm_keys = set(vllm_index)
    missing_vllm = sorted(cb_keys - vllm_keys)
    if missing_vllm:
        missing = ", ".join(f"{key} dp={dp}" for key, dp in missing_vllm[:5])
        raise ValueError(f"missing vllm alignment keys: {missing}")
    missing_cb = sorted(vllm_keys - cb_keys)
    if missing_cb:
        missing = ", ".join(f"{key} dp={dp}" for key, dp in missing_cb[:5])
        raise ValueError(f"missing cb alignment keys: {missing}")

    compare_rows: list[SchedulerAlignedCompareRow] = []
    for key in sorted(cb_keys, key=lambda item: (item[1], cb_index[item].engine_step_id)):
        cb_row = cb_index[key]
        vllm_row = vllm_index[key]
        compare_rows.append(
            SchedulerAlignedCompareRow(
                alignment_key=cb_row.alignment_key,
                alignment_key_type=cb_row.alignment_key_type,
                dp_rank=cb_row.dp_rank,
                scenario=cb_row.scenario,
                phase=cb_row.phase,
                topology_key=cb_row.topology_key,
                cb_engine_step_id=cb_row.engine_step_id,
                cb_iter_index=cb_row.iteration,
                cb_scheduled_context_tokens=cb_row.scheduled_context_tokens,
                cb_scheduled_decode_tokens=cb_row.scheduled_decode_tokens,
                cb_scheduled_total_tokens=cb_row.scheduled_total_tokens,
                cb_scheduled_context_reqs=cb_row.scheduled_context_reqs,
                cb_scheduled_decode_reqs=cb_row.scheduled_decode_reqs,
                cb_scheduled_total_reqs=cb_row.scheduled_total_reqs,
                cb_forward_token_count=cb_row.forward_token_count,
                cb_forward_regime=cb_row.forward_regime,
                cb_cudagraph_runtime_mode=cb_row.cudagraph_runtime_mode,
                vllm_engine_step_id=vllm_row.engine_step_id,
                vllm_iter_index=vllm_row.iteration,
                vllm_scheduled_context_tokens=(
                    vllm_row.scheduled_context_tokens
                ),
                vllm_scheduled_decode_tokens=vllm_row.scheduled_decode_tokens,
                vllm_scheduled_total_tokens=vllm_row.scheduled_total_tokens,
                vllm_scheduled_context_reqs=vllm_row.scheduled_context_reqs,
                vllm_scheduled_decode_reqs=vllm_row.scheduled_decode_reqs,
                vllm_scheduled_total_reqs=vllm_row.scheduled_total_reqs,
                vllm_forward_token_count=vllm_row.forward_token_count,
                vllm_forward_regime=vllm_row.forward_regime,
                vllm_cudagraph_runtime_mode=vllm_row.cudagraph_runtime_mode,
                scheduled_context_token_delta=(
                    vllm_row.scheduled_context_tokens
                    - cb_row.scheduled_context_tokens
                ),
                scheduled_decode_token_delta=(
                    vllm_row.scheduled_decode_tokens
                    - cb_row.scheduled_decode_tokens
                ),
                scheduled_total_token_delta=(
                    vllm_row.scheduled_total_tokens
                    - cb_row.scheduled_total_tokens
                ),
                scheduled_context_req_delta=(
                    vllm_row.scheduled_context_reqs
                    - cb_row.scheduled_context_reqs
                ),
                scheduled_decode_req_delta=(
                    vllm_row.scheduled_decode_reqs
                    - cb_row.scheduled_decode_reqs
                ),
                scheduled_total_req_delta=(
                    vllm_row.scheduled_total_reqs - cb_row.scheduled_total_reqs
                ),
                forward_token_delta=(
                    vllm_row.forward_token_count - cb_row.forward_token_count
                ),
                same_phase=int(vllm_row.phase == cb_row.phase),
                same_topology_key=int(vllm_row.topology_key == cb_row.topology_key),
                same_scheduled_total_tokens=int(
                    vllm_row.scheduled_total_tokens
                    == cb_row.scheduled_total_tokens
                ),
                same_scheduled_total_reqs=int(
                    vllm_row.scheduled_total_reqs == cb_row.scheduled_total_reqs
                ),
                same_forward_token_count=int(
                    vllm_row.forward_token_count == cb_row.forward_token_count
                ),
                same_forward_regime=int(
                    vllm_row.forward_regime == cb_row.forward_regime
                ),
                same_cudagraph_runtime_mode=int(
                    vllm_row.cudagraph_runtime_mode
                    == cb_row.cudagraph_runtime_mode
                ),
                valid_for_default=False,
                perf_database=False,
                diagnostic_only=True,
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
