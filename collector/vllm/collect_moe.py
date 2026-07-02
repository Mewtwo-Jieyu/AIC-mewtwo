# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import torch
import torch.nn.functional as F
from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
from vllm.model_executor.layers.fused_moe.layer import determine_expert_map
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    batched_moe_align_block_size,
    moe_align_block_size,
)
from vllm.version import __version__ as vllm_version

# wna16 (4-bit W4A16 marlin) MoE path. Mirrors vLLM's
# CompressedTensorsWNA16MarlinMoEMethod weight prep + apply so that the
# collected latency matches the real `marlin_moe_wna16` kernel served for
# compressed-tensors pack-quantized checkpoints (e.g. Kimi-K2.5 routed experts).
try:
    from vllm import _custom_ops as _vllm_ops
    from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
        batched_fused_marlin_moe,
        fused_marlin_moe,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
        marlin_moe_permute_scales,
    )
    from vllm.scalar_type import scalar_types

    _WNA16_MARLIN_AVAILABLE = True
    _WNA16_TYPE_MAP = {4: scalar_types.uint4b8, 8: scalar_types.uint8b128}
except Exception:  # pragma: no cover - older vLLM without marlin MoE
    _WNA16_MARLIN_AVAILABLE = False
    _WNA16_TYPE_MAP = {}

# Compatibility: block FP8 helpers may differ by version.
# Priority: vllm.utils.deep_gemm -> deep_gemm extension -> None.
try:
    from vllm.utils.deep_gemm import per_block_cast_to_fp8
except Exception:
    try:
        import deep_gemm  # type: ignore

        per_block_cast_to_fp8 = getattr(deep_gemm, "per_block_cast_to_fp8", None)
    except Exception:
        per_block_cast_to_fp8 = None  # type: ignore[assignment]

from collector.common_test_cases import get_common_moe_test_cases
from collector.helper import balanced_logits, benchmark_with_power, get_sm_version, log_perf, power_law_logits_v3

aic_debug = int(os.getenv("aic_moe_debug", "0"))  # noqa: SIM112

compatible_version = ["0.11.0", "0.12.0", "0.14.0", "0.19.0"]


def _is_power_law_mode(distributed):
    return distributed in (
        "power_law",
        "power_law_eplb",
        "power_law_rank0_compact",
        "power_law_batched_rank0",
    )


def _power_law_use_eplb(distributed):
    return distributed == "power_law_eplb"


def _is_rank0_compact_mode(distributed):
    return distributed == "power_law_rank0_compact"


def _is_batched_rank0_mode(distributed):
    return distributed == "power_law_batched_rank0"


def _distribution_label(distributed, power_law_alpha):
    if distributed == "power_law":
        return "power_law_" + str(power_law_alpha)
    if distributed == "power_law_eplb":
        return "power_law_eplb_" + str(power_law_alpha)
    if distributed == "power_law_rank0_compact":
        return "power_law_rank0_compact_" + str(power_law_alpha)
    if distributed == "power_law_batched_rank0":
        return "power_law_batched_rank0_" + str(power_law_alpha)
    return distributed


def get_moe_test_cases():
    """Generate MoE test cases"""

    # Quantization types supported by vLLM
    moe_list = ["float16"]
    if get_sm_version() > 86:
        moe_list += ["fp8"]
    if get_sm_version() >= 90 and per_block_cast_to_fp8 is not None:
        moe_list += ["fp8_block"]

    test_cases = []

    for common_moe_testcase in get_common_moe_test_cases():
        if common_moe_testcase.token_expert_distribution != "power_law":
            continue

        model_name = common_moe_testcase.model_name
        if model_name in ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]:
            continue

        # vllm does not support TP when EP is enabled.
        if common_moe_testcase.tp > 1 and common_moe_testcase.ep > 1:
            continue

        for moe_type in moe_list:
            # fp8_block requires hidden_size divisible by block group_size (128)
            if moe_type == "fp8_block" and (
                common_moe_testcase.hidden_size % 128 != 0 or common_moe_testcase.inter_size % 128 != 0
            ):
                continue

            test_cases.append(
                [
                    moe_type,
                    common_moe_testcase.num_tokens_list,
                    common_moe_testcase.hidden_size,
                    common_moe_testcase.inter_size,
                    common_moe_testcase.topk,
                    common_moe_testcase.num_experts,
                    common_moe_testcase.tp,
                    common_moe_testcase.ep,
                    common_moe_testcase.model_name,
                    "moe_perf.txt",
                    common_moe_testcase.token_expert_distribution,
                    common_moe_testcase.power_law_alpha,
                ]
            )

    return test_cases


def _build_marlin_wna16_experts(
    local_num_experts,
    hidden_size,
    local_inter_size,
    group_size,
    num_bits,
    params_dtype,
    device,
):
    """Construct marlin-repacked int4 (W4A16) expert weights + permuted scales.

    Replicates CompressedTensorsWNA16MarlinMoEMethod.create_weights +
    process_weights_after_loading (non-actorder branch). Values are random; only
    shapes/dtypes/layout matter for latency measurement.
    """
    packed_factor = 32 // num_bits
    # Pre-repack "checkpoint" layout (Marlin backend, is_transposed):
    #   w13: (E, hidden // packed_factor, 2 * inter)
    #   w2:  (E, inter  // packed_factor, hidden)
    w13_packed = torch.randint(
        torch.iinfo(torch.int32).min,
        torch.iinfo(torch.int32).max,
        (local_num_experts, hidden_size // packed_factor, 2 * local_inter_size),
        dtype=torch.int32,
        device=device,
    )
    w2_packed = torch.randint(
        torch.iinfo(torch.int32).min,
        torch.iinfo(torch.int32).max,
        (local_num_experts, local_inter_size // packed_factor, hidden_size),
        dtype=torch.int32,
        device=device,
    )
    num_groups_w13 = hidden_size // group_size
    num_groups_w2 = local_inter_size // group_size
    w13_scale = (
        torch.rand((local_num_experts, num_groups_w13, 2 * local_inter_size), dtype=params_dtype, device=device) * 0.01
    )
    w2_scale = torch.rand((local_num_experts, num_groups_w2, hidden_size), dtype=params_dtype, device=device) * 0.01

    empty_g_idx = torch.empty((local_num_experts, 0), dtype=torch.int32, device=device)

    marlin_w13 = _vllm_ops.gptq_marlin_moe_repack(
        w13_packed,
        empty_g_idx,
        w13_packed.shape[1] * packed_factor,
        w13_packed.shape[2],
        num_bits,
    )
    marlin_w2 = _vllm_ops.gptq_marlin_moe_repack(
        w2_packed,
        empty_g_idx,
        w2_packed.shape[1] * packed_factor,
        w2_packed.shape[2],
        num_bits,
    )
    marlin_w13_scale = marlin_moe_permute_scales(
        s=w13_scale,
        size_k=marlin_w13.shape[2],
        size_n=w13_scale.shape[2],
        group_size=group_size,
    )
    marlin_w2_scale = marlin_moe_permute_scales(
        s=w2_scale,
        size_k=w2_scale.shape[1] * group_size,
        size_n=w2_scale.shape[2],
        group_size=group_size,
    )
    workspace = marlin_make_workspace_new(torch.device(device), 4)
    return marlin_w13, marlin_w2, marlin_w13_scale, marlin_w2_scale, empty_g_idx, workspace


def run_moe_marlin_wna16(
    moe_type,
    num_tokens_lists,
    hidden_size,
    inter_size,
    topk,
    num_experts,
    moe_tp_size,
    moe_ep_size,
    model_name,
    perf_filename,
    distributed="power_law",
    power_law_alpha=0.0,
    device="cuda:0",
    group_size=32,
):
    """Benchmark 4-bit W4A16 (wna16) marlin MoE and log to perf table."""
    if not _WNA16_MARLIN_AVAILABLE:
        raise ImportError("wna16 marlin MoE path is unavailable in this vLLM build.")

    num_bits = 4
    quant_type_id = _WNA16_TYPE_MAP[num_bits].id
    params_dtype = torch.float16

    torch.cuda.set_device(device)
    torch.set_default_device(device)

    local_inter_size = inter_size // moe_tp_size
    expert_map_result = determine_expert_map(moe_ep_size, 0, num_experts)
    if isinstance(expert_map_result, tuple) and len(expert_map_result) == 3:
        local_num_experts, expert_map, _ = expert_map_result
    else:
        local_num_experts, expert_map = expert_map_result  # type: ignore[misc]

    (
        marlin_w13,
        marlin_w2,
        marlin_w13_scale,
        marlin_w2_scale,
        empty_g_idx,
        workspace,
    ) = _build_marlin_wna16_experts(
        local_num_experts,
        hidden_size,
        local_inter_size,
        group_size,
        num_bits,
        params_dtype,
        device,
    )

    for num_tokens_idx, num_tokens in enumerate(num_tokens_lists):
        print("num_tokens", num_tokens, "topk", topk)
        hidden_states = torch.randn([num_tokens, hidden_size], dtype=params_dtype, device=device)

        num_iter = 10 if _is_power_law_mode(distributed) else 1
        if _is_power_law_mode(distributed):
            topk_weights_list = []
            topk_ids_list = []
            rank0_info_list = []
            batched_hidden_states_list = []
            expert_num_tokens_list = []
            for _ in range(num_iter):
                logits_result = power_law_logits_v3(
                    num_tokens,
                    num_experts,
                    topk,
                    moe_ep_size,
                    power_law_alpha,
                    use_eplb=_power_law_use_eplb(distributed),
                    return_rank0_info=(
                        _is_rank0_compact_mode(distributed)
                        or _is_batched_rank0_mode(distributed)
                    ),
                )
                rank0_info = None
                if _is_batched_rank0_mode(distributed):
                    _, rank0_info = logits_result
                    experts_per_rank = num_experts // moe_ep_size
                    selected = rank0_info["rank0_selected_slots"]
                    local_selected = selected[selected < experts_per_rank].to(torch.int64)
                    expert_num_tokens = torch.bincount(
                        local_selected,
                        minlength=local_num_experts,
                    ).to(torch.int32)
                    max_tokens_per_expert = max(int(expert_num_tokens.max().item()), 1)
                    batched_hidden_states_list.append(
                        torch.randn(
                            [local_num_experts, max_tokens_per_expert, hidden_size],
                            dtype=params_dtype,
                            device=device,
                        )
                    )
                    expert_num_tokens_list.append(expert_num_tokens.to(device))
                    rank0_info_list.append(rank0_info)
                    continue
                if _is_rank0_compact_mode(distributed):
                    _, rank0_info = logits_result
                    logits = rank0_info["rank0_logits"].to(params_dtype).to(device)
                else:
                    logits = logits_result.to(params_dtype).to(device)
                weights, ids = torch.topk(logits, topk, dim=-1)
                # fused_marlin_moe requires float32 topk weights and int32 ids.
                topk_weights_list.append(F.softmax(weights, dim=-1).to(torch.float32))
                topk_ids_list.append(ids.to(torch.int32))
                rank0_info_list.append(rank0_info)
        elif distributed == "balanced":
            actual_logits = balanced_logits(num_tokens, num_experts, topk).to(params_dtype).to(device)
            topk_weights, topk_ids = torch.topk(actual_logits, topk, dim=-1)
            topk_weights = F.softmax(topk_weights, dim=-1).to(torch.float32)
            topk_ids = topk_ids.to(torch.int32)
        else:
            raise ValueError(f"Unsupported distributed mode: {distributed}")

        def _marlin_call(tw, ti):
            return fused_marlin_moe(
                hidden_states[: tw.shape[0]],
                marlin_w13,
                marlin_w2,
                None,
                None,
                marlin_w13_scale,
                marlin_w2_scale,
                tw,
                ti,
                quant_type_id=quant_type_id,
                global_num_experts=num_experts,
                expert_map=expert_map,
                g_idx1=empty_g_idx,
                g_idx2=empty_g_idx,
                sort_indices1=empty_g_idx,
                sort_indices2=empty_g_idx,
                workspace=workspace,
                is_k_full=True,
            )

        def _batched_marlin_call(hidden, expert_num_tokens):
            return batched_fused_marlin_moe(
                hidden,
                expert_num_tokens,
                marlin_w13,
                marlin_w2,
                None,
                None,
                marlin_w13_scale,
                marlin_w2_scale,
                quant_type_id=quant_type_id,
                global_num_experts=num_experts,
                expert_map=expert_map,
                g_idx1=empty_g_idx,
                g_idx2=empty_g_idx,
                sort_indices1=empty_g_idx,
                sort_indices2=empty_g_idx,
                workspace=workspace,
                is_k_full=True,
            )

        def _block_size_m(m, e, topk_count):
            for candidate in [8, 16, 32, 48, 64]:
                if m * topk_count / e / candidate < 0.9:
                    return candidate
            return 64

        def _print_marlin_diag(idx, tw, ti, rank0_info=None):
            if os.getenv("AIC_MOE_MARLIN_DIAG", "0") != "1":
                return
            effective_m = ti.shape[0]
            block_size_m = _block_size_m(effective_m, local_num_experts, ti.shape[1])
            _, expert_ids, num_tokens_post_padded = moe_align_block_size(
                ti,
                block_size_m,
                num_experts,
                expert_map,
                ignore_invalid_experts=True,
            )
            valid_blocks = int((expert_ids >= 0).sum().item())
            details = {
                "idx": idx,
                "distribution": _distribution_label(distributed, power_law_alpha),
                "input_tokens": num_tokens,
                "effective_m": effective_m,
                "topk": ti.shape[1],
                "local_num_experts": local_num_experts,
                "global_num_experts": num_experts,
                "block_size_m": block_size_m,
                "num_tokens_post_padded": int(num_tokens_post_padded.item()),
                "valid_blocks": valid_blocks,
            }
            if rank0_info is not None:
                details.update(
                    {
                        "rank0_num_tokens": int(rank0_info["rank0_num_tokens"]),
                        "rank0_total_selections": int(rank0_info["rank0_total_selections"]),
                    }
                )
            print("[marlin_diag] " + " ".join(f"{key}={value}" for key, value in details.items()))

        def _print_batched_marlin_diag(idx, hidden, expert_num_tokens, rank0_info):
            if os.getenv("AIC_MOE_MARLIN_DIAG", "0") != "1":
                return
            max_tokens_per_expert = hidden.shape[1]
            _, expert_ids, num_tokens_post_padded = batched_moe_align_block_size(
                max_tokens_per_expert,
                64,
                expert_num_tokens,
            )
            valid_blocks = int((expert_ids >= 0).sum().item())
            details = {
                "idx": idx,
                "distribution": _distribution_label(distributed, power_law_alpha),
                "input_tokens": num_tokens,
                "batched_local_experts": hidden.shape[0],
                "max_tokens_per_expert": max_tokens_per_expert,
                "valid_expert_tokens": int(expert_num_tokens.sum().item()),
                "block_size_m": 64,
                "num_tokens_post_padded": int(num_tokens_post_padded.item()),
                "valid_blocks": valid_blocks,
                "rank0_num_tokens": int(rank0_info["rank0_num_tokens"]),
                "rank0_total_selections": int(rank0_info["rank0_total_selections"]),
            }
            print("[marlin_batched_diag] " + " ".join(f"{key}={value}" for key, value in details.items()))

        if _is_batched_rank0_mode(distributed):
            for idx, (hidden, expert_num_tokens) in enumerate(
                zip(batched_hidden_states_list, expert_num_tokens_list)
            ):
                _print_batched_marlin_diag(idx, hidden, expert_num_tokens, rank0_info_list[idx])
        elif _is_power_law_mode(distributed):
            for idx, (tw, ti) in enumerate(zip(topk_weights_list, topk_ids_list)):
                _print_marlin_diag(idx, tw, ti, rank0_info_list[idx])
        else:
            _print_marlin_diag(0, topk_weights, topk_ids)

        def run_single_iteration():
            if _is_batched_rank0_mode(distributed):
                for hidden, expert_num_tokens in zip(batched_hidden_states_list, expert_num_tokens_list):
                    _ = _batched_marlin_call(hidden, expert_num_tokens)
            elif _is_power_law_mode(distributed):
                for tw, ti in zip(topk_weights_list, topk_ids_list):
                    _ = _marlin_call(tw, ti)
            else:
                _ = _marlin_call(topk_weights, topk_ids)

        # CUDA-graph timing: real vLLM decode replays captured graphs, so the
        # per-op latency has no kernel-launch overhead. Eager timing inflates
        # small-batch marlin MoE ~3x (launch-bound), which over-predicts
        # generation_moe. Capture the num_iter marlin calls once and time replays
        # to match the profiled `marlin_moe_wna16` self-CUDA time.
        def _time_cuda_graph():
            capture_stream = torch.cuda.Stream(device=device)
            capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(capture_stream):
                for _ in range(3):
                    run_single_iteration()
            torch.cuda.current_stream().wait_stream(capture_stream)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run_single_iteration()
            torch.cuda.synchronize()

            reps = 20
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            for _ in range(reps):
                graph.replay()
            end_evt.record()
            torch.cuda.synchronize()
            return start_evt.elapsed_time(end_evt) / reps / num_iter

        try:
            latency = _time_cuda_graph()
        except torch.OutOfMemoryError:
            if num_tokens_idx > 0:
                break
            raise
        power_stats = None

        print(f"moe latency: {latency}")

        log_perf(
            item_list=[
                {
                    "moe_dtype": moe_type,
                    "num_tokens": num_tokens,
                    "hidden_size": hidden_size,
                    "inter_size": inter_size,
                    "topk": topk,
                    "num_experts": num_experts,
                    "moe_tp_size": moe_tp_size,
                    "moe_ep_size": moe_ep_size,
                    "distribution": _distribution_label(distributed, power_law_alpha),
                    "latency": latency,
                }
            ],
            framework="VLLM",
            version=vllm_version,
            device_name=torch.cuda.get_device_name(device),
            op_name="moe",
            kernel_source="vllm_marlin_moe_wna16",
            perf_filename=perf_filename,
            power_stats=power_stats,
        )


def run_moe_torch(
    moe_type,
    num_tokens_lists,
    hidden_size,
    inter_size,
    topk,
    num_experts,
    moe_tp_size,
    moe_ep_size,
    model_name,
    perf_filename,
    distributed="power_law",
    power_law_alpha=0.0,
    device="cuda:0",
):
    """Run vLLM MoE performance benchmarking"""
    if moe_type in ("int4_wo", "w4a16", "wna16"):
        return run_moe_marlin_wna16(
            "int4_wo",
            num_tokens_lists,
            hidden_size,
            inter_size,
            topk,
            num_experts,
            moe_tp_size,
            moe_ep_size,
            model_name,
            perf_filename,
            distributed=distributed,
            power_law_alpha=power_law_alpha,
            device=device,
        )

    torch.cuda.set_device(device)
    torch.set_default_device(device)

    # Configure quantization parameters
    dtype = torch.float16
    quant_config = None
    block_shape: list[int] | None = None
    a1_scale = None
    a2_scale = None

    # Calculate local number of experts
    local_inter_size = inter_size // moe_tp_size
    expert_map_result = determine_expert_map(moe_ep_size, 0, num_experts)
    if isinstance(expert_map_result, tuple) and len(expert_map_result) == 3:
        local_num_experts, expert_map, _ = expert_map_result
    else:
        # Backward compatibility with older determine_expert_map signatures
        # that return only (local_num_experts, expert_map)
        local_num_experts, expert_map = expert_map_result  # type: ignore[misc]

    # Create weight tensors
    # w1: gate + up projection weights [num_experts, 2 * inter_size, hidden_size]
    # w2: down projection weights [num_experts, hidden_size, inter_size]
    w1 = torch.randn(
        local_num_experts,
        2 * local_inter_size,
        hidden_size,
        dtype=torch.float16,
        device=device,
    )
    w2 = torch.randn(
        local_num_experts,
        hidden_size,
        local_inter_size,
        dtype=torch.float16,
        device=device,
    )

    if moe_type in ["fp8", "fp8_block"]:
        dtype = torch.float8_e4m3fn
        if moe_type == "fp8_block":
            block_shape = [128, 128]

            if per_block_cast_to_fp8 is None:
                raise ImportError("per_block_cast_to_fp8 is unavailable; fp8_block requires a newer vLLM build.")

            w1_scale_list = []
            w2_scale_list = []
            w1_q = torch.empty_like(w1, dtype=dtype)
            w2_q = torch.empty_like(w2, dtype=dtype)
            for i in range(local_num_experts):
                w1_q[i], w1_scale_i = per_block_cast_to_fp8(w1[i], block_size=block_shape, use_ue8m0=True)
                w2_q[i], w2_scale_i = per_block_cast_to_fp8(w2[i], block_size=block_shape, use_ue8m0=True)
                w1_scale_list.append(w1_scale_i)
                w2_scale_list.append(w2_scale_i)
            w1 = w1_q
            w2 = w2_q
            w1_scale = torch.stack(w1_scale_list)
            w2_scale = torch.stack(w2_scale_list)
        else:
            w1_scale = torch.randn(local_num_experts, dtype=torch.float32, device=device)
            w2_scale = torch.randn(local_num_experts, dtype=torch.float32, device=device)
            a1_scale = torch.randn(1, dtype=torch.float32, device=device)
            a2_scale = torch.randn(1, dtype=torch.float32, device=device)

        quant_config = fp8_w8a8_moe_quant_config(
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            block_shape=block_shape,
        )

    if dtype == torch.float8_e4m3fn:
        w1 = w1.to(dtype)
        w2 = w2.to(dtype)

    # Performance testing for each token count
    for num_tokens_idx, num_tokens in enumerate(num_tokens_lists):
        print("num_tokens", num_tokens)
        print("topk", topk)
        hidden_states = torch.randn([num_tokens, hidden_size]).half().to(device)

        # Generate topk_weights and topk_ids
        num_iter = 10 if _is_power_law_mode(distributed) else 1
        if _is_power_law_mode(distributed):
            topk_weights_list = []
            topk_ids_list = []

            for _ in range(num_iter):
                logits_result = power_law_logits_v3(
                    num_tokens,
                    num_experts,
                    topk,
                    moe_ep_size,
                    power_law_alpha,
                    use_eplb=_power_law_use_eplb(distributed),
                    return_rank0_info=_is_rank0_compact_mode(distributed),
                )
                if _is_rank0_compact_mode(distributed):
                    _, rank0_info = logits_result
                    logits = rank0_info["rank0_logits"].half().to(device)
                else:
                    logits = logits_result.half().to(device)
                weights, ids = torch.topk(logits, topk, dim=-1)
                topk_weights_list.append(F.softmax(weights, dim=-1))
                topk_ids_list.append(ids)

            print("actual num_tokens: ", [topk_ids.shape[0] for topk_ids in topk_ids_list])

        elif distributed == "balanced":
            actual_logits = balanced_logits(num_tokens, num_experts, topk).half().to(device)
            topk_weights, topk_ids = torch.topk(actual_logits, topk, dim=-1)
            topk_weights = F.softmax(topk_weights, dim=-1)

        else:
            raise ValueError(f"Unsupported distributed mode: {distributed}")

        num_warmups = 3
        num_runs = 6
        if _is_power_law_mode(distributed):
            num_warmups = 1
            num_runs = 1

        def run_single_iteration():
            if _is_power_law_mode(distributed):
                for i, (tw, ti) in enumerate(zip(topk_weights_list, topk_ids_list)):
                    local_num_tokens = tw.shape[0]
                    _ = fused_experts(
                        hidden_states[:local_num_tokens],
                        w1,
                        w2,
                        tw,
                        ti,
                        inplace=False,
                        quant_config=quant_config,
                        global_num_experts=num_experts,
                        expert_map=expert_map,
                    )
            else:
                _ = fused_experts(
                    hidden_states,
                    w1,
                    w2,
                    topk_weights,
                    topk_ids,
                    inplace=False,
                    quant_config=quant_config,
                    global_num_experts=num_experts,
                    expert_map=expert_map,
                )

        def run_iterations(use_cuda_graph=False):
            # Use benchmark_with_power context manager
            with benchmark_with_power(
                device=device,
                kernel_func=run_single_iteration,
                num_warmups=num_warmups,
                num_runs=num_runs,
                repeat_n=1,
            ) as results:
                pass

            return results["latency_ms"] / num_iter, results["power_stats"]

        try:
            latency, power_stats = run_iterations(use_cuda_graph=False)
        except torch.OutOfMemoryError:
            # If OOM, check if we had at least one successful run.
            if num_tokens_idx > 0:
                break
            raise

        print(f"moe latency: {latency}")

        source = "vllm_fused_moe"

        log_perf(
            item_list=[
                {
                    "moe_dtype": moe_type,
                    "num_tokens": num_tokens,
                    "hidden_size": hidden_size,
                    "inter_size": inter_size,
                    "topk": topk,
                    "num_experts": num_experts,
                    "moe_tp_size": moe_tp_size,
                    "moe_ep_size": moe_ep_size,
                    "distribution": _distribution_label(distributed, power_law_alpha),
                    "latency": latency,
                }
            ],
            framework="VLLM",
            version=vllm_version,
            device_name=torch.cuda.get_device_name(device),
            op_name="moe",
            kernel_source=source,
            perf_filename=perf_filename,
            power_stats=power_stats,
        )


if __name__ == "__main__":
    test_cases = get_moe_test_cases()
    print(f"Total test cases: {len(test_cases)}")

    for test_case in test_cases[:4]:
        print(f"Running test case: {test_case}")
        try:
            run_moe_torch(*test_case)
        except Exception as e:
            print(f"Test case failed: {test_case}")
            print(f"Error: {e}")
            continue
