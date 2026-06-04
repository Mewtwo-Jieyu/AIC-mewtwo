# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import math
import os
import time

import torch
import vllm
from vllm.config import set_current_vllm_config
from vllm.platforms import current_platform
from vllm.version import __version__ as vllm_version

from collector.common_test_cases import get_context_mla_common_test_cases, get_generation_mla_common_test_cases
from collector.helper import get_sm_version, log_perf
from collector.vllm.utils import (
    BatchSpec,
    MockAttentionLayer,
    _Backend,
    convert_dtype_to_torch,
    create_and_prepopulate_kv_cache_mla,
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    create_vllm_config,
    get_attention_backend,
    resolve_obj_by_qualname,
    setup_distributed,
    with_exit_stack,
)


@with_exit_stack
def run_attention_torch(
    exit_stack,
    batch_size,
    input_len,
    num_heads,
    tp_size,
    q_lora_rank,
    kv_lora_rank,
    qk_rope_head_dim,
    qk_nope_head_dim,
    v_head_dim,
    block_size,
    model_name,
    use_fp8_kv_cache,
    is_context_phase,
    perf_filename,
    device="cuda:0",
):
    setup_distributed(device)
    torch.cuda.set_device(device)

    assert num_heads % tp_size == 0, "num_heads must be divisible by tp_size"
    num_heads = num_heads // tp_size

    dtype = torch.bfloat16
    model = os.path.join(os.path.dirname(__file__), "fake_mla_hf_model")
    head_dim = kv_lora_rank + qk_rope_head_dim
    num_kv_heads = num_heads

    num_kv_cache_blocks = max(
        # Number of kv cache blocks needed for number of tokens in the entire KV cache.
        # Add +1 because VLLM considers the 1st block to be the "null" block.
        1 + math.ceil((input_len + 1) / block_size) * batch_size,
        # set a reasonable minimum
        8192,
    )
    try:
        # Let vllm choose the backend.
        # defautl for vllm 0.11.0
        backend = current_platform.get_attn_backend_cls(
            None,
            head_dim,
            dtype,
            kv_cache_dtype="fp8" if use_fp8_kv_cache else None,
            block_size=block_size,
            use_v1=True,
            use_mla=True,
            has_sink=False,
            use_sparse=False,
        )
    except TypeError:
        try:
            # in the case of vllm 0.12.0 use_v1 is removed
            backend = current_platform.get_attn_backend_cls(
                None,
                head_dim,
                dtype,
                kv_cache_dtype="fp8" if use_fp8_kv_cache else None,
                block_size=block_size,
                use_mla=True,
                has_sink=False,
                use_sparse=False,
            )
        except TypeError:
            # vllm 0.14.0
            from vllm.v1.attention.selector import AttentionSelectorConfig

            attn_selector_config = AttentionSelectorConfig(
                head_size=head_dim,
                dtype=dtype,
                kv_cache_dtype="fp8" if use_fp8_kv_cache else None,
                block_size=block_size,
                use_mla=True,
                has_sink=False,
                use_sparse=False,
            )
            backend = current_platform.get_attn_backend_cls(None, attn_selector_config)

    if _Backend is not None:
        backend_name = _Backend[resolve_obj_by_qualname(backend).get_name()]
        print(f"VLLM chose MLA backend: {backend_name}")
        builder_cls, impl_cls = get_attention_backend(backend_name)
    else:
        backend_cls = resolve_obj_by_qualname(backend)
        backend_name = backend_cls.get_name()
        print(f"VLLM chose MLA backend: {backend_name}")
        builder_cls = backend_cls.get_builder_cls()
        impl_cls = backend_cls.get_impl_cls()

    if is_context_phase:
        batch_spec = BatchSpec(
            seq_lens=[input_len] * batch_size,
            query_lens=[input_len] * batch_size,
        )
    else:
        batch_spec = BatchSpec(
            seq_lens=[input_len] * batch_size,
            query_lens=[1] * batch_size,
        )

    try:
        vllm.utils.torch_utils.set_random_seed(42)
    except AttributeError:
        current_platform.seed_everything(42)

    vllm_config = create_vllm_config(
        model_name=model,
        max_model_len=max(batch_spec.seq_lens),
        block_size=block_size,
        num_gpu_blocks=num_kv_cache_blocks,
        max_num_seqs=batch_size,
        use_fp8_kv_cache=use_fp8_kv_cache,
    )
    assert convert_dtype_to_torch(vllm_config.model_config.dtype) == torch.bfloat16

    kv_cache_spec = create_standard_kv_cache_spec(vllm_config, use_fp8_kv_cache)

    # Generate data and compute SDPA reference output
    all_q_vllm, all_kv_c_vllm, all_k_pe_vllm = [], [], []
    kv_c_contexts, k_pe_contexts = [], []

    for i in range(batch_size):
        s_len = batch_spec.seq_lens[i]
        q_len = batch_spec.query_lens[i]
        context_len = s_len - q_len

        # Generate MLA tensors
        # Q has both nope and rope components:
        # [q_len, num_heads, qk_nope_head_dim + qk_rope_head_dim]
        q_c = torch.randn(q_len, num_heads, qk_nope_head_dim + qk_rope_head_dim, dtype=dtype, device=device)

        # KV_C (latent K/V): [s_len, kv_lora_rank]
        kv_c_full = torch.randn(s_len, kv_lora_rank, dtype=dtype, device=device)

        # K_PE (rope component): [s_len, 1, qk_rope_head_dim]
        k_pe_full = torch.randn(s_len, 1, qk_rope_head_dim, dtype=dtype, device=device)

        # Inputs for vLLM MLA backends are just the new tokens
        all_q_vllm.append(q_c)
        all_kv_c_vllm.append(kv_c_full[context_len:])  # New kv_c tokens
        all_k_pe_vllm.append(k_pe_full[context_len:])  # New k_pe tokens

        # Contextual K/V data used to populate the paged cache (MLA format)
        kv_c_contexts.append(kv_c_full[:context_len])
        k_pe_contexts.append(k_pe_full[:context_len])

    query_vllm = torch.cat(all_q_vllm, dim=0)
    kv_c_vllm = torch.cat(all_kv_c_vllm, dim=0)
    k_pe_vllm = torch.cat(all_k_pe_vllm, dim=0)

    common_attn_metadata = create_common_attn_metadata(batch_spec, vllm_config.cache_config.block_size, device)

    # 3. Simulate Paged KV Cache and a realistic slot_mapping
    kv_cache = create_and_prepopulate_kv_cache_mla(
        kv_c_contexts=all_kv_c_vllm,
        k_pe_contexts=all_k_pe_vllm,
        block_size=block_size,
        head_size=head_dim,
        dtype=current_platform.fp8_dtype() if use_fp8_kv_cache else dtype,
        device=device,
        num_blocks=num_kv_cache_blocks,
        common_attn_metadata=common_attn_metadata,
        randomize_blocks=True,
        kv_cache_dtype="fp8" if use_fp8_kv_cache else None,
    )

    # Build metadata
    layer_names = ["placeholder"]
    exit_stack.enter_context(set_current_vllm_config(vllm_config))

    builder = builder_cls(kv_cache_spec, layer_names, vllm_config, device)
    attn_metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common_attn_metadata,
    )

    # Create mock kv_b_proj using the same weights as reference implementation
    from vllm.model_executor.layers.linear import ColumnParallelLinear

    mock_kv_b_proj = ColumnParallelLinear(
        input_size=kv_lora_rank, output_size=num_heads * (qk_nope_head_dim + v_head_dim), bias=False
    ).to(device=device, dtype=dtype)

    # Instantiate implementation
    sliding_window = vllm_config.model_config.get_sliding_window()
    scale = 1.0 / (head_dim**0.5)

    impl = impl_cls(
        num_heads=num_heads,
        head_size=head_dim,
        scale=scale,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=sliding_window,
        kv_cache_dtype="fp8" if use_fp8_kv_cache else "auto",
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        qk_head_dim=qk_nope_head_dim + qk_rope_head_dim,
        v_head_dim=v_head_dim,
        kv_b_proj=mock_kv_b_proj,
    )

    # Process weights to create W_UK_T and W_UV attributes needed by MLA
    impl.process_weights_after_loading(dtype)

    # Create mock layer and output buffer
    mock_layer = MockAttentionLayer(device)
    output = torch.empty(
        query_vllm.shape[0],
        num_heads * v_head_dim,
        dtype=query_vllm.dtype,
        device=query_vllm.device,
    )

    # Run forward pass

    test_ite = 6
    warm_up = 3

    def run():
        impl.forward(
            mock_layer,
            query_vllm,
            kv_c_vllm,
            k_pe_vllm,
            kv_cache,
            attn_metadata,
            output=output,
        )

    # Warmup
    for i in range(warm_up):
        run()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    for i in range(test_ite):
        run()
    end_event.record()
    torch.cuda.synchronize()

    latency = start_event.elapsed_time(end_event) / test_ite
    print(f"MLA latency: {latency}")

    if is_context_phase:
        isl = input_len
        step = 0
        op_name = "context_mla"
    else:
        isl = 1
        step = input_len
        op_name = "generation_mla"

    kv_cache_dtype_str = "float16" if not use_fp8_kv_cache else "fp8"
    dtype_str = "float16"
    kernel_source = f"vllm_{backend_name}".lower()

    log_perf(
        item_list=[
            {
                "mla_dtype": dtype_str,
                "kv_cache_dtype": kv_cache_dtype_str,
                "num_heads": num_heads,
                "batch_size": batch_size,
                "isl": isl,
                "tp_size": tp_size,
                "step": step,
                "latency": latency,
            }
        ],
        framework="VLLM",
        version=vllm_version,
        device_name=torch.cuda.get_device_name(device),
        op_name=op_name,
        kernel_source=kernel_source,
        perf_filename=perf_filename,
    )


def _shape(tensor):
    return "x".join(str(dim) for dim in tensor.shape)


def _mixed_state_smoke_batch_spec() -> BatchSpec:
    return BatchSpec(
        seq_lens=[16] * 15 + [2],
        query_lens=[16] * 15 + [1],
        name="mixed_241_16_248",
    )


def _validate_mla_smoke_key(
    phase,
    topology_key,
    attention_actual_tokens,
    attention_max_query_len,
    num_tokens_padded,
):
    if (phase, attention_actual_tokens, attention_max_query_len, num_tokens_padded) != ("mixed", 241, 16, 248):
        raise SystemExit("MLA state-smoke only supports mixed 241/16/248")
    if topology_key != "tp4dp2moetp1ep8":
        raise SystemExit("MLA state-smoke only supports tp4dp2moetp1ep8")


def _percentile_ms(samples, percentile):
    ordered = sorted(samples)
    idx = min(len(ordered) - 1, round((len(ordered) - 1) * percentile / 100))
    return ordered[idx]


def _summarize_ms(samples):
    if not samples:
        raise SystemExit("MLA timing-smoke needs at least one measurement")
    return {
        "mean": sum(samples) / len(samples),
        "p50": _percentile_ms(samples, 50),
        "p99": _percentile_ms(samples, 99),
    }


def _prepare_mla_smoke_state(
    exit_stack,
    phase="mixed",
    topology_key="tp4dp2moetp1ep8",
    attention_actual_tokens=241,
    attention_max_query_len=16,
    num_tokens_padded=248,
    device="cuda:0",
):
    _validate_mla_smoke_key(
        phase,
        topology_key,
        attention_actual_tokens,
        attention_max_query_len,
        num_tokens_padded,
    )

    dtype = torch.bfloat16
    batch_spec = _mixed_state_smoke_batch_spec()
    if batch_spec.compute_num_tokens() != attention_actual_tokens:
        raise SystemExit("MLA state-smoke query token count mismatch")
    if max(batch_spec.query_lens) != attention_max_query_len:
        raise SystemExit("MLA state-smoke max query length mismatch")

    num_heads_global = 64
    tp_size = 4
    num_heads = num_heads_global // tp_size
    num_kv_heads = num_heads
    q_lora_rank = 1536
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    qk_nope_head_dim = 128
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    v_head_dim = 128
    block_size = 16
    head_dim = kv_lora_rank + qk_rope_head_dim
    num_kv_cache_blocks = 16384
    model = os.path.join(os.path.dirname(__file__), "fake_mla_hf_model")
    vllm_config = create_vllm_config(
        model_name=model,
        max_model_len=max(batch_spec.seq_lens),
        block_size=block_size,
        num_gpu_blocks=num_kv_cache_blocks,
        max_num_seqs=len(batch_spec.seq_lens),
        use_fp8_kv_cache=False,
    )

    exit_stack.enter_context(set_current_vllm_config(vllm_config))
    setup_distributed(device)
    torch.cuda.set_device(device)

    try:
        backend = current_platform.get_attn_backend_cls(
            None,
            head_dim,
            dtype,
            kv_cache_dtype=None,
            block_size=block_size,
            use_v1=True,
            use_mla=True,
            has_sink=False,
            use_sparse=False,
        )
    except TypeError:
        try:
            backend = current_platform.get_attn_backend_cls(
                None,
                head_dim,
                dtype,
                kv_cache_dtype=None,
                block_size=block_size,
                use_mla=True,
                has_sink=False,
                use_sparse=False,
            )
        except TypeError:
            from vllm.v1.attention.selector import AttentionSelectorConfig

            attn_selector_config = AttentionSelectorConfig(
                head_size=head_dim,
                dtype=dtype,
                kv_cache_dtype=None,
                block_size=block_size,
                use_mla=True,
                has_sink=False,
                use_sparse=False,
            )
            backend = current_platform.get_attn_backend_cls(None, attn_selector_config)

    if _Backend is not None:
        backend_name = _Backend[resolve_obj_by_qualname(backend).get_name()]
        builder_cls, impl_cls = get_attention_backend(backend_name)
    else:
        backend_cls = resolve_obj_by_qualname(backend)
        backend_name = backend_cls.get_name()
        builder_cls = backend_cls.get_builder_cls()
        impl_cls = backend_cls.get_impl_cls()

    try:
        vllm.utils.torch_utils.set_random_seed(42)
    except AttributeError:
        current_platform.seed_everything(42)

    assert convert_dtype_to_torch(vllm_config.model_config.dtype) == torch.bfloat16
    kv_cache_spec = create_standard_kv_cache_spec(vllm_config, use_fp8_kv_cache=False)

    all_q_vllm, all_kv_c_vllm, all_k_pe_vllm = [], [], []
    for s_len, q_len in zip(batch_spec.seq_lens, batch_spec.query_lens, strict=True):
        context_len = s_len - q_len
        q_c = torch.randn(q_len, num_heads, qk_head_dim, dtype=dtype, device=device)
        kv_c_full = torch.randn(s_len, kv_lora_rank, dtype=dtype, device=device)
        k_pe_full = torch.randn(s_len, 1, qk_rope_head_dim, dtype=dtype, device=device)
        all_q_vllm.append(q_c)
        all_kv_c_vllm.append(kv_c_full[context_len:])
        all_k_pe_vllm.append(k_pe_full[context_len:])

    query_vllm = torch.cat(all_q_vllm, dim=0)
    kv_c_vllm = torch.cat(all_kv_c_vllm, dim=0)
    k_pe_vllm = torch.cat(all_k_pe_vllm, dim=0)
    if int(query_vllm.shape[0]) != attention_actual_tokens:
        raise SystemExit("MLA state-smoke query tensor token mismatch")
    if int(kv_c_vllm.shape[0]) != attention_actual_tokens:
        raise SystemExit("MLA state-smoke kv_c tensor token mismatch")
    if int(k_pe_vllm.shape[0]) != attention_actual_tokens:
        raise SystemExit("MLA state-smoke k_pe tensor token mismatch")

    common_attn_metadata = create_common_attn_metadata(batch_spec, block_size, torch.device(device))
    common_attn_metadata.block_table_tensor = torch.zeros(
        len(batch_spec.seq_lens),
        num_kv_cache_blocks,
        dtype=torch.int32,
        device=device,
    )

    kv_cache = create_and_prepopulate_kv_cache_mla(
        kv_c_contexts=all_kv_c_vllm,
        k_pe_contexts=all_k_pe_vllm,
        block_size=block_size,
        head_size=head_dim,
        dtype=dtype,
        device=device,
        num_blocks=num_kv_cache_blocks,
        common_attn_metadata=common_attn_metadata,
        randomize_blocks=True,
        kv_cache_dtype=None,
    )

    layer_names = ["placeholder"]
    builder = builder_cls(kv_cache_spec, layer_names, vllm_config, torch.device(device))
    attn_metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common_attn_metadata,
    )

    from vllm.model_executor.layers.linear import ColumnParallelLinear

    mock_kv_b_proj = ColumnParallelLinear(
        input_size=kv_lora_rank,
        output_size=num_heads * (qk_nope_head_dim + v_head_dim),
        bias=False,
    ).to(device=device, dtype=dtype)

    impl = impl_cls(
        num_heads=num_heads,
        head_size=head_dim,
        scale=1.0 / (head_dim**0.5),
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=vllm_config.model_config.get_sliding_window(),
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        qk_head_dim=qk_head_dim,
        v_head_dim=v_head_dim,
        kv_b_proj=mock_kv_b_proj,
    )
    impl.process_weights_after_loading(dtype)
    impl.dcp_world_size = 1
    impl.dcp_rank = 0

    if attn_metadata.prefill is not None:
        kernel_entrypoint = "forward_mha"
        query_for_kernel = query_vllm
        mock_layer = None
    elif attn_metadata.decode is not None:
        kernel_entrypoint = "forward_mqa"
        query_for_kernel = torch.randn(
            query_vllm.shape[0],
            num_heads,
            head_dim,
            dtype=dtype,
            device=device,
        )
        mock_layer = MockAttentionLayer(torch.device(device))
    else:
        raise SystemExit("MLA state-smoke metadata has neither prefill nor decode path")

    return {
        "impl": impl,
        "query": query_for_kernel,
        "kv_c": kv_c_vllm,
        "k_pe": k_pe_vllm,
        "kv_cache": kv_cache,
        "attn_metadata": attn_metadata,
        "mock_layer": mock_layer,
        "kernel_entrypoint": kernel_entrypoint,
        "dtype": dtype,
        "device": device,
        "num_heads": num_heads,
        "v_head_dim": v_head_dim,
        "target_backend": type(impl).__name__,
        "phase": phase,
        "topology_key": topology_key,
        "attention_actual_tokens": attention_actual_tokens,
        "attention_max_query_len": attention_max_query_len,
        "num_tokens_padded": num_tokens_padded,
        "local_num_heads": num_heads,
        "local_num_kv_heads": num_kv_heads,
        "q_lora_rank": q_lora_rank,
        "kv_lora_rank": kv_lora_rank,
        "qk_head_dim": qk_head_dim,
        "mla_head_size": head_dim,
        "v_head_dim": v_head_dim,
        "query_shape": _shape(query_for_kernel),
        "kv_c_shape": _shape(kv_c_vllm),
        "k_pe_shape": _shape(k_pe_vllm),
        "block_table_shape": _shape(common_attn_metadata.block_table_tensor),
        "kv_cache_shape": _shape(kv_cache),
        "metadata_type": type(attn_metadata).__name__,
        "metadata_num_actual_tokens": int(attn_metadata.num_actual_tokens),
        "metadata_max_query_len": int(attn_metadata.max_query_len),
    }


def _run_mla_kernel_once(state):
    if state["kernel_entrypoint"] == "forward_mha":
        output = torch.empty(
            state["query"].shape[0],
            state["num_heads"] * state["v_head_dim"],
            dtype=state["query"].dtype,
            device=state["query"].device,
        )
        state["impl"].forward_mha(
            state["query"],
            state["kv_c"],
            state["k_pe"],
            state["kv_cache"],
            state["attn_metadata"],
            torch.ones((), dtype=torch.float32, device=state["device"]),
            output,
        )
        return output
    if state["kernel_entrypoint"] == "forward_mqa":
        output, _ = state["impl"].forward_mqa(
            state["query"],
            state["kv_cache"],
            state["attn_metadata"],
            state["mock_layer"],
        )
        return output
    raise SystemExit(f"unsupported MLA kernel entrypoint: {state['kernel_entrypoint']}")


def _build_mla_smoke_result(state, output):
    return {
        "target_backend": state["target_backend"],
        "kernel_entrypoint": state["kernel_entrypoint"],
        "phase": state["phase"],
        "topology_key": state["topology_key"],
        "attention_actual_tokens": state["attention_actual_tokens"],
        "attention_max_query_len": state["attention_max_query_len"],
        "num_tokens_padded": state["num_tokens_padded"],
        "local_num_heads": state["local_num_heads"],
        "local_num_kv_heads": state["local_num_kv_heads"],
        "q_lora_rank": state["q_lora_rank"],
        "kv_lora_rank": state["kv_lora_rank"],
        "qk_head_dim": state["qk_head_dim"],
        "mla_head_size": state["mla_head_size"],
        "v_head_dim": state["v_head_dim"],
        "query_shape": state["query_shape"],
        "kv_c_shape": state["kv_c_shape"],
        "k_pe_shape": state["k_pe_shape"],
        "block_table_shape": state["block_table_shape"],
        "kv_cache_shape": state["kv_cache_shape"],
        "metadata_type": state["metadata_type"],
        "metadata_num_actual_tokens": state["metadata_num_actual_tokens"],
        "metadata_max_query_len": state["metadata_max_query_len"],
        "output_shape": _shape(output),
        "called_attention_kernel": True,
        "called_model_forward": False,
    }


@with_exit_stack
def run_mla_state_smoke(
    exit_stack,
    phase="mixed",
    topology_key="tp4dp2moetp1ep8",
    attention_actual_tokens=241,
    attention_max_query_len=16,
    num_tokens_padded=248,
    device="cuda:0",
):
    state = _prepare_mla_smoke_state(
        exit_stack,
        phase=phase,
        topology_key=topology_key,
        attention_actual_tokens=attention_actual_tokens,
        attention_max_query_len=attention_max_query_len,
        num_tokens_padded=num_tokens_padded,
        device=device,
    )
    output = _run_mla_kernel_once(state)
    torch.cuda.synchronize()
    return _build_mla_smoke_result(state, output)


@with_exit_stack
def run_mla_timing_smoke(
    exit_stack,
    phase="mixed",
    topology_key="tp4dp2moetp1ep8",
    attention_actual_tokens=241,
    attention_max_query_len=16,
    num_tokens_padded=248,
    device="cuda:0",
    warmup_iters=20,
    measure_iters=200,
):
    if warmup_iters < 0:
        raise SystemExit("MLA timing-smoke warmup_iters must be non-negative")
    if measure_iters <= 0:
        raise SystemExit("MLA timing-smoke measure_iters must be positive")
    state = _prepare_mla_smoke_state(
        exit_stack,
        phase=phase,
        topology_key=topology_key,
        attention_actual_tokens=attention_actual_tokens,
        attention_max_query_len=attention_max_query_len,
        num_tokens_padded=num_tokens_padded,
        device=device,
    )
    if state["kernel_entrypoint"] != "forward_mqa":
        raise SystemExit("MLA timing-smoke only supports forward_mqa")

    output = None
    for _ in range(warmup_iters):
        output = _run_mla_kernel_once(state)
    torch.cuda.synchronize()

    wall_samples = []
    cuda_event_samples = []
    for _ in range(measure_iters):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_wall = time.perf_counter()
        start_event.record()
        output = _run_mla_kernel_once(state)
        end_event.record()
        torch.cuda.synchronize()
        wall_samples.append((time.perf_counter() - start_wall) * 1000.0)
        cuda_event_samples.append(start_event.elapsed_time(end_event))

    wall = _summarize_ms(wall_samples)
    cuda_event = _summarize_ms(cuda_event_samples)
    result = _build_mla_smoke_result(state, output)
    return {
        "target_backend": result["target_backend"],
        "kernel_entrypoint": result["kernel_entrypoint"],
        "phase": result["phase"],
        "topology_key": result["topology_key"],
        "attention_actual_tokens": result["attention_actual_tokens"],
        "attention_max_query_len": result["attention_max_query_len"],
        "num_tokens_padded": result["num_tokens_padded"],
        "query_shape": result["query_shape"],
        "kv_cache_shape": result["kv_cache_shape"],
        "metadata_type": result["metadata_type"],
        "output_shape": result["output_shape"],
        "warmup_iters": warmup_iters,
        "measure_iters": measure_iters,
        "wall_ms_mean": wall["mean"],
        "wall_ms_p50": wall["p50"],
        "wall_ms_p99": wall["p99"],
        "cuda_event_ms_mean": cuda_event["mean"],
        "cuda_event_ms_p50": cuda_event["p50"],
        "cuda_event_ms_p99": cuda_event["p99"],
        "called_attention_kernel": True,
        "called_model_forward": False,
    }


def _get_mla_test_cases(is_context: bool):
    test_cases = []

    kv_cache_dtype_list = [False]
    if get_sm_version() > 86:
        kv_cache_dtype_list.append(True)

    if is_context:
        common_test_cases = get_context_mla_common_test_cases()
    else:
        common_test_cases = get_generation_mla_common_test_cases()

    tp_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

    for common_mla_testcase in common_test_cases:
        for tp_size in tp_sizes:
            if common_mla_testcase.num_heads % tp_size != 0:
                continue

            for is_fp8_kv_cache in kv_cache_dtype_list:
                test_cases.append(
                    [
                        common_mla_testcase.batch_size,
                        common_mla_testcase.input_len,
                        common_mla_testcase.num_heads,
                        tp_size,
                        common_mla_testcase.q_lora_rank,
                        common_mla_testcase.kv_lora_rank,
                        common_mla_testcase.qk_rope_head_dim,
                        common_mla_testcase.qk_nope_head_dim,
                        common_mla_testcase.v_head_dim,
                        common_mla_testcase.kv_cache_block_size,
                        common_mla_testcase.model_name,
                        is_fp8_kv_cache,
                        is_context,
                        "context_mla_perf.txt" if is_context else "generation_mla_perf.txt",
                    ]
                )

    return test_cases


def get_context_mla_test_cases():
    return _get_mla_test_cases(is_context=True)


def get_generation_mla_test_cases():
    return _get_mla_test_cases(is_context=False)


if __name__ == "__main__":
    test_cases = get_context_mla_test_cases()
    test_cases = test_cases[:10]
    for test_case in test_cases:
        print(f"Running context attention test case: {test_case}")
        run_attention_torch(*test_case)

    test_cases = get_generation_mla_test_cases()
    test_cases = test_cases[:10]
    for test_case in test_cases:
        print(f"Running generation attention test case: {test_case}")
        run_attention_torch(*test_case)
