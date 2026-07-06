import json
import os
import socket
import statistics
import time
import traceback
from pathlib import Path

OUT = Path(os.environ["PHASE431_OUT"])
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--moonshotai--Kimi-K2.5",
)
BUCKETS = [8, 16, 32, 64, 128]
HIDDEN_SIZE = 7168
NUM_EXPERTS = 384
TP_SIZE = 4
DP_SIZE = 2
EXPECTED_BACKEND = "allgather_reducescatter"
EXPECTED_MANAGER = "AgRsAll2AllManager"


def normalize_backend(configured_backend: str, manager_class: str) -> str:
    if configured_backend == EXPECTED_BACKEND and manager_class == EXPECTED_MANAGER:
        return EXPECTED_BACKEND
    if configured_backend in {
        "deepep_high_throughput",
        "deepep_low_latency",
        "flashinfer_all2allv",
        "flashinfer_nvlink_two_sided",
        "flashinfer_nvlink_one_sided",
        "nixl_ep",
        "mori",
        "naive",
    }:
        return configured_backend
    if manager_class:
        return manager_class
    return "unknown_backend"


def write_rank_error(rank: int, exc: BaseException) -> None:
    payload = {
        "rank": rank,
        "error_type": exc.__class__.__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "timestamp_unix": time.time(),
    }
    (OUT / f"phase431_ep_rank_{rank}_error.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 8:
        raise RuntimeError(f"expected WORLD_SIZE=8, got {world_size}")

    import torch
    import vllm
    import torch.distributed as dist
    from vllm.config import ModelConfig, ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, parallel_state
    from vllm.forward_context import get_forward_context, set_forward_context

    torch.cuda.set_device(local_rank)

    parallel_config = ParallelConfig(
        pipeline_parallel_size=1,
        tensor_parallel_size=TP_SIZE,
        data_parallel_size=DP_SIZE,
        data_parallel_size_local=DP_SIZE,
        enable_expert_parallel=True,
        is_moe_model=True,
        distributed_executor_backend="external_launcher",
        all2all_backend=os.environ.get("PHASE431_ALL2ALL_BACKEND", EXPECTED_BACKEND),
    )
    model_config = ModelConfig(
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=262144,
        skip_tokenizer_init=True,
    )
    vllm_config = VllmConfig(model_config=model_config, parallel_config=parallel_config)

    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            backend="nccl",
        )
        parallel_state.ensure_model_parallel_initialized(
            tensor_model_parallel_size=TP_SIZE,
            pipeline_model_parallel_size=1,
            backend="nccl",
        )

        ep_group = parallel_state.get_ep_group()
        dp_group = parallel_state.get_dp_group()
        tp_group = parallel_state.get_tp_group()
        communicator = ep_group.device_communicator
        manager = getattr(communicator, "all2all_manager", None)
        manager_class = manager.__class__.__name__ if manager is not None else ""
        configured_backend = getattr(communicator, "all2all_backend", parallel_config.all2all_backend)
        backend = normalize_backend(str(configured_backend), manager_class)
        if backend != EXPECTED_BACKEND:
            raise RuntimeError(f"backend_drift:{backend}")
        if manager_class != EXPECTED_MANAGER:
            raise RuntimeError(f"manager_drift:{manager_class}")

        is_sequence_parallel = bool(parallel_config.use_sequence_parallel_moe)
        sp_size = TP_SIZE if is_sequence_parallel else 1
        results = []

        for num_tokens in BUCKETS:
            local_tokens = (num_tokens + sp_size - 1) // sp_size
            hidden_states = torch.randn(local_tokens, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
            router_logits = torch.randn(local_tokens, NUM_EXPERTS, device="cuda", dtype=torch.bfloat16)
            with set_forward_context(None, vllm_config, num_tokens=num_tokens):
                ctx = get_forward_context()
                if ctx.dp_metadata is None:
                    raise RuntimeError("dp_metadata_missing")
                with ctx.dp_metadata.sp_local_sizes(sp_size):
                    warm_hidden, _ = ep_group.dispatch_router_logits(
                        hidden_states,
                        router_logits,
                        is_sequence_parallel,
                    )
                    warm_combined = ep_group.combine(warm_hidden, is_sequence_parallel)
                    if not bool(torch.isfinite(warm_combined).all().item()):
                        raise RuntimeError(f"non_finite_warmup_bucket_{num_tokens}")
                    torch.cuda.synchronize()
                    latencies = []
                    for _ in range(20):
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        dispatched_hidden, _ = ep_group.dispatch_router_logits(
                            hidden_states,
                            router_logits,
                            is_sequence_parallel,
                        )
                        combined_hidden = ep_group.combine(dispatched_hidden, is_sequence_parallel)
                        end.record()
                        end.synchronize()
                        if not bool(torch.isfinite(combined_hidden).all().item()):
                            raise RuntimeError(f"non_finite_output_bucket_{num_tokens}")
                        latencies.append(float(start.elapsed_time(end)))

            results.append(
                {
                    "bucket_tokens": num_tokens,
                    "rank": rank,
                    "local_rank": local_rank,
                    "latency_ms_median": float(statistics.median(latencies)),
                    "latency_ms_samples": latencies,
                    "shape": (
                        "num_tokens=%d,local_tokens=%d,hidden_size=7168,"
                        "num_experts=384,top_k=8,dtype=bfloat16"
                    )
                    % (num_tokens, local_tokens),
                    "hidden_in_shape": "x".join(str(x) for x in hidden_states.shape),
                    "router_in_shape": "x".join(str(x) for x in router_logits.shape),
                    "dispatched_hidden_shape": "x".join(str(x) for x in dispatched_hidden.shape),
                    "combined_shape": "x".join(str(x) for x in combined_hidden.shape),
                    "output_all_finite": True,
                }
            )
            dist.barrier()

        payload = {
            "rank": rank,
            "local_rank": local_rank,
            "worker": socket.gethostname(),
            "hostname": socket.gethostname(),
            "vllm_version": vllm.__version__,
            "source_root": str(Path(vllm.__file__).resolve().parent),
            "measurement_boundary": "vllm_ep_group_dispatch_router_logits_plus_combine",
            "configured_backend": str(configured_backend),
            "backend": backend,
            "manager_class": manager_class,
            "world_size": world_size,
            "tp_group_size": tp_group.world_size,
            "dp_group_size": dp_group.world_size,
            "ep_group_size": ep_group.world_size,
            "ep_rank": ep_group.rank_in_group,
            "dp_rank": dp_group.rank_in_group,
            "tp_rank": tp_group.rank_in_group,
            "is_sequence_parallel": is_sequence_parallel,
            "sp_size": sp_size,
            "runtime_dispatch_scope": "fixed_model_config_hw_vllm_version_tuple",
            "bucket_results": results,
            "timestamp_unix": time.time(),
        }
        (OUT / f"phase431_ep_rank_{rank}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        dist.barrier()
        try:
            parallel_state.destroy_model_parallel()
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        write_rank_error(int(os.environ.get("RANK", "-1")), exc)
        raise
