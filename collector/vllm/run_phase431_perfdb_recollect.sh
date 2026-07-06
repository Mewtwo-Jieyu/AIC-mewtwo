#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-$(pwd)}"
OUT_ROOT="${OUT_ROOT:-/mnt/shared-storage-user/ailab-sys/zhaojieyu/aic/docs/iter_gap_investigation/phase431_perfdb_recollect}"
RUN_ID="${RUN_ID:-phase431_perfdb_recollect_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_ROOT}/${RUN_ID}"
MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--moonshotai--Kimi-K2.5}"

export PATH="/usr/local/nvidia/bin:/usr/local/bin:/usr/bin:/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda-12.9/compat:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export VLLM_ENABLE_CUDA_COMPATIBILITY="${VLLM_ENABLE_CUDA_COMPATIBILITY:-1}"
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/src:${PYTHONPATH:-}"

mkdir -p "${OUT_DIR}"

snapshot_gpu_apps() {
  local path="$1"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >"${path}" 2>/dev/null || true
}

snapshot_processes() {
  local path="$1"
  ps -eo pid,comm,args \
    | grep -E 'vllm|ray|qwen|Kimi|APIServer|run-one|phase431' \
    | grep -v grep \
    | grep -v 'run_phase431_perfdb_recollect.sh' \
    >"${path}" || true
}

wait_for_no_compute_apps() {
  for _ in $(seq 1 20); do
    local apps
    apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' || true)"
    if [[ -z "${apps}" ]]; then
      return 0
    fi
    sleep 3
  done
  return 1
}

snapshot_gpu_apps "${OUT_DIR}/gpu_compute_apps_before.txt"
snapshot_processes "${OUT_DIR}/process_snapshot_before.txt"

cat >"${OUT_DIR}/phase431_collect_mla_moe.py" <<'PY'
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

OUT = Path(os.environ["PHASE431_OUT"])
MODEL_PATH = os.environ["MODEL_PATH"]
BATCHES = [8, 16, 32, 64, 128]
MLA_RAW = OUT / "phase431_generation_mla_raw.csv"
MOE_RAW = OUT / "phase431_moe_int4_wo_raw.csv"
SUMMARY = OUT / "phase431_mla_moe_summary.json"

from collector.vllm.collect_mla import run_attention_torch
from collector.vllm.collect_moe import run_moe_marlin_wna16


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _latencies_by_batch(path: Path, batch_field: str) -> dict[int, float]:
    rows = _read_rows(path)
    return {int(row[batch_field]): float(row["latency"]) for row in rows}


def _monotonic(values: dict[int, float]) -> bool:
    ordered = [values[b] for b in sorted(values)]
    return all(right >= left for left, right in zip(ordered, ordered[1:]))


def _collect_mla_once() -> None:
    if MLA_RAW.exists():
        MLA_RAW.unlink()
    for batch in BATCHES:
        run_attention_torch(
            batch_size=batch,
            input_len=8191,
            num_heads=128,
            tp_size=8,
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            qk_nope_head_dim=128,
            v_head_dim=128,
            block_size=16,
            model_name=MODEL_PATH,
            use_fp8_kv_cache=False,
            is_context_phase=False,
            perf_filename=str(MLA_RAW),
            device="cuda:0",
        )


def _collect_moe_once() -> None:
    if MOE_RAW.exists():
        MOE_RAW.unlink()
    run_moe_marlin_wna16(
        "int4_wo",
        BATCHES,
        hidden_size=7168,
        inter_size=2048,
        topk=8,
        num_experts=384,
        moe_tp_size=1,
        moe_ep_size=8,
        model_name=MODEL_PATH,
        perf_filename=str(MOE_RAW),
        distributed="power_law",
        power_law_alpha=1.01,
        device="cuda:0",
    )


def _run_with_retry(name: str, collect, path: Path, batch_field: str, *, require_monotonic: bool) -> dict:
    attempts = []
    for attempt in range(1, 4):
        collect()
        values = _latencies_by_batch(path, batch_field)
        complete = sorted(values) == BATCHES
        monotonic = _monotonic(values)
        attempts.append(
            {
                "attempt": attempt,
                "complete": complete,
                "monotonic": monotonic,
                "values": values,
            }
        )
        if complete and (monotonic or not require_monotonic):
            return {
                "name": name,
                "ok": True,
                "attempts": attempts,
                "selected_attempt": attempt,
            }
    return {
        "name": name,
        "ok": False,
        "attempts": attempts,
        "selected_attempt": None,
    }


def main() -> None:
    summary = {
        "source": "phase431_perfdb_recollect",
        "worker": subprocess.check_output(["hostname"], text=True).strip(),
        "timestamp_unix": time.time(),
        "committed_source_root": os.environ.get("SOURCE_ROOT", ""),
        "model_path": MODEL_PATH,
        "mla": _run_with_retry("mla", _collect_mla_once, MLA_RAW, "batch_size", require_monotonic=False),
        "moe": _run_with_retry("moe", _collect_moe_once, MOE_RAW, "num_tokens", require_monotonic=True),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not summary["mla"]["ok"] or not summary["moe"]["ok"]:
        raise SystemExit("phase431 MLA/MoE collection self-check failed")


if __name__ == "__main__":
    main()
PY

cat >"${OUT_DIR}/phase431_ep8_a2a_sweep.py" <<'PY'
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
PY

cat >"${OUT_DIR}/phase431_ep8_a2a_finalize.py" <<'PY'
import csv
import json
import os
from pathlib import Path

OUT = Path(os.environ["PHASE431_OUT"])
BUCKETS = [8, 16, 32, 64, 128]
EXPECTED_BACKEND = "allgather_reducescatter"
EXPECTED_MANAGER = "AgRsAll2AllManager"
torchrun_exit = int(os.environ.get("PHASE431_TORCHRUN_EXIT", "999"))


def read_text(name: str) -> str:
    path = OUT / name
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


gpu_after = read_text("gpu_compute_apps_after.txt")
residual = read_text("process_residual_after.txt")
cleanup_ok = (not gpu_after.strip()) and (not residual.strip())
error_paths = sorted(OUT.glob("phase431_ep_rank_*_error.json"))
rank_paths = sorted(OUT.glob("phase431_ep_rank_[0-7].json"))
rank_error = len(error_paths)
base_failure = []
if torchrun_exit != 0:
    base_failure.append(f"torchrun_exit_{torchrun_exit}")
if rank_error:
    first = json.loads(error_paths[0].read_text(encoding="utf-8"))
    base_failure.append(first.get("error", first.get("error_type", "rank_error")))
    (OUT / "phase431_ep_error.txt").write_text(
        "\n\n".join(p.read_text(encoding="utf-8", errors="replace") for p in error_paths),
        encoding="utf-8",
    )
if not cleanup_ok:
    base_failure.append("cleanup_residual")

rank_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in rank_paths]
if not base_failure and len(rank_payloads) != 8:
    base_failure.append(f"rank_result_count_{len(rank_payloads)}")

rows = []
for bucket in BUCKETS:
    failure = list(base_failure)
    bucket_rank_rows = []
    for payload in rank_payloads:
        for item in payload.get("bucket_results", []):
            if int(item.get("bucket_tokens", -1)) == bucket:
                bucket_rank_rows.append((payload, item))
    if not failure and len(bucket_rank_rows) != 8:
        failure.append(f"bucket_rank_count_{len(bucket_rank_rows)}")
    backends = {payload.get("backend", "") for payload, _ in bucket_rank_rows}
    managers = {payload.get("manager_class", "") for payload, _ in bucket_rank_rows}
    if not failure and backends != {EXPECTED_BACKEND}:
        failure.append("backend_drift:" + ";".join(sorted(backends)))
    if not failure and managers != {EXPECTED_MANAGER}:
        failure.append("manager_drift:" + ";".join(sorted(managers)))
    finite_values = [item.get("output_all_finite") is True for _, item in bucket_rank_rows]
    if not failure and (not finite_values or not all(finite_values)):
        failure.append("non_finite_output")
    latencies = [float(item.get("latency_ms_median", 0.0)) for _, item in bucket_rank_rows]
    rank0_payload = bucket_rank_rows[0][0] if bucket_rank_rows else {}
    rank0_item = bucket_rank_rows[0][1] if bucket_rank_rows else {}
    row = {
        "source": "phase431_perfdb_recollect",
        "row_type": "ep_a2a_raw",
        "ok": "false" if failure else "true",
        "worker": str(rank0_payload.get("worker", "")),
        "vllm_version": str(rank0_payload.get("vllm_version", "")),
        "source_root": str(rank0_payload.get("source_root", "")),
        "measurement_boundary": "vllm_ep_group_dispatch_router_logits_plus_combine",
        "bucket_tokens": str(bucket),
        "backend": next(iter(backends)) if len(backends) == 1 else ";".join(sorted(backends)),
        "manager": next(iter(managers)) if len(managers) == 1 else ";".join(sorted(managers)),
        "configured_backend": str(rank0_payload.get("configured_backend", "")),
        "shape": str(rank0_item.get("shape", f"num_tokens={bucket},hidden_size=7168,num_experts=384,top_k=8,dtype=bfloat16")),
        "latency_ms": f"{max(latencies):.6f}" if latencies else "",
        "rank_error": str(rank_error),
        "cleanup": "true" if cleanup_ok else "false",
        "gpu_process_residue": "false" if cleanup_ok else "true",
        "failure_reason": ";".join(failure),
        "runtime_dispatch_scope": "fixed_model_config_hw_vllm_version_tuple",
        "default_readiness": "No-Go",
        "diagnostic_only": "false",
        "valid_for_default": "false",
        "perf_database": "true",
    }
    rows.append(row)

with (OUT / "phase431_ep_a2a_raw.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

passed = all(row["ok"] == "true" for row in rows)
(OUT / "phase431_ep_a2a_summary.json").write_text(
    json.dumps({"ok": passed, "rows": rows}, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
if not passed:
    raise SystemExit("phase431 EP a2a self-check failed")
PY

export PHASE431_OUT="${OUT_DIR}"
export SOURCE_ROOT
export MODEL_PATH

python3 "${OUT_DIR}/phase431_collect_mla_moe.py" 2>&1 | tee "${OUT_DIR}/phase431_collect_mla_moe.log"

set +e
torchrun --standalone --nproc_per_node=8 "${OUT_DIR}/phase431_ep8_a2a_sweep.py" >"${OUT_DIR}/phase431_ep_torchrun.log" 2>&1
TORCHRUN_EXIT=$?
set -e

wait_for_no_compute_apps || true
snapshot_gpu_apps "${OUT_DIR}/gpu_compute_apps_after.txt"
snapshot_processes "${OUT_DIR}/process_residual_after.txt"

PHASE431_TORCHRUN_EXIT="${TORCHRUN_EXIT}" python3 "${OUT_DIR}/phase431_ep8_a2a_finalize.py"

cat >"${OUT_DIR}/meta.json" <<EOF
{
  "source": "phase431_perfdb_recollect",
  "run_id": "${RUN_ID}",
  "source_root": "${SOURCE_ROOT}",
  "model_path": "${MODEL_PATH}",
  "output_dir": "${OUT_DIR}",
  "default_readiness": "No-Go",
  "perf_database": true,
  "valid_for_default": false
}
EOF

echo "${OUT_DIR}"
