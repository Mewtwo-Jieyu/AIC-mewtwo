#!/bin/bash
set -e
trap 'echo "Cleaning up..."; kill 0 2>/dev/null || true' EXIT INT TERM

export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-32B"}
export HF_TOKEN=${HF_TOKEN:-"None"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"Qwen/Qwen3-32B"}
export HEAD_NODE_IP=${HEAD_NODE_IP:-"0.0.0.0"}
export ETCD_ENDPOINTS="${HEAD_NODE_IP}:2379"
export NATS_SERVER="nats://${HEAD_NODE_IP}:4222"



PREFILL_GPU=2
PREFILL_WORKERS=4
for ((w=0; w<PREFILL_WORKERS; w++)); do
  BASE=$(( w * PREFILL_GPU ))
  GPU_LIST=$(seq -s, $BASE $((BASE+PREFILL_GPU-1)))
  ( CUDA_VISIBLE_DEVICES=$GPU_LIST python3 -m dynamo.trtllm \
    --model-path "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --extra-engine-args "/workspace/engine_configs/prefill_config.yaml" \
    --disaggregation-mode prefill \
 2>&1 | sed "s/^/[Prefill $w] /" ) &
done

wait
