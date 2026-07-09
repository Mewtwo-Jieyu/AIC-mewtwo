#!/usr/bin/env bash
set -euo pipefail

# Phase451-I DEBUG ramp observer.
#
# Measurement-only: no profiler, no vLLM patch, no simulator/runtime/PerfDB
# changes. This reruns the DP2 8k2k reference ramp with DEBUG logging so the
# first-divergence path can inspect EngineCore waiting/victim/admission records.

export VLLM_ENABLE_CUDA_COMPATIBILITY="${VLLM_ENABLE_CUDA_COMPATIBILITY:-1}"
export PATH="/opt/py3/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/kubebrain:/usr/local/nvidia/bin"
export LD_LIBRARY_PATH="/nccl/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.9/compat:/usr/local/lib/python3.12/dist-packages/torch/lib"

export PHASE_NAME="${PHASE_NAME:-phase451i_debug_ramp}"
export OUT_ROOT="${OUT_ROOT:-docs/iter_gap_investigation/phase451i_debug_ramp}"
export TMP_ROOT="${TMP_ROOT:-/tmp/phase451i_debug_ramp_$$}"
export PORT="${PORT:-20951}"
export FORCE="${FORCE:-0}"

export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-DEBUG}"

export SCENARIO="${SCENARIO:-K2.5-tp4ep8dp2-8k2k}"
export TP="${TP:-4}"
export DP="${DP:-2}"
export EP="${EP:-8}"
export ISL="${ISL:-8000}"
export OSL="${OSL:-2000}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8000}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"

export BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-128}"
export BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-128}"
export BENCH_WARMUP_REQUESTS="${BENCH_WARMUP_REQUESTS:-0}"
export BENCH_TIMEOUT_S="${BENCH_TIMEOUT_S:-7200}"

exec "$(dirname "$0")/run_phase412_arrival_sweep.sh" "$@"
