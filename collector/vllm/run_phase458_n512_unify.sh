#!/usr/bin/env bash
set -euo pipefail

# Phase458 N=512 unified-reference recollect wrapper.
#
# Runs four strictly separated vanilla sessions. No profiler, no B2b patch,
# no serving-state instrumentation. Each task delegates to the proven
# Phase412 vanilla runner, but pins the Phase458 protocol and output layout.

WORKDIR="${WORKDIR:-/mnt/shared-storage-user/zhaojieyu/backup/aic}"
OUT_ROOT="${OUT_ROOT:-docs/iter_gap_investigation/phase458_n512_unify}"
TASKS="${TASKS:-A B C D}"
FORCE="${FORCE:-0}"

export VLLM_ENABLE_CUDA_COMPATIBILITY="${VLLM_ENABLE_CUDA_COMPATIBILITY:-1}"
export PATH="/opt/py3/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/kubebrain:/usr/local/nvidia/bin"
export LD_LIBRARY_PATH="/nccl/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.9/compat:/usr/local/lib/python3.12/dist-packages/torch/lib"

log() { echo "[$(date -Is)] $*"; }

want_task() {
  local name="$1"
  [[ " ${TASKS} " == *" ${name} "* ]]
}

run_tp8_8k_reference() {
  log "phase458 task A: tp8 8k2k N=512 vanilla reference"
  WORKDIR="${WORKDIR}" \
  OUT_ROOT="${OUT_ROOT}/recollect_tp8_8k2k" \
  TMP_ROOT="/tmp/phase458_recollect_tp8_8k2k_$$" \
  FORCE="${FORCE}" \
  PHASE_NAME="phase458_recollect_tp8_8k2k" \
  PORT="21110" \
  SCENARIO="K2.5-tp8ep8-8k2k" \
  TP="8" DP="1" EP="8" \
  ISL="8000" OSL="2000" \
  MAX_NUM_BATCHED_TOKENS="8000" \
  MAX_MODEL_LEN="262144" \
  BENCH_NUM_PROMPTS="512" \
  BENCH_MAX_CONCURRENCY="128" \
  bash collector/vllm/run_phase412_arrival_sweep.sh
}

run_tp8_32k_reference() {
  log "phase458 task B: tp8 32k3k N=512 vanilla reference"
  WORKDIR="${WORKDIR}" \
  OUT_ROOT="${OUT_ROOT}/recollect_tp8_32k3k" \
  TMP_ROOT="/tmp/phase458_recollect_tp8_32k3k_$$" \
  FORCE="${FORCE}" \
  PHASE_NAME="phase458_recollect_tp8_32k3k" \
  PORT="21120" \
  SCENARIO="K2.5-tp8ep8-32k3k" \
  TP="8" DP="1" EP="8" \
  ISL="32000" OSL="3000" \
  MAX_NUM_BATCHED_TOKENS="32000" \
  MAX_MODEL_LEN="262144" \
  BENCH_NUM_PROMPTS="512" \
  BENCH_MAX_CONCURRENCY="128" \
  bash collector/vllm/run_phase412_arrival_sweep.sh
}

run_tp8_bt_reference() {
  log "phase458 task C: tp8 8k2k bt65536 N=512 vanilla reference"
  WORKDIR="${WORKDIR}" \
  OUT_ROOT="${OUT_ROOT}/recollect_tp8_8k2k_bt65536" \
  TMP_ROOT="/tmp/phase458_recollect_tp8_8k2k_bt65536_$$" \
  FORCE="${FORCE}" \
  PHASE_NAME="phase458_recollect_tp8_8k2k_bt65536" \
  PORT="21130" \
  SCENARIO="K2.5-tp8ep8-8k2k-bt65536" \
  TP="8" DP="1" EP="8" \
  ISL="8000" OSL="2000" \
  MAX_NUM_BATCHED_TOKENS="65536" \
  MAX_MODEL_LEN="262144" \
  BENCH_NUM_PROMPTS="512" \
  BENCH_MAX_CONCURRENCY="128" \
  bash collector/vllm/run_phase412_arrival_sweep.sh
}

run_dp2_bt_reference() {
  log "phase458 task D: dp2 8k2k bt65536 N=512 vanilla reference"
  WORKDIR="${WORKDIR}" \
  OUT_ROOT="${OUT_ROOT}/recollect_dp2_8k2k_bt65536" \
  TMP_ROOT="/tmp/phase458_recollect_dp2_8k2k_bt65536_$$" \
  FORCE="${FORCE}" \
  PHASE_NAME="phase458_recollect_dp2_8k2k_bt65536" \
  PORT="21140" \
  SCENARIO="K2.5-tp4ep8dp2-8k2k-bt65536" \
  TP="4" DP="2" EP="8" \
  ISL="8000" OSL="2000" \
  MAX_NUM_BATCHED_TOKENS="65536" \
  MAX_MODEL_LEN="131072" \
  BENCH_NUM_PROMPTS="512" \
  BENCH_MAX_CONCURRENCY="128" \
  bash collector/vllm/run_phase412_arrival_sweep.sh
}

main() {
  cd "${WORKDIR}"
  mkdir -p "${OUT_ROOT}"
  : >"${OUT_ROOT}/driver.log"
  log "phase458_n512_unify_start tasks=${TASKS} workdir=${WORKDIR}" | tee -a "${OUT_ROOT}/driver.log"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits >"${OUT_ROOT}/gpu_inventory.txt"
  if want_task A; then run_tp8_8k_reference 2>&1 | tee -a "${OUT_ROOT}/driver.log"; fi
  if want_task B; then run_tp8_32k_reference 2>&1 | tee -a "${OUT_ROOT}/driver.log"; fi
  if want_task C; then run_tp8_bt_reference 2>&1 | tee -a "${OUT_ROOT}/driver.log"; fi
  if want_task D; then run_dp2_bt_reference 2>&1 | tee -a "${OUT_ROOT}/driver.log"; fi
  log "phase458_n512_unify_done" | tee -a "${OUT_ROOT}/driver.log"
}

main "$@"
