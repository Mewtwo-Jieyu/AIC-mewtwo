#!/usr/bin/env bash
set -euo pipefail

# Phase454 GPU batch wrapper.
#
# Runs four sessions, strictly separated:
# A/B: vanilla N=512 reference recollects.
# C/D: B2b event-timing sessions for TP8 and dp2 bt65536 scope extension.

WORKDIR="${WORKDIR:-/mnt/shared-storage-user/zhaojieyu/backup/aic}"
OUT_ROOT="${OUT_ROOT:-docs/iter_gap_investigation/phase454_gpu_batch}"
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

run_dp2_8k_reference() {
  log "phase454 task A: dp2 8k2k N=512 vanilla reference"
  WORKDIR="${WORKDIR}" \
  OUT_ROOT="${OUT_ROOT}/recollect_dp2_8k2k" \
  TMP_ROOT="/tmp/phase454_recollect_dp2_8k2k_$$" \
  FORCE="${FORCE}" \
  PHASE_NAME="phase454_recollect_dp2_8k2k" \
  PORT="21054" \
  SCENARIO="K2.5-tp4ep8dp2-8k2k" \
  TP="4" DP="2" EP="8" \
  ISL="8000" OSL="2000" \
  MAX_NUM_BATCHED_TOKENS="8000" \
  MAX_MODEL_LEN="131072" \
  BENCH_NUM_PROMPTS="512" \
  BENCH_MAX_CONCURRENCY="128" \
  bash collector/vllm/run_phase412_arrival_sweep.sh
}

run_dp2_32k_reference() {
  log "phase454 task B: dp2 32k3k N=512 vanilla reference"
  WORKDIR="${WORKDIR}" \
  OUT_ROOT="${OUT_ROOT}/recollect_dp2_32k3k" \
  TMP_ROOT="/tmp/phase454_recollect_dp2_32k3k_$$" \
  FORCE="${FORCE}" \
  PHASE_NAME="phase454_recollect_dp2_32k3k" \
  PORT="21064" \
  SCENARIO="K2.5-tp4ep8dp2-32k3k" \
  TP="4" DP="2" EP="8" \
  ISL="32000" OSL="3000" \
  MAX_NUM_BATCHED_TOKENS="32000" \
  MAX_MODEL_LEN="262144" \
  BENCH_NUM_PROMPTS="512" \
  BENCH_MAX_CONCURRENCY="64" \
  bash collector/vllm/run_phase412_arrival_sweep.sh
}

run_tp8_b2b() {
  log "phase454 task C: TP8 8k2k B2b"
  WORKDIR="${WORKDIR}" \
  OUT_ROOT="${OUT_ROOT}/b2b_tp8_8k2k" \
  TMP_ROOT="/tmp/phase454_b2b_tp8_8k2k_$$" \
  FORCE="${FORCE}" \
  MODE="overhead_gate" \
  PORT="21074" \
  SCENARIO="K2.5-tp8ep8-8k2k" \
  TP="8" DP="1" EP="8" \
  ISL="8000" OSL="2000" \
  MAX_NUM_BATCHED_TOKENS="8000" \
  MAX_MODEL_LEN="262144" \
  BENCH_NUM_PROMPTS="512" \
  BENCH_MAX_CONCURRENCY="128" \
  bash collector/vllm/run_phase446_b2b_event_timing.sh
}

run_dp2_bt_b2b() {
  log "phase454 task D: dp2 8k2k bt65536 B2b"
  WORKDIR="${WORKDIR}" \
  OUT_ROOT="${OUT_ROOT}/b2b_dp2_8k2k_bt65536" \
  TMP_ROOT="/tmp/phase454_b2b_dp2_8k2k_bt65536_$$" \
  FORCE="${FORCE}" \
  MODE="overhead_gate" \
  PORT="21084" \
  SCENARIO="K2.5-tp4ep8dp2-8k2k-bt65536" \
  TP="4" DP="2" EP="8" \
  ISL="8000" OSL="2000" \
  MAX_NUM_BATCHED_TOKENS="65536" \
  MAX_MODEL_LEN="131072" \
  BENCH_NUM_PROMPTS="512" \
  BENCH_MAX_CONCURRENCY="128" \
  bash collector/vllm/run_phase446_b2b_event_timing.sh
}

main() {
  cd "${WORKDIR}"
  mkdir -p "${OUT_ROOT}"
  : >"${OUT_ROOT}/driver.log"
  log "phase454_gpu_batch_start tasks=${TASKS} workdir=${WORKDIR}" | tee -a "${OUT_ROOT}/driver.log"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits >"${OUT_ROOT}/gpu_inventory.txt"
  if want_task A; then run_dp2_8k_reference 2>&1 | tee -a "${OUT_ROOT}/driver.log"; fi
  if want_task B; then run_dp2_32k_reference 2>&1 | tee -a "${OUT_ROOT}/driver.log"; fi
  if want_task C; then run_tp8_b2b 2>&1 | tee -a "${OUT_ROOT}/driver.log"; fi
  if want_task D; then run_dp2_bt_b2b 2>&1 | tee -a "${OUT_ROOT}/driver.log"; fi
  log "phase454_gpu_batch_done" | tee -a "${OUT_ROOT}/driver.log"
}

main "$@"
