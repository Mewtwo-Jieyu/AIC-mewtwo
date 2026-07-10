#!/usr/bin/env bash
set -euo pipefail

# Phase461 GPU batch: one committed MLA microbenchmark and three isolated
# serving-state sessions. Task D is diagnostic-only and must not enter PerfDB.

WORKDIR="${WORKDIR:-/mnt/shared-storage-user/zhaojieyu/backup/aic}"
OUT_ROOT="${OUT_ROOT:-docs/iter_gap_investigation/phase461_cost_recollect}"
TASKS="${TASKS:-A B C D}"
FORCE="${FORCE:-0}"
COMMITTED_COLLECTOR_ROOT="${COMMITTED_COLLECTOR_ROOT:-/tmp/phase461_committed_collector}"
PHASE461_MLA_COLLECTOR="${PHASE461_MLA_COLLECTOR:-/tmp/phase461_upload/collect_phase461_mla_grid.py}"

export VLLM_ENABLE_CUDA_COMPATIBILITY="${VLLM_ENABLE_CUDA_COMPATIBILITY:-1}"
export PATH="/opt/py3/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/kubebrain:/usr/local/nvidia/bin"
export LD_LIBRARY_PATH="/nccl/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.9/compat:/usr/local/lib/python3.12/dist-packages/torch/lib"

log() { echo "[$(date -Is)] $*"; }

want_task() {
  local name="$1"
  [[ " ${TASKS} " == *" ${name} "* ]]
}

gpu_apps() {
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
}

require_idle() {
  local apps
  apps="$(gpu_apps)"
  if [[ -n "${apps}" ]]; then
    log "gpu_not_idle=${apps}"
    return 1
  fi
}

require_empty_file() {
  local path="$1"
  [[ -f "${path}" && ! -s "${path}" ]]
}

assert_b2b_result() {
  local root="$1"
  python3 - "${root}/overhead_gate.json" <<'PY'
import json
import sys

gate = json.load(open(sys.argv[1], encoding="utf-8"))
if not gate.get("passed") or float(gate["overhead_pct"]) > 2.0:
    raise SystemExit(f"B2b overhead gate failed: {gate}")
PY
  [[ -s "${root}/overhead_on/event_timing.jsonl" ]]
  require_empty_file "${root}/overhead_on/process_residual_after.txt"
  require_empty_file "${root}/overhead_on/gpu_compute_apps_after.txt"
}

run_mla_grid() {
  log "phase461 task A: committed MLA exact grid"
  [[ -f "${COMMITTED_COLLECTOR_ROOT}/collector_commit.txt" ]]
  [[ -f "${COMMITTED_COLLECTOR_ROOT}/collector/vllm/collect_mla.py" ]]
  local out="${OUT_ROOT}/mla_decode_grid"
  mkdir -p "${out}"
  require_idle
  COMMITTED_COLLECTOR_ROOT="${COMMITTED_COLLECTOR_ROOT}" \
    python3 "${PHASE461_MLA_COLLECTOR}" --out-dir "${out}" \
    2>&1 | tee "${out}/collector.log"
  gpu_apps >"${out}/gpu_compute_apps_after.txt"
  require_empty_file "${out}/gpu_compute_apps_after.txt"
  cp -f "${COMMITTED_COLLECTOR_ROOT}/collector_commit.txt" "${out}/collector_commit.txt"
}

run_b2b() {
  local label="$1" port="$2" scenario="$3" tp="$4" dp="$5"
  local isl="$6" osl="$7" max_bt="$8" max_model_len="$9"
  local out="${OUT_ROOT}/${label}"
  require_idle
  WORKDIR="${WORKDIR}" \
  OUT_ROOT="${out}" \
  TMP_ROOT="/tmp/phase461_${label}_$$" \
  PATCH_BACKUP_PATH="/tmp/gpu_model_runner.py.phase461.${label}.bak" \
  FORCE="${FORCE}" MODE="overhead_gate" PORT="${port}" \
  SCENARIO="${scenario}" TP="${tp}" DP="${dp}" EP="8" \
  ISL="${isl}" OSL="${osl}" MAX_NUM_BATCHED_TOKENS="${max_bt}" \
  MAX_MODEL_LEN="${max_model_len}" BENCH_NUM_PROMPTS="512" \
  BENCH_MAX_CONCURRENCY="128" \
  bash collector/vllm/run_phase446_b2b_event_timing.sh
  assert_b2b_result "${out}"
}

run_tp8_bt_b2b() {
  log "phase461 task B: tp8 bt65536 mixed serving-state"
  run_b2b "tp8_bt65536_mixed" "21210" "K2.5-tp8ep8-8k2k-bt65536" \
    "8" "1" "8000" "2000" "65536" "262144"
}

run_tp8_32k_b2b() {
  log "phase461 task C: tp8 32k3k mixed serving-state"
  run_b2b "tp8_32k3k_mixed" "21220" "K2.5-tp8ep8-32k3k" \
    "8" "1" "32000" "3000" "32000" "262144"
}

run_dp2_bt_diagnostic() {
  log "phase461 task D: dp2 bt65536 paired-rank diagnostic"
  local out="${OUT_ROOT}/dp2_bt65536_diagnostic"
  run_b2b "dp2_bt65536_diagnostic" "21230" "K2.5-tp4ep8dp2-8k2k-bt65536" \
    "4" "2" "8000" "2000" "65536" "131072"
  python3 - "${out}/ingest_policy.json" <<'PY'
import json
import sys

payload = {
    "measurement": "paired_rank_event_plus_iteration_wall",
    "diagnostic_only": True,
    "perf_database": False,
    "valid_for_default": False,
    "reason": "single forward_total row cannot represent the observed DP phase spread",
}
open(sys.argv[1], "w", encoding="utf-8").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

main() {
  cd "${WORKDIR}"
  mkdir -p "${OUT_ROOT}"
  touch "${OUT_ROOT}/driver.log"
  cp -f "$0" "${OUT_ROOT}/run_phase461_cost_recollect.sh"
  cp -f "${PHASE461_MLA_COLLECTOR}" "${OUT_ROOT}/collect_phase461_mla_grid.py"
  sha256sum "${OUT_ROOT}/run_phase461_cost_recollect.sh" \
    "${OUT_ROOT}/collect_phase461_mla_grid.py" >"${OUT_ROOT}/collector_files.sha256"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits \
    >"${OUT_ROOT}/gpu_inventory.txt"
  log "phase461_cost_recollect_start tasks=${TASKS}" | tee -a "${OUT_ROOT}/driver.log"
  if want_task A; then run_mla_grid 2>&1 | tee -a "${OUT_ROOT}/driver.log"; fi
  if want_task B; then run_tp8_bt_b2b 2>&1 | tee -a "${OUT_ROOT}/driver.log"; fi
  if want_task C; then run_tp8_32k_b2b 2>&1 | tee -a "${OUT_ROOT}/driver.log"; fi
  if want_task D; then run_dp2_bt_diagnostic 2>&1 | tee -a "${OUT_ROOT}/driver.log"; fi
  require_idle
  gpu_apps >"${OUT_ROOT}/gpu_compute_apps_after.txt"
  require_empty_file "${OUT_ROOT}/gpu_compute_apps_after.txt"
  log "phase461_cost_recollect_done" | tee -a "${OUT_ROOT}/driver.log"
}

main "$@"
