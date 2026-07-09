#!/usr/bin/env bash
set -euo pipefail

# Phase452: logging-only dual observation.
# The temporary patch records API-server routing snapshots and scheduler
# preemption/resume events. It does not change scheduling decisions.

WORKDIR="${WORKDIR:-/mnt/shared-storage-user/ailab-sys/zhaojieyu/aic}"
OUT_ROOT="${OUT_ROOT:-docs/iter_gap_investigation/phase452_dual_observation}"
TMP_ROOT="${TMP_ROOT:-/tmp/phase452_dual_observation_$$}"
PORT="${PORT:-20952}"
FORCE="${FORCE:-0}"

CORE_CLIENT_PATH="${CORE_CLIENT_PATH:-/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core_client.py}"
SCHEDULER_PATH="${SCHEDULER_PATH:-/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py}"
PATCH_BACKUP_DIR="${PATCH_BACKUP_DIR:-/tmp/phase452_dual_observation_backups}"
PATCHER_REL="${PATCHER_REL:-collector/vllm/phase452_dual_observation_patch.py}"

SCENARIO="${SCENARIO:-K2.5-tp4ep8dp2-8k2k}"
ISL="${ISL:-8000}"
OSL="${OSL:-2000}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8000}"
BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-128}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"

export PATH="/opt/py3/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/kubebrain:/usr/local/nvidia/bin"
export LD_LIBRARY_PATH="/nccl/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.9/compat:/usr/local/lib/python3.12/dist-packages/torch/lib"
export VLLM_ENABLE_CUDA_COMPATIBILITY="${VLLM_ENABLE_CUDA_COMPATIBILITY:-1}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

PATCH_APPLIED=0

log() { echo "[$(date -Is)] $*"; }

patch_apply() {
  cd "${WORKDIR}"
  python3 "${PATCHER_REL}" \
    --core-client-target "${CORE_CLIENT_PATH}" \
    --scheduler-target "${SCHEDULER_PATH}" \
    --backup-dir "${PATCH_BACKUP_DIR}" \
    --apply
  python3 "${PATCHER_REL}" \
    --core-client-target "${CORE_CLIENT_PATH}" \
    --scheduler-target "${SCHEDULER_PATH}" \
    --backup-dir "${PATCH_BACKUP_DIR}" \
    --check >/dev/null
  PATCH_APPLIED=1
  sha256sum "${CORE_CLIENT_PATH}" >"${OUT_ROOT}/core_client_patched.sha256"
  sha256sum "${SCHEDULER_PATH}" >"${OUT_ROOT}/scheduler_patched.sha256"
  sha256sum "${PATCH_BACKUP_DIR}/core_client.py.phase452.dual_observation.bak" >"${OUT_ROOT}/core_client_backup.sha256"
  sha256sum "${PATCH_BACKUP_DIR}/scheduler.py.phase452.dual_observation.bak" >"${OUT_ROOT}/scheduler_backup.sha256"
}

patch_restore() {
  if [[ "${PATCH_APPLIED}" == "1" ]]; then
    cd "${WORKDIR}"
    python3 "${PATCHER_REL}" \
      --core-client-target "${CORE_CLIENT_PATH}" \
      --scheduler-target "${SCHEDULER_PATH}" \
      --backup-dir "${PATCH_BACKUP_DIR}" \
      --restore || true
    PATCH_APPLIED=0
  fi
}

trap patch_restore EXIT

write_meta() {
  mkdir -p "${OUT_ROOT}"
  python3 - "${OUT_ROOT}/meta.json" <<PY
import json
from pathlib import Path

payload = {
    "phase": "phase452_dual_observation",
    "scenario": "${SCENARIO}",
    "isl": int("${ISL}"),
    "osl": int("${OSL}"),
    "max_num_batched_tokens": int("${MAX_NUM_BATCHED_TOKENS}"),
    "bench_num_prompts": int("${BENCH_NUM_PROMPTS}"),
    "bench_max_concurrency": int("${BENCH_MAX_CONCURRENCY}"),
    "max_model_len": int("${MAX_MODEL_LEN}"),
    "measurement": "api_route_and_scheduler_preemption_jsonl",
    "runtime_change": "temporary_logging_patch_only",
    "default_readiness": "No-Go",
}
Path("${OUT_ROOT}/meta.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\\n",
    encoding="utf-8",
)
PY
}

self_check_jsonl() {
  local path="$1"
  local out_json="$2"
  python3 - "${path}" "${out_json}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

path = Path(sys.argv[1])
out = Path(sys.argv[2])
counts = Counter()
if path.exists():
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            counts[json.loads(line).get("kind", "unknown")] += 1
payload = {
    "route_rows": counts["route"],
    "stats_overwrite_rows": counts["stats_overwrite"],
    "preempt_decision_rows": counts["preempt_decision"],
    "preempt_after_free_rows": counts["preempt_after_free"],
    "victim_reschedule_rows": counts["victim_reschedule"],
    "route_gate_passed": counts["route"] >= 120,
    "stats_gate_passed": counts["stats_overwrite"] > 0,
    "victim_gate_passed": counts["preempt_decision"] > 0,
}
payload["passed"] = (
    payload["route_gate_passed"]
    and payload["stats_gate_passed"]
    and payload["victim_gate_passed"]
)
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
raise SystemExit(0 if payload["passed"] else 1)
PY
}

main() {
  cd "${WORKDIR}"
  mkdir -p "${OUT_ROOT}" "${TMP_ROOT}"
  : >"${OUT_ROOT}/driver.log"
  write_meta

  patch_apply
  export AIC_PHASE452_OBS_JSONL="${OUT_ROOT}/dual_observation.jsonl"
  : >"${AIC_PHASE452_OBS_JSONL}"
  log "run dual observation scenario=${SCENARIO}" | tee -a "${OUT_ROOT}/driver.log"

  PHASE_NAME="phase452_dual_observation" \
  OUT_ROOT="${OUT_ROOT}" \
  TMP_ROOT="${TMP_ROOT}/run" \
  PORT="${PORT}" \
  FORCE="${FORCE}" \
  SCENARIO="${SCENARIO}" \
  ISL="${ISL}" \
  OSL="${OSL}" \
  MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS}" \
  BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS}" \
  BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY}" \
  MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  "$(dirname "$0")/run_phase425_8k2k_sweep.sh" \
    2>&1 | tee -a "${OUT_ROOT}/driver.log"

  self_check_jsonl "${AIC_PHASE452_OBS_JSONL}" "${OUT_ROOT}/self_check.json" \
    | tee -a "${OUT_ROOT}/driver.log"
}

main "$@"
