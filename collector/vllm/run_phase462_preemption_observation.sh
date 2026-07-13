#!/usr/bin/env bash
set -euo pipefail

# Phase462 Step2b: logging-only preemption decision observation.

WORKDIR="${WORKDIR:-/mnt/shared-storage-user/zhaojieyu/backup/aic}"
OUT_ROOT="${OUT_ROOT:-docs/iter_gap_investigation/phase462_preemption_observation}"
TMP_ROOT="${TMP_ROOT:-/tmp/phase462_preemption_observation_$$}"
PATCHER_REL="${PATCHER_REL:-collector/vllm/phase462_preemption_observation_patch.py}"
PATCH_BACKUP_DIR="${PATCH_BACKUP_DIR:-/tmp/phase462_preemption_observation_backups_$$}"
SCHEDULER_PATH="${SCHEDULER_PATH:-/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py}"
KV_PATH="${KV_PATH:-/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_manager.py}"
PORT="${PORT:-21462}"

SCENARIO="K2.5-tp8ep8-32k3k"
ISL=32000
OSL=1200
NUM_PROMPTS=128
CONCURRENCY=128
MAX_BT=32000
MAX_MODEL_LEN=262144
MAX_OVERHEAD_PCT=2.0

export PATH="/opt/py3/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/kubebrain:/usr/local/nvidia/bin"
export LD_LIBRARY_PATH="/nccl/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.9/compat:/usr/local/lib/python3.12/dist-packages/torch/lib"
export VLLM_ENABLE_CUDA_COMPATIBILITY=1

PATCH_APPLIED=0

restore_patch() {
  if [[ "${PATCH_APPLIED}" == "1" ]]; then
    python3 "${PATCHER_REL}" \
      --scheduler-target "${SCHEDULER_PATH}" \
      --kv-target "${KV_PATH}" \
      --backup-dir "${PATCH_BACKUP_DIR}" \
      --restore || true
    PATCH_APPLIED=0
  fi
}

trap restore_patch EXIT

run_variant() {
  local variant="$1"
  local variant_root="${OUT_ROOT}/overhead_${variant}"
  local variant_tmp="${TMP_ROOT}/overhead_${variant}"
  WORKDIR="${WORKDIR}" \
  PHASE_NAME="phase462_preemption_observation_${variant}" \
  OUT_ROOT="${variant_root}" \
  TMP_ROOT="${variant_tmp}" \
  PORT="${PORT}" \
  FORCE=1 \
  SCENARIO="${SCENARIO}" \
  TP=8 DP=1 EP=8 \
  ISL="${ISL}" OSL="${OSL}" \
  MAX_NUM_BATCHED_TOKENS="${MAX_BT}" \
  BENCH_NUM_PROMPTS="${NUM_PROMPTS}" \
  BENCH_MAX_CONCURRENCY="${CONCURRENCY}" \
  MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  BENCH_TIMEOUT_S=3600 \
  "$(dirname "$0")/run_phase412_arrival_sweep.sh"
}

self_check() {
  python3 - "${OUT_ROOT}/preemption_observation.jsonl" "${OUT_ROOT}/self_check.json" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
output = Path(sys.argv[2])
rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
failures = [row for row in rows if row.get("kind") == "allocate_failure"]
decisions = [row for row in rows if row.get("kind") == "preempt_decision"]
pairs = []
used = set()
for decision in decisions:
    candidates = [
        (idx, row)
        for idx, row in enumerate(failures)
        if idx not in used
        and row.get("pid") == decision.get("pid")
        and row.get("trigger_request_id") == decision.get("trigger_request_id")
        and int(row.get("ts_ns", 0)) <= int(decision.get("ts_ns", 0))
    ]
    if not candidates:
        continue
    idx, failure = max(candidates, key=lambda item: int(item[1].get("ts_ns", 0)))
    used.add(idx)
    pairs.append({"allocate_failure": failure, "preempt_decision": decision})

required_failure = {"ts_ns", "trigger_request_id", "requested_blocks", "free_blocks"}
required_decision = {
    "ts_ns",
    "trigger_request_id",
    "victim_request_id",
    "victim_position",
    "running_count_after_pop",
    "waiting_count",
    "free_blocks_before_victim_free",
}
complete = [
    pair
    for pair in pairs
    if required_failure <= pair["allocate_failure"].keys()
    and required_decision <= pair["preempt_decision"].keys()
    and all(pair["allocate_failure"].get(key) is not None for key in required_failure)
    and all(pair["preempt_decision"].get(key) is not None for key in required_decision)
]
payload = {
    "schema": "phase462_preemption_observation_self_check_v1",
    "allocate_failure_rows": len(failures),
    "preempt_decision_rows": len(decisions),
    "paired_rows": len(pairs),
    "complete_pairs": len(complete),
    "passed": len(complete) > 0,
    "first_complete_pair": complete[0] if complete else None,
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
raise SystemExit(0 if payload["passed"] else 1)
PY
}

write_overhead_gate() {
  python3 - \
    "${OUT_ROOT}/overhead_off/${SCENARIO}/bench_result.json" \
    "${OUT_ROOT}/overhead_on/${SCENARIO}/bench_result.json" \
    "${OUT_ROOT}/overhead_gate.json" \
    "${MAX_OVERHEAD_PCT}" <<'PY'
import json
import sys
from pathlib import Path

off = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["output_tok_s"]
on = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))["output_tok_s"]
limit = float(sys.argv[4])
delta_pct = abs(on - off) / off * 100.0
payload = {
    "gate": "phase462_logging_only_overhead",
    "metric": "output_tok_s",
    "off_output_tok_s": off,
    "on_output_tok_s": on,
    "absolute_delta_pct": delta_pct,
    "max_overhead_pct": limit,
    "passed": delta_pct <= limit,
}
Path(sys.argv[3]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
raise SystemExit(0 if payload["passed"] else 1)
PY
}

main() {
  cd "${WORKDIR}"
  mkdir -p "${OUT_ROOT}" "${TMP_ROOT}"
  cat >"${OUT_ROOT}/meta.json" <<EOF
{
  "phase": "phase462_preemption_observation",
  "scenario": "${SCENARIO}",
  "isl": ${ISL},
  "osl": ${OSL},
  "num_prompts": ${NUM_PROMPTS},
  "concurrency": ${CONCURRENCY},
  "max_num_batched_tokens": ${MAX_BT},
  "runtime_change": "temporary_logging_patch_only",
  "diagnostic_only": true,
  "valid_for_default": false,
  "perf_database": false
}
EOF
  sha256sum "${PATCHER_REL}" >"${OUT_ROOT}/patcher.sha256"

  run_variant off

  python3 "${PATCHER_REL}" \
    --scheduler-target "${SCHEDULER_PATH}" \
    --kv-target "${KV_PATH}" \
    --backup-dir "${PATCH_BACKUP_DIR}" \
    --apply
  PATCH_APPLIED=1
  python3 "${PATCHER_REL}" \
    --scheduler-target "${SCHEDULER_PATH}" \
    --kv-target "${KV_PATH}" \
    --backup-dir "${PATCH_BACKUP_DIR}" \
    --check
  sha256sum "${SCHEDULER_PATH}" >"${OUT_ROOT}/scheduler_patched.sha256"
  sha256sum "${KV_PATH}" >"${OUT_ROOT}/kv_cache_manager_patched.sha256"

  export AIC_PHASE462_PREEMPT_JSONL="${OUT_ROOT}/preemption_observation.jsonl"
  : >"${AIC_PHASE462_PREEMPT_JSONL}"
  run_variant on
  self_check
  write_overhead_gate

  restore_patch
  sha256sum "${SCHEDULER_PATH}" >"${OUT_ROOT}/scheduler_restored.sha256"
  sha256sum "${KV_PATH}" >"${OUT_ROOT}/kv_cache_manager_restored.sha256"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
    >"${OUT_ROOT}/gpu_compute_apps_after.txt" 2>/dev/null || true
  pgrep -af "vllm.entrypoints.cli.main serve|VLLM::|raylet|gcs_server|run_openai_fixed_shape_benchmark" \
    >"${OUT_ROOT}/process_residual_after.txt" || true
}

main "$@"
