#!/usr/bin/env bash
set -euo pipefail

# Phase462 Step2a-3b: logging-only tokenizer-to-scheduler arrival observation.

WORKDIR="${WORKDIR:-/mnt/shared-storage-user/zhaojieyu/backup/aic}"
OUT_ROOT="${OUT_ROOT:-docs/iter_gap_investigation/phase462_arrival_observation}"
TMP_ROOT="${TMP_ROOT:-/tmp/phase462_arrival_observation_$$}"
PATCHER_REL="${PATCHER_REL:-collector/vllm/phase462_arrival_observation_patch.py}"
ANALYZER_REL="${ANALYZER_REL:-scripts/analyze_phase462_arrival_observation.py}"
PATCH_BACKUP_DIR="${PATCH_BACKUP_DIR:-/tmp/phase462_arrival_observation_backups_$$}"
ASYNC_PATH="${ASYNC_PATH:-/usr/local/lib/python3.12/dist-packages/vllm/utils/async_utils.py}"
COMPLETION_PATH="${COMPLETION_PATH:-/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/completion/serving.py}"
ENGINE_PATH="${ENGINE_PATH:-/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py}"
SCHEDULER_PATH="${SCHEDULER_PATH:-/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py}"
PORT="${PORT:-21463}"
MAX_OVERHEAD_PCT=2.0
NUM_PROMPTS=512
CONCURRENCY=128
OSL=1200
REUSE_VALID_OFF="${REUSE_VALID_OFF:-0}"
REUSE_VALID_ON="${REUSE_VALID_ON:-0}"

export PATH="/opt/py3/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/kubebrain:/usr/local/nvidia/bin"
export LD_LIBRARY_PATH="/nccl/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.9/compat:/usr/local/lib/python3.12/dist-packages/torch/lib"
export VLLM_ENABLE_CUDA_COMPATIBILITY=1

PATCH_APPLIED=0

patch_args() {
  printf '%s\n' \
    --async-target "${ASYNC_PATH}" \
    --completion-target "${COMPLETION_PATH}" \
    --engine-target "${ENGINE_PATH}" \
    --scheduler-target "${SCHEDULER_PATH}" \
    --backup-dir "${PATCH_BACKUP_DIR}"
}

restore_patch() {
  if [[ "${PATCH_APPLIED}" == "1" ]]; then
    mapfile -t args < <(patch_args)
    python3 "${PATCHER_REL}" "${args[@]}" --restore || true
    PATCH_APPLIED=0
  fi
}

trap restore_patch EXIT

source_hashes() {
  local suffix="$1"
  sha256sum \
    "${ASYNC_PATH}" \
    "${COMPLETION_PATH}" \
    "${ENGINE_PATH}" \
    "${SCHEDULER_PATH}" >"${OUT_ROOT}/source_${suffix}.sha256"
}

apply_patch() {
  mapfile -t args < <(patch_args)
  python3 "${PATCHER_REL}" "${args[@]}" --apply
  PATCH_APPLIED=1
  python3 "${PATCHER_REL}" "${args[@]}" --check
}

run_variant() {
  local label="$1"
  local isl="$2"
  local max_bt="$3"
  local variant="$4"
  local scenario="K2.5-tp8ep8-${label}"
  local variant_root="${OUT_ROOT}/overhead_${variant}"
  local variant_tmp="${TMP_ROOT}/overhead_${variant}"
  local trace_prefix=""
  local observation_path=""

  local reuse_requested=0
  if [[ "${variant}" == "off" && "${REUSE_VALID_OFF}" == "1" ]]; then
    reuse_requested=1
  elif [[ "${variant}" == "on" && "${REUSE_VALID_ON}" == "1" && -s "${OUT_ROOT}/${label}.jsonl" ]]; then
    reuse_requested=1
  fi
  if [[ "${reuse_requested}" == "1" ]]; then
    if python3 - "${variant_root}/${scenario}/bench_result.json" "${NUM_PROMPTS}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
if not path.exists():
    raise SystemExit(1)
payload = json.loads(path.read_text(encoding="utf-8"))
raise SystemExit(
    0
    if int(payload.get("ok_requests", 0)) == expected
    and int(payload.get("failed_requests", 0)) == 0
    else 1
)
PY
    then
      echo "reuse_valid_${variant}=${scenario}"
      return 0
    fi
  fi

  if [[ "${variant}" == "on" ]]; then
    trace_prefix="phase462-${label}"
    observation_path="/dev/shm/phase462_arrival_observation_$$_${label}.jsonl"
    : >"${observation_path}"
  fi

  AIC_PHASE462_ARRIVAL_JSONL="${observation_path}" \
  AIC_PHASE462_SCENARIO="${label}" \
  WORKDIR="${WORKDIR}" \
  PHASE_NAME="phase462_arrival_observation_${variant}_${label}" \
  OUT_ROOT="${variant_root}" \
  TMP_ROOT="${variant_tmp}" \
  PORT="${PORT}" \
  FORCE=1 \
  SCENARIO="${scenario}" \
  TP=8 DP=1 EP=8 \
  ISL="${isl}" OSL="${OSL}" \
  MAX_NUM_BATCHED_TOKENS="${max_bt}" \
  BENCH_NUM_PROMPTS="${NUM_PROMPTS}" \
  BENCH_MAX_CONCURRENCY="${CONCURRENCY}" \
  BENCH_WARMUP_REQUESTS=0 \
  BENCH_REQUEST_ID_PREFIX="${trace_prefix}" \
  MAX_MODEL_LEN=262144 \
  BENCH_TIMEOUT_S=7200 \
  "$(dirname "$0")/run_phase412_arrival_sweep.sh"

  python3 - "${variant_root}/${scenario}/bench_result.json" "${NUM_PROMPTS}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = int(sys.argv[2])
ok = int(payload.get("ok_requests", 0))
failed = int(payload.get("failed_requests", 0))
if ok != expected or failed != 0:
    raise SystemExit(f"benchmark_incomplete:ok={ok}:failed={failed}:expected={expected}")
PY

  if [[ "${variant}" == "on" ]]; then
    cp -f "${observation_path}" "${OUT_ROOT}/${label}.jsonl"
    rm -f "${observation_path}"
  fi
}

write_overhead_gate() {
  python3 - "${OUT_ROOT}" "${MAX_OVERHEAD_PCT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
limit = float(sys.argv[2])
rows = []
for label in ("8k2k", "32k3k"):
    scenario = f"K2.5-tp8ep8-{label}"
    off = json.loads(
        (root / "overhead_off" / scenario / "bench_result.json").read_text()
    )["output_tok_s"]
    on = json.loads(
        (root / "overhead_on" / scenario / "bench_result.json").read_text()
    )["output_tok_s"]
    delta_pct = abs(on - off) / off * 100.0
    rows.append(
        {
            "scenario": label,
            "off_output_tok_s": off,
            "on_output_tok_s": on,
            "absolute_delta_pct": delta_pct,
            "max_overhead_pct": limit,
            "passed": delta_pct <= limit,
        }
    )
payload = {
    "gate": "phase462_arrival_logging_overhead",
    "rows": rows,
    "passed": all(row["passed"] for row in rows),
}
(root / "overhead_gate.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(payload, sort_keys=True))
raise SystemExit(0 if payload["passed"] else 1)
PY
}

main() {
  cd "${WORKDIR}"
  mkdir -p "${OUT_ROOT}" "${TMP_ROOT}"
  cat >"${OUT_ROOT}/meta.json" <<EOF
{
  "phase": "phase462_arrival_observation",
  "scenarios": ["8k2k", "32k3k"],
  "isl": [8000, 32000],
  "osl": ${OSL},
  "num_prompts": ${NUM_PROMPTS},
  "concurrency": ${CONCURRENCY},
  "runtime_change": "temporary_logging_patch_only",
  "diagnostic_only": true,
  "valid_for_default": false,
  "perf_database": false
}
EOF
  sha256sum "${PATCHER_REL}" "${ANALYZER_REL}" >"${OUT_ROOT}/tools.sha256"
  source_hashes original

  run_variant 8k2k 8000 8000 off
  apply_patch
  source_hashes patched_8k2k
  run_variant 8k2k 8000 8000 on
  restore_patch
  source_hashes restored_8k2k

  run_variant 32k3k 32000 32000 off
  apply_patch
  source_hashes patched_32k3k
  run_variant 32k3k 32000 32000 on
  restore_patch
  source_hashes restored_32k3k

  python3 "${ANALYZER_REL}" \
    "${OUT_ROOT}/8k2k.jsonl" \
    "${OUT_ROOT}/32k3k.jsonl" \
    --required-scenario 8k2k \
    --required-scenario 32k3k \
    --output "${OUT_ROOT}/self_check.json"
  write_overhead_gate

  if ! cmp -s "${OUT_ROOT}/source_original.sha256" "${OUT_ROOT}/source_restored_32k3k.sha256"; then
    echo "source_restore_hash_mismatch" >&2
    return 1
  fi
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
    >"${OUT_ROOT}/gpu_compute_apps_after.txt" 2>/dev/null || true
  pgrep -af "vllm.entrypoints.cli.main serve|VLLM::|raylet|gcs_server|run_openai_fixed_shape_benchmark" \
    >"${OUT_ROOT}/process_residual_after.txt" || true
}

main "$@"
