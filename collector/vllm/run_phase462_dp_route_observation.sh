#!/usr/bin/env bash
set -euo pipefail

# Phase462 DP Step3b: logging-only route -> receive -> first-admit observation.

WORKDIR="${WORKDIR:-/mnt/shared-storage-user/zhaojieyu/backup/aic}"
OUT_ROOT="${OUT_ROOT:-docs/iter_gap_investigation/phase462_dp_route_observation}"
TMP_ROOT="${TMP_ROOT:-/tmp/phase462_dp_route_observation_$$}"
PATCHER_REL="${PATCHER_REL:-collector/vllm/phase462_dp_route_observation_patch.py}"
ANALYZER_REL="${ANALYZER_REL:-scripts/analyze_phase462_dp_route_observation.py}"
PATCH_BACKUP_DIR="${PATCH_BACKUP_DIR:-/tmp/phase462_dp_route_backups_$$}"
CORE_CLIENT_PATH="${CORE_CLIENT_PATH:-/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core_client.py}"
ENGINE_PATH="${ENGINE_PATH:-/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py}"
PORT="${PORT:-21530}"
FORCE="${FORCE:-0}"

MAX_OVERHEAD_PCT=2.0
OVERHEAD_PROMPTS=128
CAPTURE_PROMPTS=512
CONCURRENCY=128
ISL=8000
OSL=2000
MAX_BT=65536
REQUEST_PREFIX="phase462-dp-route"

export PATH="/opt/py3/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/kubebrain:/usr/local/nvidia/bin"
export LD_LIBRARY_PATH="/nccl/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.9/compat:/usr/local/lib/python3.12/dist-packages/torch/lib"
export VLLM_ENABLE_CUDA_COMPATIBILITY=1

PATCH_APPLIED=0
ORIGINAL_HASHES=""

patch_args() {
  printf '%s\n' \
    --core-client-target "${CORE_CLIENT_PATH}" \
    --engine-target "${ENGINE_PATH}" \
    --backup-dir "${PATCH_BACKUP_DIR}"
}

source_hashes() {
  sha256sum "${CORE_CLIENT_PATH}" "${ENGINE_PATH}"
}

restore_patch() {
  if [[ "${PATCH_APPLIED}" == "1" ]]; then
    mapfile -t args < <(patch_args)
    python3 "${PATCHER_REL}" "${args[@]}" --restore
    PATCH_APPLIED=0
  fi
}

record_residuals() {
  mkdir -p "${OUT_ROOT}"
  if ! nvidia-smi \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader \
    >"${OUT_ROOT}/gpu_compute_apps_after.txt"; then
    echo "gpu_residual_probe_failed" >&2
    return 1
  fi
  local pgrep_status=0
  pgrep -af "vllm.entrypoints.cli.main serve|VLLM::|run_openai_fixed_shape_benchmark" \
    >"${OUT_ROOT}/process_residual_after.txt" || pgrep_status=$?
  if [[ "${pgrep_status}" != "0" && "${pgrep_status}" != "1" ]]; then
    echo "process_residual_probe_failed:${pgrep_status}" >&2
    return "${pgrep_status}"
  fi
}

cleanup() {
  local status=$?
  restore_patch || status=1
  if [[ -n "${ORIGINAL_HASHES}" ]]; then
    source_hashes >"${OUT_ROOT}/source_restored.sha256" || status=1
    if ! cmp -s <(printf '%s\n' "${ORIGINAL_HASHES}") "${OUT_ROOT}/source_restored.sha256"; then
      echo "source_restore_hash_mismatch" >&2
      status=1
    fi
  fi
  record_residuals || status=1
  exit "${status}"
}

trap cleanup EXIT

apply_observation_patch() {
  mapfile -t args < <(patch_args)
  python3 "${PATCHER_REL}" "${args[@]}" --apply
  PATCH_APPLIED=1
  python3 "${PATCHER_REL}" "${args[@]}" --check
  python3 -m py_compile "${CORE_CLIENT_PATH}" "${ENGINE_PATH}"
}

run_variant() {
  local label="$1"
  local num_prompts="$2"
  local trace_enabled="$3"
  local scenario="K2.5-tp4ep8dp2-8k2k-bt65536-${label}"
  local variant_root="${OUT_ROOT}/${label}"
  local variant_tmp="${TMP_ROOT}/${label}"
  local trace_dir=""

  if [[ "${trace_enabled}" == "1" ]]; then
    trace_dir="${variant_root}/trace"
    mkdir -p "${trace_dir}"
  fi

  AIC_PHASE462_DP_ROUTE_TRACE_DIR="${trace_dir}" \
  AIC_PHASE462_DP_REQUEST_PREFIX="${REQUEST_PREFIX}" \
  WORKDIR="${WORKDIR}" \
  PHASE_NAME="phase462_dp_route_${label}" \
  OUT_ROOT="${variant_root}" \
  TMP_ROOT="${variant_tmp}" \
  PORT="${PORT}" \
  FORCE=1 \
  SCENARIO="${scenario}" \
  TP=4 DP=2 EP=8 \
  ISL="${ISL}" OSL="${OSL}" \
  MAX_NUM_BATCHED_TOKENS="${MAX_BT}" \
  MAX_MODEL_LEN=131072 \
  BENCH_NUM_PROMPTS="${num_prompts}" \
  BENCH_MAX_CONCURRENCY="${CONCURRENCY}" \
  BENCH_WARMUP_REQUESTS=0 \
  BENCH_REQUEST_ID_PREFIX="${REQUEST_PREFIX}" \
  BENCH_TIMEOUT_S=7200 \
  bash collector/vllm/run_phase412_arrival_sweep.sh

  python3 - "${variant_root}/${scenario}/bench_result.json" "${num_prompts}" <<'PY'
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
}

write_overhead_gate() {
  python3 - "${OUT_ROOT}" "${MAX_OVERHEAD_PCT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
limit = float(sys.argv[2])

def rate(label):
    scenario = f"K2.5-tp4ep8dp2-8k2k-bt65536-{label}"
    return float(json.loads(
        (root / label / scenario / "bench_result.json").read_text(encoding="utf-8")
    )["output_tok_s"])

off = rate("overhead_off")
on = rate("overhead_on")
delta_pct = abs(on - off) / off * 100.0
payload = {
    "gate": "phase462_dp_route_logging_overhead",
    "off_output_tok_s": off,
    "on_output_tok_s": on,
    "absolute_delta_pct": delta_pct,
    "max_overhead_pct": limit,
    "passed": delta_pct <= limit,
}
(root / "overhead_gate.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(payload, sort_keys=True))
raise SystemExit(0 if payload["passed"] else 1)
PY
}

judge_capture() {
  local trace_dir="${OUT_ROOT}/capture/trace"
  mapfile -t trace_paths < <(find "${trace_dir}" -maxdepth 1 -type f -name '*.jsonl' | sort)
  if [[ "${#trace_paths[@]}" == "0" ]]; then
    echo "phase462_dp_route_trace_missing" >&2
    return 1
  fi
  python3 "${ANALYZER_REL}" "${trace_paths[@]}" \
    --expected-requests "${CAPTURE_PROMPTS}" \
    --output-csv "${OUT_ROOT}/phase462_dp_route_judgement.csv" \
    --output-report "${OUT_ROOT}/phase462_dp_route_judgement.md"
}

main() {
  cd "${WORKDIR}"
  if [[ -e "${OUT_ROOT}" ]]; then
    if [[ "${FORCE}" != "1" ]]; then
      echo "abort_existing_artifact=${OUT_ROOT}; set FORCE=1 to overwrite" >&2
      return 1
    fi
    rm -rf "${OUT_ROOT}"
  fi
  mkdir -p "${OUT_ROOT}" "${TMP_ROOT}"
  cat >"${OUT_ROOT}/meta.json" <<EOF
{
  "phase": "phase462_dp_step3b_route_observation",
  "protocol": "dp2-bt65536/N512/C128/ISL8k/OSL2k/max_bt65536",
  "runtime_change": "temporary_logging_patch_only",
  "diagnostic_only": true,
  "valid_for_default": false,
  "perf_database": false
}
EOF
  sha256sum "${PATCHER_REL}" "${ANALYZER_REL}" "$0" >"${OUT_ROOT}/tools.sha256"
  ORIGINAL_HASHES="$(source_hashes)"
  printf '%s\n' "${ORIGINAL_HASHES}" >"${OUT_ROOT}/source_original.sha256"

  run_variant overhead_off "${OVERHEAD_PROMPTS}" 0
  apply_observation_patch
  source_hashes >"${OUT_ROOT}/source_patched.sha256"
  run_variant overhead_on "${OVERHEAD_PROMPTS}" 1
  write_overhead_gate
  run_variant capture "${CAPTURE_PROMPTS}" 1
  judge_capture

  restore_patch
  source_hashes >"${OUT_ROOT}/source_restored.sha256"
  cmp -s "${OUT_ROOT}/source_original.sha256" "${OUT_ROOT}/source_restored.sha256"
  record_residuals
}

main "$@"
