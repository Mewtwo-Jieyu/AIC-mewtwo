#!/usr/bin/env bash
set -euo pipefail

# Phase446 B2b v2: low-overhead graph-outer CUDA event timing.
#
# The patch records CUDA event pairs during each forward step, drains completed
# timings into memory, and writes JSONL only at process shutdown. No CUPTI,
# torch profiler, per-step file IO, or in-step event synchronization is used.

WORKDIR="${WORKDIR:-/mnt/shared-storage-user/ailab-sys/zhaojieyu/aic}"
OUT_ROOT="${OUT_ROOT:-docs/iter_gap_investigation/phase446_b2b_event_timing}"
TMP_ROOT="${TMP_ROOT:-/tmp/phase446_b2b_event_timing_$$}"
PORT="${PORT:-20946}"
FORCE="${FORCE:-0}"
MODE="${MODE:-overhead_gate}"

GPU_MODEL_RUNNER_PATH="${GPU_MODEL_RUNNER_PATH:-/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py}"
PATCH_BACKUP_PATH="${PATCH_BACKUP_PATH:-/tmp/gpu_model_runner.py.phase446.b2b.bak}"
PATCHER_REL="${PATCHER_REL:-collector/vllm/phase446_b2b_event_timing_patch.py}"
OVERHEAD_MAX_PCT="${OVERHEAD_MAX_PCT:-2.0}"

SCENARIO="${SCENARIO:-K2.5-tp4ep8dp2-8k2k}"
ISL="${ISL:-8000}"
OSL="${OSL:-2000}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8000}"
BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-512}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"

BENCH_NUM_PROMPTS_32K="${BENCH_NUM_PROMPTS_32K:-256}"

export PATH="/opt/py3/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/kubebrain:/usr/local/nvidia/bin"
export LD_LIBRARY_PATH="/nccl/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.9/compat:/usr/local/lib/python3.12/dist-packages/torch/lib"
export VLLM_ENABLE_CUDA_COMPATIBILITY="${VLLM_ENABLE_CUDA_COMPATIBILITY:-1}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export AIC_PHASE446_FLUSH_INTERVAL="${AIC_PHASE446_FLUSH_INTERVAL:-64}"

PATCH_APPLIED=0

log() { echo "[$(date -Is)] $*"; }

patch_apply() {
  cd "${WORKDIR}"
  python3 "${PATCHER_REL}" --target "${GPU_MODEL_RUNNER_PATH}" --backup "${PATCH_BACKUP_PATH}" --apply
  python3 "${PATCHER_REL}" --target "${GPU_MODEL_RUNNER_PATH}" --check >/dev/null
  PATCH_APPLIED=1
  sha256sum "${GPU_MODEL_RUNNER_PATH}" >"${OUT_ROOT}/gpu_model_runner_patched.sha256"
  sha256sum "${PATCH_BACKUP_PATH}" >"${OUT_ROOT}/gpu_model_runner_backup.sha256"
}

patch_restore() {
  if [[ "${PATCH_APPLIED}" == "1" ]]; then
    cd "${WORKDIR}"
    python3 "${PATCHER_REL}" --target "${GPU_MODEL_RUNNER_PATH}" --backup "${PATCH_BACKUP_PATH}" --restore || true
    PATCH_APPLIED=0
  fi
}

trap patch_restore EXIT

run_point() {
  local label="$1"
  local event_enabled="$2"
  local port="$3"
  local scenario="$4"
  local isl="$5"
  local osl="$6"
  local max_bt="$7"
  local prompts="$8"
  local concurrency="$9"
  local max_model_len="${10}"

  local label_root="${OUT_ROOT}/${label}"
  mkdir -p "${label_root}"
  local pre_stop_hook=""
  if [[ "${event_enabled}" == "1" ]]; then
    export AIC_PHASE446_EVENT_JSONL="${label_root}/event_timing.jsonl"
    : >"${AIC_PHASE446_EVENT_JSONL}"
    pre_stop_hook="${label_root}/flush_phase446_events.sh"
    cat >"${pre_stop_hook}" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
pkill -USR1 -f "VLLM::EngineCore" 2>/dev/null || true
pkill -USR1 -f "VLLM::Worker" 2>/dev/null || true
sleep "${AIC_PHASE446_SIGNAL_FLUSH_WAIT_S:-10}"
SH
    chmod +x "${pre_stop_hook}"
  else
    unset AIC_PHASE446_EVENT_JSONL
  fi

  log "run_point label=${label} event_enabled=${event_enabled} scenario=${scenario}"
  PHASE_NAME="phase446_b2b" \
  OUT_ROOT="${label_root}" \
  TMP_ROOT="${TMP_ROOT}/${label}" \
  PORT="${port}" \
  FORCE="${FORCE}" \
  SCENARIO="${scenario}" \
  ISL="${isl}" \
  OSL="${osl}" \
  MAX_NUM_BATCHED_TOKENS="${max_bt}" \
  BENCH_NUM_PROMPTS="${prompts}" \
  BENCH_MAX_CONCURRENCY="${concurrency}" \
  MAX_MODEL_LEN="${max_model_len}" \
  PRE_STOP_HOOK_SCRIPT="${pre_stop_hook}" \
  "$(dirname "$0")/run_phase425_8k2k_sweep.sh"

  if [[ "${event_enabled}" == "1" ]]; then
    local events
    events="$(wc -l <"${AIC_PHASE446_EVENT_JSONL}" | tr -d ' ')"
    if [[ "${events}" == "0" ]]; then
      log "event_jsonl_empty=${AIC_PHASE446_EVENT_JSONL}"
      return 1
    fi
    log "event_jsonl_rows=${events} path=${AIC_PHASE446_EVENT_JSONL}"
  fi
}

bench_json_for() {
  local label="$1"
  local scenario="$2"
  echo "${OUT_ROOT}/${label}/${scenario}/bench_result.json"
}

compare_overhead() {
  local off_json="$1"
  local on_json="$2"
  local out_json="$3"
  python3 - "${off_json}" "${on_json}" "${out_json}" "${OVERHEAD_MAX_PCT}" <<'PY'
import json
import sys

off_path, on_path, out_path, max_pct = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
off = json.load(open(off_path, encoding="utf-8"))
on = json.load(open(on_path, encoding="utf-8"))
off_tput = float(off["output_tok_s"])
on_tput = float(on["output_tok_s"])
overhead_pct = max(0.0, (off_tput - on_tput) / off_tput * 100.0)
payload = {
    "gate": "phase446_b2b_event_timing_overhead",
    "metric": "output_tok_s",
    "off_output_tok_s": off_tput,
    "on_output_tok_s": on_tput,
    "overhead_pct": overhead_pct,
    "max_overhead_pct": max_pct,
    "passed": overhead_pct <= max_pct,
}
open(out_path, "w", encoding="utf-8").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, sort_keys=True))
raise SystemExit(0 if payload["passed"] else 1)
PY
}

write_meta() {
  mkdir -p "${OUT_ROOT}"
  python3 - "${OUT_ROOT}/meta.json" "${MODE}" "${SCENARIO}" "${ISL}" "${OSL}" "${MAX_NUM_BATCHED_TOKENS}" "${BENCH_NUM_PROMPTS}" "${BENCH_MAX_CONCURRENCY}" "${MAX_MODEL_LEN}" "${AIC_PHASE446_FLUSH_INTERVAL}" <<'PY'
import json
import sys

(
    path,
    mode,
    scenario,
    isl,
    osl,
    max_bt,
    prompts,
    concurrency,
    max_model_len,
    flush_interval,
) = sys.argv[1:]
payload = {
    "phase": "phase446_b2b_event_timing",
    "mode": mode,
    "scenario": scenario,
    "isl": int(isl),
    "osl": int(osl),
    "max_num_batched_tokens": int(max_bt),
    "bench_num_prompts": int(prompts),
    "bench_max_concurrency": int(concurrency),
    "max_model_len": int(max_model_len),
    "measurement": "model_forward_cuda_event_outer_delayed",
    "flush_interval": int(flush_interval),
    "runtime_file_io": "shutdown_only",
    "attention_busy_ms": "not_split_by_patch",
    "default_readiness": "No-Go",
}
open(path, "w", encoding="utf-8").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

main() {
  cd "${WORKDIR}"
  mkdir -p "${OUT_ROOT}" "${TMP_ROOT}"
  : >"${OUT_ROOT}/driver.log"
  write_meta

  case "${MODE}" in
    overhead_gate)
      run_point "overhead_off" "0" "${PORT}" "${SCENARIO}" "${ISL}" "${OSL}" \
        "${MAX_NUM_BATCHED_TOKENS}" "${BENCH_NUM_PROMPTS}" "${BENCH_MAX_CONCURRENCY}" "${MAX_MODEL_LEN}" \
        2>&1 | tee -a "${OUT_ROOT}/driver.log"
      patch_apply
      run_point "overhead_on" "1" "$((PORT + 1))" "${SCENARIO}" "${ISL}" "${OSL}" \
        "${MAX_NUM_BATCHED_TOKENS}" "${BENCH_NUM_PROMPTS}" "${BENCH_MAX_CONCURRENCY}" "${MAX_MODEL_LEN}" \
        2>&1 | tee -a "${OUT_ROOT}/driver.log"
      compare_overhead \
        "$(bench_json_for overhead_off "${SCENARIO}")" \
        "$(bench_json_for overhead_on "${SCENARIO}")" \
        "${OUT_ROOT}/overhead_gate.json" | tee -a "${OUT_ROOT}/driver.log"
      ;;
    event_on_8k)
      patch_apply
      run_point "event_on_8k" "1" "${PORT}" "K2.5-tp4ep8dp2-8k2k" "8000" "2000" \
        "8000" "${BENCH_NUM_PROMPTS}" "${BENCH_MAX_CONCURRENCY}" "131072" \
        2>&1 | tee -a "${OUT_ROOT}/driver.log"
      ;;
    event_on_32k)
      patch_apply
      run_point "event_on_32k" "1" "${PORT}" "K2.5-tp4ep8dp2-32k3k" "32000" "3000" \
        "32000" "${BENCH_NUM_PROMPTS_32K}" "${BENCH_MAX_CONCURRENCY}" "262144" \
        2>&1 | tee -a "${OUT_ROOT}/driver.log"
      ;;
    collect_8k_32k)
      patch_apply
      run_point "event_on_8k" "1" "${PORT}" "K2.5-tp4ep8dp2-8k2k" "8000" "2000" \
        "8000" "${BENCH_NUM_PROMPTS}" "${BENCH_MAX_CONCURRENCY}" "131072" \
        2>&1 | tee -a "${OUT_ROOT}/driver.log"
      run_point "event_on_32k" "1" "$((PORT + 1))" "K2.5-tp4ep8dp2-32k3k" "32000" "3000" \
        "32000" "${BENCH_NUM_PROMPTS_32K}" "${BENCH_MAX_CONCURRENCY}" "262144" \
        2>&1 | tee -a "${OUT_ROOT}/driver.log"
      ;;
    restore_only)
      PATCH_APPLIED=1
      patch_restore
      ;;
    *)
      echo "unknown MODE=${MODE}" >&2
      return 2
      ;;
  esac
}

main "$@"
