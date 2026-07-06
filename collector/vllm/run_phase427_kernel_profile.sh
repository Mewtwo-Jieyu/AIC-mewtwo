#!/usr/bin/env bash
set -euo pipefail

# Phase427 DP2 8k2k kernel profiler driver.
#
# Measurement-only. vLLM 0.19 torch profiler cannot be reused reliably in one
# process after stop_profile, so each window gets a fresh serve process.

WORKDIR="${WORKDIR:-/mnt/shared-storage-user/ailab-sys/zhaojieyu/aic}"
MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--moonshotai--Kimi-K2.5}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-kimi-k2.5}"
OUT_ROOT="${OUT_ROOT:-docs/iter_gap_investigation/phase427_kernel_profile}"
TMP_ROOT="${TMP_ROOT:-/tmp/phase427_kernel_profile_$$}"
PORT="${PORT:-20927}"
FORCE="${FORCE:-0}"

SCENARIO="${SCENARIO:-K2.5-tp4ep8dp2-8k2k}"
TP="${TP:-4}"
DP="${DP:-2}"
EP="${EP:-8}"
ISL="${ISL:-8000}"
OSL="${OSL:-2000}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8000}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-512}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-128}"
BENCH_WARMUP_REQUESTS="${BENCH_WARMUP_REQUESTS:-0}"
BENCH_TIMEOUT_S="${BENCH_TIMEOUT_S:-14400}"

PROFILE_MIXED_DELAY_S="${PROFILE_MIXED_DELAY_S:-70}"
PROFILE_DECODE_DELAY_S="${PROFILE_DECODE_DELAY_S:-240}"
PROFILE_WINDOW_S="${PROFILE_WINDOW_S:-35}"
PROF_WARMUP_ITERS="${PROF_WARMUP_ITERS:-2}"
PROF_ACTIVE_ITERS="${PROF_ACTIVE_ITERS:-35}"

METRICS_POLL_INTERVAL_S="${METRICS_POLL_INTERVAL_S:-2}"
READY_TIMEOUT_POLLS="${READY_TIMEOUT_POLLS:-480}"
DRAIN_POLL_SECONDS="${DRAIN_POLL_SECONDS:-5}"
DRAIN_STABLE_POLLS="${DRAIN_STABLE_POLLS:-3}"
DRAIN_TIMEOUT_SECONDS="${DRAIN_TIMEOUT_SECONDS:-600}"
GRACEFUL_TERM_SECONDS="${GRACEFUL_TERM_SECONDS:-120}"

export PATH="/usr/local/nvidia/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export VLLM_ENABLE_CUDA_COMPATIBILITY="${VLLM_ENABLE_CUDA_COMPATIBILITY:-1}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
unset VLLM_NUM_GPU_BLOCKS_OVERRIDE
unset NUM_GPU_BLOCKS_OVERRIDE

SERVICE_PID=""
BENCH_PID=""
METRICS_PID=""
BASELINE_PIDS=""

log() { echo "[$(date -Is)] $*"; }

gpu_busy_pids() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '
}

gpu_busy_pids_mine() {
  local p
  for p in $(gpu_busy_pids); do
    case " ${BASELINE_PIDS} " in
      *" ${p} "*) ;;
      *) echo "${p}" ;;
    esac
  done
}

snapshot_gpu_apps() {
  local path="$1"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >"${path}" 2>/dev/null || true
}

stop_metrics_poller() {
  if [[ -n "${METRICS_PID}" ]]; then
    kill -TERM "${METRICS_PID}" 2>/dev/null || true
    wait "${METRICS_PID}" 2>/dev/null || true
    METRICS_PID=""
  fi
}

stop_service() {
  stop_metrics_poller
  if [[ -n "${BENCH_PID}" ]]; then
    kill -TERM "${BENCH_PID}" 2>/dev/null || true
    wait "${BENCH_PID}" 2>/dev/null || true
    BENCH_PID=""
  fi
  if [[ -n "${SERVICE_PID}" ]]; then
    kill -TERM "${SERVICE_PID}" 2>/dev/null || true
  fi
  pkill -TERM -f "vllm.entrypoints.cli.main serve ${MODEL_PATH}" 2>/dev/null || true
  local waited=0
  while (( waited < GRACEFUL_TERM_SECONDS )); do
    if [[ -n "${SERVICE_PID}" ]] && kill -0 "${SERVICE_PID}" 2>/dev/null; then
      :
    elif [[ -z "$(gpu_busy_pids_mine)" ]]; then
      break
    fi
    sleep 3
    waited=$((waited + 3))
  done
  if [[ -n "${SERVICE_PID}" ]]; then
    kill -KILL "${SERVICE_PID}" 2>/dev/null || true
  fi
  pkill -KILL -f "vllm.entrypoints.cli.main serve ${MODEL_PATH}" 2>/dev/null || true
  pkill -KILL -f "VLLM::" 2>/dev/null || true
  ray stop --force >/dev/null 2>&1 || true
  local leftover
  leftover="$(gpu_busy_pids_mine)"
  if [[ -n "${leftover}" ]]; then
    log "force-killing lingering gpu pids: [$(echo "${leftover}" | tr '\n' ' ')]"
    # shellcheck disable=SC2086
    kill -KILL ${leftover} 2>/dev/null || true
  fi
  SERVICE_PID=""
}

trap stop_service EXIT

wait_for_gpu_drain() {
  local stable=0 elapsed=0 sample
  while (( elapsed <= DRAIN_TIMEOUT_SECONDS )); do
    sample="$(gpu_busy_pids_mine)"
    if [[ -z "${sample}" ]]; then
      stable=$((stable + 1))
    else
      stable=0
    fi
    if (( stable >= DRAIN_STABLE_POLLS )); then
      log "gpu_drained"
      return 0
    fi
    sleep "${DRAIN_POLL_SECONDS}"
    elapsed=$((elapsed + DRAIN_POLL_SECONDS))
  done
  log "gpu_drain_timeout mine_busy_pids=[$(gpu_busy_pids_mine | tr '\n' ' ')]"
  return 1
}

wait_for_service() {
  local serve_log="$1" poll
  for poll in $(seq 1 "${READY_TIMEOUT_POLLS}"); do
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      log "service_ready port=${PORT} poll=${poll}"
      return 0
    fi
    if [[ -n "${SERVICE_PID}" ]] && ! kill -0 "${SERVICE_PID}" 2>/dev/null; then
      log "service_exited_before_ready port=${PORT}"
      tail -n 160 "${serve_log}" || true
      return 1
    fi
    sleep 5
  done
  log "service_ready_timeout port=${PORT}"
  tail -n 160 "${serve_log}" || true
  return 1
}

start_metrics_poller() {
  local metrics_jsonl="$1"
  python3 - "${PORT}" "${metrics_jsonl}" "${METRICS_POLL_INTERVAL_S}" <<'PY' &
import datetime as dt
import json
import sys
import time
import urllib.error
import urllib.request

port = int(sys.argv[1])
path = sys.argv[2]
interval = float(sys.argv[3])
url = f"http://127.0.0.1:{port}/metrics"

with open(path, "a", encoding="utf-8") as f:
    while True:
        record = {"ts": dt.datetime.now(dt.UTC).isoformat(), "url": url}
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                record["status"] = resp.status
                record["body"] = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            record["status"] = exc.code
            record["error"] = str(exc)
            record["body"] = exc.read().decode("utf-8", errors="replace")
        except Exception as exc:
            record["status"] = None
            record["error"] = repr(exc)
            record["body"] = ""
        f.write(json.dumps(record, ensure_ascii=True) + "\n")
        f.flush()
        time.sleep(interval)
PY
  METRICS_PID=$!
  log "metrics_poller_pid=${METRICS_PID}"
}

copy_profile_window() {
  local window="$1" prof_dir="$2" out_dir="$3"
  local dst="${out_dir}/prof_${window}"
  rm -rf "${dst}"
  mkdir -p "${dst}"
  cp -f "${prof_dir}"/* "${dst}/" 2>/dev/null || true
  local nfiles ntext
  nfiles="$(find "${dst}" -type f | wc -l | tr -d ' ')"
  ntext="$(find "${dst}" -type f -name 'profiler_out_*.txt' | wc -l | tr -d ' ')"
  log "copied_${window}_profile_files=${nfiles} text_files=${ntext}"
  echo "files=${nfiles}" >"${dst}/profile_done.txt"
  echo "text_files=${ntext}" >>"${dst}/profile_done.txt"
  if (( ntext == 0 )); then
    log "abort: missing profiler_out files for ${window}"
    return 1
  fi
}

profile_window() {
  local window="$1" target_delay_s="$2" start_epoch="$3" prof_dir="$4" out_dir="$5"
  local now_epoch elapsed_s sleep_s
  now_epoch="$(date +%s)"
  elapsed_s=$((now_epoch - start_epoch))
  sleep_s=$((target_delay_s - elapsed_s))
  if (( sleep_s > 0 )); then
    log "profile_${window}_waiting=${sleep_s}s"
    sleep "${sleep_s}"
  fi
  if [[ -n "${BENCH_PID}" ]] && ! kill -0 "${BENCH_PID}" 2>/dev/null; then
    log "profile_${window}_skipped_benchmark_finished"
    return 1
  fi
  log "profile_${window}_start"
  curl -fsS -X POST "http://127.0.0.1:${PORT}/start_profile"
  sleep "${PROFILE_WINDOW_S}"
  log "profile_${window}_stop"
  curl -fsS -X POST "http://127.0.0.1:${PORT}/stop_profile" || log "warn: ${window} stop_profile returned non-zero; copying profiler files if present"
  sleep 20
  copy_profile_window "${window}" "${prof_dir}" "${out_dir}"
}

write_meta() {
  local out_dir="$1"
  cat >"${out_dir}/meta.json" <<EOF
{"name": "${SCENARIO}", "phase": "phase427", "bench_num_prompts": ${BENCH_NUM_PROMPTS}, "tp": ${TP}, "dp": ${DP}, "ep": ${EP}, "isl": ${ISL}, "osl": ${OSL}, "max_num_batched_tokens": ${MAX_NUM_BATCHED_TOKENS}, "batch_size": ${BENCH_MAX_CONCURRENCY}, "world_size": $((TP * DP)), "port": ${PORT}, "prefix_caching": false, "gpu_memory_utilization": ${GPU_MEMORY_UTILIZATION}, "max_model_len": ${MAX_MODEL_LEN}, "max_num_seqs": ${MAX_NUM_SEQS}, "enable_logging_iteration_details": true, "cudagraph_metrics": true, "profile_mixed_delay_s": ${PROFILE_MIXED_DELAY_S}, "profile_decode_delay_s": ${PROFILE_DECODE_DELAY_S}, "profile_window_s": ${PROFILE_WINDOW_S}, "prof_warmup_iters": ${PROF_WARMUP_ITERS}, "prof_active_iters": ${PROF_ACTIVE_ITERS}, "profile_process_model": "fresh_serve_per_window", "benchmark_completion_required": false}
EOF
}

run_window() {
  local window="$1" delay_s="$2"
  local out_dir="${OUT_ROOT}/${SCENARIO}"
  local tmp_dir="${TMP_ROOT}/${SCENARIO}/${window}"
  local prof_dir="${tmp_dir}/prof"
  local serve_log="${out_dir}/serve_${window}.log"
  local metrics_jsonl="${out_dir}/metrics_${window}.jsonl"
  local bench_json="${tmp_dir}/bench_result.json"
  local bench_rec="${tmp_dir}/bench_records.jsonl"
  local bench_log="${tmp_dir}/bench.log"
  local profiler_json

  mkdir -p "${tmp_dir}" "${prof_dir}"
  : >"${serve_log}"
  : >"${metrics_jsonl}"
  rm -rf "${out_dir}/prof_${window}"

  profiler_json="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${prof_dir}\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":true,\"warmup_iterations\":${PROF_WARMUP_ITERS},\"active_iterations\":${PROF_ACTIVE_ITERS},\"wait_iterations\":0}"

  local serve_cmd=(
    python3 -m vllm.entrypoints.cli.main serve "${MODEL_PATH}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --port "${PORT}"
    --tensor-parallel-size "${TP}"
    --data-parallel-size "${DP}"
    --enable-expert-parallel
    --max-model-len "${MAX_MODEL_LEN}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --enable-chunked-prefill
    --no-enable-prefix-caching
    --enable-logging-iteration-details
    --cudagraph-metrics
    --profiler-config "${profiler_json}"
    --mm-encoder-tp-mode data
    --skip-mm-profiling
    --trust-remote-code
    --enable-auto-tool-choice
    --tool-call-parser kimi_k2
    --reasoning-parser kimi_k2
  )

  log "window=${window} serve_command: ${serve_cmd[*]}" | tee -a "${OUT_ROOT}/driver.log"
  nohup "${serve_cmd[@]}" >"${serve_log}" 2>&1 &
  SERVICE_PID=$!
  log "window=${window} service_pid=${SERVICE_PID}" | tee -a "${OUT_ROOT}/driver.log"

  wait_for_service "${serve_log}"
  start_metrics_poller "${metrics_jsonl}"

  python3 scripts/run_openai_fixed_shape_benchmark.py \
    --host 127.0.0.1 --port "${PORT}" \
    --model "${SERVED_MODEL_NAME}" --tokenizer "${MODEL_PATH}" \
    --num-prompts "${BENCH_NUM_PROMPTS}" --max-concurrency "${BENCH_MAX_CONCURRENCY}" \
    --input-len "${ISL}" --output-len "${OSL}" \
    --warmup-requests "${BENCH_WARMUP_REQUESTS}" --timeout-s "${BENCH_TIMEOUT_S}" \
    --result-json "${bench_json}" --records-jsonl "${bench_rec}" >"${bench_log}" 2>&1 &
  BENCH_PID=$!
  local bench_start_epoch
  bench_start_epoch="$(date +%s)"
  log "window=${window} bench_pid=${BENCH_PID}"

  profile_window "${window}" "${delay_s}" "${bench_start_epoch}" "${prof_dir}" "${out_dir}"

  stop_metrics_poller
  if [[ -n "${BENCH_PID}" ]]; then
    kill -TERM "${BENCH_PID}" 2>/dev/null || true
    wait "${BENCH_PID}" 2>/dev/null || true
    BENCH_PID=""
  fi
  cp -f "${bench_log}" "${out_dir}/bench_${window}.log" 2>/dev/null || true
  cp -f "${bench_rec}" "${out_dir}/bench_records_${window}.jsonl" 2>/dev/null || true
  if [[ -f "${bench_json}" ]]; then
    cp -f "${bench_json}" "${out_dir}/bench_result_${window}.json"
  fi

  stop_service
  wait_for_gpu_drain || { log "warn: drain timeout after ${window}"; }
  log "window=${window} done" | tee -a "${OUT_ROOT}/driver.log"
}

main() {
  cd "${WORKDIR}"
  local out_dir="${OUT_ROOT}/${SCENARIO}"

  if [[ "${FORCE}" == "1" ]]; then
    rm -rf "${out_dir}/prof_mixed" "${out_dir}/prof_decode"
    rm -f "${out_dir}/serve.log" "${out_dir}/serve_mixed.log" "${out_dir}/serve_decode.log"
    rm -f "${out_dir}/metrics.jsonl" "${out_dir}/metrics_mixed.jsonl" "${out_dir}/metrics_decode.jsonl"
    rm -f "${out_dir}/bench.log" "${out_dir}/bench_mixed.log" "${out_dir}/bench_decode.log"
    rm -f "${out_dir}/bench_records.jsonl" "${out_dir}/bench_records_mixed.jsonl" "${out_dir}/bench_records_decode.jsonl"
    rm -f "${out_dir}/bench_result.json" "${out_dir}/bench_result_mixed.json" "${out_dir}/bench_result_decode.json"
  fi
  mkdir -p "${out_dir}" "${OUT_ROOT}"
  write_meta "${out_dir}"

  python3 -c 'import vllm; print("vllm_version="+getattr(vllm,"__version__","unknown"))' | tee "${OUT_ROOT}/runtime.txt"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits >"${OUT_ROOT}/gpu_inventory.txt"
  sed -n '1p' "${OUT_ROOT}/gpu_inventory.txt" | tee -a "${OUT_ROOT}/runtime.txt"

  BASELINE_PIDS="$(gpu_busy_pids | tr '\n' ' ')"
  log "baseline_gpu_pids=[${BASELINE_PIDS}]" | tee -a "${OUT_ROOT}/driver.log"
  snapshot_gpu_apps "${OUT_ROOT}/gpu_compute_apps_before.txt"
  wait_for_gpu_drain || { log "abort: my gpu procs still busy"; return 1; }

  run_window "mixed" "${PROFILE_MIXED_DELAY_S}"
  run_window "decode" "${PROFILE_DECODE_DELAY_S}"

  {
    echo "===== serve_mixed.log ====="
    cat "${out_dir}/serve_mixed.log" 2>/dev/null || true
    echo "===== serve_decode.log ====="
    cat "${out_dir}/serve_decode.log" 2>/dev/null || true
  } >"${out_dir}/serve.log"

  snapshot_gpu_apps "${OUT_ROOT}/gpu_compute_apps_after.txt"
  pgrep -af "vllm.entrypoints.cli.main serve|ray::|raylet|gcs_server|VLLM::APIServer|VLLM::EngineCore|run_openai_fixed_shape_benchmark" >"${OUT_ROOT}/process_residual_after.txt" || true
  log "PHASE427_DONE" | tee -a "${OUT_ROOT}/driver.log"
}

main "$@"
