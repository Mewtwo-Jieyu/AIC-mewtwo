#!/usr/bin/env bash
set -euo pipefail

# Phase433 32k3k prefill execution forensics.
# Measurement-only. Prefer nsys when available; otherwise use vLLM torch
# profiler chrome traces with CPU/runtime activities as the precision-limited
# fallback.

WORKDIR="${WORKDIR:-/mnt/shared-storage-user/ailab-sys/zhaojieyu/aic}"
MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--moonshotai--Kimi-K2.5}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-kimi-k2.5}"
OUT_ROOT="${OUT_ROOT:-docs/iter_gap_investigation/phase433_nsys_prefill}"
TMP_ROOT="${TMP_ROOT:-/tmp/phase433_nsys_prefill_$$}"
PORT="${PORT:-20933}"
FORCE="${FORCE:-0}"

SCENARIO="${SCENARIO:-K2.5-tp4ep8dp2-32k3k}"
TP="${TP:-4}"
DP="${DP:-2}"
EP="${EP:-8}"
ISL="${ISL:-32000}"
OSL="${OSL:-3000}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-256}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-128}"
BENCH_WARMUP_REQUESTS="${BENCH_WARMUP_REQUESTS:-0}"
BENCH_TIMEOUT_S="${BENCH_TIMEOUT_S:-14400}"

METRICS_POLL_INTERVAL_S="${METRICS_POLL_INTERVAL_S:-2}"
READY_TIMEOUT_POLLS="${READY_TIMEOUT_POLLS:-480}"
DRAIN_POLL_SECONDS="${DRAIN_POLL_SECONDS:-5}"
DRAIN_STABLE_POLLS="${DRAIN_STABLE_POLLS:-3}"
DRAIN_TIMEOUT_SECONDS="${DRAIN_TIMEOUT_SECONDS:-600}"
GRACEFUL_TERM_SECONDS="${GRACEFUL_TERM_SECONDS:-120}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
ACTIVE_ITERS="${ACTIVE_ITERS:-24}"
WINDOW_SECONDS="${WINDOW_SECONDS:-180}"
MIN_CTX_TOKENS="${MIN_CTX_TOKENS:-31000}"

export PATH="/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/cuda-12.9/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export VLLM_ENABLE_CUDA_COMPATIBILITY="${VLLM_ENABLE_CUDA_COMPATIBILITY:-1}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
unset VLLM_NUM_GPU_BLOCKS_OVERRIDE
unset NUM_GPU_BLOCKS_OVERRIDE

SERVICE_PID=""
BENCH_PID=""
METRICS_PID=""
BASELINE_PIDS=""
NSYS_AVAILABLE="false"
PROFILER_MODE="torch_profiler_cpu_fallback"

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

probe_nsys() {
  if command -v nsys >/dev/null 2>&1; then
    NSYS_AVAILABLE="true"
    PROFILER_MODE="nsys"
    nsys --version >"${OUT_ROOT}/nsys_version.txt" 2>&1 || true
  else
    NSYS_AVAILABLE="false"
    PROFILER_MODE="torch_profiler_cpu_fallback"
    echo "nsys_not_found" >"${OUT_ROOT}/nsys_version.txt"
  fi
  log "nsys_available=${NSYS_AVAILABLE} profiler_mode=${PROFILER_MODE}" | tee -a "${OUT_ROOT}/driver.log"
}

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

start_service() {
  local prof_dir="$1" serve_log="$2" attempt="$3"
  local profiler_json
  profiler_json="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${prof_dir}\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":true,\"warmup_iterations\":1,\"active_iterations\":${ACTIVE_ITERS},\"wait_iterations\":0}"
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
  log "attempt=${attempt} serve_command: ${serve_cmd[*]}" | tee -a "${OUT_ROOT}/driver.log"
  nohup "${serve_cmd[@]}" >"${serve_log}" 2>&1 &
  SERVICE_PID=$!
  log "attempt=${attempt} service_pid=${SERVICE_PID}" | tee -a "${OUT_ROOT}/driver.log"
  wait_for_service "${serve_log}"
}

start_bench() {
  local bench_json="$1" bench_rec="$2" bench_log="$3"
  python3 scripts/run_openai_fixed_shape_benchmark.py \
    --host 127.0.0.1 --port "${PORT}" \
    --model "${SERVED_MODEL_NAME}" --tokenizer "${MODEL_PATH}" \
    --num-prompts "${BENCH_NUM_PROMPTS}" --max-concurrency "${BENCH_MAX_CONCURRENCY}" \
    --input-len "${ISL}" --output-len "${OSL}" \
    --warmup-requests "${BENCH_WARMUP_REQUESTS}" --timeout-s "${BENCH_TIMEOUT_S}" \
    --result-json "${bench_json}" --records-jsonl "${bench_rec}" >"${bench_log}" 2>&1 &
  BENCH_PID=$!
  log "bench_pid=${BENCH_PID}"
}

copy_profile_window() {
  local prof_dir="$1" out_dir="$2"
  local dst="${out_dir}/prof_prefill"
  rm -rf "${dst}"
  mkdir -p "${dst}"
  cp -f "${prof_dir}"/* "${dst}/" 2>/dev/null || true
  local nfiles ntrace
  nfiles="$(find "${dst}" -type f | wc -l | tr -d ' ')"
  ntrace="$(find "${dst}" -type f -name '*.pt.trace.json.gz' | wc -l | tr -d ' ')"
  log "copied_prefill_profile_files=${nfiles} trace_files=${ntrace}"
  echo "files=${nfiles}" >"${dst}/profile_done.txt"
  echo "trace_files=${ntrace}" >>"${dst}/profile_done.txt"
  if (( ntrace == 0 )); then
    log "abort: missing chrome trace files"
    return 1
  fi
}

run_window_check() {
  local out_dir="$1" attempt="$2"
  local check_file="${out_dir}/check_prefill_attempt${attempt}.json"
  local rc=0
  python3 scripts/analyze_phase433_nsys_prefill.py \
    --artifact-root "${OUT_ROOT}" \
    --scenario "${SCENARIO}" \
    --window prefill \
    --min-ctx-tokens "${MIN_CTX_TOKENS}" \
    --check-window >"${check_file}" || rc=$?
  python3 - "${out_dir}/window_checks.jsonl" "${attempt}" "${check_file}" "${rc}" <<'PY'
import json
import sys

out_path, attempt, check_file, rc = sys.argv[1:]
with open(check_file, encoding="utf-8") as f:
    payload = json.loads(f.read())
payload["attempt"] = int(attempt)
payload["passed"] = rc == "0"
with open(out_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(payload, sort_keys=True) + "\n")
PY
  if (( rc == 0 )); then
    log "prefill self_check_passed $(cat "${check_file}")"
    return 0
  fi
  log "prefill self_check_failed $(cat "${check_file}" 2>/dev/null || true)"
  return 1
}

profile_once() {
  local attempt="$1"
  local out_dir="${OUT_ROOT}/${SCENARIO}"
  local tmp_dir="${TMP_ROOT}/${SCENARIO}/attempt${attempt}"
  local prof_dir="${tmp_dir}/prof"
  local serve_log="${out_dir}/serve_prefill_attempt${attempt}.log"
  local metrics_jsonl="${out_dir}/metrics_prefill_attempt${attempt}.jsonl"
  local bench_json="${tmp_dir}/bench_result.json"
  local bench_rec="${tmp_dir}/bench_records.jsonl"
  local bench_log="${tmp_dir}/bench.log"

  mkdir -p "${tmp_dir}" "${prof_dir}" "${out_dir}"
  : >"${serve_log}"
  : >"${metrics_jsonl}"
  rm -rf "${out_dir}/prof_prefill"

  start_service "${prof_dir}" "${serve_log}" "${attempt}"
  start_metrics_poller "${metrics_jsonl}"
  log "profile_start_before_bench attempt=${attempt}"
  curl --max-time 30 -fsS -X POST "http://127.0.0.1:${PORT}/start_profile"
  start_bench "${bench_json}" "${bench_rec}" "${bench_log}"
  sleep "${WINDOW_SECONDS}"
  log "profile_stop attempt=${attempt}"
  curl --max-time 30 -fsS -X POST "http://127.0.0.1:${PORT}/stop_profile" || log "warn: stop_profile returned non-zero; copying profiler files if present"
  sleep 20
  copy_profile_window "${prof_dir}" "${out_dir}"

  stop_metrics_poller
  if [[ -n "${BENCH_PID}" ]]; then
    kill -TERM "${BENCH_PID}" 2>/dev/null || true
    wait "${BENCH_PID}" 2>/dev/null || true
    BENCH_PID=""
  fi
  cp -f "${bench_log}" "${out_dir}/bench_prefill_attempt${attempt}.log" 2>/dev/null || true
  cp -f "${bench_rec}" "${out_dir}/bench_records_prefill_attempt${attempt}.jsonl" 2>/dev/null || true
  if [[ -f "${bench_json}" ]]; then
    cp -f "${bench_json}" "${out_dir}/bench_result_prefill_attempt${attempt}.json"
  fi

  stop_service
  wait_for_gpu_drain || { log "warn: drain timeout after attempt=${attempt}"; }
  run_window_check "${out_dir}" "${attempt}"
}

write_meta() {
  local out_dir="$1"
  cat >"${out_dir}/meta.json" <<EOF
{"name":"${SCENARIO}","phase":"phase433","profiler_mode":"${PROFILER_MODE}","nsys_available":${NSYS_AVAILABLE},"bench_num_prompts":${BENCH_NUM_PROMPTS},"bench_max_concurrency":${BENCH_MAX_CONCURRENCY},"tp":${TP},"dp":${DP},"ep":${EP},"isl":${ISL},"osl":${OSL},"max_num_batched_tokens":${MAX_NUM_BATCHED_TOKENS},"world_size":$((TP * DP)),"port":${PORT},"prefix_caching":false,"gpu_memory_utilization":${GPU_MEMORY_UTILIZATION},"max_model_len":${MAX_MODEL_LEN},"max_num_seqs":${MAX_NUM_SEQS},"enable_logging_iteration_details":true,"cudagraph_metrics":true,"active_iterations":${ACTIVE_ITERS},"window_seconds":${WINDOW_SECONDS},"min_ctx_tokens":${MIN_CTX_TOKENS}}
EOF
}

main() {
  cd "${WORKDIR}"
  mkdir -p "${OUT_ROOT}"
  if [[ "${FORCE}" == "1" ]]; then
    rm -rf "${OUT_ROOT}/${SCENARIO}"
    rm -f "${OUT_ROOT}/driver.log" "${OUT_ROOT}/runtime.txt" "${OUT_ROOT}/gpu_inventory.txt"
  fi

  probe_nsys
  local out_dir="${OUT_ROOT}/${SCENARIO}"
  mkdir -p "${out_dir}"
  : >"${out_dir}/window_checks.jsonl"
  write_meta "${out_dir}"

  python3 -c 'import vllm; print("vllm_version="+getattr(vllm,"__version__","unknown"))' | tee "${OUT_ROOT}/runtime.txt"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits >"${OUT_ROOT}/gpu_inventory.txt"
  sed -n '1p' "${OUT_ROOT}/gpu_inventory.txt" | tee -a "${OUT_ROOT}/runtime.txt"

  BASELINE_PIDS="$(gpu_busy_pids | tr '\n' ' ')"
  log "baseline_gpu_pids=[${BASELINE_PIDS}]" | tee -a "${OUT_ROOT}/driver.log"
  snapshot_gpu_apps "${OUT_ROOT}/gpu_compute_apps_before.txt"
  wait_for_gpu_drain || { log "abort: my gpu procs still busy"; return 1; }

  local attempt
  for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
    log "attempt=${attempt} starting"
    if profile_once "${attempt}"; then
      log "attempt=${attempt} accepted" | tee -a "${OUT_ROOT}/driver.log"
      break
    fi
    if (( attempt == MAX_ATTEMPTS )); then
      log "failed_all_attempts" | tee -a "${OUT_ROOT}/driver.log"
      return 1
    fi
    log "attempt=${attempt} rejected; retrying" | tee -a "${OUT_ROOT}/driver.log"
    stop_service
    wait_for_gpu_drain || true
  done

  {
    for file in "${out_dir}"/serve_prefill_attempt*.log; do
      [[ -f "${file}" ]] || continue
      echo "===== ${file} ====="
      cat "${file}"
    done
  } >"${out_dir}/serve.log"

  snapshot_gpu_apps "${OUT_ROOT}/gpu_compute_apps_after.txt"
  pgrep -af "vllm.entrypoints.cli.main serve|ray::|raylet|gcs_server|VLLM::APIServer|VLLM::EngineCore|run_openai_fixed_shape_benchmark" >"${OUT_ROOT}/process_residual_after.txt" || true
  log "PHASE433_DONE" | tee -a "${OUT_ROOT}/driver.log"
}

main "$@"
