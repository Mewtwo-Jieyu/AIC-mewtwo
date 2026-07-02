#!/usr/bin/env bash
set -euo pipefail

# Phase397l measurement driver: per-op decode breakdown for vLLM 0.19 ep8.
#
# For each config point it starts one vLLM 0.19 service with the native torch
# profiler enabled (--profiler-config), plus NCCL COLL tracing, runs a
# closed-loop fixed-shape benchmark, then -- once the run is in steady-state
# decode -- triggers /start_profile, captures a bounded window of decode
# iterations, and /stop_profile to dump per-rank chrome traces. The traces +
# NCCL logs are copied back to the repo out_dir for offline analysis.
#
# Measurement-only: does NOT touch the perf database, the simulator, or any
# gate constants. Mirrors the two 8k/2k MULTI_CONFIG_DATA topologies.

WORKDIR="${WORKDIR:-/mnt/shared-storage-user/ailab-sys/zhaojieyu/aic}"
MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--moonshotai--Kimi-K2.5}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-kimi-k2.5}"
OUT_ROOT="${OUT_ROOT:-docs/iter_gap_investigation/phase397l_decode_profile}"
TMP_ROOT="${TMP_ROOT:-/tmp/phase397l_$$}"
PORT_BASE="${PORT_BASE:-19800}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-128}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-128}"
BENCH_WARMUP_REQUESTS="${BENCH_WARMUP_REQUESTS:-0}"
BENCH_TIMEOUT_S="${BENCH_TIMEOUT_S:-7200}"

# Profiler window: after the benchmark has been running PROFILE_DELAY_S (so all
# prompts have finished prefill and are in steady-state decode), trigger the
# torch profiler for a bounded schedule (warmup + active iterations), then wait
# PROFILE_WINDOW_S before /stop_profile flushes the traces.
PROFILE_DELAY_S="${PROFILE_DELAY_S:-45}"
PROFILE_WINDOW_S="${PROFILE_WINDOW_S:-25}"
PROF_WARMUP_ITERS="${PROF_WARMUP_ITERS:-3}"
PROF_ACTIVE_ITERS="${PROF_ACTIVE_ITERS:-25}"

READY_TIMEOUT_POLLS="${READY_TIMEOUT_POLLS:-360}"   # 360 * 5s = 30 min for weight load
DRAIN_POLL_SECONDS="${DRAIN_POLL_SECONDS:-5}"
DRAIN_STABLE_POLLS="${DRAIN_STABLE_POLLS:-3}"
DRAIN_TIMEOUT_SECONDS="${DRAIN_TIMEOUT_SECONDS:-300}"
GRACEFUL_TERM_SECONDS="${GRACEFUL_TERM_SECONDS:-75}"

export PATH="/usr/local/nvidia/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export VLLM_ENABLE_CUDA_COMPATIBILITY="${VLLM_ENABLE_CUDA_COMPATIBILITY:-1}"
# Low-level NCCL collective trace (secondary cross-check for comm volume).
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-COLL,INIT}"

# name tp dp ep isl osl max_bt  (the two 8k/2k MULTI_CONFIG_DATA topologies)
POINTS=(
  "K2.5-tp8ep8-8k2k 8 1 8 8000 2000 8000"
  "K2.5-tp4ep8dp2-8k2k 4 2 8 8000 2000 8000"
)

SERVICE_PID=""
BENCH_PID=""
ONLY_POINTS="${ONLY_POINTS:-}"
CUR_PORT=""

log() { echo "[$(date -Is)] $*"; }

BASELINE_PIDS=""

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

stop_service() {
  if [[ -n "${BENCH_PID}" ]]; then
    kill -TERM "${BENCH_PID}" 2>/dev/null || true
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
  local port="$1" serve_log="$2" poll
  for poll in $(seq 1 "${READY_TIMEOUT_POLLS}"); do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      log "service_ready port=${port} poll=${poll}"
      return 0
    fi
    if [[ -n "${SERVICE_PID}" ]] && ! kill -0 "${SERVICE_PID}" 2>/dev/null; then
      log "service_exited_before_ready port=${port}"
      tail -n 120 "${serve_log}" || true
      return 1
    fi
    sleep 5
  done
  log "service_ready_timeout port=${port}"
  tail -n 120 "${serve_log}" || true
  return 1
}

run_point() {
  local idx="$1" spec="$2"
  read -r name tp dp ep isl osl max_bt <<<"${spec}"
  local port=$((PORT_BASE + idx))
  CUR_PORT="${port}"
  local out_dir="${OUT_ROOT}/${name}"
  local tmp_dir="${TMP_ROOT}/${name}"
  local prof_dir="${tmp_dir}/prof"
  local nccl_dir="${tmp_dir}/nccl_logs"
  local serve_log="${out_dir}/serve.log"

  if [[ -f "${out_dir}/prof_done.txt" ]]; then
    log "skip (already done): ${name}"
    return 0
  fi
  mkdir -p "${out_dir}" "${tmp_dir}" "${prof_dir}" "${nccl_dir}"

  log "=== point ${idx}: ${name} tp=${tp} dp=${dp} ep=${ep} isl=${isl} osl=${osl} max_bt=${max_bt} port=${port} ==="

  local busy
  busy="$(gpu_busy_pids_mine)"
  if [[ -n "${busy}" ]]; then
    log "gpu_busy_before_point mine_busy_pids=[$(echo "${busy}" | tr '\n' ' ')] - waiting to drain"
    wait_for_gpu_drain || { log "abort: my gpu procs still busy"; return 1; }
  fi

  # Torch profiler: schedule-based (warmup+active) so /start_profile captures a
  # bounded number of steady-state decode iterations. record_shapes=true so op
  # input shapes can be matched back to the perf DB; with_stack=false to keep
  # traces small.
  local profiler_json
  profiler_json="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${prof_dir}\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":true,\"warmup_iterations\":${PROF_WARMUP_ITERS},\"active_iterations\":${PROF_ACTIVE_ITERS},\"wait_iterations\":0}"

  export NCCL_DEBUG_FILE="${nccl_dir}/nccl.%h.%p.log"

  local serve_cmd=(
    python3 -m vllm.entrypoints.cli.main serve "${MODEL_PATH}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --port "${port}"
    --tensor-parallel-size "${tp}"
    --enable-expert-parallel
    --max-model-len "${MAX_MODEL_LEN}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-num-batched-tokens "${max_bt}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --enable-chunked-prefill
    --enable-prefix-caching
    --mm-encoder-tp-mode data
    --skip-mm-profiling
    --trust-remote-code
    --enable-auto-tool-choice
    --tool-call-parser kimi_k2
    --reasoning-parser kimi_k2
    --enable-logging-iteration-details
    --profiler-config "${profiler_json}"
  )
  if [[ "${dp}" != "1" ]]; then
    serve_cmd+=(--data-parallel-size "${dp}")
  fi

  cat > "${out_dir}/meta.json" <<EOF
{"name": "${name}", "tp": ${tp}, "dp": ${dp}, "ep": ${ep}, "isl": ${isl}, "osl": ${osl}, "max_num_batched_tokens": ${max_bt}, "batch_size": ${BENCH_MAX_CONCURRENCY}, "world_size": $((tp * dp)), "port": ${port}, "profile_delay_s": ${PROFILE_DELAY_S}, "profile_window_s": ${PROFILE_WINDOW_S}, "prof_warmup_iters": ${PROF_WARMUP_ITERS}, "prof_active_iters": ${PROF_ACTIVE_ITERS}}
EOF

  log "serve_command: ${serve_cmd[*]}"
  nohup "${serve_cmd[@]}" > "${serve_log}" 2>&1 &
  SERVICE_PID=$!
  log "service_pid=${SERVICE_PID}"

  if ! wait_for_service "${port}" "${serve_log}"; then
    stop_service
    wait_for_gpu_drain || true
    return 1
  fi

  # Fire the closed-loop benchmark in the background so the driver can time the
  # profiler window against steady-state decode.
  local bench_json="${tmp_dir}/bench_result.json"
  local bench_log="${tmp_dir}/bench.log"
  python3 scripts/run_openai_fixed_shape_benchmark.py \
    --host 127.0.0.1 --port "${port}" \
    --model "${SERVED_MODEL_NAME}" --tokenizer "${MODEL_PATH}" \
    --num-prompts "${BENCH_NUM_PROMPTS}" --max-concurrency "${BENCH_MAX_CONCURRENCY}" \
    --input-len "${isl}" --output-len "${osl}" \
    --warmup-requests "${BENCH_WARMUP_REQUESTS}" --timeout-s "${BENCH_TIMEOUT_S}" \
    --result-json "${bench_json}" > "${bench_log}" 2>&1 &
  BENCH_PID=$!
  log "bench_pid=${BENCH_PID} - waiting ${PROFILE_DELAY_S}s for steady-state decode"

  sleep "${PROFILE_DELAY_S}"
  if ! kill -0 "${BENCH_PID}" 2>/dev/null; then
    log "warn: benchmark finished before profiler window; tail:"
    tail -n 5 "${bench_log}" || true
  fi

  log "start_profile"
  curl -fsS -X POST "http://127.0.0.1:${port}/start_profile" || log "warn: start_profile failed"
  sleep "${PROFILE_WINDOW_S}"
  log "stop_profile"
  curl -fsS -X POST "http://127.0.0.1:${port}/stop_profile" || log "warn: stop_profile failed"

  # Give the profiler a moment to dump per-rank traces.
  sleep 20

  # Stop the benchmark (we only need the decode window, not full completion).
  if [[ -n "${BENCH_PID}" ]]; then
    kill -TERM "${BENCH_PID}" 2>/dev/null || true
    BENCH_PID=""
  fi

  # Copy artifacts off node-local tmp into repo out_dir.
  cp -f "${bench_json}" "${out_dir}/bench_result.json" 2>/dev/null || true
  cp -f "${bench_log}" "${out_dir}/bench.log" 2>/dev/null || true
  mkdir -p "${out_dir}/prof" "${out_dir}/nccl_logs"
  cp -f "${prof_dir}"/* "${out_dir}/prof/" 2>/dev/null || true
  cp -f "${nccl_dir}"/* "${out_dir}/nccl_logs/" 2>/dev/null || true
  local ntraces
  ntraces="$(find "${out_dir}/prof" -type f | wc -l | tr -d ' ')"
  log "copied prof traces=${ntraces} to ${out_dir}/prof"
  echo "traces=${ntraces}" > "${out_dir}/prof_done.txt"

  stop_service
  wait_for_gpu_drain || { log "warn: drain timeout after point ${name}"; }
  CUR_PORT=""
  return 0
}

main() {
  cd "${WORKDIR}"
  mkdir -p "${OUT_ROOT}" "${TMP_ROOT}"
  python3 -c 'import vllm; print("vllm_version="+getattr(vllm,"__version__","unknown"))' | tee "${OUT_ROOT}/runtime.txt"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits | head -n1 | tee -a "${OUT_ROOT}/runtime.txt"

  BASELINE_PIDS="$(gpu_busy_pids | tr '\n' ' ')"
  log "baseline_gpu_pids=[${BASELINE_PIDS}]"

  local rc=0 idx=0 spec name
  for spec in "${POINTS[@]}"; do
    if [[ -n "${ONLY_POINTS}" && ! " ${ONLY_POINTS} " == *" ${idx} "* ]]; then
      idx=$((idx + 1))
      continue
    fi
    if ! run_point "${idx}" "${spec}"; then
      read -r name _ <<<"${spec}"
      log "POINT_FAILED name=${name}"
      rc=1
    fi
    idx=$((idx + 1))
  done
  log "ALL_POINTS_DONE rc=${rc}"
  return "${rc}"
}

main "$@"
