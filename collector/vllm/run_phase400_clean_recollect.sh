#!/usr/bin/env bash
set -euo pipefail

# Phase400 clean recollect driver.
#
# Runs the four 0.19-real 8-GPU MULTI_CONFIG points with prefix caching
# disabled and server stats enabled. This is measurement-only: it does not
# change simulator code, perf tables, or gate thresholds.

WORKDIR="${WORKDIR:-/mnt/shared-storage-user/ailab-sys/zhaojieyu/aic}"
MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--moonshotai--Kimi-K2.5}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-kimi-k2.5}"
OUT_ROOT="${OUT_ROOT:-docs/iter_gap_investigation/phase400_clean_recollect}"
TMP_ROOT="${TMP_ROOT:-/tmp/phase400_clean_recollect_$$}"
PORT_BASE="${PORT_BASE:-20700}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-128}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-128}"
BENCH_WARMUP_REQUESTS="${BENCH_WARMUP_REQUESTS:-0}"
BENCH_TIMEOUT_S="${BENCH_TIMEOUT_S:-14400}"

READY_TIMEOUT_POLLS="${READY_TIMEOUT_POLLS:-480}"   # 480 * 5s = 40 min
DRAIN_POLL_SECONDS="${DRAIN_POLL_SECONDS:-5}"
DRAIN_STABLE_POLLS="${DRAIN_STABLE_POLLS:-3}"
DRAIN_TIMEOUT_SECONDS="${DRAIN_TIMEOUT_SECONDS:-600}"
GRACEFUL_TERM_SECONDS="${GRACEFUL_TERM_SECONDS:-120}"

export PATH="/usr/local/nvidia/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export VLLM_ENABLE_CUDA_COMPATIBILITY="${VLLM_ENABLE_CUDA_COMPATIBILITY:-1}"
unset VLLM_NUM_GPU_BLOCKS_OVERRIDE
unset NUM_GPU_BLOCKS_OVERRIDE

POINTS=(
  "K2.5-tp8ep8-8k2k 8 1 8 8000 2000 8000"
  "K2.5-tp8ep8-32k3k 8 1 8 32000 3000 32000"
  "K2.5-tp4ep8dp2-8k2k 4 2 8 8000 2000 8000"
  "K2.5-tp4ep8dp2-32k3k 4 2 8 32000 3000 32000"
)

SERVICE_PID=""
ONLY_POINTS="${ONLY_POINTS:-}"
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

stop_service() {
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
      tail -n 160 "${serve_log}" || true
      return 1
    fi
    sleep 5
  done
  log "service_ready_timeout port=${port}"
  tail -n 160 "${serve_log}" || true
  return 1
}

run_point() {
  local idx="$1" spec="$2"
  read -r name tp dp ep isl osl max_bt <<<"${spec}"
  local port=$((PORT_BASE + idx))
  local out_dir="${OUT_ROOT}/${name}"
  local tmp_dir="${TMP_ROOT}/${name}"
  local serve_log="${out_dir}/serve.log"

  if [[ -f "${out_dir}/bench_result.json" ]]; then
    log "skip (already done): ${name}"
    return 0
  fi
  mkdir -p "${out_dir}" "${tmp_dir}"

  log "=== point ${idx}: ${name} tp=${tp} dp=${dp} ep=${ep} isl=${isl} osl=${osl} max_bt=${max_bt} port=${port} ==="

  local busy
  busy="$(gpu_busy_pids_mine)"
  if [[ -n "${busy}" ]]; then
    log "gpu_busy_before_point mine_busy_pids=[$(echo "${busy}" | tr '\n' ' ')] - waiting to drain"
    wait_for_gpu_drain || { log "abort: my gpu procs still busy"; return 1; }
  fi

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
    --no-enable-prefix-caching
    --mm-encoder-tp-mode data
    --skip-mm-profiling
    --trust-remote-code
    --enable-auto-tool-choice
    --tool-call-parser kimi_k2
    --reasoning-parser kimi_k2
  )
  if [[ "${dp}" != "1" ]]; then
    serve_cmd+=(--data-parallel-size "${dp}")
  fi

  cat >"${out_dir}/meta.json" <<EOF
{"name": "${name}", "phase": "phase400", "tp": ${tp}, "dp": ${dp}, "ep": ${ep}, "isl": ${isl}, "osl": ${osl}, "max_num_batched_tokens": ${max_bt}, "batch_size": ${BENCH_MAX_CONCURRENCY}, "world_size": $((tp * dp)), "port": ${port}, "prefix_caching": false, "gpu_memory_utilization": ${GPU_MEMORY_UTILIZATION}, "max_model_len": ${MAX_MODEL_LEN}, "max_num_seqs": ${MAX_NUM_SEQS}}
EOF

  log "serve_command: ${serve_cmd[*]}"
  nohup "${serve_cmd[@]}" >"${serve_log}" 2>&1 &
  SERVICE_PID=$!
  log "service_pid=${SERVICE_PID}"

  if ! wait_for_service "${port}" "${serve_log}"; then
    stop_service
    wait_for_gpu_drain || true
    return 1
  fi

  local bench_json="${tmp_dir}/bench_result.json"
  local bench_rec="${tmp_dir}/bench_records.jsonl"
  local bench_log="${tmp_dir}/bench.log"
  local bench_exit=0
  set +e
  python3 scripts/run_openai_fixed_shape_benchmark.py \
    --host 127.0.0.1 --port "${port}" \
    --model "${SERVED_MODEL_NAME}" --tokenizer "${MODEL_PATH}" \
    --num-prompts "${BENCH_NUM_PROMPTS}" --max-concurrency "${BENCH_MAX_CONCURRENCY}" \
    --input-len "${isl}" --output-len "${osl}" \
    --warmup-requests "${BENCH_WARMUP_REQUESTS}" --timeout-s "${BENCH_TIMEOUT_S}" \
    --result-json "${bench_json}" --records-jsonl "${bench_rec}" >"${bench_log}" 2>&1
  bench_exit="$?"
  set -e
  log "benchmark_exit_code=${bench_exit}"

  cp -f "${bench_log}" "${out_dir}/bench.log" 2>/dev/null || true
  cp -f "${bench_rec}" "${out_dir}/bench_records.jsonl" 2>/dev/null || true
  if [[ -f "${bench_json}" ]]; then
    python3 - "${bench_json}" "${out_dir}/bench_result.json" <<'PY'
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
data = json.loads(open(src, encoding="utf-8").read())
data.pop("records", None)
open(dst, "w", encoding="utf-8").write(json.dumps(data, indent=2, ensure_ascii=False))
PY
    log "wrote ${out_dir}/bench_result.json"
  else
    log "missing_bench_json=${bench_json}"
  fi

  stop_service
  wait_for_gpu_drain || { log "warn: drain timeout after point ${name}"; }
  return "${bench_exit}"
}

main() {
  cd "${WORKDIR}"
  mkdir -p "${OUT_ROOT}" "${TMP_ROOT}"
  python3 -c 'import vllm; print("vllm_version="+getattr(vllm,"__version__","unknown"))' | tee "${OUT_ROOT}/runtime.txt"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits >"${OUT_ROOT}/gpu_inventory.txt"
  sed -n '1p' "${OUT_ROOT}/gpu_inventory.txt" | tee -a "${OUT_ROOT}/runtime.txt"

  BASELINE_PIDS="$(gpu_busy_pids | tr '\n' ' ')"
  log "baseline_gpu_pids=[${BASELINE_PIDS}]" | tee -a "${OUT_ROOT}/driver.log"
  snapshot_gpu_apps "${OUT_ROOT}/gpu_compute_apps_before.txt"

  local rc=0 idx=0 spec name
  for spec in "${POINTS[@]}"; do
    if [[ -n "${ONLY_POINTS}" && ! " ${ONLY_POINTS} " == *" ${idx} "* ]]; then
      idx=$((idx + 1))
      continue
    fi
    if ! run_point "${idx}" "${spec}" 2>&1 | tee -a "${OUT_ROOT}/driver.log"; then
      read -r name _ <<<"${spec}"
      log "POINT_FAILED name=${name}" | tee -a "${OUT_ROOT}/driver.log"
      rc=1
      break
    fi
    idx=$((idx + 1))
  done

  snapshot_gpu_apps "${OUT_ROOT}/gpu_compute_apps_after.txt"
  pgrep -af "vllm|ray|APIServer|EngineCore|run_openai_fixed_shape_benchmark" >"${OUT_ROOT}/process_residual_after.txt" || true
  log "ALL_POINTS_DONE rc=${rc}" | tee -a "${OUT_ROOT}/driver.log"
  return "${rc}"
}

main "$@"
