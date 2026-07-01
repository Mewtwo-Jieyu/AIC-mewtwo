#!/usr/bin/env bash
set -euo pipefail

# Phase397k measurement driver.
#
# Freshly measures vLLM 0.19 aggregate serving throughput on a single 8-GPU
# H200 node across the six MULTI_CONFIG_DATA points (mirrors
# scripts/validate_cb_simulator.py). Each point: start one vLLM service, run one
# fixed-shape closed-loop benchmark, save bench_result.json + meta.json, then
# stop the service and wait for the GPUs to drain before the next point.
#
# Unlike collector/vllm/run_phase164_clean_budget_benchmark.sh this driver
# parametrizes ISL/OSL/max_num_batched_tokens per point, so it also covers the
# two 32k/3k (bt32000) points the phase164 harness does not name.
#
# Measurement-only: does NOT touch the perf database or the simulator.

WORKDIR="${WORKDIR:-/mnt/shared-storage-user/ailab-sys/zhaojieyu/aic}"
MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--moonshotai--Kimi-K2.5}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-kimi-k2.5}"
OUT_ROOT="${OUT_ROOT:-docs/iter_gap_investigation/phase397k_measured_0190}"
TMP_ROOT="${TMP_ROOT:-/tmp/phase397k_$$}"
PORT_BASE="${PORT_BASE:-19100}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-128}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-128}"
BENCH_WARMUP_REQUESTS="${BENCH_WARMUP_REQUESTS:-0}"
BENCH_TIMEOUT_S="${BENCH_TIMEOUT_S:-7200}"

READY_TIMEOUT_POLLS="${READY_TIMEOUT_POLLS:-360}"   # 360 * 5s = 30 min for weight load
DRAIN_POLL_SECONDS="${DRAIN_POLL_SECONDS:-5}"
DRAIN_STABLE_POLLS="${DRAIN_STABLE_POLLS:-3}"
DRAIN_TIMEOUT_SECONDS="${DRAIN_TIMEOUT_SECONDS:-300}"

export PATH="/usr/local/nvidia/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export VLLM_ENABLE_CUDA_COMPATIBILITY="${VLLM_ENABLE_CUDA_COMPATIBILITY:-1}"

# name tp dp ep isl osl max_bt  (mirrors validate_cb_simulator.MULTI_CONFIG_DATA)
POINTS=(
  "K2.5-tp8ep8-8k2k 8 1 8 8000 2000 8000"
  "K2.5-tp8ep8-32k3k 8 1 8 32000 3000 32000"
  "K2.5-tp4ep8dp2-8k2k 4 2 8 8000 2000 8000"
  "K2.5-tp4ep8dp2-32k3k 4 2 8 32000 3000 32000"
  "K2.5-tp8ep8-8k2k-bt65536 8 1 8 8000 2000 65536"
  "K2.5-tp4ep8dp2-8k2k-bt65536 4 2 8 8000 2000 65536"
)

SERVICE_PID=""
# Space-separated list of point indices to run; empty = all.
ONLY_POINTS="${ONLY_POINTS:-}"
# Seconds to wait for a graceful SIGTERM shutdown before SIGKILL. DP
# (--data-parallel-size) teardown needs a long graceful window: an early
# SIGKILL orphans the DP workers and leaks their VRAM (unreclaimable without a
# GPU reset), which blocks every subsequent point on the node.
GRACEFUL_TERM_SECONDS="${GRACEFUL_TERM_SECONDS:-75}"

log() { echo "[$(date -Is)] $*"; }

# GPU compute-app pids present before this driver started (pre-existing orphans
# / other tenants). They are ignored by the drain/teardown logic; only pids we
# spawn ("mine") gate progress. Populated in main().
BASELINE_PIDS=""

gpu_busy_pids() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '
}

# GPU compute-app pids that are NOT in the baseline set (i.e. spawned by us).
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
  # Graceful first: give vLLM (and any DP workers) time to release VRAM.
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
  # Hard kill anything that is left.
  if [[ -n "${SERVICE_PID}" ]]; then
    kill -KILL "${SERVICE_PID}" 2>/dev/null || true
  fi
  pkill -KILL -f "vllm.entrypoints.cli.main serve ${MODEL_PATH}" 2>/dev/null || true
  pkill -KILL -f "VLLM::" 2>/dev/null || true
  ray stop --force >/dev/null 2>&1 || true
  # Last resort: kill whatever still holds the GPU, by pid (ours only).
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
    --enable-prefix-caching
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

  # meta for downstream manifest build
  cat > "${out_dir}/meta.json" <<EOF
{"name": "${name}", "tp": ${tp}, "dp": ${dp}, "ep": ${ep}, "isl": ${isl}, "osl": ${osl}, "max_num_batched_tokens": ${max_bt}, "batch_size": ${BENCH_MAX_CONCURRENCY}, "world_size": $((tp * dp)), "port": ${port}}
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

  local bench_json="${tmp_dir}/bench_result.json"
  local bench_rec="${tmp_dir}/bench_records.jsonl"
  local bench_exit=0
  set +e
  python3 scripts/run_openai_fixed_shape_benchmark.py \
    --host 127.0.0.1 --port "${port}" \
    --model "${SERVED_MODEL_NAME}" --tokenizer "${MODEL_PATH}" \
    --num-prompts "${BENCH_NUM_PROMPTS}" --max-concurrency "${BENCH_MAX_CONCURRENCY}" \
    --input-len "${isl}" --output-len "${osl}" \
    --warmup-requests "${BENCH_WARMUP_REQUESTS}" --timeout-s "${BENCH_TIMEOUT_S}" \
    --result-json "${bench_json}" --records-jsonl "${bench_rec}"
  bench_exit="$?"
  set -e
  log "benchmark_exit_code=${bench_exit}"

  # Copy off node-local tmp into repo out_dir (guards against NFS lock issues
  # during the write). Strip the bulky per-request records array.
  if [[ -f "${bench_json}" ]]; then
    python3 - "${bench_json}" "${out_dir}/bench_result.json" <<'PY'
import json, sys
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
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits | head -n1 | tee -a "${OUT_ROOT}/runtime.txt"

  # Record pre-existing GPU compute-app pids (orphaned allocations / other
  # tenants) so drain/teardown ignore them and only gate on our own processes.
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
