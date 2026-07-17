#!/usr/bin/env bash
set -Eeuo pipefail

# Phase463 fixed-shape latency recollect.
# Measurement only: diagnostic_only=true, valid_for_default=false,
# perf_database=false, default_readiness=No-Go.

WORKDIR="${WORKDIR:-/mnt/shared-storage-user/zhaojieyu/backup/aic}"
MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--moonshotai--Kimi-K2.5}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-kimi-k2.5}"
OUT_ROOT="${OUT_ROOT:-phase463_six_point_latency_recollect_263a969}"
TMP_ROOT="${TMP_ROOT:-/tmp/phase463_six_point_latency_recollect_$$}"
PORT_BASE="${PORT_BASE:-21260}"
ONLY_POINTS="${ONLY_POINTS:-}"

BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-512}"
BENCH_WARMUP_REQUESTS="${BENCH_WARMUP_REQUESTS:-0}"
BENCH_TIMEOUT_S="${BENCH_TIMEOUT_S:-21600}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
SAMPLE_INTERVAL_S="${SAMPLE_INTERVAL_S:-2}"
READY_TIMEOUT_POLLS="${READY_TIMEOUT_POLLS:-480}"
DRAIN_POLL_SECONDS="${DRAIN_POLL_SECONDS:-5}"
DRAIN_STABLE_POLLS="${DRAIN_STABLE_POLLS:-3}"
DRAIN_TIMEOUT_SECONDS="${DRAIN_TIMEOUT_SECONDS:-600}"
GRACEFUL_TERM_SECONDS="${GRACEFUL_TERM_SECONDS:-120}"

export PATH="/opt/py3/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/kubebrain:/usr/local/nvidia/bin"
export LD_LIBRARY_PATH="/nccl/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.9/compat:/usr/local/lib/python3.12/dist-packages/torch/lib"
export VLLM_ENABLE_CUDA_COMPATIBILITY="${VLLM_ENABLE_CUDA_COMPATIBILITY:-1}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
unset VLLM_NUM_GPU_BLOCKS_OVERRIDE
unset NUM_GPU_BLOCKS_OVERRIDE

POINTS=(
  "K2.5-tp8ep8-8k2k-canary-nonstream canary_nonstream 0 8 1 8 8000 2000 8000 262144 128"
  "K2.5-tp8ep8-8k2k formal 1 8 1 8 8000 2000 8000 262144 128"
  "K2.5-tp8ep8-32k3k formal 1 8 1 8 32000 3000 32000 262144 128"
  "K2.5-tp4ep8dp2-8k2k formal 1 4 2 8 8000 2000 8000 131072 128"
  "K2.5-tp4ep8dp2-32k3k formal 1 4 2 8 32000 3000 32000 262144 128"
  "K2.5-tp8ep8-8k2k-bt65536 formal 1 8 1 8 8000 2000 65536 262144 128"
  "K2.5-tp4ep8dp2-8k2k-bt65536 formal 1 4 2 8 8000 2000 65536 131072 128"
  "K2.5-tp4ep8dp2-32k3k-c64-diagnostic diagnostic 1 4 2 8 32000 3000 32000 262144 64"
)

SERVICE_PID=""
METRICS_PID=""
GPU_PID=""
CURRENT_POINT="preflight"

log() { echo "[$(date -Is)] $*"; }

gpu_busy_pids() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '
}

service_process_pids() {
  pgrep -f "vllm.entrypoints.cli.main serve|ray::|raylet|gcs_server|VLLM::APIServer|VLLM::EngineCore" || true
}

snapshot_gpu_apps() {
  local path="$1"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >"${path}" 2>/dev/null || true
}

stop_samplers() {
  local pid
  for pid in "${METRICS_PID}" "${GPU_PID}"; do
    if [[ -n "${pid}" ]]; then
      kill -TERM "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
  METRICS_PID=""
  GPU_PID=""
}

stop_service() {
  stop_samplers
  if [[ -n "${SERVICE_PID}" ]]; then
    kill -TERM "${SERVICE_PID}" 2>/dev/null || true
  fi
  pkill -TERM -f "vllm.entrypoints.cli.main serve ${MODEL_PATH}" 2>/dev/null || true
  local waited=0
  while (( waited < GRACEFUL_TERM_SECONDS )); do
    if [[ -z "$(gpu_busy_pids)" && -z "$(service_process_pids)" ]]; then
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
  leftover="$(gpu_busy_pids)"
  if [[ -n "${leftover}" ]]; then
    # shellcheck disable=SC2086
    kill -KILL ${leftover} 2>/dev/null || true
  fi
  SERVICE_PID=""
}

point_failed() {
  local rc=$?
  trap - ERR
  log "POINT_FAILED name=${CURRENT_POINT} rc=${rc}"
  stop_service
  exit "${rc}"
}

trap stop_service EXIT
trap point_failed ERR

wait_for_gpu_drain() {
  local stable=0 elapsed=0 sample
  while (( elapsed <= DRAIN_TIMEOUT_SECONDS )); do
    sample="$(gpu_busy_pids)"
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
  log "gpu_drain_timeout busy_pids=[$(gpu_busy_pids | tr '\n' ' ')]"
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

start_metrics_sampler() {
  local port="$1" path="$2"
  python3 - "${port}" "${path}" "${SAMPLE_INTERVAL_S}" <<'PY' &
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
            with urllib.request.urlopen(url, timeout=5) as response:
                record["status"] = response.status
                record["body"] = response.read().decode("utf-8", errors="replace")
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
}

start_gpu_sampler() {
  local path="$1"
  (
    while true; do
      nvidia-smi \
        --query-gpu=timestamp,index,utilization.gpu,power.draw,memory.used \
        --format=csv,noheader,nounits >>"${path}"
      sleep "${SAMPLE_INTERVAL_S}"
    done
  ) &
  GPU_PID=$!
}

should_run_point() {
  local idx="$1" name="$2"
  [[ -z "${ONLY_POINTS}" || " ${ONLY_POINTS} " == *" ${idx} "* || " ${ONLY_POINTS} " == *" ${name} "* ]]
}

validate_canary() {
  local nonstream="${OUT_ROOT}/K2.5-tp8ep8-8k2k-canary-nonstream/bench_result.json"
  local stream="${OUT_ROOT}/K2.5-tp8ep8-8k2k/bench_result.json"
  python3 - "${nonstream}" "${stream}" <<'PY'
import json
import sys

nonstream = float(json.load(open(sys.argv[1], encoding="utf-8"))["output_tok_s"])
stream = float(json.load(open(sys.argv[2], encoding="utf-8"))["output_tok_s"])
delta = abs(stream / nonstream - 1.0) * 100.0
if delta > 2.0:
    raise SystemExit(f"stream_canary_throughput_delta:{delta:.6f}>2.0")
print(json.dumps({"nonstream_output_tok_s": nonstream, "stream_output_tok_s": stream, "delta_pct": delta, "passed": True}, indent=2))
PY
}

run_point() {
  local idx="$1" spec="$2"
  local name role stream tp dp ep isl osl max_bt max_model_len concurrency
  read -r name role stream tp dp ep isl osl max_bt max_model_len concurrency <<<"${spec}"
  local port=$((PORT_BASE + idx))
  local out_dir="${OUT_ROOT}/${name}"
  local tmp_dir="${TMP_ROOT}/${name}"
  local serve_log="${out_dir}/serve.log"

  if [[ -f "${out_dir}/scenario_analysis.json" ]]; then
    log "skip_validated_point=${name}"
    return 0
  fi
  if [[ -e "${out_dir}" ]]; then
    log "abort_partial_artifact=${out_dir}; archive it before retry"
    return 1
  fi
  mkdir -p "${out_dir}" "${tmp_dir}"
  : >"${serve_log}"
  : >"${out_dir}/metrics.jsonl"
  : >"${out_dir}/gpu.csv"

  if [[ -n "$(gpu_busy_pids)" || -n "$(service_process_pids)" ]]; then
    log "preflight_residue_detected gpu=[$(gpu_busy_pids | tr '\n' ' ')] process=[$(service_process_pids | tr '\n' ' ')]"
    return 1
  fi

  local serve_cmd=(
    python3 -m vllm.entrypoints.cli.main serve "${MODEL_PATH}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --port "${port}"
    --tensor-parallel-size "${tp}"
    --data-parallel-size "${dp}"
    --enable-expert-parallel
    --max-model-len "${max_model_len}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-num-batched-tokens "${max_bt}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --enable-chunked-prefill
    --no-enable-prefix-caching
    --enable-logging-iteration-details
    --cudagraph-metrics
    --mm-encoder-tp-mode data
    --skip-mm-profiling
    --trust-remote-code
    --enable-auto-tool-choice
    --tool-call-parser kimi_k2
    --reasoning-parser kimi_k2
  )

  cat >"${out_dir}/meta.json" <<EOF
{"name":"${name}","phase":"phase463","role":"${role}","stream":$([[ "${stream}" == "1" ]] && echo true || echo false),"bench_num_prompts":${BENCH_NUM_PROMPTS},"tp":${tp},"dp":${dp},"ep":${ep},"isl":${isl},"osl":${osl},"max_num_batched_tokens":${max_bt},"batch_size":${concurrency},"world_size":$((tp * dp)),"port":${port},"prefix_caching":false,"gpu_memory_utilization":${GPU_MEMORY_UTILIZATION},"max_model_len":${max_model_len},"max_num_seqs":${MAX_NUM_SEQS},"sample_interval_s":${SAMPLE_INTERVAL_S},"diagnostic_only":true,"valid_for_default":false,"perf_database":false,"default_readiness":"No-Go"}
EOF

  log "point_start idx=${idx} name=${name} role=${role} stream=${stream}"
  log "serve_command: ${serve_cmd[*]}"
  nohup "${serve_cmd[@]}" >"${serve_log}" 2>&1 &
  SERVICE_PID=$!
  wait_for_service "${port}" "${serve_log}" || return 1

  curl -fsS "http://127.0.0.1:${port}/metrics" >"${out_dir}/metrics_before.prom"
  start_metrics_sampler "${port}" "${out_dir}/metrics.jsonl"
  start_gpu_sampler "${out_dir}/gpu.csv"

  local stream_args=()
  if [[ "${stream}" == "1" ]]; then
    stream_args=(--stream)
  fi
  local bench_json="${tmp_dir}/bench_result.json"
  local bench_records="${out_dir}/bench_records.jsonl"
  local bench_log="${out_dir}/bench.log"
  local bench_exit=0
  set +e
  python3 scripts/run_openai_fixed_shape_benchmark.py \
    --host 127.0.0.1 --port "${port}" \
    --model "${SERVED_MODEL_NAME}" --tokenizer "${MODEL_PATH}" \
    --num-prompts "${BENCH_NUM_PROMPTS}" --max-concurrency "${concurrency}" \
    --input-len "${isl}" --output-len "${osl}" \
    --warmup-requests "${BENCH_WARMUP_REQUESTS}" --timeout-s "${BENCH_TIMEOUT_S}" \
    --skip-prompt-token-id-probe "${stream_args[@]}" \
    --result-json "${bench_json}" --records-jsonl "${bench_records}" >"${bench_log}" 2>&1
  bench_exit=$?
  set -e
  stop_samplers
  curl -fsS "http://127.0.0.1:${port}/metrics" >"${out_dir}/metrics_after.prom"
  if ! kill -0 "${SERVICE_PID}" 2>/dev/null; then
    log "server_restarted_or_exited_during_benchmark=${name}"
    return 1
  fi
  if [[ "${bench_exit}" != "0" || ! -f "${bench_json}" ]]; then
    log "benchmark_failed name=${name} exit=${bench_exit}"
    tail -n 120 "${bench_log}" || true
    return 1
  fi

  python3 - "${bench_json}" "${out_dir}/bench_result.json" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
data.pop("records", None)
with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
    f.write("\n")
PY

  python3 scripts/analyze_phase463_six_point_latency_recollect.py \
    --scenario-dir "${out_dir}" \
    --output-json "${out_dir}/scenario_analysis.json"

  stop_service
  wait_for_gpu_drain || return 1
  snapshot_gpu_apps "${out_dir}/gpu_compute_apps_after.txt"
  service_process_pids >"${out_dir}/process_residue_after.txt"
  if [[ -s "${out_dir}/gpu_compute_apps_after.txt" || -s "${out_dir}/process_residue_after.txt" ]]; then
    log "cleanup_residue name=${name}"
    return 1
  fi
  gzip -f "${out_dir}/metrics.jsonl" "${out_dir}/bench_records.jsonl" "${out_dir}/serve.log"
  log "point_done name=${name}"
}

main() {
  cd "${WORKDIR}"
  if [[ "${BENCH_WARMUP_REQUESTS}" != "0" ]]; then
    log "warmup_requests_must_be_zero for exact Prometheus deltas"
    return 1
  fi
  mkdir -p "${OUT_ROOT}" "${TMP_ROOT}"
  touch "${OUT_ROOT}/driver.log"
  python3 -c 'import vllm; print("vllm_version="+getattr(vllm,"__version__","unknown"))' | tee "${OUT_ROOT}/runtime.txt"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits >"${OUT_ROOT}/gpu_inventory.txt"
  snapshot_gpu_apps "${OUT_ROOT}/gpu_compute_apps_before.txt"
  if [[ -s "${OUT_ROOT}/gpu_compute_apps_before.txt" || -n "$(service_process_pids)" ]]; then
    log "abort_nonempty_worker" | tee -a "${OUT_ROOT}/driver.log"
    return 1
  fi

  local idx=0 spec name
  for spec in "${POINTS[@]}"; do
    read -r name _ <<<"${spec}"
    if should_run_point "${idx}" "${name}"; then
      CURRENT_POINT="${name}"
      run_point "${idx}" "${spec}" > >(tee -a "${OUT_ROOT}/driver.log") 2>&1
      if [[ "${idx}" == "1" ]]; then
        validate_canary | tee "${OUT_ROOT}/canary_gate.json" | tee -a "${OUT_ROOT}/driver.log"
      fi
    fi
    idx=$((idx + 1))
  done
  snapshot_gpu_apps "${OUT_ROOT}/gpu_compute_apps_after.txt"
  service_process_pids >"${OUT_ROOT}/process_residue_after.txt"
  if [[ -s "${OUT_ROOT}/gpu_compute_apps_after.txt" || -s "${OUT_ROOT}/process_residue_after.txt" ]]; then
    log "final_cleanup_residue" | tee -a "${OUT_ROOT}/driver.log"
    return 1
  fi
  log "phase463_done" | tee -a "${OUT_ROOT}/driver.log"
}

main "$@"
