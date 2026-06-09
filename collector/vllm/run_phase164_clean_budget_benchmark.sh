#!/usr/bin/env bash
set -euo pipefail

# Phase164 clean benchmark runner. Preflight prints commands only. run-one
# starts one vLLM service, runs one fixed-shape benchmark, writes clean metadata,
# and cleans up vLLM/Ray residue.

MODE="${1:-preflight}"
SCENARIO="${2:-${SCENARIO:-}}"

WORKDIR="${WORKDIR:-/mnt/nvme1n1/ml_research/jieyu/aic}"
MODEL_PATH="${MODEL_PATH:-/mnt/cfs/models/kimi-2.5-fp8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-kimi-k2.5}"
PORT_BASE="${PORT_BASE:-18000}"
OUT_ROOT="${OUT_ROOT:-docs/iter_gap_investigation/phase164_clean_gpu_budget_benchmark}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-128}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-128}"
BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-8000}"
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-2000}"
BENCH_WARMUP_REQUESTS="${BENCH_WARMUP_REQUESTS:-0}"
BENCH_TIMEOUT_S="${BENCH_TIMEOUT_S:-7200}"
ALLOW_EXISTING_GPU_APPS="${ALLOW_EXISTING_GPU_APPS:-0}"

export PATH="/usr/local/nvidia/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export VLLM_ENABLE_CUDA_COMPATIBILITY="${VLLM_ENABLE_CUDA_COMPATIBILITY:-1}"

SERVICE_PID=""
CLEANUP_LOG=""

usage() {
  cat <<'EOF'
Usage:
  run_phase164_clean_budget_benchmark.sh preflight
  run_phase164_clean_budget_benchmark.sh run-one <scenario>
  run_phase164_clean_budget_benchmark.sh cleanup

Scenarios:
  tp8ep8-bt8000
  tp8ep8-bt65536
  tp4dp2ep8-bt8000
  tp4dp2ep8-bt65536
EOF
}

scenario_index() {
  case "$1" in
    tp8ep8-bt8000) echo 0 ;;
    tp8ep8-bt65536) echo 1 ;;
    tp4dp2ep8-bt8000) echo 2 ;;
    tp4dp2ep8-bt65536) echo 3 ;;
    *) echo "unknown_scenario=$1" >&2; exit 2 ;;
  esac
}

scenario_tp() {
  case "$1" in
    tp8ep8-bt8000|tp8ep8-bt65536) echo 8 ;;
    tp4dp2ep8-bt8000|tp4dp2ep8-bt65536) echo 4 ;;
    *) echo "unknown_scenario=$1" >&2; exit 2 ;;
  esac
}

scenario_dp() {
  case "$1" in
    tp8ep8-bt8000|tp8ep8-bt65536) echo 1 ;;
    tp4dp2ep8-bt8000|tp4dp2ep8-bt65536) echo 2 ;;
    *) echo "unknown_scenario=$1" >&2; exit 2 ;;
  esac
}

scenario_ep() {
  case "$1" in
    tp8ep8-bt8000|tp8ep8-bt65536|tp4dp2ep8-bt8000|tp4dp2ep8-bt65536) echo 8 ;;
    *) echo "unknown_scenario=$1" >&2; exit 2 ;;
  esac
}

scenario_max_bt() {
  case "$1" in
    tp8ep8-bt8000|tp4dp2ep8-bt8000) echo 8000 ;;
    tp8ep8-bt65536|tp4dp2ep8-bt65536) echo 65536 ;;
    *) echo "unknown_scenario=$1" >&2; exit 2 ;;
  esac
}

scenario_port() {
  if [[ -n "${PORT:-}" ]]; then
    echo "${PORT}"
  else
    echo $((PORT_BASE + $(scenario_index "$1")))
  fi
}

scenario_dir() {
  echo "${OUT_ROOT}/$1"
}

quote_cmd() {
  printf "%q " "$@"
}

gpu_apps() {
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
}

check_gpu_state() {
  local out_file="$1"
  gpu_apps > "${out_file}"
  if [[ -s "${out_file}" && "${ALLOW_EXISTING_GPU_APPS}" != "1" ]]; then
    echo "gpu_busy=true" >&2
    cat "${out_file}" >&2
    exit 1
  fi
}

build_serve_cmd() {
  local scenario="$1"
  local tp dp max_bt port
  tp="$(scenario_tp "${scenario}")"
  dp="$(scenario_dp "${scenario}")"
  max_bt="$(scenario_max_bt "${scenario}")"
  port="$(scenario_port "${scenario}")"

  SERVE_CMD=(
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
    SERVE_CMD+=(--data-parallel-size "${dp}")
  fi
}

build_benchmark_cmd() {
  local scenario="$1"
  local port out_dir
  port="$(scenario_port "${scenario}")"
  out_dir="$(scenario_dir "${scenario}")"
  BENCH_CMD=(
    python3 scripts/run_openai_fixed_shape_benchmark.py
    --host 127.0.0.1
    --port "${port}"
    --model "${SERVED_MODEL_NAME}"
    --tokenizer "${MODEL_PATH}"
    --num-prompts "${BENCH_NUM_PROMPTS}"
    --max-concurrency "${BENCH_MAX_CONCURRENCY}"
    --input-len "${BENCH_INPUT_LEN}"
    --output-len "${BENCH_OUTPUT_LEN}"
    --warmup-requests "${BENCH_WARMUP_REQUESTS}"
    --timeout-s "${BENCH_TIMEOUT_S}"
    --result-json "${out_dir}/bench_result.json"
    --records-jsonl "${out_dir}/bench_records.jsonl"
  )
}

print_commands_for_scenario() {
  local scenario="$1"
  build_serve_cmd "${scenario}"
  build_benchmark_cmd "${scenario}"
  echo "scenario=${scenario}"
  echo "tp=$(scenario_tp "${scenario}") dp=$(scenario_dp "${scenario}") ep=$(scenario_ep "${scenario}") max_bt=$(scenario_max_bt "${scenario}") port=$(scenario_port "${scenario}")"
  echo "serve_command=$(quote_cmd "${SERVE_CMD[@]}")"
  echo "benchmark_command=$(quote_cmd "${BENCH_CMD[@]}")"
}

stop_service() {
  if [[ -n "${CLEANUP_LOG}" ]]; then
    echo "stop_service_start=$(date -Is)" >> "${CLEANUP_LOG}"
  fi
  if [[ -n "${SERVICE_PID}" ]]; then
    kill -TERM "${SERVICE_PID}" 2>/dev/null || true
    sleep 5
    kill -KILL "${SERVICE_PID}" 2>/dev/null || true
  fi
  pkill -f "vllm.entrypoints.cli.main serve ${MODEL_PATH}" 2>/dev/null || true
  pkill -f "vllm serve ${MODEL_PATH}" 2>/dev/null || true
  pkill -f "VLLM::DPCoordinator" 2>/dev/null || true
  ray stop --force >/dev/null 2>&1 || true
  if [[ -n "${CLEANUP_LOG}" ]]; then
    echo "stop_service_done=$(date -Is)" >> "${CLEANUP_LOG}"
  fi
}

cleanup() {
  set +e
  if [[ -n "${CLEANUP_LOG}" ]]; then
    echo "cleanup_trap_start=$(date -Is)" >> "${CLEANUP_LOG}"
  fi
  stop_service
  if [[ -n "${CLEANUP_LOG}" ]]; then
    echo "cleanup_trap_done=$(date -Is)" >> "${CLEANUP_LOG}"
  fi
}
trap cleanup EXIT

wait_for_service() {
  local port="$1"
  local serve_log="$2"
  local ready_log="$3"
  local poll response
  : > "${ready_log}"
  echo "ready_probe_start=$(date -Is) port=${port}" | tee -a "${ready_log}"
  for poll in $(seq 1 180); do
    if response="$(curl -fsS "http://127.0.0.1:${port}/v1/models" 2>&1)"; then
      echo "service_ready=true poll=${poll}" | tee -a "${ready_log}"
      printf '%s\n' "${response}" >> "${ready_log}"
      return
    fi
    echo "service_ready=false poll=${poll}" >> "${ready_log}"
    if [[ -n "${SERVICE_PID}" ]] && ! kill -0 "${SERVICE_PID}" 2>/dev/null; then
      echo "service_exited_before_ready=true" | tee -a "${ready_log}" >&2
      tail -n 200 "${serve_log}" >&2 || true
      exit 1
    fi
    sleep 5
  done
  echo "service_ready_timeout=true" | tee -a "${ready_log}" >&2
  tail -n 200 "${serve_log}" >&2 || true
  exit 1
}

write_result_json() {
  local scenario="$1"
  local out_dir="$2"
  local serve_command="$3"
  local benchmark_command="$4"
  local vllm_version gpu_csv cuda_version
  vllm_version="$(python3 - <<'PY'
import vllm
print(getattr(vllm, "__version__", "unknown"))
PY
)"
  gpu_csv="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits | head -n 1)"
  cuda_version="$(nvidia-smi | sed -n 's/.*CUDA Version: \([^ ]*\).*/\1/p' | head -n 1)"
  [[ -n "${cuda_version}" ]] || { echo "missing_cuda_version_from_nvidia_smi=true" >&2; exit 1; }

  PHASE164_SCENARIO="${scenario}" \
  PHASE164_TP="$(scenario_tp "${scenario}")" \
  PHASE164_DP="$(scenario_dp "${scenario}")" \
  PHASE164_EP="$(scenario_ep "${scenario}")" \
  PHASE164_MAX_BT="$(scenario_max_bt "${scenario}")" \
  PHASE164_WORLD_SIZE="$(( $(scenario_tp "${scenario}") * $(scenario_dp "${scenario}") ))" \
  PHASE164_GPU_COUNT="$(( $(scenario_tp "${scenario}") * $(scenario_dp "${scenario}") ))" \
  PHASE164_MODEL_PATH="${MODEL_PATH}" \
  PHASE164_VLLM_VERSION="${vllm_version}" \
  PHASE164_DTYPE="${DTYPE:-auto}" \
  PHASE164_QUANTIZATION="${QUANTIZATION:-fp8}" \
  PHASE164_GPU_CSV="${gpu_csv}" \
  PHASE164_CUDA_VERSION="${cuda_version}" \
  PHASE164_SERVE_COMMAND="${serve_command}" \
  PHASE164_BENCHMARK_COMMAND="${benchmark_command}" \
  PHASE164_BENCH_JSON="${out_dir}/bench_result.json" \
  PHASE164_OUT_JSON="${out_dir}/phase164_result.json" \
  PHASE164_MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
  PHASE164_BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS}" \
  python3 - <<'PY'
import json
import os
from pathlib import Path

bench_path = Path(os.environ["PHASE164_BENCH_JSON"])
bench = json.loads(bench_path.read_text(encoding="utf-8"))
if int(bench.get("failed_requests", -1)) != 0:
    raise SystemExit(f"failed_requests={bench.get('failed_requests')}")
if int(bench.get("ok_requests", 0)) != int(os.environ["PHASE164_BENCH_NUM_PROMPTS"]):
    raise SystemExit("ok_requests does not match requested prompt count")

gpu_parts = [part.strip() for part in os.environ["PHASE164_GPU_CSV"].split(",")]
if len(gpu_parts) < 3:
    raise SystemExit("invalid nvidia-smi gpu csv")

bench_clean = {k: v for k, v in bench.items() if k != "records"}
payload = {
    "scenario": os.environ["PHASE164_SCENARIO"],
    "bench_result": bench_clean,
    "shape": {
        "isl": int(bench["input_len"]),
        "osl": int(bench["output_len"]),
        "batch_size": int(bench["max_concurrency"]),
        "max_num_batched_tokens": int(os.environ["PHASE164_MAX_BT"]),
        "max_num_seqs": int(os.environ["PHASE164_MAX_NUM_SEQS"]),
    },
    "parallelism": {
        "tp": int(os.environ["PHASE164_TP"]),
        "dp": int(os.environ["PHASE164_DP"]),
        "ep": int(os.environ["PHASE164_EP"]),
        "world_size": int(os.environ["PHASE164_WORLD_SIZE"]),
        "gpu_count": int(os.environ["PHASE164_GPU_COUNT"]),
    },
    "runtime": {
        "vllm_version": os.environ["PHASE164_VLLM_VERSION"],
        "model_path": os.environ["PHASE164_MODEL_PATH"],
        "dtype": os.environ["PHASE164_DTYPE"],
        "quantization": os.environ["PHASE164_QUANTIZATION"],
        "serve_command": os.environ["PHASE164_SERVE_COMMAND"],
        "benchmark_command": os.environ["PHASE164_BENCHMARK_COMMAND"],
    },
    "hardware": {
        "gpu_model": gpu_parts[0],
        "gpu_memory_gb": round(float(gpu_parts[1]) / 1024.0, 3),
        "driver_version": gpu_parts[2],
        "cuda_version": os.environ["PHASE164_CUDA_VERSION"],
    },
    "diagnostic_only": True,
    "valid_for_default": False,
    "perf_database": False,
}
Path(os.environ["PHASE164_OUT_JSON"]).write_text(
    json.dumps(payload, indent=2, sort_keys=True),
    encoding="utf-8",
)
PY
}

run_preflight() {
  cd "${WORKDIR}"
  mkdir -p "${OUT_ROOT}"
  check_gpu_state "${OUT_ROOT}/gpu_compute_apps_preflight.txt"
  [[ -f scripts/run_openai_fixed_shape_benchmark.py ]] || {
    echo "missing_benchmark_script=scripts/run_openai_fixed_shape_benchmark.py" >&2
    exit 2
  }
  echo "phase164_mode=preflight"
  echo "workdir=${WORKDIR}"
  echo "model_path=${MODEL_PATH}"
  echo "output_root=${OUT_ROOT}"
  for scenario in tp8ep8-bt8000 tp8ep8-bt65536 tp4dp2ep8-bt8000 tp4dp2ep8-bt65536; do
    print_commands_for_scenario "${scenario}"
  done
  echo "required_result_schema=throughput,success_fail,shape,runtime,gpu,serve_command,benchmark_command,diagnostic_flags"
  echo "diagnostic_only=true valid_for_default=false perf_database=false"
  echo "preflight_benchmark_started=false"
}

run_one() {
  local scenario="$1"
  local out_dir serve_log bench_json serve_command benchmark_command port ready_log run_log
  [[ -n "${scenario}" ]] || { usage >&2; exit 2; }
  scenario_index "${scenario}" >/dev/null

  cd "${WORKDIR}"
  [[ -f scripts/run_openai_fixed_shape_benchmark.py ]] || {
    echo "missing_benchmark_script=scripts/run_openai_fixed_shape_benchmark.py" >&2
    exit 2
  }
  out_dir="$(scenario_dir "${scenario}")"
  mkdir -p "${out_dir}"

  port="$(scenario_port "${scenario}")"
  serve_log="${out_dir}/serve.log"
  ready_log="${out_dir}/ready_probe_${port}.log"
  CLEANUP_LOG="${out_dir}/cleanup_${scenario}.log"
  run_log="${out_dir}/run_one_${scenario}.log"
  : > "${run_log}"
  echo "service_log=${serve_log}"
  echo "runner_log=${run_log}"
  echo "ready_probe_log=${ready_log}"
  echo "cleanup_log=${CLEANUP_LOG}"
  echo "realtime_tail_command=tail -f ${serve_log} ${run_log} ${ready_log} ${CLEANUP_LOG}"
  exec >> "${run_log}" 2>&1

  check_gpu_state "${out_dir}/gpu_compute_apps_before.txt"

  build_serve_cmd "${scenario}"
  build_benchmark_cmd "${scenario}"
  serve_command="$(quote_cmd "${SERVE_CMD[@]}")"
  benchmark_command="$(quote_cmd "${BENCH_CMD[@]}")"
  bench_json="${out_dir}/bench_result.json"

  echo "scenario=${scenario}"
  echo "runner_log_start=$(date -Is)"
  echo "service_log=${serve_log}"
  echo "runner_log=${run_log}"
  echo "ready_probe_log=${ready_log}"
  echo "cleanup_log=${CLEANUP_LOG}"
  echo "realtime_tail_command=tail -f ${serve_log} ${run_log} ${ready_log} ${CLEANUP_LOG}"
  echo "serve_command=${serve_command}"
  echo "benchmark_command=${benchmark_command}"
  nohup "${SERVE_CMD[@]}" > "${serve_log}" 2>&1 &
  SERVICE_PID=$!
  echo "service_pid=${SERVICE_PID}"
  wait_for_service "${port}" "${serve_log}" "${ready_log}"

  "${BENCH_CMD[@]}"
  write_result_json "${scenario}" "${out_dir}" "${serve_command}" "${benchmark_command}"
  stop_service
  SERVICE_PID=""
  gpu_apps > "${out_dir}/gpu_compute_apps_after.txt" || true
  if [[ -s "${out_dir}/gpu_compute_apps_after.txt" ]]; then
    echo "gpu_residue_after_run=true" >&2
    cat "${out_dir}/gpu_compute_apps_after.txt" >&2
    exit 1
  fi
  [[ -f "${bench_json}" ]] || { echo "missing_bench_json=${bench_json}" >&2; exit 1; }
  echo "phase164_result_json=${out_dir}/phase164_result.json"
}

case "${MODE}" in
  preflight)
    run_preflight
    ;;
  run-one)
    run_one "${SCENARIO}"
    ;;
  cleanup)
    cleanup
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
