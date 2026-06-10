#!/usr/bin/env bash
set -euo pipefail

# Phase178 holdout runner. Preflight prints commands only. run-one is reserved
# for a later GPU phase and keeps the same log/cleanup contract as Phase164.

MODE="${1:-preflight}"
SCENARIO="${2:-${SCENARIO:-}}"

WORKDIR="${WORKDIR:-/mnt/nvme1n1/ml_research/jieyu/aic}"
MODEL_PATH="${MODEL_PATH:-/mnt/cfs/models/kimi-2.5-fp8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-kimi-k2.5}"
PORT_BASE="${PORT_BASE:-18100}"
OUT_ROOT="${OUT_ROOT:-docs/iter_gap_investigation/phase178_budget_mechanism_holdout}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-128}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-128}"
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-2000}"
BENCH_WARMUP_REQUESTS="${BENCH_WARMUP_REQUESTS:-0}"
BENCH_TIMEOUT_S="${BENCH_TIMEOUT_S:-7200}"
ALLOW_EXISTING_GPU_APPS="${ALLOW_EXISTING_GPU_APPS:-0}"
GPU_DRAIN_POLL_SECONDS="${GPU_DRAIN_POLL_SECONDS:-5}"
GPU_DRAIN_STABLE_POLLS="${GPU_DRAIN_STABLE_POLLS:-3}"
GPU_DRAIN_TIMEOUT_SECONDS="${GPU_DRAIN_TIMEOUT_SECONDS:-300}"

export PATH="/usr/local/nvidia/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export VLLM_ENABLE_CUDA_COMPATIBILITY="${VLLM_ENABLE_CUDA_COMPATIBILITY:-1}"

SERVICE_PID=""
CLEANUP_LOG=""

SCENARIOS=(
  tp8ep8-4k2k-bt4000
  tp8ep8-4k2k-bt65536
  tp4dp2ep8-4k2k-bt4000
  tp4dp2ep8-4k2k-bt65536
  tp8ep8-12k2k-bt12000
  tp8ep8-12k2k-bt65536
  tp4dp2ep8-12k2k-bt12000
  tp4dp2ep8-12k2k-bt65536
)

usage() {
  cat <<'EOF'
Usage:
  run_phase178_budget_mechanism_holdout.sh preflight
  run_phase178_budget_mechanism_holdout.sh run-one <scenario>
  run_phase178_budget_mechanism_holdout.sh cleanup

Scenarios:
  tp8ep8-4k2k-bt4000
  tp8ep8-4k2k-bt65536
  tp4dp2ep8-4k2k-bt4000
  tp4dp2ep8-4k2k-bt65536
  tp8ep8-12k2k-bt12000
  tp8ep8-12k2k-bt65536
  tp4dp2ep8-12k2k-bt12000
  tp4dp2ep8-12k2k-bt65536
EOF
}

scenario_index() {
  case "$1" in
    tp8ep8-4k2k-bt4000) echo 0 ;;
    tp8ep8-4k2k-bt65536) echo 1 ;;
    tp4dp2ep8-4k2k-bt4000) echo 2 ;;
    tp4dp2ep8-4k2k-bt65536) echo 3 ;;
    tp8ep8-12k2k-bt12000) echo 4 ;;
    tp8ep8-12k2k-bt65536) echo 5 ;;
    tp4dp2ep8-12k2k-bt12000) echo 6 ;;
    tp4dp2ep8-12k2k-bt65536) echo 7 ;;
    *) echo "unknown_scenario=$1" >&2; exit 2 ;;
  esac
}

scenario_tp() {
  case "$1" in
    tp8ep8-*) echo 8 ;;
    tp4dp2ep8-*) echo 4 ;;
    *) echo "unknown_scenario=$1" >&2; exit 2 ;;
  esac
}

scenario_dp() {
  case "$1" in
    tp8ep8-*) echo 1 ;;
    tp4dp2ep8-*) echo 2 ;;
    *) echo "unknown_scenario=$1" >&2; exit 2 ;;
  esac
}

scenario_ep() {
  scenario_index "$1" >/dev/null
  echo 8
}

scenario_isl() {
  case "$1" in
    *-4k2k-*) echo 4000 ;;
    *-12k2k-*) echo 12000 ;;
    *) echo "unknown_scenario=$1" >&2; exit 2 ;;
  esac
}

scenario_max_bt() {
  case "$1" in
    *-bt4000) echo 4000 ;;
    *-bt12000) echo 12000 ;;
    *-bt65536) echo 65536 ;;
    *) echo "unknown_scenario=$1" >&2; exit 2 ;;
  esac
}

scenario_topology_key() {
  case "$1" in
    tp8ep8-*) echo tp8_dp1_ep8 ;;
    tp4dp2ep8-*) echo tp4_dp2_ep8 ;;
    *) echo "unknown_scenario=$1" >&2; exit 2 ;;
  esac
}

scenario_shape_key() {
  case "$(scenario_isl "$1")" in
    4000) echo isl4000_osl2000_batch128 ;;
    12000) echo isl12000_osl2000_batch128 ;;
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
      return 0
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
  [[ -n "${PHASE178_STEADY_STATE_TIME_MS:-}" ]] || {
    echo "missing_phase178_steady_state_time_ms=true" >&2
    exit 2
  }
  vllm_version="$(python3 - <<'PYVLLM'
import vllm
print(getattr(vllm, "__version__", "unknown"))
PYVLLM
)"
  gpu_csv="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits | head -n 1)"
  cuda_version="$(nvidia-smi | sed -n 's/.*CUDA Version: \([^ ]*\).*/\1/p' | head -n 1)"
  [[ -n "${cuda_version}" ]] || { echo "missing_cuda_version_from_nvidia_smi=true" >&2; exit 1; }

  PHASE178_SCENARIO="${scenario}" \
  PHASE178_TOPOLOGY_KEY="$(scenario_topology_key "${scenario}")" \
  PHASE178_SHAPE_KEY="$(scenario_shape_key "${scenario}")" \
  PHASE178_TP="$(scenario_tp "${scenario}")" \
  PHASE178_DP="$(scenario_dp "${scenario}")" \
  PHASE178_EP="$(scenario_ep "${scenario}")" \
  PHASE178_ISL="$(scenario_isl "${scenario}")" \
  PHASE178_MAX_BT="$(scenario_max_bt "${scenario}")" \
  PHASE178_WORLD_SIZE="$(( $(scenario_tp "${scenario}") * $(scenario_dp "${scenario}") ))" \
  PHASE178_GPU_COUNT="$(( $(scenario_tp "${scenario}") * $(scenario_dp "${scenario}") ))" \
  PHASE178_MODEL_PATH="${MODEL_PATH}" \
  PHASE178_VLLM_VERSION="${vllm_version}" \
  PHASE178_DTYPE="${DTYPE:-auto}" \
  PHASE178_QUANTIZATION="${QUANTIZATION:-fp8}" \
  PHASE178_GPU_CSV="${gpu_csv}" \
  PHASE178_CUDA_VERSION="${cuda_version}" \
  PHASE178_SERVE_COMMAND="${serve_command}" \
  PHASE178_BENCHMARK_COMMAND="${benchmark_command}" \
  PHASE178_BENCH_JSON="${out_dir}/bench_result.json" \
  PHASE178_OUT_JSON="${out_dir}/phase178_result.json" \
  PHASE178_MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
  PHASE178_BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS}" \
  python3 - <<'PYJSON'
import json
import os
from pathlib import Path

bench_path = Path(os.environ["PHASE178_BENCH_JSON"])
bench = json.loads(bench_path.read_text(encoding="utf-8"))
if int(bench.get("failed_requests", -1)) != 0:
    raise SystemExit(f"failed_requests={bench.get('failed_requests')}")
if int(bench.get("ok_requests", 0)) != int(os.environ["PHASE178_BENCH_NUM_PROMPTS"]):
    raise SystemExit("ok_requests does not match requested prompt count")

gpu_parts = [part.strip() for part in os.environ["PHASE178_GPU_CSV"].split(",")]
if len(gpu_parts) < 3:
    raise SystemExit("invalid nvidia-smi gpu csv")

bench_clean = {k: v for k, v in bench.items() if k != "records"}
payload = {
    "source": "phase178_budget_mechanism_holdout",
    "scenario": os.environ["PHASE178_SCENARIO"],
    "topology_key": os.environ["PHASE178_TOPOLOGY_KEY"],
    "shape_key": os.environ["PHASE178_SHAPE_KEY"],
    "bench_result": bench_clean,
    "shape": {
        "isl": int(os.environ["PHASE178_ISL"]),
        "osl": int(bench["output_len"]),
        "batch_size": int(bench["max_concurrency"]),
        "max_num_batched_tokens": int(os.environ["PHASE178_MAX_BT"]),
        "max_num_seqs": int(os.environ["PHASE178_MAX_NUM_SEQS"]),
    },
    "parallelism": {
        "tp": int(os.environ["PHASE178_TP"]),
        "dp": int(os.environ["PHASE178_DP"]),
        "ep": int(os.environ["PHASE178_EP"]),
        "world_size": int(os.environ["PHASE178_WORLD_SIZE"]),
        "gpu_count": int(os.environ["PHASE178_GPU_COUNT"]),
    },
    "scheduler": {
        "steady_state_time_ms": float(os.environ["PHASE178_STEADY_STATE_TIME_MS"]),
    },
    "runtime": {
        "vllm_version": os.environ["PHASE178_VLLM_VERSION"],
        "model_path": os.environ["PHASE178_MODEL_PATH"],
        "dtype": os.environ["PHASE178_DTYPE"],
        "quantization": os.environ["PHASE178_QUANTIZATION"],
        "serve_command": os.environ["PHASE178_SERVE_COMMAND"],
        "benchmark_command": os.environ["PHASE178_BENCHMARK_COMMAND"],
    },
    "hardware": {
        "gpu_model": gpu_parts[0],
        "gpu_memory_gb": round(float(gpu_parts[1]) / 1024.0, 3),
        "driver_version": gpu_parts[2],
        "cuda_version": os.environ["PHASE178_CUDA_VERSION"],
    },
    "diagnostic_only": True,
    "valid_for_default": False,
    "perf_database": False,
}
Path(os.environ["PHASE178_OUT_JSON"]).write_text(
    json.dumps(payload, indent=2, sort_keys=True),
    encoding="utf-8",
)
PYJSON
}

wait_for_gpu_drain() {
  local out_dir="$1"
  local final_file="${out_dir}/gpu_compute_apps_after.txt"
  local timeout_file="${out_dir}/gpu_compute_apps_after_timeout.txt"
  local sample_file="${out_dir}/gpu_compute_apps_after.poll"
  local drain_log="${out_dir}/gpu_compute_apps_drain.log"
  local stable=0
  local elapsed=0
  local poll=0
  local query_exit=0
  local bytes=0

  : > "${drain_log}"
  rm -f "${final_file}" "${timeout_file}" "${sample_file}"
  echo "gpu_drain_start=$(date -Is)" >> "${drain_log}"
  echo "gpu_drain_poll_seconds=${GPU_DRAIN_POLL_SECONDS}" >> "${drain_log}"
  echo "gpu_drain_stable_polls=${GPU_DRAIN_STABLE_POLLS}" >> "${drain_log}"
  echo "gpu_drain_timeout_seconds=${GPU_DRAIN_TIMEOUT_SECONDS}" >> "${drain_log}"

  while (( elapsed <= GPU_DRAIN_TIMEOUT_SECONDS )); do
    poll=$((poll + 1))
    if gpu_apps > "${sample_file}"; then
      query_exit=0
    else
      query_exit="$?"
      echo "gpu_apps_query_failed_exit=${query_exit}" > "${sample_file}"
    fi
    bytes="$(wc -c < "${sample_file}" | tr -d ' ')"
    if [[ ! -s "${sample_file}" ]]; then
      stable=$((stable + 1))
    else
      stable=0
    fi
    echo "poll=${poll} elapsed_seconds=${elapsed} query_exit=${query_exit} bytes=${bytes} stable_empty_polls=${stable}" >> "${drain_log}"
    if (( stable >= GPU_DRAIN_STABLE_POLLS )); then
      : > "${final_file}"
      rm -f "${sample_file}"
      echo "gpu_drain_close_reason=stable" >> "${drain_log}"
      echo "gpu_drain_done=$(date -Is)" >> "${drain_log}"
      return 0
    fi
    sleep "${GPU_DRAIN_POLL_SECONDS}"
    elapsed=$((elapsed + GPU_DRAIN_POLL_SECONDS))
  done

  cp "${sample_file}" "${timeout_file}" 2>/dev/null || true
  echo "gpu_drain_close_reason=timeout" >> "${drain_log}"
  echo "gpu_drain_done=$(date -Is)" >> "${drain_log}"
  return 1
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
    --input-len "$(scenario_isl "${scenario}")"
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
  echo "topology_key=$(scenario_topology_key "${scenario}") shape_key=$(scenario_shape_key "${scenario}") tp=$(scenario_tp "${scenario}") dp=$(scenario_dp "${scenario}") ep=$(scenario_ep "${scenario}") isl=$(scenario_isl "${scenario}") osl=${BENCH_OUTPUT_LEN} batch=${BENCH_MAX_CONCURRENCY} max_bt=$(scenario_max_bt "${scenario}") port=$(scenario_port "${scenario}")"
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

run_preflight() {
  echo "phase178_mode=preflight"
  echo "workdir=${WORKDIR}"
  echo "model_path=${MODEL_PATH}"
  echo "output_root=${OUT_ROOT}"
  for scenario in "${SCENARIOS[@]}"; do
    print_commands_for_scenario "${scenario}"
  done
  echo "required_result_schema=throughput,success_fail,shape,topology,scheduler,runtime,gpu,serve_command,benchmark_command,diagnostic_flags"
  echo "diagnostic_only=true valid_for_default=false perf_database=false"
  echo "preflight_benchmark_started=false"
}

run_one() {
  local scenario="$1"
  local out_dir serve_log ready_log run_log port serve_command benchmark_command bench_json
  [[ -n "${scenario}" ]] || { usage >&2; exit 2; }
  scenario_index "${scenario}" >/dev/null
  [[ -n "${PHASE178_STEADY_STATE_TIME_MS:-}" ]] || {
    echo "missing_phase178_steady_state_time_ms=true" >&2
    exit 2
  }

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
  : > "${CLEANUP_LOG}"
  trap cleanup EXIT

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
  echo "serve_command=${serve_command}"
  echo "benchmark_command=${benchmark_command}"
  nohup "${SERVE_CMD[@]}" > "${serve_log}" 2>&1 &
  SERVICE_PID=$!
  echo "service_pid=${SERVICE_PID}"
  wait_for_service "${port}" "${serve_log}" "${ready_log}"

  local benchmark_exit_code=0
  local write_result_exit_code=0
  set +e
  "${BENCH_CMD[@]}"
  benchmark_exit_code="$?"
  set -e
  echo "benchmark_exit_code=${benchmark_exit_code}"

  if [[ -f "${bench_json}" ]]; then
    set +e
    write_result_json "${scenario}" "${out_dir}" "${serve_command}" "${benchmark_command}"
    write_result_exit_code="$?"
    set -e
    echo "write_result_exit_code=${write_result_exit_code}"
  else
    echo "missing_bench_json=${bench_json}" >&2
    write_result_exit_code=1
  fi

  stop_service
  SERVICE_PID=""
  if ! wait_for_gpu_drain "${out_dir}"; then
    echo "gpu_drain_timeout=true" >&2
    cat "${out_dir}/gpu_compute_apps_drain.log" >&2
    exit 1
  fi
  if [[ "${write_result_exit_code}" != "0" ]]; then
    exit "${write_result_exit_code}"
  fi
  if [[ "${benchmark_exit_code}" != "0" ]]; then
    exit "${benchmark_exit_code}"
  fi
  echo "phase178_result_json=${out_dir}/phase178_result.json"
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
