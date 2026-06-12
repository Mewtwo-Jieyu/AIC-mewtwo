#!/usr/bin/env bash
set -euo pipefail

# Phase232 scheduler trace preflight. This script prints the source/runner
# contract only; it does not patch vLLM, start service, run benchmark, or clean.

MODE="${1:-preflight}"
SCENARIO="${2:-${SCENARIO:-}}"

WORKDIR="${WORKDIR:-/mnt/nvme1n1/ml_research/jieyu/aic}"
MODEL_PATH="${MODEL_PATH:-/mnt/cfs/models/kimi-2.5-fp8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-kimi-k2.5}"
PORT_BASE="${PORT_BASE:-18200}"
OUT_ROOT="${OUT_ROOT:-docs/iter_gap_investigation/phase232_scheduler_trace}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-128}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-128}"
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-2000}"
BENCH_WARMUP_REQUESTS="${BENCH_WARMUP_REQUESTS:-0}"
BENCH_TIMEOUT_S="${BENCH_TIMEOUT_S:-7200}"

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

TRACE_SCHEMA=(
  scenario
  iteration
  phase
  scheduled_context_tokens
  scheduled_decode_tokens
  scheduled_total_tokens
  scheduled_context_reqs
  scheduled_decode_reqs
  max_num_batched_tokens
  max_num_seqs
  forward_token_count
  tp
  dp
  ep
  topology_key
  shape_key
)

usage() {
  cat <<'EOF'
Usage:
  run_phase232_scheduler_trace_preflight.sh preflight [scenario]
  run_phase232_scheduler_trace_preflight.sh help

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

join_by_comma() {
  local IFS=","
  echo "$*"
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
  echo $((PORT_BASE + $(scenario_index "$1")))
}

scenario_dir() {
  echo "${OUT_ROOT}/$1"
}

quote_cmd() {
  printf "%q " "$@"
}

build_serve_cmd() {
  local scenario="$1"
  local tp dp max_bt port
  tp="$(scenario_tp "${scenario}")"
  dp="$(scenario_dp "${scenario}")"
  max_bt="$(scenario_max_bt "${scenario}")"
  port="$(scenario_port "${scenario}")"

  SERVE_CMD=(
    env
    AIC_PHASE232_SCHEDULER_TRACE_LOG=1
    "AIC_PHASE232_SCENARIO=${scenario}"
    "AIC_PHASE232_TOPOLOGY_KEY=$(scenario_topology_key "${scenario}")"
    "AIC_PHASE232_SHAPE_KEY=$(scenario_shape_key "${scenario}")"
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

print_scenario_contract() {
  local scenario="$1"
  scenario_index "${scenario}" >/dev/null
  build_serve_cmd "${scenario}"
  build_benchmark_cmd "${scenario}"
  echo "scenario=${scenario}"
  echo "topology_key=$(scenario_topology_key "${scenario}")"
  echo "shape_key=$(scenario_shape_key "${scenario}")"
  echo "tp=$(scenario_tp "${scenario}") dp=$(scenario_dp "${scenario}") ep=$(scenario_ep "${scenario}")"
  echo "isl=$(scenario_isl "${scenario}") osl=${BENCH_OUTPUT_LEN} batch=${BENCH_MAX_CONCURRENCY}"
  echo "max_num_batched_tokens=$(scenario_max_bt "${scenario}")"
  echo "max_num_seqs=${MAX_NUM_SEQS}"
  echo "trace_jsonl=$(scenario_dir "${scenario}")/scheduler_trace.jsonl"
  echo "serve_command=$(quote_cmd "${SERVE_CMD[@]}")"
  echo "benchmark_command=$(quote_cmd "${BENCH_CMD[@]}")"
}

run_preflight() {
  local scenarios=()
  if [[ -n "${SCENARIO}" ]]; then
    scenario_index "${SCENARIO}" >/dev/null
    scenarios=("${SCENARIO}")
  else
    scenarios=("${SCENARIOS[@]}")
  fi

  echo "phase232_mode=preflight"
  echo "workdir=${WORKDIR}"
  echo "model_path=${MODEL_PATH}"
  echo "output_root=${OUT_ROOT}"
  echo "scenario_count=${#scenarios[@]}"
  echo "trace_source=from vllm.v1.utils import compute_iteration_details; compute_iteration_details(scheduler_output)"
  echo "trace_marker=AIC_PHASE232_SCHEDULER_TRACE_ROW"
  echo "trace_schema=$(join_by_comma "${TRACE_SCHEMA[@]}")"
  echo "diagnostic_only=true valid_for_default=false perf_database=false"
  for scenario in "${scenarios[@]}"; do
    print_scenario_contract "${scenario}"
  done
  echo "source_patch_started=false"
  echo "scheduler_trace_started=false"
  echo "gpu_benchmark_started=false"
}

case "${MODE}" in
  preflight)
    run_preflight
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
