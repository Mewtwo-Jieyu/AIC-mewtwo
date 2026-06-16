#!/usr/bin/env bash
set -euo pipefail

# Phase272 deeper scheduler trace source-map. source-check/preflight are
# local-only and must not patch vLLM, start a service, or run a benchmark.

MODE="${1:-preflight}"
SCENARIO="${2:-${SCENARIO:-}}"

WORKDIR="${WORKDIR:-/mnt/shared-storage-user/ailab-sys/zhaojieyu/aic/phase237_scheduler_trace_preflight_27545205}"
MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--moonshotai--Kimi-K2.5}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-kimi-k2.5}"
PORT_BASE="${PORT_BASE:-18400}"
OUT_ROOT="${OUT_ROOT:-docs/iter_gap_investigation/phase272_deeper_scheduler_trace}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-128}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-128}"
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-2000}"
BENCH_WARMUP_REQUESTS="${BENCH_WARMUP_REQUESTS:-0}"
BENCH_TIMEOUT_S="${BENCH_TIMEOUT_S:-7200}"

SCENARIOS=(
  tp8ep8-12k2k-bt12000
  tp8ep8-12k2k-bt65536
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
  iteration_start_ns
  forward_start_ns
  forward_end_ns
  iteration_end_ns
  iteration_elapsed_ns
  forward_elapsed_ns
  scheduler_running_reqs
  scheduler_waiting_reqs
  scheduler_scheduled_new_reqs
  scheduler_scheduled_decode_reqs
  scheduler_scheduled_prefill_reqs
  active_request_count
  context_drain_tokens
  mixed_iteration_index
  tail_decode_tokens
)

usage() {
  cat <<'EOF'
Usage:
  run_phase272_deeper_trace_source_map.sh source-check
  run_phase272_deeper_trace_source_map.sh preflight [scenario]
  run_phase272_deeper_trace_source_map.sh run-one <scenario>
  run_phase272_deeper_trace_source_map.sh help

Scenarios:
  tp8ep8-12k2k-bt12000
  tp8ep8-12k2k-bt65536
EOF
}

join_by_comma() {
  local IFS=","
  echo "$*"
}

scenario_index() {
  case "$1" in
    tp8ep8-12k2k-bt12000) echo 0 ;;
    tp8ep8-12k2k-bt65536) echo 1 ;;
    *) echo "unknown_scenario=$1" >&2; exit 2 ;;
  esac
}

scenario_tp() {
  scenario_index "$1" >/dev/null
  echo 8
}

scenario_dp() {
  scenario_index "$1" >/dev/null
  echo 1
}

scenario_ep() {
  scenario_index "$1" >/dev/null
  echo 8
}

scenario_isl() {
  scenario_index "$1" >/dev/null
  echo 12000
}

scenario_max_bt() {
  case "$1" in
    tp8ep8-12k2k-bt12000) echo 12000 ;;
    tp8ep8-12k2k-bt65536) echo 65536 ;;
    *) echo "unknown_scenario=$1" >&2; exit 2 ;;
  esac
}

scenario_topology_key() {
  scenario_index "$1" >/dev/null
  echo tp8_dp1_ep8
}

scenario_shape_key() {
  scenario_index "$1" >/dev/null
  echo isl12000_osl2000_batch128
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

locate_runner_path() {
  local candidates=(
    "${WORKDIR}/vllm/vllm/v1/worker/gpu_model_runner.py"
    "${WORKDIR}/vllm/v1/worker/gpu_model_runner.py"
    "${WORKDIR}/site-packages/vllm/v1/worker/gpu_model_runner.py"
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py"
  )
  local path
  for path in "${candidates[@]}"; do
    if [[ -f "${path}" ]]; then
      echo "${path}"
      return 0
    fi
  done
  return 1
}

source_map_contract_text() {
  cat <<'EOF'
compute_iteration_details(scheduler_output)
scheduler_output.total_num_scheduled_tokens
self.scheduler_config.max_num_batched_tokens
self.scheduler_config.max_num_seqs
batch_desc.num_tokens
time.perf_counter_ns
running
waiting
scheduled_new_reqs
scheduled_decode_reqs
scheduled_prefill_reqs
num_ctx_requests
num_generation_requests
num_ctx_tokens
num_generation_tokens
EOF
}

run_source_check() {
  local contract source_path
  contract="$(source_map_contract_text)"
  [[ "${contract}" == *"compute_iteration_details(scheduler_output)"* ]] || { echo "missing_compute_iteration_details_contract=true" >&2; exit 1; }
  [[ "${contract}" == *"time.perf_counter_ns"* ]] || { echo "missing_timing_contract=true" >&2; exit 1; }
  [[ "${contract}" == *"running"* ]] || { echo "missing_running_queue_contract=true" >&2; exit 1; }
  [[ "${contract}" == *"waiting"* ]] || { echo "missing_waiting_queue_contract=true" >&2; exit 1; }

  echo "phase272_mode=source-check"
  echo "patch_anchor_compute_iteration_details=true"
  echo "patch_anchor_scheduler_output=true"
  echo "patch_anchor_forward_timing=true"
  echo "source_map_forward_timing=available"
  echo "source_map_queue_state=available"
  echo "source_map_boundary_timeline=available"
  echo "trace_schema=$(join_by_comma "${TRACE_SCHEMA[@]}")"
  echo "diagnostic_only=true valid_for_default=false perf_database=false"

  source_path="$(locate_runner_path 2>/dev/null)" || { echo "missing_vllm_source=true" >&2; exit 1; }
  [[ -f "${source_path}" ]] || { echo "missing_vllm_source=true" >&2; exit 1; }
  grep -q "scheduler_output" "${source_path}" || { echo "missing_scheduler_output_anchor=${source_path}" >&2; exit 1; }
  grep -q "batch_desc.num_tokens" "${source_path}" || { echo "missing_batch_desc_anchor=${source_path}" >&2; exit 1; }
  echo "vllm_source_present=true"
  echo "vllm_source_path=${source_path}"
  echo "source_patch_started=false"
  echo "scheduler_trace_started=false"
  echo "gpu_benchmark_started=false"
  echo "source_check=PASS"
}

build_serve_cmd() {
  local scenario="$1"
  local max_bt port
  max_bt="$(scenario_max_bt "${scenario}")"
  port="$(scenario_port "${scenario}")"

  SERVE_CMD=(
    env
    AIC_PHASE272_DEEPER_TRACE_LOG=1
    "AIC_PHASE272_SCENARIO=${scenario}"
    "AIC_PHASE272_TOPOLOGY_KEY=$(scenario_topology_key "${scenario}")"
    "AIC_PHASE272_SHAPE_KEY=$(scenario_shape_key "${scenario}")"
    "AIC_PHASE272_TP=$(scenario_tp "${scenario}")"
    "AIC_PHASE272_DP=$(scenario_dp "${scenario}")"
    "AIC_PHASE272_EP=$(scenario_ep "${scenario}")"
    python3 -m vllm.entrypoints.cli.main serve "${MODEL_PATH}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --port "${port}"
    --tensor-parallel-size "$(scenario_tp "${scenario}")"
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
  echo "trace_jsonl=$(scenario_dir "${scenario}")/deeper_scheduler_trace.jsonl"
  echo "phase272_result_json=$(scenario_dir "${scenario}")/phase272_result.json"
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

  echo "phase272_mode=preflight"
  echo "workdir=${WORKDIR}"
  echo "model_path=${MODEL_PATH}"
  echo "output_root=${OUT_ROOT}"
  echo "scenario_count=${#scenarios[@]}"
  echo "trace_source=phase234 scheduler trace plus forward timing, scheduler queue state, boundary timeline"
  echo "trace_marker=AIC_PHASE272_DEEPER_TRACE_ROW"
  echo "trace_schema=$(join_by_comma "${TRACE_SCHEMA[@]}")"
  echo "diagnostic_only=true valid_for_default=false perf_database=false"
  for scenario in "${scenarios[@]}"; do
    print_scenario_contract "${scenario}"
  done
  echo "source_patch_started=false"
  echo "scheduler_trace_started=false"
  echo "gpu_benchmark_started=false"
}

run_one() {
  local scenario="${SCENARIO}"
  [[ -n "${scenario}" ]] || { echo "missing_scenario=true" >&2; exit 2; }
  scenario_index "${scenario}" >/dev/null
  if [[ "${PHASE272_ALLOW_GPU_RUN:-0}" != "1" ]]; then
    echo "phase272_allow_gpu_run_required=true" >&2
    exit 2
  fi
  echo "phase272_run_one_source_map_only=true" >&2
  echo "phase272_guarded_run_one_not_implemented=true" >&2
  exit 2
}

case "${MODE}" in
  source-check)
    run_source_check
    ;;
  preflight)
    run_preflight
    ;;
  run-one)
    run_one
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
