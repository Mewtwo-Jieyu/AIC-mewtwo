#!/usr/bin/env bash
set -euo pipefail

# Phase274 deeper scheduler trace runner. source-check/preflight are local-only.
# run-one is a real GPU entry and requires PHASE274_ALLOW_GPU_RUN=1.

MODE="${1:-preflight}"
SCENARIO="${2:-${SCENARIO:-}}"

WORKDIR="${WORKDIR:-/mnt/shared-storage-user/ailab-sys/zhaojieyu/aic/phase237_scheduler_trace_preflight_27545205}"
MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--moonshotai--Kimi-K2.5}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-kimi-k2.5}"
PORT_BASE="${PORT_BASE:-18500}"
OUT_ROOT="${OUT_ROOT:-docs/iter_gap_investigation/phase274_deeper_scheduler_trace}"
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
RUNNER_PATH="${PHASE274_GPU_MODEL_RUNNER_PATH:-}"
RUNNER_BACKUP_PATH=""

SCENARIOS=(
  tp8ep8-12k2k-bt12000
  tp8ep8-12k2k-bt65536
  tp4dp2ep8-12k2k-bt12000
  tp4dp2ep8-12k2k-bt65536
)

RAW_TRACE_SCHEMA=(
  source
  scenario
  iteration
  phase
  scheduled_context_tokens
  scheduled_decode_tokens
  scheduled_total_tokens
  scheduled_context_reqs
  scheduled_decode_reqs
  scheduled_total_reqs
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
  active_request_count
  scheduled_new_req_count
  scheduled_cached_req_count
  scheduled_resumed_req_count
  finished_req_count
  diagnostic_only
  valid_for_default
  perf_database
)

usage() {
  cat <<'EOF'
Usage:
  run_phase274_deeper_scheduler_trace.sh source-check
  run_phase274_deeper_scheduler_trace.sh preflight [scenario]
  run_phase274_deeper_scheduler_trace.sh run-one <scenario>
  run_phase274_deeper_scheduler_trace.sh cleanup
  run_phase274_deeper_scheduler_trace.sh __test-validate-trace <scenario>
  run_phase274_deeper_scheduler_trace.sh __test-patch-source
  run_phase274_deeper_scheduler_trace.sh __test-capture-startup-diagnostics <scenario>
  run_phase274_deeper_scheduler_trace.sh help

Scenarios:
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
    tp8ep8-12k2k-bt12000) echo 0 ;;
    tp8ep8-12k2k-bt65536) echo 1 ;;
    tp4dp2ep8-12k2k-bt12000) echo 2 ;;
    tp4dp2ep8-12k2k-bt65536) echo 3 ;;
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
  scenario_index "$1" >/dev/null
  echo 12000
}

scenario_max_bt() {
  case "$1" in
    tp8ep8-12k2k-bt12000) echo 12000 ;;
    tp8ep8-12k2k-bt65536) echo 65536 ;;
    tp4dp2ep8-12k2k-bt12000) echo 12000 ;;
    tp4dp2ep8-12k2k-bt65536) echo 65536 ;;
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
  scenario_index "$1" >/dev/null
  echo isl12000_osl2000_batch128
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

locate_runner_path() {
  if [[ -n "${RUNNER_PATH}" ]]; then
    echo "${RUNNER_PATH}"
    return
  fi
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
  python3 - <<'PY'
import inspect
import vllm.v1.worker.gpu_model_runner as runner

path = inspect.getsourcefile(runner)
if not path:
    raise SystemExit("missing gpu_model_runner source path")
print(path)
PY
}

run_source_check() {
  local source_path
  echo "phase274_mode=source-check"
  echo "raw_trace_schema=$(join_by_comma "${RAW_TRACE_SCHEMA[@]}")"
  echo "result_derived_fields=boundary_timeline,mixed_sequence,tail_decode_tokens"
  echo "diagnostic_only=true valid_for_default=false perf_database=false"
  source_path="$(locate_runner_path 2>/dev/null)" || { echo "missing_vllm_source=true" >&2; exit 1; }
  [[ -f "${source_path}" ]] || { echo "missing_vllm_source=true" >&2; exit 1; }
  grep -q "scheduler_output" "${source_path}" || { echo "missing_scheduler_output_anchor=${source_path}" >&2; exit 1; }
  grep -q "batch_desc.num_tokens" "${source_path}" || { echo "missing_batch_desc_anchor=${source_path}" >&2; exit 1; }
  grep -q "maybe_create_ubatch_slices" "${source_path}" || { echo "missing_maybe_create_ubatch_slices_anchor=${source_path}" >&2; exit 1; }
  grep -q "time" "${source_path}" || { echo "missing_timing_anchor=${source_path}" >&2; exit 1; }
  PYTHONPATH="${WORKDIR}/vllm:${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}" python3 - <<'PY'
from vllm.v1.core.sched.output import CachedRequestData
from vllm.v1.utils import compute_iteration_details

if not callable(compute_iteration_details):
    raise SystemExit("compute_iteration_details_not_callable=true")
print("compute_iteration_details_import=true")

cached_request_fields = set(getattr(CachedRequestData, "__annotations__", {}))
cached_request_fields.update(getattr(CachedRequestData, "__slots__", ()))
if hasattr(CachedRequestData, "num_reqs"):
    cached_request_fields.add("num_reqs")
if hasattr(CachedRequestData, "resumed_req_ids"):
    cached_request_fields.add("resumed_req_ids")
if "num_reqs" not in cached_request_fields:
    raise SystemExit("cached_request_data_num_reqs_missing=true")
if "resumed_req_ids" not in cached_request_fields:
    raise SystemExit("cached_request_data_resumed_req_ids_missing=true")
print("cached_request_data_num_reqs=true")
print("cached_request_data_resumed_req_ids=true")
PY
  echo "vllm_source_present=true"
  echo "vllm_source_path=${source_path}"
  echo "patch_anchor_scheduler_output=true"
  echo "patch_anchor_batch_desc_num_tokens=true"
  echo "patch_anchor_forward_timing=true"
  echo "patch_anchor_maybe_create_ubatch_slices=true"
  echo "source_patch_started=false"
  echo "scheduler_trace_started=false"
  echo "gpu_benchmark_started=false"
  echo "source_check=PASS"
}

build_serve_cmd() {
  local scenario="$1"
  local max_bt port dp
  max_bt="$(scenario_max_bt "${scenario}")"
  port="$(scenario_port "${scenario}")"
  dp="$(scenario_dp "${scenario}")"
  SERVE_CMD=(
    env
    AIC_PHASE274_DEEPER_TRACE_LOG=1
    "AIC_PHASE274_SCENARIO=${scenario}"
    "AIC_PHASE274_TOPOLOGY_KEY=$(scenario_topology_key "${scenario}")"
    "AIC_PHASE274_SHAPE_KEY=$(scenario_shape_key "${scenario}")"
    "AIC_PHASE274_TP=$(scenario_tp "${scenario}")"
    "AIC_PHASE274_DP=$(scenario_dp "${scenario}")"
    "AIC_PHASE274_EP=$(scenario_ep "${scenario}")"
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
  echo "trace_jsonl=$(scenario_dir "${scenario}")/deeper_scheduler_trace.jsonl"
  echo "phase274_result_json=$(scenario_dir "${scenario}")/phase274_result.json"
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
  echo "phase274_mode=preflight"
  echo "workdir=${WORKDIR}"
  echo "model_path=${MODEL_PATH}"
  echo "output_root=${OUT_ROOT}"
  echo "scenario_count=${#scenarios[@]}"
  echo "trace_marker=AIC_PHASE274_DEEPER_TRACE_ROW"
  echo "raw_trace_schema=$(join_by_comma "${RAW_TRACE_SCHEMA[@]}")"
  echo "result_derived_fields=boundary_timeline,mixed_sequence,tail_decode_tokens"
  echo "diagnostic_only=true valid_for_default=false perf_database=false"
  for scenario in "${scenarios[@]}"; do
    print_scenario_contract "${scenario}"
  done
  echo "source_patch_started=false"
  echo "scheduler_trace_started=false"
  echo "gpu_benchmark_started=false"
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

patch_source() {
  RUNNER_PATH="$(locate_runner_path)"
  RUNNER_BACKUP_PATH="${RUNNER_PATH}.phase274.bak"
  [[ -f "${RUNNER_PATH}" ]] || { echo "missing_gpu_model_runner=${RUNNER_PATH}" >&2; exit 1; }
  [[ ! -e "${RUNNER_BACKUP_PATH}" ]] || { echo "existing_phase274_backup=${RUNNER_BACKUP_PATH}" >&2; exit 1; }
  cp "${RUNNER_PATH}" "${RUNNER_BACKUP_PATH}"
  PHASE274_RUNNER_PATH="${RUNNER_PATH}" python3 - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["PHASE274_RUNNER_PATH"])
text = path.read_text(encoding="utf-8")


def replace_once(src: str, old: str, new: str) -> str:
    count = src.count(old)
    if count != 1:
        raise SystemExit(f"expected one match, got {count}: {old[:120]!r}")
    return src.replace(old, new)


if "import json\n" not in text:
    text = replace_once(text, "import functools\n", "import functools\nimport json\n")
if "import os\n" not in text:
    text = replace_once(text, "import json\n", "import json\nimport os\n")
if "import time\n" not in text:
    text = replace_once(text, "import os\n", "import os\nimport time\n")
if "from vllm.v1.utils import compute_iteration_details\n" not in text:
    text = replace_once(
        text,
        "from vllm.v1.worker.utils import is_residual_scattered_for_sp\n",
        "from vllm.v1.worker.utils import is_residual_scattered_for_sp\n"
        "from vllm.v1.utils import compute_iteration_details\n",
    )

old = (
    "        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens\n"
    "        with (\n"
)
new = (
    "        aic_phase274_log_enabled = os.environ.get(\"AIC_PHASE274_DEEPER_TRACE_LOG\") == \"1\"\n"
    "        aic_phase274_iteration_start_ns = time.perf_counter_ns()\n"
    "        aic_phase274_iter = int(getattr(self, \"_aic_phase274_iter\", 0))\n"
    "        self._aic_phase274_iter = aic_phase274_iter + 1\n"
    "\n"
    "        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens\n"
    "        with (\n"
)
text = replace_once(text, old, new)

old = (
    "            num_tokens_padded = batch_desc.num_tokens\n"
    "            num_reqs_padded = (\n"
    "                batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs\n"
    "            )\n"
    "            ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(\n"
)
new = (
    "            num_tokens_padded = batch_desc.num_tokens\n"
    "            num_reqs_padded = (\n"
    "                batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs\n"
    "            )\n"
    "            if aic_phase274_log_enabled:\n"
    "                aic_details = compute_iteration_details(scheduler_output)\n"
    "                aic_ctx_req = int(aic_details.num_ctx_requests)\n"
    "                aic_ctx_tok = int(aic_details.num_ctx_tokens)\n"
    "                aic_gen_req = int(aic_details.num_generation_requests)\n"
    "                aic_gen_tok = int(aic_details.num_generation_tokens)\n"
    "                if aic_ctx_req and aic_gen_req:\n"
    "                    aic_phase = \"mixed\"\n"
    "                elif aic_ctx_req:\n"
    "                    aic_phase = \"prefill\"\n"
    "                elif aic_gen_req:\n"
    "                    aic_phase = \"pure_decode\"\n"
    "                else:\n"
    "                    aic_phase = \"idle\"\n"
    "                aic_active_request_count = int(len(self.requests))\n"
    "                aic_scheduled_new_req_count = int(len(scheduler_output.scheduled_new_reqs))\n"
    "                aic_scheduled_cached_req_count = int(scheduler_output.scheduled_cached_reqs.num_reqs)\n"
    "                aic_scheduled_resumed_req_count = int(len(scheduler_output.scheduled_cached_reqs.resumed_req_ids))\n"
    "                aic_finished_req_count = int(len(scheduler_output.finished_req_ids))\n"
    "            ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(\n"
)
text = replace_once(text, old, new)

old = "            model_output = self._model_forward(\n"
new = (
    "            aic_phase274_forward_start_ns = time.perf_counter_ns()\n"
    "            model_output = self._model_forward(\n"
)
text = replace_once(text, old, new)

old = (
    "                intermediate_tensors=intermediate_tensors,\n"
    "                inputs_embeds=inputs_embeds,\n"
    "                **model_kwargs,\n"
    "            )\n"
)
new = (
    "                intermediate_tensors=intermediate_tensors,\n"
    "                inputs_embeds=inputs_embeds,\n"
    "                **model_kwargs,\n"
    "            )\n"
    "            aic_phase274_forward_end_ns = time.perf_counter_ns()\n"
)
text = replace_once(text, old, new)

old = (
    "        if deferred_state_corrections_fn:\n"
    "            deferred_state_corrections_fn()\n"
    "\n"
    "        return None\n"
)
new = (
    "        if deferred_state_corrections_fn:\n"
    "            deferred_state_corrections_fn()\n"
    "\n"
    "        if aic_phase274_log_enabled and aic_phase != \"idle\":\n"
    "            aic_phase274_iteration_end_ns = time.perf_counter_ns()\n"
    "            aic_payload = {\n"
    "                \"source\": \"phase274_deeper_scheduler_trace\",\n"
    "                \"scenario\": os.environ.get(\"AIC_PHASE274_SCENARIO\", \"unknown\"),\n"
    "                \"iteration\": aic_phase274_iter,\n"
    "                \"phase\": aic_phase,\n"
    "                \"scheduled_context_tokens\": aic_ctx_tok,\n"
    "                \"scheduled_decode_tokens\": aic_gen_tok,\n"
    "                \"scheduled_total_tokens\": int(scheduler_output.total_num_scheduled_tokens),\n"
    "                \"scheduled_context_reqs\": aic_ctx_req,\n"
    "                \"scheduled_decode_reqs\": aic_gen_req,\n"
    "                \"scheduled_total_reqs\": aic_ctx_req + aic_gen_req,\n"
    "                \"max_num_batched_tokens\": int(self.scheduler_config.max_num_batched_tokens),\n"
    "                \"max_num_seqs\": int(self.scheduler_config.max_num_seqs),\n"
    "                \"forward_token_count\": int(batch_desc.num_tokens),\n"
    "                \"tp\": int(os.environ.get(\"AIC_PHASE274_TP\", getattr(self.parallel_config, \"tensor_parallel_size\", 1))),\n"
    "                \"dp\": int(os.environ.get(\"AIC_PHASE274_DP\", getattr(self.parallel_config, \"data_parallel_size\", 1))),\n"
    "                \"ep\": int(os.environ.get(\"AIC_PHASE274_EP\", \"8\")),\n"
    "                \"topology_key\": os.environ.get(\"AIC_PHASE274_TOPOLOGY_KEY\", \"unknown\"),\n"
    "                \"shape_key\": os.environ.get(\"AIC_PHASE274_SHAPE_KEY\", \"unknown\"),\n"
    "                \"iteration_start_ns\": aic_phase274_iteration_start_ns,\n"
    "                \"forward_start_ns\": aic_phase274_forward_start_ns,\n"
    "                \"forward_end_ns\": aic_phase274_forward_end_ns,\n"
    "                \"iteration_end_ns\": aic_phase274_iteration_end_ns,\n"
    "                \"iteration_elapsed_ns\": aic_phase274_iteration_end_ns - aic_phase274_iteration_start_ns,\n"
    "                \"forward_elapsed_ns\": aic_phase274_forward_end_ns - aic_phase274_forward_start_ns,\n"
    "                \"active_request_count\": aic_active_request_count,\n"
    "                \"scheduled_new_req_count\": aic_scheduled_new_req_count,\n"
    "                \"scheduled_cached_req_count\": aic_scheduled_cached_req_count,\n"
    "                \"scheduled_resumed_req_count\": aic_scheduled_resumed_req_count,\n"
    "                \"finished_req_count\": aic_finished_req_count,\n"
    "                \"diagnostic_only\": True,\n"
    "                \"valid_for_default\": False,\n"
    "                \"perf_database\": False,\n"
    "            }\n"
    "            print(\"AIC_PHASE274_DEEPER_TRACE_ROW \" + json.dumps(aic_payload, sort_keys=True), flush=True)\n"
    "\n"
    "        return None\n"
)
text = replace_once(text, old, new)
path.write_text(text, encoding="utf-8")
PY
}

restore_source() {
  if [[ -n "${RUNNER_BACKUP_PATH}" && -f "${RUNNER_BACKUP_PATH}" && -n "${RUNNER_PATH}" ]]; then
    mv "${RUNNER_BACKUP_PATH}" "${RUNNER_PATH}"
  fi
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
  pkill -f "vllm.entrypoints.cli.main serve" 2>/dev/null || true
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
  restore_source
  if [[ -n "${CLEANUP_LOG}" ]]; then
    echo "cleanup_trap_done=$(date -Is)" >> "${CLEANUP_LOG}"
  fi
}

capture_startup_diagnostics() {
  local out_dir="$1"
  local scenario="$2"
  local reason="$3"
  local port summary process_snapshot gpu_snapshot nvidia_snapshot ray_tail serve_tail ready_tail
  local runner_path_value ray_root latest_session ray_logs_found
  local ray_logs=()
  local ray_log

  mkdir -p "${out_dir}" 2>/dev/null || true
  port="$(scenario_port "${scenario}")"
  summary="${out_dir}/startup_failure_summary.txt"
  process_snapshot="${out_dir}/startup_process_snapshot.txt"
  gpu_snapshot="${out_dir}/startup_gpu_compute_apps.txt"
  nvidia_snapshot="${out_dir}/startup_nvidia_smi.txt"
  ray_tail="${out_dir}/startup_ray_logs_tail.txt"
  serve_tail="${out_dir}/startup_serve_tail.txt"
  ready_tail="${out_dir}/startup_ready_tail.txt"

  runner_path_value="${RUNNER_PATH:-}"
  if [[ -z "${runner_path_value}" ]]; then
    runner_path_value="$(locate_runner_path 2>/dev/null || true)"
  fi

  {
    echo "scenario=${scenario}"
    echo "reason=${reason}"
    echo "timestamp=$(date -Is)"
    echo "service_pid=${SERVICE_PID:-}"
    echo "port=${port}"
    echo "workdir=${WORKDIR}"
    echo "runner_path=${runner_path_value:-unavailable}"
  } > "${summary}" 2>&1 || true

  {
    echo "timestamp=$(date -Is)"
    ps -eo pid=,ppid=,stat=,args= | awk '/vllm.entrypoints.cli.main serve|VLLM::|raylet|gcs_server|qwen|Kimi|DPCoordinator/ && $0 !~ /awk/ {print}'
  } > "${process_snapshot}" 2>&1 || true

  if ! gpu_apps > "${gpu_snapshot}" 2>&1; then
    echo "gpu_compute_apps_unavailable=true" >> "${gpu_snapshot}" 2>/dev/null || true
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    if ! nvidia-smi > "${nvidia_snapshot}" 2>&1; then
      echo "nvidia_smi_unavailable=true" >> "${nvidia_snapshot}" 2>/dev/null || true
    fi
  else
    echo "nvidia_smi_unavailable=true" > "${nvidia_snapshot}" 2>/dev/null || true
  fi

  ray_root="${PHASE274_TEST_RAY_ROOT:-/tmp/ray}"
  latest_session="$(find "${ray_root}" -maxdepth 1 -type d -name 'session_*' 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -n "${latest_session}" && -d "${latest_session}/logs" ]]; then
    while IFS= read -r ray_log; do
      ray_logs+=("${ray_log}")
    done < <(find "${latest_session}/logs" -maxdepth 1 -type f 2>/dev/null | sort | tail -n 20)
  fi
  ray_logs_found="${#ray_logs[@]}"
  {
    echo "ray_root=${ray_root}"
    echo "ray_latest_session=${latest_session:-}"
    echo "ray_logs_found=${ray_logs_found}"
    if (( ray_logs_found > 0 )); then
      for ray_log in "${ray_logs[@]}"; do
        echo "== ${ray_log} =="
        tail -n 120 "${ray_log}" 2>/dev/null || true
      done
    fi
  } > "${ray_tail}" 2>&1 || true

  if [[ -f "${out_dir}/serve.log" ]]; then
    tail -n 300 "${out_dir}/serve.log" > "${serve_tail}" 2>&1 || true
  else
    echo "serve_log_missing=true" > "${serve_tail}" 2>/dev/null || true
  fi

  if [[ -f "${out_dir}/ready_probe_${port}.log" ]]; then
    tail -n 300 "${out_dir}/ready_probe_${port}.log" > "${ready_tail}" 2>&1 || true
  else
    echo "ready_probe_log_missing=true" > "${ready_tail}" 2>/dev/null || true
  fi
}

wait_for_service() {
  local port="$1"
  local serve_log="$2"
  local ready_log="$3"
  local out_dir="$4"
  local scenario="$5"
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
      capture_startup_diagnostics "${out_dir}" "${scenario}" "service_exited_before_ready" || true
      tail -n 200 "${serve_log}" >&2 || true
      exit 1
    fi
    sleep 5
  done
  echo "service_ready_timeout=true" | tee -a "${ready_log}" >&2
  capture_startup_diagnostics "${out_dir}" "${scenario}" "service_ready_timeout" || true
  tail -n 200 "${serve_log}" >&2 || true
  exit 1
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
  local bytes=0

  : > "${drain_log}"
  rm -f "${final_file}" "${timeout_file}" "${sample_file}"
  echo "gpu_drain_start=$(date -Is)" >> "${drain_log}"
  while (( elapsed <= GPU_DRAIN_TIMEOUT_SECONDS )); do
    poll=$((poll + 1))
    gpu_apps > "${sample_file}" || true
    bytes="$(wc -c < "${sample_file}" | tr -d ' ')"
    if [[ ! -s "${sample_file}" ]]; then
      stable=$((stable + 1))
    else
      stable=0
    fi
    echo "poll=${poll} elapsed_seconds=${elapsed} bytes=${bytes} stable_empty_polls=${stable}" >> "${drain_log}"
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

extract_trace_jsonl() {
  local serve_log="$1"
  local trace_jsonl="$2"
  PHASE274_SERVE_LOG="${serve_log}" PHASE274_TRACE_JSONL="${trace_jsonl}" python3 - <<'PY'
import json
import os
from pathlib import Path

marker = "AIC_PHASE274_DEEPER_TRACE_ROW "
serve_log = Path(os.environ["PHASE274_SERVE_LOG"])
trace_jsonl = Path(os.environ["PHASE274_TRACE_JSONL"])
rows = []
for line in serve_log.read_text(encoding="utf-8", errors="replace").splitlines():
    if marker not in line:
        continue
    payload = line.split(marker, 1)[1]
    rows.append(json.loads(payload))
if not rows:
    raise SystemExit("missing_phase274_deeper_trace_rows=true")
with trace_jsonl.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, sort_keys=True) + "\n")
print(f"deeper_scheduler_trace_rows={len(rows)}")
PY
}

write_result_json() {
  local scenario="$1"
  local out_dir="$2"
  local serve_command="$3"
  local benchmark_command="$4"
  PHASE274_SCENARIO="${scenario}" \
  PHASE274_TOPOLOGY_KEY="$(scenario_topology_key "${scenario}")" \
  PHASE274_SHAPE_KEY="$(scenario_shape_key "${scenario}")" \
  PHASE274_TP="$(scenario_tp "${scenario}")" \
  PHASE274_DP="$(scenario_dp "${scenario}")" \
  PHASE274_EP="$(scenario_ep "${scenario}")" \
  PHASE274_ISL="$(scenario_isl "${scenario}")" \
  PHASE274_MAX_BT="$(scenario_max_bt "${scenario}")" \
  PHASE274_MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
  PHASE274_BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS}" \
  PHASE274_SERVE_COMMAND="${serve_command}" \
  PHASE274_BENCHMARK_COMMAND="${benchmark_command}" \
  PHASE274_BENCH_JSON="${PHASE274_TEST_BENCH_JSON:-${out_dir}/bench_result.json}" \
  PHASE274_TRACE_JSONL="${PHASE274_TEST_TRACE_JSONL:-${out_dir}/deeper_scheduler_trace.jsonl}" \
  PHASE274_OUT_JSON="${PHASE274_TEST_OUT_JSON:-${out_dir}/phase274_result.json}" \
  python3 - <<'PY'
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

bench = json.loads(Path(os.environ["PHASE274_BENCH_JSON"]).read_text(encoding="utf-8"))
trace_rows = [
    json.loads(line)
    for line in Path(os.environ["PHASE274_TRACE_JSONL"]).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if int(bench.get("failed_requests", -1)) != 0:
    raise SystemExit(f"failed_requests={bench.get('failed_requests')}")
if int(bench.get("ok_requests", 0)) != int(os.environ["PHASE274_BENCH_NUM_PROMPTS"]):
    raise SystemExit("ok_requests does not match requested prompt count")
if not trace_rows:
    raise SystemExit("missing deeper scheduler trace rows")

required_fields = {
    "source",
    "scenario",
    "iteration",
    "phase",
    "scheduled_context_tokens",
    "scheduled_decode_tokens",
    "scheduled_total_tokens",
    "scheduled_context_reqs",
    "scheduled_decode_reqs",
    "scheduled_total_reqs",
    "max_num_batched_tokens",
    "max_num_seqs",
    "forward_token_count",
    "tp",
    "dp",
    "ep",
    "topology_key",
    "shape_key",
    "iteration_start_ns",
    "forward_start_ns",
    "forward_end_ns",
    "iteration_end_ns",
    "iteration_elapsed_ns",
    "forward_elapsed_ns",
    "active_request_count",
    "scheduled_new_req_count",
    "scheduled_cached_req_count",
    "scheduled_resumed_req_count",
    "finished_req_count",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
}
non_negative_integer_fields = {
    "iteration",
    "scheduled_context_tokens",
    "scheduled_decode_tokens",
    "scheduled_total_tokens",
    "scheduled_context_reqs",
    "scheduled_decode_reqs",
    "scheduled_total_reqs",
    "max_num_batched_tokens",
    "max_num_seqs",
    "forward_token_count",
    "tp",
    "dp",
    "ep",
    "iteration_start_ns",
    "forward_start_ns",
    "forward_end_ns",
    "iteration_end_ns",
    "iteration_elapsed_ns",
    "forward_elapsed_ns",
    "active_request_count",
    "scheduled_new_req_count",
    "scheduled_cached_req_count",
    "scheduled_resumed_req_count",
    "finished_req_count",
}
expected_values = {
    "scenario": os.environ["PHASE274_SCENARIO"],
    "topology_key": os.environ["PHASE274_TOPOLOGY_KEY"],
    "shape_key": os.environ["PHASE274_SHAPE_KEY"],
    "max_num_batched_tokens": int(os.environ["PHASE274_MAX_BT"]),
    "max_num_seqs": int(os.environ["PHASE274_MAX_NUM_SEQS"]),
    "tp": int(os.environ["PHASE274_TP"]),
    "dp": int(os.environ["PHASE274_DP"]),
    "ep": int(os.environ["PHASE274_EP"]),
}
allowed_phases = {"prefill", "mixed", "pure_decode"}


def _is_non_negative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


for row_index, row in enumerate(trace_rows):
    missing = sorted(required_fields - row.keys())
    if missing:
        raise SystemExit(f"trace_row_missing_field=row_{row_index}:{','.join(missing)}")
    if row["source"] != "phase274_deeper_scheduler_trace":
        raise SystemExit(f"trace_row_source_mismatch=row_{row_index}")
    for field in non_negative_integer_fields:
        if not _is_non_negative_int(row[field]):
            raise SystemExit(f"trace_row_non_negative_integer_mismatch=row_{row_index}:{field}")
    for field, expected_value in expected_values.items():
        if row[field] != expected_value:
            raise SystemExit(
                f"trace_row_{field}_mismatch=row_{row_index}:"
                f"actual={row[field]} expected={expected_value}"
            )
    if row["phase"] not in allowed_phases:
        raise SystemExit(f"trace_row_phase_mismatch=row_{row_index}:{row['phase']}")
    if (
        row["diagnostic_only"] is not True
        or row["valid_for_default"] is not False
        or row["perf_database"] is not False
    ):
        raise SystemExit(f"trace_row_flag_mismatch=row_{row_index}")
    if row["scheduled_total_tokens"] != row["scheduled_context_tokens"] + row["scheduled_decode_tokens"]:
        raise SystemExit(f"trace_row_token_sum_mismatch=row_{row_index}")
    if row["scheduled_total_reqs"] != row["scheduled_context_reqs"] + row["scheduled_decode_reqs"]:
        raise SystemExit(f"trace_row_request_sum_mismatch=row_{row_index}")
    if row["forward_end_ns"] < row["forward_start_ns"]:
        raise SystemExit(f"trace_row_timing_order_mismatch=row_{row_index}:forward")
    if row["iteration_end_ns"] < row["iteration_start_ns"]:
        raise SystemExit(f"trace_row_timing_order_mismatch=row_{row_index}:iteration")

tp = int(os.environ["PHASE274_TP"])
dp = int(os.environ["PHASE274_DP"])
worker_rows_per_iteration = tp * dp
timing_fields = {
    "iteration_start_ns",
    "forward_start_ns",
    "forward_end_ns",
    "iteration_end_ns",
    "iteration_elapsed_ns",
    "forward_elapsed_ns",
}
payload_signature_fields = tuple(sorted(required_fields - timing_fields))
rows_by_iteration = defaultdict(list)
for row in trace_rows:
    rows_by_iteration[int(row["iteration"])].append(row)

iterations = sorted(rows_by_iteration)
if iterations != list(range(iterations[0], iterations[-1] + 1)):
    raise SystemExit("trace_iteration_gap_mismatch=true")
if iterations[0] != 0:
    raise SystemExit("trace_iteration_start_mismatch=true")


def _payload_signature(row: dict) -> tuple:
    return tuple(row[field] for field in payload_signature_fields)


aggregated_iterations = []
for iteration in iterations:
    worker_rows = rows_by_iteration[iteration]
    if len(worker_rows) != worker_rows_per_iteration:
        raise SystemExit(
            f"trace_iteration_worker_count_mismatch=iteration_{iteration}:"
            f"actual={len(worker_rows)} expected={worker_rows_per_iteration}"
        )
    payloads = {}
    for row in worker_rows:
        signature = _payload_signature(row)
        if signature not in payloads:
            payloads[signature] = {"row": row, "count": 0}
        payloads[signature]["count"] += 1

    if dp == 1 and len(payloads) != 1:
        raise SystemExit(
            f"trace_iteration_payload_count_mismatch=iteration_{iteration}:"
            f"actual={len(payloads)} expected=1"
        )
    if dp > 1 and len(payloads) > dp:
        raise SystemExit(
            f"trace_iteration_payload_count_mismatch=iteration_{iteration}:"
            f"actual={len(payloads)} max={dp}"
        )

    rank_sum_context_tokens = 0
    rank_sum_decode_tokens = 0
    rank_sum_total_tokens = 0
    rank_sum_context_reqs = 0
    rank_sum_decode_reqs = 0
    rank_sum_total_reqs = 0
    rank_sum_forward_tokens = 0
    payload_rank_count = 0
    for payload in payloads.values():
        count = int(payload["count"])
        if count % tp != 0:
            raise SystemExit(
                f"trace_iteration_payload_worker_count_mismatch=iteration_{iteration}:"
                f"count={count} tp={tp}"
            )
        rank_count = count // tp
        payload_rank_count += rank_count
        row = payload["row"]
        rank_sum_context_tokens += int(row["scheduled_context_tokens"]) * rank_count
        rank_sum_decode_tokens += int(row["scheduled_decode_tokens"]) * rank_count
        rank_sum_total_tokens += int(row["scheduled_total_tokens"]) * rank_count
        rank_sum_context_reqs += int(row["scheduled_context_reqs"]) * rank_count
        rank_sum_decode_reqs += int(row["scheduled_decode_reqs"]) * rank_count
        rank_sum_total_reqs += int(row["scheduled_total_reqs"]) * rank_count
        rank_sum_forward_tokens += int(row["forward_token_count"]) * rank_count
    if payload_rank_count != dp:
        raise SystemExit(
            f"trace_iteration_payload_rank_count_mismatch=iteration_{iteration}:"
            f"actual={payload_rank_count} expected={dp}"
        )

    if rank_sum_context_reqs and rank_sum_decode_reqs:
        phase = "mixed"
    elif rank_sum_context_reqs:
        phase = "prefill"
    elif rank_sum_decode_reqs:
        phase = "pure_decode"
    else:
        phase = "idle"
    if phase == "idle":
        raise SystemExit(f"trace_iteration_idle_mismatch=iteration_{iteration}")

    aggregated_iterations.append(
        {
            "iteration": iteration,
            "phase": phase,
            "payload_count": len(payloads),
            "rank_sum_context_tokens": rank_sum_context_tokens,
            "rank_sum_decode_tokens": rank_sum_decode_tokens,
            "rank_sum_total_tokens": rank_sum_total_tokens,
            "rank_sum_forward_tokens": rank_sum_forward_tokens,
            "rank_max_forward_elapsed_ns": max(int(row["forward_elapsed_ns"]) for row in worker_rows),
            "rank_max_iteration_elapsed_ns": max(int(row["iteration_elapsed_ns"]) for row in worker_rows),
            "rank_max_overhead_ns": max(
                int(row["iteration_elapsed_ns"]) - int(row["forward_elapsed_ns"])
                for row in worker_rows
            ),
        }
    )

mixed_iterations = [
    item["iteration"] for item in aggregated_iterations if item["phase"] == "mixed"
]
pure_iterations = [
    item["iteration"] for item in aggregated_iterations if item["phase"] == "pure_decode"
]
tail_decode_tokens = [
    item["rank_sum_decode_tokens"]
    for item in sorted(
        (item for item in aggregated_iterations if item["phase"] == "pure_decode"),
        key=lambda item: item["iteration"],
    )[-5:]
]
payload = {
    "source": "phase274_deeper_scheduler_trace",
    "scenario": os.environ["PHASE274_SCENARIO"],
    "topology_key": os.environ["PHASE274_TOPOLOGY_KEY"],
    "shape_key": os.environ["PHASE274_SHAPE_KEY"],
    "bench_result": {k: v for k, v in bench.items() if k != "records"},
    "shape": {
        "isl": int(os.environ["PHASE274_ISL"]),
        "osl": int(bench["output_len"]),
        "batch_size": int(bench["max_concurrency"]),
        "max_num_batched_tokens": int(os.environ["PHASE274_MAX_BT"]),
        "max_num_seqs": int(os.environ["PHASE274_MAX_NUM_SEQS"]),
    },
    "parallelism": {
        "tp": tp,
        "dp": dp,
        "ep": int(os.environ["PHASE274_EP"]),
    },
    "scheduler_trace": {
        "trace_rows": len(trace_rows),
        "unique_iterations": len(aggregated_iterations),
        "worker_rows_per_iteration": worker_rows_per_iteration,
        "max_payloads_per_iteration": max(item["payload_count"] for item in aggregated_iterations),
        "phase_counts": dict(Counter(item["phase"] for item in aggregated_iterations)),
        "max_scheduled_total_tokens": max(
            int(item["rank_sum_total_tokens"]) for item in aggregated_iterations
        ),
        "max_forward_token_count": max(
            int(item["rank_sum_forward_tokens"]) for item in aggregated_iterations
        ),
        "rank_sum_max_scheduled_total_tokens": max(
            int(item["rank_sum_total_tokens"]) for item in aggregated_iterations
        ),
        "rank_sum_max_forward_token_count": max(
            int(item["rank_sum_forward_tokens"]) for item in aggregated_iterations
        ),
        "rank_max_forward_elapsed_ns": max(
            int(item["rank_max_forward_elapsed_ns"]) for item in aggregated_iterations
        ),
        "rank_max_iteration_elapsed_ns": max(
            int(item["rank_max_iteration_elapsed_ns"]) for item in aggregated_iterations
        ),
        "rank_max_overhead_ns": max(
            int(item["rank_max_overhead_ns"]) for item in aggregated_iterations
        ),
    },
    "boundary_timeline": {
        "first_iteration": min(iterations),
        "last_iteration": max(iterations),
        "mixed_iterations": mixed_iterations,
        "first_pure_decode_iteration": min(pure_iterations) if pure_iterations else None,
        "tail_decode_tokens": tail_decode_tokens,
    },
    "runtime": {
        "serve_command": os.environ["PHASE274_SERVE_COMMAND"],
        "benchmark_command": os.environ["PHASE274_BENCHMARK_COMMAND"],
    },
    "diagnostic_only": True,
    "valid_for_default": False,
    "perf_database": False,
}
Path(os.environ["PHASE274_OUT_JSON"]).write_text(
    json.dumps(payload, indent=2, sort_keys=True),
    encoding="utf-8",
)
PY
}

run_one() {
  local scenario="$1"
  local out_dir port serve_log ready_log run_log bench_json trace_jsonl
  local serve_command benchmark_command benchmark_exit_code=0 write_result_exit_code=0
  [[ -n "${scenario}" ]] || { usage >&2; exit 2; }
  scenario_index "${scenario}" >/dev/null
  if [[ "${PHASE274_ALLOW_GPU_RUN:-0}" != "1" ]]; then
    echo "phase274_allow_gpu_run_required=true" >&2
    exit 2
  fi

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
  run_log="${out_dir}/run_one_${scenario}.log"
  CLEANUP_LOG="${out_dir}/cleanup_${scenario}.log"
  bench_json="${out_dir}/bench_result.json"
  trace_jsonl="${out_dir}/deeper_scheduler_trace.jsonl"
  : > "${run_log}"
  : > "${CLEANUP_LOG}"
  trap cleanup EXIT

  echo "service_log=${serve_log}"
  echo "runner_log=${run_log}"
  echo "ready_probe_log=${ready_log}"
  echo "trace_jsonl=${trace_jsonl}"
  echo "cleanup_log=${CLEANUP_LOG}"
  echo "realtime_tail_command=tail -f ${serve_log} ${run_log} ${ready_log} ${CLEANUP_LOG}"
  exec >> "${run_log}" 2>&1

  check_gpu_state "${out_dir}/gpu_compute_apps_before.txt"
  patch_source
  python3 -m py_compile "${RUNNER_PATH}"
  build_serve_cmd "${scenario}"
  build_benchmark_cmd "${scenario}"
  serve_command="$(quote_cmd "${SERVE_CMD[@]}")"
  benchmark_command="$(quote_cmd "${BENCH_CMD[@]}")"

  echo "scenario=${scenario}"
  echo "runner_log_start=$(date -Is)"
  echo "serve_command=${serve_command}"
  echo "benchmark_command=${benchmark_command}"
  nohup "${SERVE_CMD[@]}" > "${serve_log}" 2>&1 &
  SERVICE_PID=$!
  echo "service_pid=${SERVICE_PID}"
  wait_for_service "${port}" "${serve_log}" "${ready_log}" "${out_dir}" "${scenario}"

  set +e
  "${BENCH_CMD[@]}"
  benchmark_exit_code="$?"
  set -e
  echo "benchmark_exit_code=${benchmark_exit_code}"

  stop_service
  SERVICE_PID=""
  extract_trace_jsonl "${serve_log}" "${trace_jsonl}"
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
  echo "phase274_result_json=${out_dir}/phase274_result.json"
}

run_test_validate_trace() {
  local scenario="$1"
  [[ -n "${scenario}" ]] || { usage >&2; exit 2; }
  scenario_index "${scenario}" >/dev/null
  [[ -n "${PHASE274_TEST_BENCH_JSON:-}" ]] || { echo "missing_phase274_test_bench_json=true" >&2; exit 2; }
  [[ -n "${PHASE274_TEST_TRACE_JSONL:-}" ]] || { echo "missing_phase274_test_trace_jsonl=true" >&2; exit 2; }
  [[ -n "${PHASE274_TEST_OUT_JSON:-}" ]] || { echo "missing_phase274_test_out_json=true" >&2; exit 2; }
  write_result_json "${scenario}" "$(dirname "${PHASE274_TEST_OUT_JSON}")" "test_serve_command" "test_benchmark_command"
  echo "phase274_trace_guard=PASS"
}

run_test_patch_source() {
  [[ -n "${PHASE274_GPU_MODEL_RUNNER_PATH:-}" ]] || {
    echo "missing_phase274_gpu_model_runner_path=true" >&2
    exit 2
  }
  trap restore_source EXIT
  patch_source
  grep -q "AIC_PHASE274_DEEPER_TRACE_ROW" "${RUNNER_PATH}" || {
    echo "missing_phase274_trace_marker_after_patch=true" >&2
    exit 1
  }
  grep -q "aic_phase274_forward_end_ns = time.perf_counter_ns()" "${RUNNER_PATH}" || {
    echo "missing_phase274_forward_end_timing_after_patch=true" >&2
    exit 1
  }
  restore_source
  trap - EXIT
  echo "phase274_patch_source=PASS"
}

run_test_capture_startup_diagnostics() {
  local scenario="$1"
  local out_dir
  [[ -n "${scenario}" ]] || { usage >&2; exit 2; }
  scenario_index "${scenario}" >/dev/null
  out_dir="${PHASE274_TEST_OUT_DIR:-$(mktemp -d)}"
  capture_startup_diagnostics "${out_dir}" "${scenario}" "test_startup_failure"
  echo "phase274_startup_diagnostics=PASS"
  echo "startup_diagnostics_dir=${out_dir}"
}

case "${MODE}" in
  source-check)
    run_source_check
    ;;
  preflight)
    run_preflight
    ;;
  run-one)
    run_one "${SCENARIO}"
    ;;
  cleanup)
    cleanup
    ;;
  __test-validate-trace)
    run_test_validate_trace "${SCENARIO}"
    ;;
  __test-patch-source)
    run_test_patch_source
    ;;
  __test-capture-startup-diagnostics)
    run_test_capture_startup_diagnostics "${SCENARIO}"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
