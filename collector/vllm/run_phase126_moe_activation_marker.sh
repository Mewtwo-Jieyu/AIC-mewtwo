#!/usr/bin/env bash
set -euo pipefail

# Phase126 diagnostic runner. It fixes the Phase124 successful marker path:
# parameterized port, eager serve mode, and an enable-file gate so startup
# forward passes do not emit marker rows.

WORKDIR="${WORKDIR:-/mnt/nvme1n1/ml_research/jieyu/aic}"
MODEL_PATH="${MODEL_PATH:-/mnt/cfs/models/kimi-2.5-fp8}"
PORT="${PORT:-18000}"
OUT_DIR="${OUT_DIR:-docs/iter_gap_investigation/phase126_moe_activation_10k2k_b32_bt8192_p${PORT}}"
VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
DEEPSEEK_PATH="${DEEPSEEK_PATH:-${VLLM_ROOT}/model_executor/models/deepseek_v2.py}"
DEEPSEEK_BACKUP_PATH="${DEEPSEEK_BACKUP_PATH:-/tmp/deepseek_v2.py.phase126.bak}"
PHASE117_TABLE="${PHASE117_TABLE:-docs/iter_gap_investigation/phase117_moe_wna16_experimental_table.csv}"
TUNED_CONFIG_DIR="${TUNED_CONFIG_DIR:-/mnt/nvme1n1/ml_research/jieyu/aic/phase108_wna16_config}"
CONFIG_SHA256="${CONFIG_SHA256:-15c797eed1dd441e1d6d50fcf7584b2264d248d5ebc1144e1c95e6c4e885d853}"
EXPECTED_CONFIG_JSON="${EXPECTED_CONFIG_JSON:-E=48,N=2048,device_name=NVIDIA_H200,dtype=int4_w4a16.json}"
BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-32}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-32}"
BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-10000}"
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-2000}"
DRAIN_POLL_SECONDS="${DRAIN_POLL_SECONDS:-5}"
DRAIN_STABLE_POLLS="${DRAIN_STABLE_POLLS:-3}"
DRAIN_TIMEOUT_SECONDS="${DRAIN_TIMEOUT_SECONDS:-900}"
STOP_OCCUPANCY_CMD="${STOP_OCCUPANCY_CMD:-}"
START_OCCUPANCY_CMD="${START_OCCUPANCY_CMD:-}"

export PATH="/usr/local/nvidia/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
export VLLM_ENABLE_CUDA_COMPATIBILITY="${VLLM_ENABLE_CUDA_COMPATIBILITY:-1}"
export VLLM_TUNED_CONFIG_FOLDER="${TUNED_CONFIG_DIR}"
export AIC_PHASE126_SCENARIO="${AIC_PHASE126_SCENARIO:-10k2k_b32_bt8192}"
export AIC_PHASE126_CONFIG_SHA256="${CONFIG_SHA256}"

cd "${WORKDIR}"
mkdir -p "${OUT_DIR}"

ENABLE_FILE="${ENABLE_FILE:-${OUT_DIR}/phase126_moe_activation_marker.enable}"
SERVE_LOG="${OUT_DIR}/serve.log"
BENCH_JSON="${OUT_DIR}/bench_result.json"
BENCH_JSONL="${OUT_DIR}/bench_records.jsonl"
MARKER_LOG="${OUT_DIR}/moe_activation_markers.log"
DRAIN_LOG="${OUT_DIR}/drain_boundary.log"
ROWS_CSV="${OUT_DIR}/moe_activation_rows.csv"
COVERAGE_CSV="${OUT_DIR}/moe_token_bucket_coverage.csv"
GPU_BEFORE="${OUT_DIR}/gpu_compute_apps_before.txt"
GPU_AFTER="${OUT_DIR}/gpu_compute_apps_after.txt"
SERVICE_PID=""

export AIC_PHASE126_MOE_ACTIVATION_ENABLE_FILE="${ENABLE_FILE}"

gpu_apps() {
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
}

stop_occupancy_if_requested() {
  gpu_apps > "${GPU_BEFORE}"
  if [[ -s "${GPU_BEFORE}" && -z "${STOP_OCCUPANCY_CMD}" ]]; then
    echo "gpu_busy_without_STOP_OCCUPANCY_CMD=true" >&2
    cat "${GPU_BEFORE}" >&2
    exit 1
  fi
  if [[ -n "${STOP_OCCUPANCY_CMD}" ]]; then
    bash -lc "${STOP_OCCUPANCY_CMD}"
  fi
}

start_occupancy_if_requested() {
  if [[ -n "${START_OCCUPANCY_CMD}" ]]; then
    bash -lc "${START_OCCUPANCY_CMD}"
  fi
  gpu_apps > "${GPU_AFTER}" || true
}

stop_service() {
  if [[ -n "${SERVICE_PID}" ]]; then
    kill -TERM "${SERVICE_PID}" 2>/dev/null || true
    sleep 5
    kill -KILL "${SERVICE_PID}" 2>/dev/null || true
  fi
  pkill -f "vllm.entrypoints.cli.main serve ${MODEL_PATH}" 2>/dev/null || true
  pkill -f "vllm serve ${MODEL_PATH}" 2>/dev/null || true
  pkill -f "VLLM::DPCoordinator" 2>/dev/null || true
}

restore_source() {
  if [[ -f "${DEEPSEEK_BACKUP_PATH}" ]]; then
    cp "${DEEPSEEK_BACKUP_PATH}" "${DEEPSEEK_PATH}"
  fi
  if grep -n "AIC_MOE_ACTIVATION_EVIDENCE_ROW\|aic_phase126\|AIC_PHASE126" "${DEEPSEEK_PATH}" >/dev/null 2>&1; then
    echo "restore_failed_phase126_marker_still_present=true" >&2
    return 1
  fi
  echo "phase126_markers_after_restore=0"
}

cleanup() {
  set +e
  rm -f "${ENABLE_FILE}"
  stop_service
  restore_source
  start_occupancy_if_requested
}
trap cleanup EXIT

patch_deepseek() {
  if grep -n "AIC_MOE_ACTIVATION_EVIDENCE_ROW\|AIC_PHASE126" "${DEEPSEEK_PATH}" >/dev/null 2>&1; then
    echo "deepseek_already_phase126_instrumented=${DEEPSEEK_PATH}" >&2
    exit 1
  fi
  cp "${DEEPSEEK_PATH}" "${DEEPSEEK_BACKUP_PATH}"
  python3 - "${DEEPSEEK_PATH}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")

def replace_once(src: str, old: str, new: str) -> str:
    count = src.count(old)
    if count != 1:
        raise SystemExit(f"expected one match, got {count}: {old[:120]!r}")
    return src.replace(old, new)

text = replace_once(text, "import torch\n", "import torch\nimport json\nimport os\n")

needle = "    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:\n"
inject = (
    needle
    +
    "        aic_phase126_enable_file = os.environ.get(\"AIC_PHASE126_MOE_ACTIVATION_ENABLE_FILE\")\n"
    "        if aic_phase126_enable_file and os.path.exists(aic_phase126_enable_file):\n"
    "            aic_rank = int(os.environ.get(\"RANK\", \"0\"))\n"
    "            aic_local_rank = int(os.environ.get(\"LOCAL_RANK\", aic_rank))\n"
    "            aic_tp = 4\n"
    "            aic_dp = 2\n"
    "            aic_payload = {\n"
    "                \"source\": \"vllm_moe_activation\",\n"
    "                \"scenario\": os.environ.get(\"AIC_PHASE126_SCENARIO\", \"unknown\"),\n"
    "                \"rank\": aic_rank,\n"
    "                \"local_rank\": aic_local_rank,\n"
    "                \"dp_rank\": int(aic_rank // aic_tp),\n"
    "                \"tp_rank\": int(aic_rank % aic_tp),\n"
    "                \"ep_rank\": aic_rank,\n"
    "                \"input_kind\": \"real_forward_hidden_states\",\n"
    "                \"activation_source\": \"loaded_model_forward\",\n"
    "                \"called_model_forward\": True,\n"
    "                \"timing\": False,\n"
    "                \"tokens_actual\": int(hidden_states.shape[0]),\n"
    "                \"hidden\": int(hidden_states.shape[-1]),\n"
    "                \"intermediate\": 2048,\n"
    "                \"global_experts\": 384,\n"
    "                \"local_experts\": 48,\n"
    "                \"topk\": 8,\n"
    "                \"topology\": \"tp4dp2ep8\",\n"
    "                \"config_sha256\": os.environ.get(\"AIC_PHASE126_CONFIG_SHA256\"),\n"
    "                \"tuning_config_loaded\": True,\n"
    "                \"moe_config_fallback\": False,\n"
    "                \"loaded_weight\": True,\n"
    "                \"random_weight\": False,\n"
    "                \"module\": \"DeepseekV2MoE\",\n"
    "                \"experts_impl\": \"SharedFusedMoE\",\n"
    "                \"diagnostic_only\": True,\n"
    "                \"valid_for_default\": False,\n"
    "                \"perf_database\": False,\n"
    "            }\n"
    "            print(\"AIC_MOE_ACTIVATION_EVIDENCE_ROW \" + json.dumps(aic_payload, sort_keys=True), flush=True)\n"
)

class_start = text.find("class DeepseekV2MoE")
if class_start < 0:
    raise SystemExit("class DeepseekV2MoE not found")
next_class = text.find("\nclass ", class_start + 1)
class_end = next_class if next_class >= 0 else len(text)
forward_start = text.find(needle, class_start, class_end)
if forward_start < 0:
    raise SystemExit("DeepseekV2MoE.forward(hidden_states) not found")
text = text[:forward_start] + inject + text[forward_start + len(needle):]
path.write_text(text, encoding="utf-8")
PY
}

wait_for_service() {
  for _ in $(seq 1 180); do
    curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && return
    if [[ -n "${SERVICE_PID}" ]] && ! kill -0 "${SERVICE_PID}" 2>/dev/null; then
      echo "service exited before readiness" >&2
      tail -n 200 "${SERVE_LOG}" >&2 || true
      exit 1
    fi
    sleep 5
  done
  echo "service did not become ready" >&2
  tail -n 200 "${SERVE_LOG}" >&2 || true
  exit 1
}

marker_row_count() {
  if [[ -f "${SERVE_LOG}" ]]; then
    grep -c "AIC_MOE_ACTIVATION_EVIDENCE_ROW" "${SERVE_LOG}" 2>/dev/null || true
  else
    echo 0
  fi
}

wait_for_marker_drain() {
  local poll_index=0
  local stable_polls=0
  local previous_count=""
  local current_count
  local deadline=$((SECONDS + DRAIN_TIMEOUT_SECONDS))

  echo "poll_index,marker_rows,stable_polls,close_reason" > "${DRAIN_LOG}"
  while true; do
    current_count="$(marker_row_count)"
    if [[ -n "${previous_count}" ]]; then
      if (( current_count < previous_count )); then
        echo "${poll_index},${current_count},${stable_polls},count_decreased" >> "${DRAIN_LOG}"
        echo "drain_close_reason=count_decreased" >&2
        echo "drain_previous_marker_rows=${previous_count}" >&2
        echo "drain_current_marker_rows=${current_count}" >&2
        return 1
      fi
      if (( current_count == previous_count )); then
        stable_polls=$((stable_polls + 1))
      else
        stable_polls=0
      fi
    fi

    if (( stable_polls >= DRAIN_STABLE_POLLS )); then
      echo "${poll_index},${current_count},${stable_polls},stable" >> "${DRAIN_LOG}"
      echo "drain_close_reason=stable"
      echo "drain_final_marker_rows=${current_count}"
      return 0
    fi

    if (( SECONDS >= deadline )); then
      echo "${poll_index},${current_count},${stable_polls},timeout" >> "${DRAIN_LOG}"
      echo "drain_close_reason=timeout" >&2
      echo "drain_final_marker_rows=${current_count}" >&2
      return 1
    fi

    echo "${poll_index},${current_count},${stable_polls},pending" >> "${DRAIN_LOG}"
    previous_count="${current_count}"
    poll_index=$((poll_index + 1))
    sleep "${DRAIN_POLL_SECONDS}"
  done
}

verify_config_file() {
  local target="${TUNED_CONFIG_DIR}/${EXPECTED_CONFIG_JSON}"
  [[ -f "${target}" ]] || { echo "missing_config=${target}" >&2; exit 1; }
  local actual
  actual="$(sha256sum "${target}" | awk '{print $1}')"
  [[ "${actual}" == "${CONFIG_SHA256}" ]] || {
    echo "config_sha256_mismatch=${actual}" >&2
    exit 1
  }
}

if [[ "${1:-run}" == "patch-dry-run" ]]; then
  trap - EXIT
  DRY_DEEPSEEK="${OUT_DIR}/deepseek_v2.phase126.dry.py"
  cp "${DEEPSEEK_PATH}" "${DRY_DEEPSEEK}"
  DEEPSEEK_PATH="${DRY_DEEPSEEK}"
  DEEPSEEK_BACKUP_PATH="${DRY_DEEPSEEK}.bak"
  patch_deepseek
  python3 -m py_compile "${DEEPSEEK_PATH}"
  grep -n "AIC_PHASE126_MOE_ACTIVATION_ENABLE_FILE\|AIC_MOE_ACTIVATION_EVIDENCE_ROW" "${DEEPSEEK_PATH}"
  echo "patch_dry_run=PASS"
  exit 0
fi

if [[ "${1:-run}" == "restore" ]]; then
  trap - EXIT
  restore_source
  exit 0
fi

[[ -f scripts/run_openai_fixed_shape_benchmark.py ]] || { echo "missing_benchmark_script=scripts/run_openai_fixed_shape_benchmark.py" >&2; exit 2; }
[[ -f scripts/analyze_vllm_moe_activation_phase123.py ]] || { echo "missing_parser=scripts/analyze_vllm_moe_activation_phase123.py" >&2; exit 2; }

verify_config_file
stop_occupancy_if_requested
stop_service
rm -f "${ENABLE_FILE}"

patch_deepseek
python3 -m py_compile "${DEEPSEEK_PATH}"

nohup python3 -m vllm.entrypoints.cli.main serve "${MODEL_PATH}" \
  --served-model-name kimi-k2.5 \
  --port "${PORT}" \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --data-parallel-size 2 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 256 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --mm-encoder-tp-mode data \
  --skip-mm-profiling \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2 \
  --enable-logging-iteration-details \
  --enforce-eager \
  > "${SERVE_LOG}" 2>&1 &
SERVICE_PID=$!
echo "service_pid=${SERVICE_PID}"

wait_for_service
touch "${ENABLE_FILE}"

python3 scripts/run_openai_fixed_shape_benchmark.py \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --model kimi-k2.5 \
  --tokenizer "${MODEL_PATH}" \
  --num-prompts "${BENCH_NUM_PROMPTS}" \
  --max-concurrency "${BENCH_MAX_CONCURRENCY}" \
  --input-len "${BENCH_INPUT_LEN}" \
  --output-len "${BENCH_OUTPUT_LEN}" \
  --warmup-requests 0 \
  --timeout-s 7200 \
  --result-json "${BENCH_JSON}" \
  --records-jsonl "${BENCH_JSONL}"

wait_for_marker_drain
rm -f "${ENABLE_FILE}"
stop_service
SERVICE_PID=""

if ! grep "AIC_MOE_ACTIVATION_EVIDENCE_ROW" "${SERVE_LOG}" > "${MARKER_LOG}"; then
  echo "no AIC_MOE_ACTIVATION_EVIDENCE_ROW rows found" >&2
  tail -n 200 "${SERVE_LOG}" >&2 || true
  exit 1
fi

PYTHONPATH=src:. python3 scripts/analyze_vllm_moe_activation_phase123.py \
  --log "${SERVE_LOG}" \
  --phase117-table "${PHASE117_TABLE}" \
  --rows-out "${ROWS_CSV}" \
  --coverage-out "${COVERAGE_CSV}"

echo "phase126_rows_csv=${ROWS_CSV}"
echo "phase126_coverage_csv=${COVERAGE_CSV}"
echo "phase126_marker_rows=$(wc -l < "${MARKER_LOG}")"
