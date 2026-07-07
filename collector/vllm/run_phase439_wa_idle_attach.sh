#!/usr/bin/env bash
set -euo pipefail

# Phase439 W-A: attach torch profiler while vLLM is idle, then record the
# ramp-to-steady interval and offline-filter steady steps.

WORKDIR="${WORKDIR:-/mnt/shared-storage-user/ailab-sys/zhaojieyu/aic}"
MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--moonshotai--Kimi-K2.5}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-kimi-k2.5}"
OUT_ROOT="${OUT_ROOT:-docs/iter_gap_investigation/phase439_wa_idle_attach}"
PORT="${PORT:-20939}"
FORCE="${FORCE:-0}"

SCENARIO="${SCENARIO:-K2.5-tp4ep8dp2-8k2k-wa-idle-attach}"
TP="${TP:-4}"
DP="${DP:-2}"
EP="${EP:-8}"
ISL="${ISL:-8000}"
OSL="${OSL:-2000}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"

BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-192}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-128}"
BENCH_WARMUP_REQUESTS="${BENCH_WARMUP_REQUESTS:-0}"
BENCH_TIMEOUT_S="${BENCH_TIMEOUT_S:-14400}"
SMOKE_NUM_PROMPTS="${SMOKE_NUM_PROMPTS:-8}"
SMOKE_MAX_CONCURRENCY="${SMOKE_MAX_CONCURRENCY:-4}"
SMOKE_INPUT_LEN="${SMOKE_INPUT_LEN:-64}"
SMOKE_OUTPUT_LEN="${SMOKE_OUTPUT_LEN:-8}"

ACTIVE_ITERS="${ACTIVE_ITERS:-4000}"
TORCH_PROFILER_RECORD_SHAPES="${TORCH_PROFILER_RECORD_SHAPES:-false}"
TRACE_FLUSH_WAIT_SECONDS="${TRACE_FLUSH_WAIT_SECONDS:-90}"
STEADY_RECORD_SECONDS="${STEADY_RECORD_SECONDS:-60}"
MIN_CTX_TOKENS="${MIN_CTX_TOKENS:-7500}"
MIN_MIXED_DECODE_BATCH="${MIN_MIXED_DECODE_BATCH:-56}"
STEADY_RUNNING_MIN="${STEADY_RUNNING_MIN:-56}"
STEADY_STABLE_POLLS="${STEADY_STABLE_POLLS:-3}"
STEADY_TIMEOUT_SECONDS="${STEADY_TIMEOUT_SECONDS:-900}"
METRICS_POLL_INTERVAL_S="${METRICS_POLL_INTERVAL_S:-2}"

READY_TIMEOUT_POLLS="${READY_TIMEOUT_POLLS:-480}"
DRAIN_POLL_SECONDS="${DRAIN_POLL_SECONDS:-5}"
DRAIN_STABLE_POLLS="${DRAIN_STABLE_POLLS:-3}"
DRAIN_TIMEOUT_SECONDS="${DRAIN_TIMEOUT_SECONDS:-600}"
GRACEFUL_TERM_SECONDS="${GRACEFUL_TERM_SECONDS:-120}"

export PATH="/opt/py3/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/kubebrain:/usr/local/nvidia/bin:${PATH}"
export LD_LIBRARY_PATH="/nccl/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.9/compat:/usr/local/lib/python3.12/dist-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export VLLM_ENABLE_CUDA_COMPATIBILITY="${VLLM_ENABLE_CUDA_COMPATIBILITY:-1}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
unset VLLM_NUM_GPU_BLOCKS_OVERRIDE
unset NUM_GPU_BLOCKS_OVERRIDE

SERVICE_PID=""
BENCH_PID=""
METRICS_PID=""
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

stop_metrics_poller() {
  if [[ -n "${METRICS_PID}" ]]; then
    kill -TERM "${METRICS_PID}" 2>/dev/null || true
    wait "${METRICS_PID}" 2>/dev/null || true
    METRICS_PID=""
  fi
}

stop_bench() {
  if [[ -n "${BENCH_PID}" ]]; then
    kill -TERM "${BENCH_PID}" 2>/dev/null || true
    wait "${BENCH_PID}" 2>/dev/null || true
    BENCH_PID=""
  fi
}

stop_service() {
  stop_metrics_poller
  stop_bench
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
  local serve_log="$1" poll
  for poll in $(seq 1 "${READY_TIMEOUT_POLLS}"); do
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      log "service_ready port=${PORT} poll=${poll}"
      return 0
    fi
    if [[ -n "${SERVICE_PID}" ]] && ! kill -0 "${SERVICE_PID}" 2>/dev/null; then
      log "service_exited_before_ready port=${PORT}"
      tail -n 160 "${serve_log}" || true
      return 1
    fi
    sleep 5
  done
  log "service_ready_timeout port=${PORT}"
  tail -n 160 "${serve_log}" || true
  return 1
}

start_metrics_poller() {
  local metrics_jsonl="$1"
  python3 - "${PORT}" "${metrics_jsonl}" "${METRICS_POLL_INTERVAL_S}" <<'PY' &
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
            with urllib.request.urlopen(url, timeout=5) as resp:
                record["status"] = resp.status
                record["body"] = resp.read().decode("utf-8", errors="replace")
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
  log "metrics_poller_pid=${METRICS_PID}"
}

wait_for_steady_running() {
  local out_dir="$1"
  python3 - "${PORT}" "${out_dir}/steady_running_checks.jsonl" "${STEADY_RUNNING_MIN}" \
    "${STEADY_STABLE_POLLS}" "${STEADY_TIMEOUT_SECONDS}" "${METRICS_POLL_INTERVAL_S}" \
    "${BENCH_PID}" <<'PY'
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.request

port = int(sys.argv[1])
out_path = sys.argv[2]
threshold = float(sys.argv[3])
stable_target = int(sys.argv[4])
timeout_s = float(sys.argv[5])
interval_s = float(sys.argv[6])
bench_pid = int(sys.argv[7])
url = f"http://127.0.0.1:{port}/metrics"
running_re = re.compile(r'^vllm:num_requests_running\{[^}]*engine="(?P<engine>[^"]+)"[^}]*\}\s+(?P<value>[0-9.]+)$')

stable = 0
start = time.monotonic()
with open(out_path, "a", encoding="utf-8") as f:
    while True:
        if time.monotonic() - start > timeout_s:
            print(f"steady_running_timeout threshold={threshold} stable={stable}/{stable_target}", file=sys.stderr)
            raise SystemExit(1)
        try:
            os.kill(bench_pid, 0)
        except OSError:
            print("benchmark_finished_before_steady_running", file=sys.stderr)
            raise SystemExit(1)
        values = {}
        error = None
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            for line in body.splitlines():
                match = running_re.match(line)
                if match:
                    values[match.group("engine")] = float(match.group("value"))
        except Exception as exc:
            error = repr(exc)
        ok = bool(values) and len(values) >= 2 and all(value >= threshold for value in values.values())
        stable = stable + 1 if ok else 0
        payload = {
            "ts": dt.datetime.now(dt.UTC).isoformat(),
            "threshold": threshold,
            "stable": stable,
            "stable_target": stable_target,
            "values": values,
            "ok": ok,
            "error": error,
        }
        f.write(json.dumps(payload, sort_keys=True) + "\n")
        f.flush()
        if stable >= stable_target:
            print(json.dumps(payload, sort_keys=True))
            raise SystemExit(0)
        time.sleep(interval_s)
PY
}

start_service() {
  local prof_dir="$1" serve_log="$2"
  local profiler_json
  profiler_json="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${prof_dir}\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":${TORCH_PROFILER_RECORD_SHAPES},\"warmup_iterations\":0,\"active_iterations\":${ACTIVE_ITERS},\"wait_iterations\":0}"
  export VLLM_TORCH_PROFILER_DIR="${prof_dir}"
  local serve_cmd=(
    python3 -m vllm.entrypoints.cli.main serve "${MODEL_PATH}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --port "${PORT}"
    --tensor-parallel-size "${TP}"
    --data-parallel-size "${DP}"
    --enable-expert-parallel
    --max-model-len "${MAX_MODEL_LEN}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --enable-chunked-prefill
    --no-enable-prefix-caching
    --enable-logging-iteration-details
    --cudagraph-metrics
    --profiler-config "${profiler_json}"
    --mm-encoder-tp-mode data
    --skip-mm-profiling
    --trust-remote-code
    --enable-auto-tool-choice
    --tool-call-parser kimi_k2
    --reasoning-parser kimi_k2
  )
  log "serve_command: ${serve_cmd[*]}" | tee -a "${OUT_ROOT}/driver.log"
  nohup "${serve_cmd[@]}" >"${serve_log}" 2>&1 &
  SERVICE_PID=$!
  log "service_pid=${SERVICE_PID}" | tee -a "${OUT_ROOT}/driver.log"
  wait_for_service "${serve_log}"
}

start_bench() {
  local num_prompts="$1" max_concurrency="$2" input_len="$3" output_len="$4"
  local result_json="$5" records_jsonl="$6" bench_log="$7"
  python3 scripts/run_openai_fixed_shape_benchmark.py \
    --host 127.0.0.1 --port "${PORT}" \
    --model "${SERVED_MODEL_NAME}" --tokenizer "${MODEL_PATH}" \
    --num-prompts "${num_prompts}" --max-concurrency "${max_concurrency}" \
    --input-len "${input_len}" --output-len "${output_len}" \
    --warmup-requests "${BENCH_WARMUP_REQUESTS}" --timeout-s "${BENCH_TIMEOUT_S}" \
    --prompt-variant-mode rotating \
    --result-json "${result_json}" --records-jsonl "${records_jsonl}" >"${bench_log}" 2>&1 &
  BENCH_PID=$!
  log "bench_pid=${BENCH_PID} prompts=${num_prompts} concurrency=${max_concurrency} isl=${input_len} osl=${output_len}"
}

profile_start_idle() {
  local gate="$1"
  log "profile_start_idle gate=${gate}"
  curl --max-time 30 -fsS -X POST "http://127.0.0.1:${PORT}/start_profile" >/dev/null
}

profile_stop() {
  local gate="$1"
  log "profile_stop gate=${gate}"
  curl --max-time 60 -fsS -X POST "http://127.0.0.1:${PORT}/stop_profile" >/dev/null
}

copy_profile_window() {
  local prof_dir="$1" out_dir="$2" window="$3"
  local dst="${out_dir}/prof_${window}"
  rm -rf "${dst}"
  mkdir -p "${dst}"
  cp -f "${prof_dir}"/* "${dst}/" 2>/dev/null || true
  local nfiles ntrace
  nfiles="$(find "${dst}" -type f | wc -l | tr -d ' ')"
  ntrace="$(find "${dst}" -type f -name '*.pt.trace.json.gz' | wc -l | tr -d ' ')"
  echo "files=${nfiles}" >"${dst}/profile_done.txt"
  echo "trace_files=${ntrace}" >>"${dst}/profile_done.txt"
  log "copied_profile_files window=${window} files=${nfiles} trace_files=${ntrace}"
  (( ntrace > 0 ))
}

check_trace_window() {
  local out_dir="$1" window="$2" gate="$3"
  local prof_dir="${out_dir}/prof_${window}"
  local check_file="${out_dir}/check_${window}.json"
  local rc=0
  python3 - "${prof_dir}" "${window}" "${gate}" "${MIN_CTX_TOKENS}" "${MIN_MIXED_DECODE_BATCH}" >"${check_file}" <<'PY' || rc=$?
import gzip
import json
import re
import sys
from pathlib import Path

prof_dir = Path(sys.argv[1])
window = sys.argv[2]
gate = sys.argv[3]
min_ctx = int(sys.argv[4])
min_batch = int(sys.argv[5])
execute_context_re = re.compile(
    rb"execute_context_\d+\((?P<ctx_tokens>\d+)\)_generation_\d+\((?P<gen_tokens>\d+)\)"
)

def scan(path):
    carry = b""
    with gzip.open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            data = carry + chunk
            for match in execute_context_re.finditer(data):
                yield int(match.group("ctx_tokens")), int(match.group("gen_tokens"))
            carry = data[-256:]

paths = sorted(p for p in prof_dir.glob("*.pt.trace.json.gz") if p.name.startswith("dp"))
prefill = 0
hit_mixed = 0
mixed_batch_max = 0
decode_batch_max = 0
for path in paths:
    for ctx, gen in scan(path):
        if ctx > 0:
            prefill += 1
            if ctx >= min_ctx:
                mixed_batch_max = max(mixed_batch_max, gen)
                if gen >= min_batch:
                    hit_mixed += 1
        elif gen > 0:
            decode_batch_max = max(decode_batch_max, gen)

if gate == "PHASE439_GATE1_SMOKE":
    passed = bool(paths)
    reason = "trace_files_seen" if passed else "trace_files_missing"
else:
    passed = bool(paths) and hit_mixed > 0
    reason = "mixed_prefill_decode_batch_ge_min_seen" if passed else "mixed_prefill_high_batch_missing"
payload = {
    "window": window,
    "gate": gate,
    "trace_files": len(paths),
    "prefill_rank_steps": prefill,
    "mixed_decode_batch_ge_min_steps": hit_mixed,
    "mixed_decode_batch_max": mixed_batch_max,
    "decode_batch_max": decode_batch_max,
    "passed": passed,
    "reason": reason,
}
print(json.dumps(payload, sort_keys=True))
raise SystemExit(0 if passed else 1)
PY
  python3 - "${out_dir}/window_checks.jsonl" "${check_file}" <<'PY'
import json
import sys

out_path, check_file = sys.argv[1:]
with open(check_file, encoding="utf-8") as f:
    payload = json.load(f)
with open(out_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(payload, sort_keys=True) + "\n")
PY
  if (( rc == 0 )); then
    log "window_check_passed $(cat "${check_file}")"
    return 0
  fi
  log "window_check_failed $(cat "${check_file}" 2>/dev/null || true)"
  return 1
}

clear_prof_dir() {
  local prof_dir="$1"
  mkdir -p "${prof_dir}"
  find "${prof_dir}" -mindepth 1 -maxdepth 1 -type f -delete
}

run_gate1_smoke() {
  local out_dir="$1" prof_dir="$2"
  local tmp_dir="${out_dir}/tmp_gate1"
  mkdir -p "${tmp_dir}"
  clear_prof_dir "${prof_dir}"
  log "PHASE439_GATE1_SMOKE start"
  profile_start_idle "PHASE439_GATE1_SMOKE"
  start_bench "${SMOKE_NUM_PROMPTS}" "${SMOKE_MAX_CONCURRENCY}" "${SMOKE_INPUT_LEN}" "${SMOKE_OUTPUT_LEN}" \
    "${tmp_dir}/bench_result.json" "${tmp_dir}/bench_records.jsonl" "${tmp_dir}/bench.log"
  wait "${BENCH_PID}"
  BENCH_PID=""
  profile_stop "PHASE439_GATE1_SMOKE"
  log "profile_flush_wait=${TRACE_FLUSH_WAIT_SECONDS}s gate=PHASE439_GATE1_SMOKE"
  sleep "${TRACE_FLUSH_WAIT_SECONDS}"
  copy_profile_window "${prof_dir}" "${out_dir}" "gate1_smoke"
  check_trace_window "${out_dir}" "gate1_smoke" "PHASE439_GATE1_SMOKE"
}

run_gate2_wa() {
  local out_dir="$1" prof_dir="$2"
  local tmp_dir="${out_dir}/tmp_gate2"
  mkdir -p "${tmp_dir}"
  clear_prof_dir "${prof_dir}"
  log "PHASE439_GATE2_WA start"
  profile_start_idle "PHASE439_GATE2_WA"
  start_bench "${BENCH_NUM_PROMPTS}" "${BENCH_MAX_CONCURRENCY}" "${ISL}" "${OSL}" \
    "${tmp_dir}/bench_result.json" "${tmp_dir}/bench_records.jsonl" "${tmp_dir}/bench.log"
  log "waiting_for_steady_running threshold=${STEADY_RUNNING_MIN} stable_polls=${STEADY_STABLE_POLLS}"
  wait_for_steady_running "${out_dir}"
  log "steady_record_seconds=${STEADY_RECORD_SECONDS}"
  sleep "${STEADY_RECORD_SECONDS}"
  profile_stop "PHASE439_GATE2_WA"
  log "profile_flush_wait=${TRACE_FLUSH_WAIT_SECONDS}s gate=PHASE439_GATE2_WA"
  sleep "${TRACE_FLUSH_WAIT_SECONDS}"
  copy_profile_window "${prof_dir}" "${out_dir}" "wa_idle_attach"
  cp -f "${tmp_dir}/bench.log" "${out_dir}/bench_wa_idle_attach.log" 2>/dev/null || true
  cp -f "${tmp_dir}/bench_records.jsonl" "${out_dir}/bench_records_wa_idle_attach.jsonl" 2>/dev/null || true
  cp -f "${tmp_dir}/bench_result.json" "${out_dir}/bench_result_wa_idle_attach.json" 2>/dev/null || true
  stop_bench
  check_trace_window "${out_dir}" "wa_idle_attach" "PHASE439_GATE2_WA"
}

write_meta() {
  local out_dir="$1"
  cat >"${out_dir}/meta.json" <<EOF
{"name":"${SCENARIO}","phase":"phase439","window":"wa_idle_attach","tp":${TP},"dp":${DP},"ep":${EP},"world_size":$((TP * DP)),"isl":${ISL},"osl":${OSL},"bench_num_prompts":${BENCH_NUM_PROMPTS},"bench_max_concurrency":${BENCH_MAX_CONCURRENCY},"max_num_batched_tokens":${MAX_NUM_BATCHED_TOKENS},"port":${PORT},"prefix_caching":false,"gpu_memory_utilization":${GPU_MEMORY_UTILIZATION},"max_model_len":${MAX_MODEL_LEN},"max_num_seqs":${MAX_NUM_SEQS},"enable_logging_iteration_details":true,"cudagraph_metrics":true,"torch_profiler_record_shapes":${TORCH_PROFILER_RECORD_SHAPES},"profile_process_model":"idle_attach_then_ramp_to_steady","steady_running_min":${STEADY_RUNNING_MIN},"steady_stable_polls":${STEADY_STABLE_POLLS},"active_iterations":${ACTIVE_ITERS},"steady_record_seconds":${STEADY_RECORD_SECONDS},"trace_flush_wait_seconds":${TRACE_FLUSH_WAIT_SECONDS},"self_check":"mixed_prefill_decode_batch_ge_${MIN_MIXED_DECODE_BATCH}","default_readiness":"No-Go"}
EOF
}

main() {
  cd "${WORKDIR}"
  local out_dir="${OUT_ROOT}/${SCENARIO}"
  local prof_dir="${out_dir}/profiler_raw"
  local serve_log="${out_dir}/serve.log"
  local metrics_jsonl="${out_dir}/metrics.jsonl"

  if [[ "${FORCE}" == "1" ]]; then
    rm -rf "${out_dir}"
    rm -f "${OUT_ROOT}/driver.log" "${OUT_ROOT}/runtime.txt" "${OUT_ROOT}/gpu_inventory.txt"
  fi
  mkdir -p "${out_dir}" "${prof_dir}" "${OUT_ROOT}"
  : >"${OUT_ROOT}/driver.log"
  : >"${out_dir}/window_checks.jsonl"
  : >"${out_dir}/steady_running_checks.jsonl"
  : >"${metrics_jsonl}"
  : >"${serve_log}"
  write_meta "${out_dir}"

  rm -rf /tmp/phase439_smoke
  python3 -c 'import vllm; print("vllm_version="+getattr(vllm,"__version__","unknown"))' | tee "${OUT_ROOT}/runtime.txt"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits >"${OUT_ROOT}/gpu_inventory.txt"
  sed -n '1p' "${OUT_ROOT}/gpu_inventory.txt" | tee -a "${OUT_ROOT}/runtime.txt"

  BASELINE_PIDS="$(gpu_busy_pids | tr '\n' ' ')"
  log "baseline_gpu_pids=[${BASELINE_PIDS}]" | tee -a "${OUT_ROOT}/driver.log"
  snapshot_gpu_apps "${OUT_ROOT}/gpu_compute_apps_before.txt"
  wait_for_gpu_drain || { log "abort: my gpu procs still busy"; return 1; }

  start_service "${prof_dir}" "${serve_log}"
  start_metrics_poller "${metrics_jsonl}"
  log "PHASE439_GATE0 service_ready profiler_dir=${prof_dir}" | tee -a "${OUT_ROOT}/driver.log"
  run_gate1_smoke "${out_dir}" "${prof_dir}"
  run_gate2_wa "${out_dir}" "${prof_dir}"

  snapshot_gpu_apps "${OUT_ROOT}/gpu_compute_apps_after.txt"
  pgrep -af "vllm.entrypoints.cli.main serve|ray::|raylet|gcs_server|VLLM::APIServer|VLLM::EngineCore|run_openai_fixed_shape_benchmark" >"${OUT_ROOT}/process_residual_after.txt" || true
  log "PHASE439_WA_IDLE_ATTACH_DONE" | tee -a "${OUT_ROOT}/driver.log"
}

main "$@"
