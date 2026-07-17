#!/usr/bin/env bash
set -uo pipefail

WORKDIR="${WORKDIR:-/mnt/shared-storage-user/zhaojieyu/backup/aic}"
OUT_ROOT="${OUT_ROOT:-phase463_six_point_latency_recollect_263a969}"
RUNNER="${RUNNER:-collector/vllm/run_phase463_six_point_latency_recollect.sh}"
WAIT_PID="${WAIT_PID:-}"
START_INDEX="${START_INDEX:-2}"
WAIT_INTERVAL_S="${WAIT_INTERVAL_S:-30}"

POINT_NAMES=(
  "K2.5-tp8ep8-8k2k-canary-nonstream"
  "K2.5-tp8ep8-8k2k"
  "K2.5-tp8ep8-32k3k"
  "K2.5-tp4ep8dp2-8k2k"
  "K2.5-tp4ep8dp2-32k3k"
  "K2.5-tp8ep8-8k2k-bt65536"
  "K2.5-tp4ep8dp2-8k2k-bt65536"
  "K2.5-tp4ep8dp2-32k3k-c64-diagnostic"
)

log() { echo "[$(date -Is)] $*"; }

record_result() {
  printf '%s\t%s\t%s\n' "$(date -Is)" "$1" "$2" >>"${OUT_ROOT}/supervisor_results.tsv"
}

archive_partial() {
  local out_dir="$1"
  local archived="${out_dir}-failed-$(date +%Y%m%dT%H%M%S)"
  mv "${out_dir}" "${archived}"
  log "archived_partial=${archived}"
}

main() {
  cd "${WORKDIR}"
  mkdir -p "${OUT_ROOT}"
  touch "${OUT_ROOT}/supervisor_results.tsv"

  if [[ -n "${WAIT_PID}" ]]; then
    log "waiting_for_existing_runner pid=${WAIT_PID}"
    while kill -0 "${WAIT_PID}" 2>/dev/null; do
      sleep "${WAIT_INTERVAL_S}"
    done
    log "existing_runner_finished pid=${WAIT_PID}"
  fi

  local idx name out_dir rc
  for idx in "${!POINT_NAMES[@]}"; do
    if (( idx < START_INDEX )); then
      continue
    fi
    name="${POINT_NAMES[$idx]}"
    out_dir="${OUT_ROOT}/${name}"
    if [[ -f "${out_dir}/scenario_analysis.json" ]]; then
      log "already_complete idx=${idx} name=${name}"
      record_result "${name}" "already_complete"
      continue
    fi
    if [[ -e "${out_dir}" ]]; then
      archive_partial "${out_dir}"
      log "POINT_SKIPPED_AFTER_FAILURE idx=${idx} name=${name} reason=existing_partial"
      record_result "${name}" "skipped_existing_partial"
      continue
    fi

    log "background_point_start idx=${idx} name=${name}"
    ONLY_POINTS="${idx}" OUT_ROOT="${OUT_ROOT}" bash "${RUNNER}"
    rc=$?
    if [[ "${rc}" != "0" || ! -f "${out_dir}/scenario_analysis.json" ]]; then
      if [[ -e "${out_dir}" ]]; then
        archive_partial "${out_dir}"
      fi
      log "POINT_SKIPPED_AFTER_FAILURE idx=${idx} name=${name} rc=${rc}"
      record_result "${name}" "failed_rc_${rc}"
      continue
    fi
    log "background_point_done idx=${idx} name=${name}"
    record_result "${name}" "completed"
  done
  log "background_supervisor_done"
}

main "$@"
