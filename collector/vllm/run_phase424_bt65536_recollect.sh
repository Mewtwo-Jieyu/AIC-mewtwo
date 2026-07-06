#!/usr/bin/env bash
set -euo pipefail

# Phase424 DP2 bt65536 clean recollect driver.
# Only the protocol label, output directory, temp directory, and max_model_len
# differ from Phase423; the clean measurement protocol is inherited.

export PHASE_NAME="${PHASE_NAME:-phase424}"
export OUT_ROOT="${OUT_ROOT:-docs/iter_gap_investigation/phase424_bt65536_recollect}"
export TMP_ROOT="${TMP_ROOT:-/tmp/phase424_bt65536_recollect_$$}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"

exec "$(dirname "$0")/run_phase423_bt65536_recollect.sh" "$@"
