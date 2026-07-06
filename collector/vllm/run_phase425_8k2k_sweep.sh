#!/usr/bin/env bash
set -euo pipefail

# Phase425 wrapper: reuse the Phase412 sustained-arrival trace driver for the
# DP2 8k2k point. Measurement-only; it does not change runtime or DB state.

export PHASE_NAME="${PHASE_NAME:-phase425}"
export OUT_ROOT="${OUT_ROOT:-docs/iter_gap_investigation/phase425_8k2k_sweep}"
export TMP_ROOT="${TMP_ROOT:-/tmp/phase425_8k2k_sweep_$$}"
export PORT="${PORT:-20925}"

export SCENARIO="${SCENARIO:-K2.5-tp4ep8dp2-8k2k}"
export ISL="${ISL:-8000}"
export OSL="${OSL:-2000}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8000}"

export BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-512}"
export BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-128}"

exec "$(dirname "$0")/run_phase412_arrival_sweep.sh" "$@"
