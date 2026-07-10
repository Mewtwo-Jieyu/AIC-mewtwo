#!/usr/bin/env bash
set -euo pipefail
pkill -USR1 -f "VLLM::EngineCore" 2>/dev/null || true
pkill -USR1 -f "VLLM::Worker" 2>/dev/null || true
sleep "${AIC_PHASE446_SIGNAL_FLUSH_WAIT_S:-10}"
