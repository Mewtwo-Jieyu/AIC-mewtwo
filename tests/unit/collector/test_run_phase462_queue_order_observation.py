#!/usr/bin/env python3
"""Contract tests for the Phase462 queue-order GPU runner."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "collector/vllm/run_phase462_queue_order_observation.sh"


def test_runner_keeps_protocol_and_diagnostic_boundaries_fixed() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert "NUM_PROMPTS=128" in text
    assert "CONCURRENCY=128" in text
    assert "ISL=32000" in text
    assert "OSL=1200" in text
    assert "MAX_BT=32000" in text
    assert "MAX_OVERHEAD_PCT=2.0" in text
    assert 'local request_prefix="phase462-queue-order"' in text
    assert '"diagnostic_only": true' in text
    assert '"valid_for_default": false' in text
    assert '"perf_database": false' in text


def test_runner_gates_formal_capture_after_overhead() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    main = text.split("main() {", maxsplit=1)[1]

    off = main.index("run_variant overhead_off")
    on = main.index("run_variant overhead_on")
    gate = main.index("write_overhead_gate")
    capture = main.index("run_variant capture")
    integrity = main.index("run_integrity_gate")

    assert off < on < gate < capture < integrity
    assert "raise SystemExit(0 if payload[\"passed\"] else 1)" in text


def test_runner_restores_all_five_sources_and_checks_residuals() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    for name in (
        "ASYNC_PATH",
        "COMPLETION_PATH",
        "ENGINE_PATH",
        "SCHEDULER_PATH",
        "KV_PATH",
    ):
        assert name in text
    assert "source_restored.sha256" in text
    assert "process_residual_after.txt" in text
    assert "gpu_compute_apps_after.txt" in text
