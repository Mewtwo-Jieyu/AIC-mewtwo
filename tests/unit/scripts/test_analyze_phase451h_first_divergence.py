#!/usr/bin/env python3
"""Tests for Phase451-H first-divergence analyzer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase451h_first_divergence.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase451h", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_find_first_divergence_reports_first_state_difference() -> None:
    mod = _load_module()
    real = [
        {"axis": 0, "running": 1, "free_blocks": 900, "waiting": 4, "phase": "decode"},
        {"axis": 1, "running": 2, "free_blocks": 800, "waiting": 3, "phase": "mixed_prefill"},
        {"axis": 2, "running": 3, "free_blocks": 700, "waiting": 2, "phase": "decode"},
    ]
    sim = [
        {"axis": 0, "running": 1, "free_blocks": 900, "waiting": 4, "phase": "decode"},
        {"axis": 1, "running": 2, "free_blocks": 800, "waiting": 3, "phase": "mixed_prefill"},
        {"axis": 2, "running": 4, "free_blocks": 650, "waiting": 1, "phase": "mixed_prefill"},
    ]

    divergence = mod.find_first_divergence(real, sim)

    assert divergence["status"] == "found"
    assert divergence["axis"] == 2
    assert divergence["first_differing_fields"] == ["running", "free_blocks", "waiting", "phase"]
    assert divergence["semantic_hint"] == "admission_or_capacity_gate"


def test_classify_latency_outlier_separates_recovery_paths() -> None:
    mod = _load_module()
    latencies = [100.0, 102.0, 105.0, 9000.0, 101.0, 9800.0]

    immediate = mod.classify_latency_outlier(
        latency_ms=9400.0,
        baseline_ms=100.0,
        prefill_ms=9000.0,
        completion_wave_ms=80000.0,
    )
    delayed = mod.classify_latency_outlier(
        latency_ms=80500.0,
        baseline_ms=100.0,
        prefill_ms=9000.0,
        completion_wave_ms=80000.0,
    )
    summary = mod.summarize_latency_outliers(
        latencies,
        baseline_ms=100.0,
        prefill_ms=9000.0,
        completion_wave_ms=80000.0,
    )

    assert immediate == "immediate_refill_like"
    assert delayed == "wait_for_completion_wave_like"
    assert summary["immediate_refill_like"] == 2
    assert summary["wait_for_completion_wave_like"] == 0


if __name__ == "__main__":
    test_find_first_divergence_reports_first_state_difference()
    test_classify_latency_outlier_separates_recovery_paths()
