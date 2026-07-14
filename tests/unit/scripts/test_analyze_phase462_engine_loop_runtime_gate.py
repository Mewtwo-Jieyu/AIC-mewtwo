#!/usr/bin/env python3
"""Tests for Phase462 Step2c-2 runtime gates."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts/analyze_phase462_engine_loop_runtime_gate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_phase462_engine_loop_runtime_gate", MODULE_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_runtime_gate_requires_zero_steady_self_preemptions() -> None:
    mod = _load_module()

    blocked = mod.evaluate_runtime_gate(
        drain_match=1.0,
        first_schedule_match=0.9375,
        first_16_match=1.0,
        preemptions=10,
        steady_self_preemptions=1,
        repeat_victim_events=0,
    )
    passed = mod.evaluate_runtime_gate(
        drain_match=1.0,
        first_schedule_match=0.9375,
        first_16_match=1.0,
        preemptions=10,
        steady_self_preemptions=0,
        repeat_victim_events=0,
    )

    assert blocked.status == "blocked"
    assert blocked.reason == "steady_state_signature_failed"
    assert blocked.default_enable_allowed is False
    assert passed.status == "pass"
    assert passed.default_enable_allowed is True


def test_report_boundaries_keep_default_aic_no_go() -> None:
    mod = _load_module()

    assert mod.REPORT_BOUNDARIES == {
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
        "default_aic": "No-Go",
    }
