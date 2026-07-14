#!/usr/bin/env python3
"""Tests for the Phase462 fully predictive EngineCore prototype."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts/analyze_phase462_predictive_engine_loop.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_phase462_predictive_engine_loop", MODULE_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_prediction_dependency_whitelist_rejects_target_timestamps() -> None:
    mod = _load_module()

    mod.validate_prediction_dependencies(mod.REQUIRED_PREDICTION_DEPENDENCIES)
    with pytest.raises(ValueError, match="judge-only input"):
        mod.validate_prediction_dependencies(
            (*mod.REQUIRED_PREDICTION_DEPENDENCIES, "target_run_timestamps")
        )


def test_prediction_dependency_whitelist_requires_every_source_input() -> None:
    mod = _load_module()

    with pytest.raises(ValueError, match="missing prediction dependencies"):
        mod.validate_prediction_dependencies(
            tuple(
                item
                for item in mod.REQUIRED_PREDICTION_DEPENDENCIES
                if item != "tokenizer_primitive"
            )
        )


def test_initial_closed_loop_cohort_forms_source_defined_microbatches() -> None:
    mod = _load_module()
    protocol = mod.BenchProtocol(
        num_requests=128,
        concurrency=128,
        prompt_tokens=32_000,
    )
    tokenizer = mod.TokenizerRuntimeSpec(
        max_batch_size=32,
        wait_timeout_ms=2.0,
        workers=1,
    )
    primitive = mod.ApprovedTokenizerPrimitive(
        intercept_ms=0.0,
        ms_per_prompt_token=0.001,
        ms_per_request=0.0,
        weighted_mape=0.05,
    )

    prediction = mod.build_predictive_tokenizer_arrivals(
        protocol=protocol,
        tokenizer=tokenizer,
        primitive=primitive,
    )

    assert [batch.request_ids for batch in prediction.batches] == [
        tuple(range(0, 32)),
        tuple(range(32, 64)),
        tuple(range(64, 96)),
        tuple(range(96, 128)),
    ]
    assert [batch.complete_ms for batch in prediction.batches] == [
        1024.0,
        2048.0,
        3072.0,
        4096.0,
    ]
    assert len(prediction.arrivals) == 128


def test_incomplete_tail_batch_waits_after_worker_becomes_available() -> None:
    mod = _load_module()
    prediction = mod.build_predictive_tokenizer_arrivals(
        protocol=mod.BenchProtocol(
            num_requests=33,
            concurrency=33,
            prompt_tokens=1,
        ),
        tokenizer=mod.TokenizerRuntimeSpec(
            max_batch_size=32,
            wait_timeout_ms=2.0,
            workers=1,
        ),
        primitive=mod.ApprovedTokenizerPrimitive(
            intercept_ms=0.0,
            ms_per_prompt_token=1 / 32,
            ms_per_request=0.0,
            weighted_mape=0.0,
        ),
    )

    assert prediction.batches[0].complete_ms == 1.0
    assert prediction.batches[1].start_ms == 3.0


def test_predictive_gate_uses_preregistered_discrete_thresholds() -> None:
    mod = _load_module()

    passed = mod.predictive_gate(
        first_schedule_step_match=0.85,
        first_16_match=14 / 16,
        preemptions=12,
        self_preemptions=0,
        repeat_victim_events=0,
        target_preemptions=10,
    )
    schedule_fail = mod.predictive_gate(
        first_schedule_step_match=0.849,
        first_16_match=14 / 16,
        preemptions=10,
        self_preemptions=0,
        repeat_victim_events=0,
        target_preemptions=10,
    )
    self_preempt_fail = mod.predictive_gate(
        first_schedule_step_match=0.90,
        first_16_match=1.0,
        preemptions=10,
        self_preemptions=1,
        repeat_victim_events=0,
        target_preemptions=10,
    )

    assert passed["passed"] is True
    assert schedule_fail["passed"] is False
    assert self_preempt_fail["passed"] is False
    assert passed["first_16_min_matches"] == 14


def test_report_boundaries_keep_failed_prediction_out_of_runtime() -> None:
    mod = _load_module()

    assert mod.REPORT_BOUNDARIES == {
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
        "default_aic": "No-Go",
    }
    assert mod.runtime_scope_verdict(prediction_gate_passed=False) == {
        "step2c": "locked",
        "verdict": "structure_correct_prediction_limited",
        "runtime_change_allowed": False,
    }
