#!/usr/bin/env python3
"""Tests for the Phase462 workload-exposure stop-loss experiment."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase462_exposure_sensitivity.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_phase462_exposure_sensitivity", MODULE_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_steady_window_uses_state_boundaries_not_fixed_steps() -> None:
    mod = _load_module()

    window = mod.derive_steady_window(
        first_prefill_completion_steps={0: 3, 1: 7, 2: 5},
        request_completion_steps={0: 20, 1: 24, 2: 22},
    )

    assert window.start_step == 8
    assert window.end_step == 20
    assert window.mode == "fully_admitted_before_drain"


def test_steady_window_uses_replacement_plateau_when_kv_queue_outlives_first_request() -> None:
    mod = _load_module()

    window = mod.derive_steady_window(
        first_prefill_completion_steps={0: 3, 1: 30, 2: 22},
        request_completion_steps={0: 20, 1: 44, 2: 42},
    )

    assert window.start_step == 21
    assert window.end_step == 31
    assert window.mode == "replacement_plateau_while_waiting_nonempty"


def test_paired_noise_band_is_computed_from_request_differences() -> None:
    mod = _load_module()

    identical = mod.paired_variation_noise_band(
        [0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]
    )
    shifted = mod.paired_variation_noise_band(
        [0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0]
    )

    assert identical.delta == 0.0
    assert identical.equivalent is True
    assert shifted.noise_band == 0.0
    assert shifted.equivalent is False


def test_phase_noise_uses_multinomial_sampling_band() -> None:
    mod = _load_module()

    same = mod.phase_distribution_comparison(
        {0: 50, 1: 50},
        {0: 50, 1: 50},
    )
    different = mod.phase_distribution_comparison(
        {0: 100},
        {1: 100},
    )

    assert same.delta == 0.0
    assert same.equivalent is True
    assert different.delta == 1.0
    assert different.delta > different.noise_band
    assert different.equivalent is False


def test_throughput_samples_use_source_block_period() -> None:
    mod = _load_module()
    rows = [
        {"decode_reqs": 16, "latency_ms": 2.0}
        for _ in range(32)
    ]

    samples = mod.block_throughput_samples(rows, block_size=16)

    assert samples == [8000.0, 8000.0]


def test_scope_requires_every_experiment_to_be_steady_equivalent() -> None:
    mod = _load_module()

    passed = mod.scope_verdict({"short_32k": True, "bt65536": True})
    failed = mod.scope_verdict({"short_32k": True, "bt65536": False})

    assert passed == {
        "step2c": "candidate",
        "scope": "engine_loop_with_t0_workload_exposure",
        "ramp_boundary": "first_schedule_trajectory_not_modeled",
        "runtime_change_allowed": False,
    }
    assert failed == {
        "step2c": "locked",
        "scope": "one_client_source_audit_or_narrow_engine_loop_scope",
        "ramp_boundary": "steady_state_is_exposure_sensitive",
        "runtime_change_allowed": False,
    }


def test_oracle_exposure_is_experiment_only() -> None:
    mod = _load_module()

    assert mod.EXPOSURE_BOUNDARIES == {
        "t0_source_model": "delivery_candidate",
        "oracle_target_timestamps": "experiment_only",
        "target_timestamps_in_runtime": False,
    }
    assert mod.REPORT_BOUNDARIES["diagnostic_only"] is True
    assert mod.REPORT_BOUNDARIES["valid_for_default"] is False
    assert mod.REPORT_BOUNDARIES["perf_database"] is False
    assert mod.REPORT_BOUNDARIES["default_aic"] == "No-Go"


def test_markdown_keeps_sign_predictions_and_zero_giant_scope(tmp_path) -> None:
    mod = _load_module()
    _, details = mod.build_report()
    output = tmp_path / "report.md"

    mod.write_markdown(output, details)

    report = output.read_text(encoding="utf-8")
    assert "## 六点符号预测" in report
    assert "K2.5-tp8ep8-8k2k" in report
    assert "只证明两种暴露输入之间不敏感" in report
    assert "不能替代 N=512 动态验收" in report
