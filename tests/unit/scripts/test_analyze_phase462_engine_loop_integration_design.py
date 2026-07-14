#!/usr/bin/env python3
"""Tests for the Phase462 Step2c-1 engine-loop integration design."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT / "scripts" / "analyze_phase462_engine_loop_integration_design.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_phase462_engine_loop_integration_design", MODULE_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_design_scope_covers_every_runtime_loop_without_policy_changes() -> None:
    mod = _load_module()

    design = mod.build_design_spec()

    assert {item["path"] for item in design["runtime_paths"]} == {
        "single_replica",
        "multi_replica",
        "dp_lockstep",
    }
    assert all(item["change"] == "engine_loop_state_machine" for item in design["runtime_paths"])
    assert design["arrival_layer"]["workload_exposure"] == "t0_closed_loop"
    assert design["arrival_layer"]["tokenizer_primitive"] == "measured_32_requests_2ms"
    assert design["unchanged"] == {
        "scheduler_admission_policy",
        "preemption_trigger",
        "victim_selection",
        "max_bt_keying",
        "perf_database",
        "validate_protocol",
        "multi_config_gate",
    }


def test_design_preregisters_stop_lines_and_no_fallback() -> None:
    mod = _load_module()

    design = mod.build_design_spec()

    assert design["stop_lines"] == {
        "oracle_structure_regression",
        "steady_self_preemption_nonzero",
        "unexpected_ab_sign",
        "out_of_scope_file_change",
    }
    assert design["steady_gate"]["self_preemption"] == 0
    assert design["steady_gate"]["repeat_victim"] == 0
    assert design["steady_gate"]["failure_action"] == "stop_before_ab"
    assert "giant_bucket" not in design["steady_gate"]
    assert design["dynamic_ab_gate"]["giant_bucket"] == "n512_trace_only"
    assert design["fallback"] == "none"


def test_sign_registry_has_four_worsen_and_two_improve() -> None:
    mod = _load_module()

    signs = mod.load_sign_predictions(mod.SIGN_CSV)

    assert len(signs) == 6
    assert sum(item["prediction"] == "worsen" for item in signs) == 4
    assert sum(item["prediction"] == "improve_until_crossing" for item in signs) == 2
    assert all(item["status"] == "pending_dynamic_ab" for item in signs)


def test_source_audit_finds_existing_serial_loops_and_backend_routing() -> None:
    mod = _load_module()

    audit = mod.audit_source_surfaces(REPO_ROOT)

    assert audit["status"] == "pass"
    assert audit["single_replica_serial_loop"] is True
    assert audit["multi_replica_serial_loop"] is True
    assert audit["dp_lockstep_serial_loop"] is True
    assert audit["dp_legacy_guard_present"] is True


def test_review_verdict_is_report_only_and_keeps_default_no_go() -> None:
    mod = _load_module()

    verdict = mod.design_review_verdict(
        mod.build_design_spec(), mod.audit_source_surfaces(REPO_ROOT)
    )

    assert verdict == {
        "step": "phase462_step2c1",
        "status": "ready_for_user_review",
        "runtime_change_allowed": False,
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
        "default_aic": "No-Go",
    }


def test_markdown_contains_required_review_and_cascade_boundaries(tmp_path) -> None:
    mod = _load_module()
    rows, details = mod.build_report()
    output = tmp_path / "design.md"

    mod.write_markdown(output, details)

    report = output.read_text(encoding="utf-8")
    assert rows
    assert "## 改动范围" in report
    assert "## 明确不改" in report
    assert "## 事件顺序" in report
    assert "## 分层红绿" in report
    assert "## post-2c 预注册" in report
    assert "短期计分板变差不是事故" in report
    assert "tp8-8k2k" in report
    assert "138" in report
    assert "8.54x" in report
    assert "95%" in report
    assert "diagnostic_only=true" in report
    assert "runtime_change_allowed=false" in report
