"""Tests for Phase462 Step2c-2b self-preemption first divergence."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "scripts/analyze_phase462_self_preemption_first_divergence.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_phase462_self_preemption_first_divergence", MODULE_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_source_legality_requires_trigger_to_be_running_tail() -> None:
    mod = _load_module()

    legal = mod.evaluate_source_legality(
        allocation_failed=True,
        trigger_request_id=96,
        victim_request_id=96,
        running_tail_request_id=96,
    )
    illegal = mod.evaluate_source_legality(
        allocation_failed=True,
        trigger_request_id=96,
        victim_request_id=96,
        running_tail_request_id=97,
    )

    assert legal.status == "source_legal"
    assert legal.self_preemption_branch_reachable is True
    assert illegal.status == "source_illegal"
    assert illegal.self_preemption_branch_reachable is False


def test_gate_cannot_relax_without_real_trigger_tail_evidence() -> None:
    mod = _load_module()

    verdict = mod.decide_gate_action(
        source_status="source_legal",
        real_relation_observable=True,
        real_trigger_tail_events=0,
        phase458_relation_observable=False,
    )

    assert verdict.action == "keep_gate_and_investigate_state_evolution"
    assert verdict.gate_change_allowed is False
    assert verdict.runtime_change_allowed is False
    assert verdict.gpu_followup_required is True


def test_real_macro_match_uses_queue_and_block_state_not_request_ids() -> None:
    mod = _load_module()
    records = [
        {
            "kind": "allocate_failure",
            "pid": 1,
            "trigger_request_id": "real-trigger",
            "requested_blocks": 1,
            "free_blocks": 0,
        },
        {
            "kind": "preempt_decision",
            "pid": 1,
            "trigger_request_id": "real-trigger",
            "victim_request_id": "real-tail",
            "running_count_after_pop": 13,
            "waiting_count": 31,
        },
    ]

    matches = mod.select_real_macro_matches(
        records,
        waiting_count=31,
        running_count=14,
        block_shortage=1,
    )

    assert len(matches) == 1
    assert matches[0]["trigger_is_tail"] is False


def test_source_illegal_event_requires_implementation_fix() -> None:
    mod = _load_module()

    verdict = mod.decide_gate_action(
        source_status="source_illegal",
        real_relation_observable=True,
        real_trigger_tail_events=0,
        phase458_relation_observable=False,
    )

    assert verdict.action == "fix_engine_loop_implementation"
    assert verdict.gate_change_allowed is False
    assert verdict.runtime_change_allowed is True
    assert verdict.gpu_followup_required is False


def test_report_boundaries_remain_diagnostic_only() -> None:
    mod = _load_module()

    assert mod.REPORT_BOUNDARIES == {
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
        "default_aic": "No-Go",
    }
