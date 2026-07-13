#!/usr/bin/env python3
"""Tests for Phase462 Step2b preemption observation analysis."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts/analyze_phase462_preemption_observation.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase462_observation", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pair_records_matches_allocate_failure_to_following_decision() -> None:
    mod = _load_module()
    records = [
        {
            "kind": "allocate_failure",
            "pid": 1,
            "ts_ns": 10,
            "trigger_request_id": "a",
            "requested_blocks": 1,
            "free_blocks": 0,
        },
        {
            "kind": "preempt_decision",
            "pid": 1,
            "ts_ns": 11,
            "trigger_request_id": "a",
            "victim_request_id": "b",
            "victim_position": 13,
            "trigger_num_computed_tokens": 32_928,
            "victim_num_computed_tokens": 32_927,
        },
    ]

    pairs = mod.pair_records(records)

    assert len(pairs) == 1
    assert pairs[0].requested_blocks == 1
    assert pairs[0].victim_request_id == "b"


def test_first_divergence_rejects_allocation_threshold_as_root() -> None:
    mod = _load_module()
    real = [
        mod.DecisionPair("a", "b", 1, 0, 13, 10, 32_928, 32_927),
        mod.DecisionPair("c", "d", 1, 0, 13, 62_000_000_010, 32_928, 32_927),
    ]
    sim = [
        {
            "local_iter": 944,
            "trigger_req_id": 13,
            "victim_req_id": 13,
            "victim_preemptions_before": 0,
            "over_blocks_before": 1,
        },
        {
            "local_iter": 946,
            "trigger_req_id": 13,
            "victim_req_id": 13,
            "victim_preemptions_before": 1,
            "over_blocks_before": 1,
        },
    ]

    verdict = mod.first_divergence_verdict(real, sim)

    assert verdict["allocation_condition"] == "aligned_one_block_short"
    assert verdict["first_divergence"] == "victim_relation_and_recurrence"
    assert verdict["candidate"] == "per_request_decode_phase_evolution"
    assert verdict["runtime_fix_allowed"] is False


def test_decision_summary_counts_repeat_and_self_preemption() -> None:
    mod = _load_module()
    events = [
        {
            "trigger_req_id": 1,
            "victim_req_id": 1,
            "victim_preemptions_before": 0,
            "over_blocks_before": 1,
        },
        {
            "trigger_req_id": 1,
            "victim_req_id": 1,
            "victim_preemptions_before": 1,
            "over_blocks_before": 1,
        },
        {
            "trigger_req_id": 2,
            "victim_req_id": 3,
            "victim_preemptions_before": 0,
            "over_blocks_before": 1,
        },
    ]

    summary = mod.summarize_sim_decisions(events)

    assert summary["preemptions"] == 3
    assert summary["unique_victims"] == 2
    assert summary["self_preemptions"] == 2
    assert summary["repeat_victim_events"] == 1
