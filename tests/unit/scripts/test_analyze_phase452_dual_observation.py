#!/usr/bin/env python3
"""Tests for Phase452 dual observation analyzer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase452_dual_observation.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase452", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_route_interleaving_detects_stats_overwrite_loss() -> None:
    mod = _load_module()
    records = [
        {
            "kind": "route",
            "ts_ns": 10,
            "pid": 1,
            "client_index": 0,
            "chosen_engine_index": 0,
            "pre_counts": [[0, 0], [0, 0]],
            "post_counts": [[2, 0], [0, 0]],
        },
        {
            "kind": "stats_overwrite",
            "ts_ns": 15,
            "pid": 1,
            "client_index": 0,
            "before_counts": [[2, 0], [0, 0]],
            "after_counts": [[0, 0], [0, 0]],
        },
        {
            "kind": "route",
            "ts_ns": 20,
            "pid": 1,
            "client_index": 0,
            "chosen_engine_index": 0,
            "pre_counts": [[0, 0], [0, 0]],
            "post_counts": [[2, 0], [0, 0]],
        },
    ]

    summary = mod.summarize_route_interleaving(records)

    assert summary.route_rows == 2
    assert summary.stats_between_pairs == 1
    assert summary.local_increment_lost_pairs == 1
    assert summary.route_counts == {0: 2}


def test_victim_summary_classifies_recompute_and_cached_resume() -> None:
    mod = _load_module()
    records = [
        {
            "kind": "preempt_decision",
            "ts_ns": 30,
            "victim": {"request_id": "a"},
        },
        {
            "kind": "victim_reschedule",
            "ts_ns": 40,
            "request": {"request_id": "a"},
            "num_computed_tokens_for_schedule": 0,
        },
        {
            "kind": "preempt_decision",
            "ts_ns": 50,
            "victim": {"request_id": "b"},
        },
        {
            "kind": "victim_reschedule",
            "ts_ns": 60,
            "request": {"request_id": "b"},
            "num_computed_tokens_for_schedule": 512,
        },
        {
            "kind": "preempt_decision",
            "ts_ns": 70,
            "victim": {"request_id": "c"},
        },
    ]

    summary = mod.summarize_victim_path(records)

    assert summary.preempt_decisions == 3
    assert summary.unique_victims == 3
    assert summary.recompute_from_zero == 1
    assert summary.resume_with_cached_tokens == 1
    assert summary.missing_reschedule == 1


def test_build_rows_exposes_two_line_decisions() -> None:
    mod = _load_module()
    records = []
    for idx in range(121):
        records.append(
            {
                "kind": "route",
                "ts_ns": idx * 10,
                "pid": 1,
                "client_index": 0,
                "chosen_engine_index": idx % 2,
                "pre_counts": [[0, 0], [0, 0]],
                "post_counts": [[2, 0], [0, 0]],
            }
        )
        if idx < 120:
            records.append(
                {
                    "kind": "stats_overwrite",
                    "ts_ns": idx * 10 + 5,
                    "pid": 1,
                    "client_index": 0,
                    "after_counts": [[0, 0], [0, 0]],
                }
            )
    records.extend(
        [
            {"kind": "preempt_decision", "ts_ns": 2000, "victim": {"request_id": "x"}},
            {
                "kind": "victim_reschedule",
                "ts_ns": 2010,
                "request": {"request_id": "x"},
                "num_computed_tokens_for_schedule": 0,
            },
        ]
    )

    rows = mod.build_rows(records)
    verdicts = {
        (row["section"], row["metric"]): row for row in rows if row["metric"] == "verdict"
    }

    assert verdicts[("route", "verdict")]["value"] == "stats_overwrite_interleaves_add"
    assert verdicts[("route", "verdict")]["status"] == "pass"
    assert verdicts[("victim", "verdict")]["value"] == "victim_recompute_from_zero"
    assert verdicts[("victim", "verdict")]["status"] == "pass"


if __name__ == "__main__":
    test_route_interleaving_detects_stats_overwrite_loss()
    test_victim_summary_classifies_recompute_and_cached_resume()
    test_build_rows_exposes_two_line_decisions()
