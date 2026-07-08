#!/usr/bin/env python3
"""Tests for Phase451-G wave ground-truth analyzer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase451g_wave_groundtruth.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase451g", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_reconstruct_closed_loop_arrivals_uses_earliest_free_worker() -> None:
    mod = _load_module()
    records = [
        {"request_index": 0, "latency_ms": 10.0, "ok": 1},
        {"request_index": 1, "latency_ms": 20.0, "ok": 1},
        {"request_index": 2, "latency_ms": 5.0, "ok": 1},
        {"request_index": 3, "latency_ms": 7.0, "ok": 1},
        {"request_index": 4, "latency_ms": 3.0, "ok": 1},
    ]

    timeline = mod.reconstruct_closed_loop_timeline(records, max_concurrency=2)

    assert [round(row["start_ms"], 3) for row in timeline] == [0.0, 0.0, 10.0, 15.0, 20.0]
    assert [round(row["finish_ms"], 3) for row in timeline] == [10.0, 20.0, 15.0, 22.0, 23.0]


def test_wave_summary_reports_generation_spread_growth() -> None:
    mod = _load_module()
    timeline = [
        {"request_index": 0, "start_ms": 0.0, "finish_ms": 100.0},
        {"request_index": 1, "start_ms": 0.0, "finish_ms": 102.0},
        {"request_index": 2, "start_ms": 0.0, "finish_ms": 104.0},
        {"request_index": 3, "start_ms": 0.0, "finish_ms": 106.0},
        {"request_index": 4, "start_ms": 100.0, "finish_ms": 250.0},
        {"request_index": 5, "start_ms": 102.0, "finish_ms": 290.0},
        {"request_index": 6, "start_ms": 104.0, "finish_ms": 330.0},
        {"request_index": 7, "start_ms": 106.0, "finish_ms": 370.0},
    ]

    rows = mod.summarize_client_waves(timeline, max_concurrency=4)
    by_wave = {row["wave_id"]: row for row in rows}

    assert by_wave[0]["finish_span_ms"] == 6.0
    assert by_wave[1]["start_span_ms"] == 6.0
    assert by_wave[1]["finish_span_ms"] == 120.0
    assert by_wave[1]["desync_growth_vs_wave0"] == 20.0


def test_replay_verdict_separates_arrival_from_scheduler_dynamics() -> None:
    mod = _load_module()

    verdict = mod.decide_replay_responsibility(
        real_arrival_preemptions=0,
        real_arrival_mixed_share=0.026,
        closed_loop_preemptions=824,
        closed_loop_mixed_share=0.0633,
        target_mixed_share=0.0243,
    )

    assert verdict["responsibility"] == "arrival_timing_model"
    assert verdict["wave_model_gate"] == "ready_for_nonparametric_arrival_replay_design"


if __name__ == "__main__":
    test_reconstruct_closed_loop_arrivals_uses_earliest_free_worker()
    test_wave_summary_reports_generation_spread_growth()
    test_replay_verdict_separates_arrival_from_scheduler_dynamics()
