from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[3] / "scripts" / "analyze_phase438_serving_state_merge.py"
    spec = importlib.util.spec_from_file_location("phase438_merge", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_phase438_union_merge_reports_conflict_without_overwrite():
    phase438 = _load_module()
    baseline = [
        phase438.ServingRow(
            source="phase436",
            phase="mixed_prefill",
            category="ep_a2a",
            bucket_tokens=8000,
            decode_batch=34,
            latency_ms=100.0,
            provenance="old",
        )
    ]
    candidate = [
        phase438.ServingRow(
            source="phase437",
            phase="mixed_prefill",
            category="ep_a2a",
            bucket_tokens=8000,
            decode_batch=34,
            latency_ms=140.0,
            provenance="new",
        )
    ]

    merged, conflicts = phase438.merge_rows(baseline, candidate, conflict_threshold_pct=15.0)

    assert len(merged) == 1
    assert merged[0].latency_ms == 100.0
    assert len(conflicts) == 1
    assert conflicts[0]["decision"] == "keep_existing_report_conflict"


def test_phase438_axis_sensitivity_refuses_high_spread_reduction():
    phase438 = _load_module()
    rows = [
        phase438.ServingRow("p", "mixed_prefill", "ep_a2a", 8000, 1, 100.0, "a"),
        phase438.ServingRow("p", "mixed_prefill", "ep_a2a", 8000, 2, 130.0, "b"),
    ]

    summary = phase438.axis_decisions(rows, tolerance_pct=15.0)

    decision = summary[("mixed_prefill", "ep_a2a")]
    assert decision["decode_batch_spread_pct"] > 15.0
    assert decision["decision"] == "keep_2d"


def test_phase438_window_recommendation_keeps_only_needed_windows():
    phase438 = _load_module()
    audit_rows = [
        {
            "scenario": "K2.5-tp4ep8dp2-8k2k",
            "phase": "mixed_prefill",
            "miss_reason": "decode_batch_above_range",
            "count": "87",
        },
        {
            "scenario": "K2.5-tp4ep8dp2-32k3k",
            "phase": "mixed_prefill",
            "miss_reason": "hit",
            "count": "10",
        },
    ]

    windows = phase438.recommend_gpu_windows(audit_rows)

    assert [row["window"] for row in windows] == ["W-A"]
    assert windows[0]["reason"] == "8k mixed prefill decode_batch above range"


def test_phase438_window_recommendation_ignores_32k_bucket_below_when_ab_is_clean():
    phase438 = _load_module()
    audit_rows = [
        {
            "scenario": "K2.5-tp4ep8dp2-32k3k",
            "phase": "mixed_prefill",
            "miss_reason": "bucket_below_range",
            "count": "128",
        }
    ]

    assert phase438.recommend_gpu_windows(audit_rows) == []
