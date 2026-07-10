#!/usr/bin/env python3
"""Tests for Phase461 Step4a-3 offline anchor validation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase461_step4_anchor_validation.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase461_step4_anchor_validation", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_anchor_selection_uses_common_cells_and_sample_support_not_drift() -> None:
    mod = _load_module()
    event = {
        (100, 1): mod.CellStats(10.0, 8),
        (200, 2): mod.CellStats(20.0, 3),
        (300, 3): mod.CellStats(30.0, 5),
    }
    same_wall = {
        (100, 1): mod.CellStats(11.0, 7),
        (200, 2): mod.CellStats(21.0, 9),
        (300, 3): mod.CellStats(31.0, 4),
    }
    reference = {
        (100, 1): mod.CellStats(10.5, 6),
        (200, 2): mod.CellStats(20.5, 2),
        (400, 4): mod.CellStats(40.0, 20),
    }

    anchors = mod.select_anchor_cells(event, same_wall, reference, count=2)

    assert anchors == [(100, 1), (200, 2)]


def test_anchor_comparison_reports_three_distinct_measurement_ratios() -> None:
    mod = _load_module()

    result = mod.compare_anchor(
        (100, 4),
        mod.CellStats(100.0, 8),
        mod.CellStats(105.0, 4),
        mod.CellStats(110.0, 6),
    )

    assert result["event_over_same_wall"] == 100.0 / 105.0
    assert result["same_wall_over_reference_wall"] == 105.0 / 110.0
    assert result["event_over_reference_wall"] == 100.0 / 110.0
    assert result["max_relative_drift"] == 1.0 - 100.0 / 110.0


def test_anchor_gate_requires_five_cells_and_every_cell_within_ten_percent() -> None:
    mod = _load_module()
    passing = [{"max_relative_drift": 0.09} for _ in range(5)]

    assert mod.anchor_gate(passing, min_cells=5, max_drift=0.10) == "pass"
    assert mod.anchor_gate(passing[:4], min_cells=5, max_drift=0.10) == "blocked"
    assert mod.anchor_gate(passing[:-1] + [{"max_relative_drift": 0.11}], min_cells=5, max_drift=0.10) == (
        "blocked"
    )


def test_query_cell_influence_requires_both_token_and_batch_brackets() -> None:
    mod = _load_module()
    table = {
        100: {4: 10.0, 8: 12.0},
        200: {4: 20.0, 8: 24.0},
    }

    assert mod.query_uses_cell(table, bucket_tokens=150, decode_batch=6, cell=(100, 4))
    assert not mod.query_uses_cell(table, bucket_tokens=150, decode_batch=9, cell=(100, 4))
    assert not mod.query_uses_cell(table, bucket_tokens=250, decode_batch=6, cell=(100, 4))


def test_ingest_unlock_requires_anchor_and_adjusted_wall_coverage() -> None:
    mod = _load_module()

    assert mod.ingest_unlock(anchor_verdict="pass", adjusted_wall_coverage=0.99) == "pass"
    assert mod.ingest_unlock(anchor_verdict="blocked", adjusted_wall_coverage=0.99) == "blocked"
    assert mod.ingest_unlock(anchor_verdict="pass", adjusted_wall_coverage=0.94) == "blocked"


def test_qualified_wall_coverage_excludes_queries_influenced_by_blocked_cell() -> None:
    mod = _load_module()
    table = {
        100: {4: 10.0, 8: 12.0},
        200: {4: 20.0, 8: 24.0},
    }
    observations = [
        mod.QueryObservation(150, 6, 90.0),
        mod.QueryObservation(200, 8, 10.0),
    ]

    result = mod.qualified_wall_coverage(observations, table, blocked_cells=[(100, 4)])

    assert result["qualified_wall_coverage"] == 0.10
    assert result["blocked_influence_count"] == 1
    assert result["blocked_influence_wall_weight"] == 0.90
