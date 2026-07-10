#!/usr/bin/env python3
"""Tests for Phase461 uncovered serving-state cell triage."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase461_step4_cell_triage.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase461_step4_cell_triage", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_weighted_coverage_uses_query_wall_contribution() -> None:
    mod = _load_module()
    table = {100: {4: 10.0}, 200: {4: 20.0}}
    observations = [
        mod.QueryObservation(150, 4, 90.0),
        mod.QueryObservation(250, 4, 10.0),
    ]

    summary = mod.summarize_weighted_coverage(observations, table)

    assert summary["query_coverage_ratio"] == 0.5
    assert summary["wall_coverage_ratio"] == 0.9
    assert summary["uncovered_wall_ms"] == 10.0


def test_uncovered_cell_is_real_observed_only_on_exact_reference_match() -> None:
    mod = _load_module()
    real = {
        (250, 4): [12.0, 14.0],
        (300, 5): [20.0],
    }
    cells = {
        (250, 4): {"query_count": 2, "sim_wall_ms": 10.0},
        (251, 4): {"query_count": 1, "sim_wall_ms": 5.0},
    }

    rows = mod.classify_uncovered_cells(cells, real)

    assert rows[(250, 4)]["classification"] == "real_observed"
    assert rows[(250, 4)]["real_samples"] == 2
    assert rows[(250, 4)]["real_wall_median_ms"] == 13.0
    assert rows[(251, 4)]["classification"] == "sim_only_for_phase458_reference"
    assert rows[(251, 4)]["real_samples"] == 0


def test_collection_targets_stop_after_weighted_gate_is_reached() -> None:
    mod = _load_module()
    rows = [
        {"classification": "real_observed", "wall_weight": 0.08},
        {"classification": "real_observed", "wall_weight": 0.04},
        {"classification": "sim_only_for_phase458_reference", "wall_weight": 0.20},
    ]

    result = mod.assign_decisions(rows, current_coverage=0.86, target_coverage=0.95)

    assert [row["decision"] for row in result] == [
        "collect_anchor_target",
        "collect_anchor_target",
        "phase462_dynamics",
    ]
    assert result[1]["projected_coverage"] == 0.98


def test_attainable_coverage_excludes_sim_only_reference_cells() -> None:
    mod = _load_module()
    rows = [
        {"classification": "real_observed", "wall_weight": 0.10},
        {"classification": "sim_only_for_phase458_reference", "wall_weight": 0.25},
    ]

    assert mod.attainable_coverage(rows, current_coverage=0.60) == 0.70


def test_next_action_separates_anchor_only_from_dynamics_first() -> None:
    mod = _load_module()

    assert mod.next_action(current_coverage=0.96, attainable=0.96, target=0.95) == (
        "anchor_calibration_only"
    )
    assert mod.next_action(current_coverage=0.58, attainable=0.58, target=0.95) == (
        "phase462_dynamics_first"
    )
    assert mod.next_action(current_coverage=0.80, attainable=0.97, target=0.95) == (
        "collect_real_observed_then_anchor"
    )
