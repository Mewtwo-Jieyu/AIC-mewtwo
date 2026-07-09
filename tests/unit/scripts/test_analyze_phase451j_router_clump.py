#!/usr/bin/env python3
"""Tests for Phase451-J router semantics audit."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase451j_router_clump.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase451j", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_source_dplb_optimistic_increment_does_not_clump_64_requests() -> None:
    mod = _load_module()

    result = mod.simulate_dplb_source_routes(
        request_count=64,
        initial_counts=[[0, 1], [0, 0]],
        client_count=2,
        client_index=0,
    )

    assert result["routes"] == [32, 32]
    assert result["final_counts"] == [[64, 1], [64, 0]]


def test_decision_blocks_no_increment_runtime_change() -> None:
    mod = _load_module()

    rows = mod.build_rows(debug_summary={})
    decisions = {
        row["metric"]: row for row in rows if row["section"] == "j2_decision"
    }

    assert decisions["remove_optimistic_increment"]["status"] == "blocked"
    assert "source contradicts" in decisions["remove_optimistic_increment"]["note"]


if __name__ == "__main__":
    test_source_dplb_optimistic_increment_does_not_clump_64_requests()
    test_decision_blocks_no_increment_runtime_change()
