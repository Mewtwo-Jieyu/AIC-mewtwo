#!/usr/bin/env python3
"""Tests for Phase451-K router harness audit."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase451k_router_harness.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase451k", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_harness_separates_route_loop_from_stats_overwrite() -> None:
    mod = _load_module()

    persistent = mod.run_route_harness(
        request_count=64,
        initial_counts=[[0, 1], [0, 0]],
        client_count=2,
        client_index=0,
        overwrite_mode="none",
    )
    overwritten = mod.run_route_harness(
        request_count=64,
        initial_counts=[[0, 1], [0, 0]],
        client_count=2,
        client_index=0,
        overwrite_mode="before_each_request",
    )

    assert persistent["routes"] == [32, 32]
    assert persistent["classification"] == "alternating"
    assert overwritten["routes"] == [0, 64]
    assert overwritten["classification"] == "clustered"


def test_two_api_servers_still_balance_when_local_increments_persist() -> None:
    mod = _load_module()

    result = mod.run_two_api_server_harness(
        request_count_per_client=64,
        initial_counts=[[0, 0], [0, 0]],
        client_count=2,
    )

    assert result["routes"] == [64, 64]
    assert result["classification"] == "alternating"


def test_decision_points_to_stats_overwrite_not_no_increment() -> None:
    mod = _load_module()

    rows = mod.build_rows(
        debug_summary={
            "first_asymmetric_burst_counts": "[[0,1],[63,1]]",
            "first_symmetric_burst_counts": "[[62,2],[62,2]]",
        }
    )
    decisions = {row["metric"]: row for row in rows if row["section"] == "k5_decision"}

    assert decisions["sim_route_alignment"]["status"] == "blocked"
    assert "stats-overwrite interleaving" in decisions["sim_route_alignment"]["note"]


if __name__ == "__main__":
    test_harness_separates_route_loop_from_stats_overwrite()
    test_two_api_servers_still_balance_when_local_increments_persist()
    test_decision_points_to_stats_overwrite_not_no_increment()
