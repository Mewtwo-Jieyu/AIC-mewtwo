#!/usr/bin/env python3
"""Tests for the Phase461 Step4 pre-ingest gates."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase461_step4_ingest_gate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase461_step4_ingest_gate", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_query_coverage_requires_a_complete_two_axis_bracket() -> None:
    mod = _load_module()
    table = {
        100: {4: 10.0, 8: 12.0},
        200: {4: 20.0, 8: 24.0},
    }

    assert mod.query_is_covered(table, bucket_tokens=150, decode_batch=6)
    assert not mod.query_is_covered(table, bucket_tokens=50, decode_batch=6)
    assert not mod.query_is_covered(table, bucket_tokens=150, decode_batch=9)


def test_coverage_summary_counts_weighted_queries_and_unique_cells() -> None:
    mod = _load_module()
    table = {
        100: {4: 10.0, 8: 12.0},
        200: {4: 20.0, 8: 24.0},
    }
    records = [
        {"bucket_tokens": 150, "decode_batch": 6},
        {"bucket_tokens": 150, "decode_batch": 6},
        {"bucket_tokens": 250, "decode_batch": 6},
    ]

    summary = mod.summarize_query_coverage(records, table)

    assert summary["query_count"] == 3
    assert summary["unique_query_cells"] == 2
    assert summary["covered_query_count"] == 2
    assert summary["uncovered_unique_cells"] == 1
    assert summary["coverage_ratio"] == 2 / 3


def test_session_comparability_requires_five_exact_cells_and_bounded_drift() -> None:
    mod = _load_module()
    reference = {(index, index): 100.0 for index in range(1, 7)}
    comparable = {(index, index): 108.0 for index in range(1, 6)}

    passed = mod.summarize_session_comparability(
        reference,
        comparable,
        min_overlap_cells=5,
        max_relative_drift=0.15,
    )
    assert passed["overlap_cells"] == 5
    assert passed["max_relative_drift"] == 0.08
    assert passed["passed"] is True

    insufficient = mod.summarize_session_comparability(
        reference,
        {(1, 1): 100.0},
        min_overlap_cells=5,
        max_relative_drift=0.15,
    )
    assert insufficient["passed"] is False

    drifted = mod.summarize_session_comparability(
        reference,
        {(index, index): 130.0 for index in range(1, 6)},
        min_overlap_cells=5,
        max_relative_drift=0.15,
    )
    assert drifted["passed"] is False


def test_ingest_gate_fails_when_either_coverage_or_comparability_fails() -> None:
    mod = _load_module()

    assert mod.ingest_gate(coverage_ratio=1.0, comparability_passed=True) == "pass"
    assert mod.ingest_gate(coverage_ratio=0.99, comparability_passed=True) == "blocked"
    assert mod.ingest_gate(coverage_ratio=1.0, comparability_passed=False) == "blocked"
