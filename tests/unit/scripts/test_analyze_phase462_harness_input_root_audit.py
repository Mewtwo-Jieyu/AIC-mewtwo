#!/usr/bin/env python3
"""Tests for Phase462 DP Step 3d harness input root audit."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase462_harness_input_root_audit.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase462_step3d", MODULE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_initial_wave_shape_uses_actual_isl_and_partial_chunk() -> None:
    mod = _load_module()

    shape = mod.initial_wave_shape(max_num_batched_tokens=65536, isl=8000)

    assert shape.full_requests == 8
    assert shape.partial_tokens == 1536
    assert shape.context_requests == 9


def test_dp_gap_decomposition_closes_official_to_o0() -> None:
    mod = _load_module()

    result = mod.decompose_dp_gap(
        official_error=1.2000380491947868,
        short_single_error=1.1282020766530387,
        o0_error=1.0895477855343982,
        matched_multi_error=1.2244742837153453,
    )

    assert abs(sum(row.delta_error for row in result.rows) - result.raw_gap) < 1e-12
    assert result.rows[0].segment == "per_replica_observation_horizon"
    assert result.rows[1].segment == "multi_replica_execution_and_assembly"
    assert result.verdict == "observation_horizon_and_topology_not_arrival"


def test_input_contract_rejects_engine_visibility_claim_without_timestamp() -> None:
    mod = _load_module()

    verdict = mod.assess_input_contract(
        explicit_start_timestamp=False,
        reconstructed_initial_zero_count=128,
        expected_concurrency=128,
        real_first_context_requests=1,
        controlled_error=1.2518,
        acceptance_limit=1.15,
    )

    assert verdict == "engine_visibility_boundary_not_observed_keep_official_protocol"


def test_first_window_exact_fraction_is_positional() -> None:
    mod = _load_module()
    real = [(1, 8000, 0), (0, 0, 1), (9, 65535, 1)]

    assert mod.first_window_exact_fraction(real, real) == 1.0
    assert mod.first_window_exact_fraction(real, real[1:]) == 0.0
