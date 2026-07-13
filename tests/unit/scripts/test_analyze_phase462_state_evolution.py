#!/usr/bin/env python3
"""Tests for Phase462 Step2a-2 state-evolution audit."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase462_state_evolution.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_phase462_state_evolution", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(iteration: int, context_reqs: int, context_tokens: int, decode: int):
    mod = _load_module()
    return mod.IterationRow(
        iteration=iteration,
        context_requests=context_reqs,
        context_tokens=context_tokens,
        generation_requests=decode,
        generation_tokens=decode,
        elapsed_ms=1.0,
    )


def test_parse_iteration_rows_extracts_scheduler_shape() -> None:
    mod = _load_module()
    lines = [
        "(EngineCore pid=1) INFO [core.py:359] Iteration(7): "
        "2 context requests, 31998 context tokens, 2 generation requests, "
        "2 generation tokens, iteration elapsed time: 2412.71 ms\n"
    ]

    rows = mod.parse_iteration_rows(lines)

    assert rows == [
        mod.IterationRow(7, 2, 31998, 2, 2, 2412.71),
    ]


def test_single_real_idle_prefill_gap_explains_following_trace() -> None:
    mod = _load_module()
    sim = [
        _row(0, 1, 32000, 0),
        _row(1, 1, 31999, 1),
        _row(2, 2, 31999, 1),
        _row(3, 2, 31998, 2),
        _row(4, 0, 0, 3),
    ]
    real = [
        _row(0, 1, 32000, 0),
        _row(1, 0, 0, 1),
        _row(2, 1, 31999, 1),
        _row(3, 2, 31999, 1),
        _row(4, 2, 31998, 2),
        _row(5, 0, 0, 3),
    ]

    result = mod.find_single_gap_alignment(real, sim)

    assert result.gap_real_iteration == 1
    assert result.compared_steps == 5
    assert result.matched_steps == 5
    assert result.match_rate == 1.0


def test_verdict_falsifies_scheduler_packing_chain() -> None:
    mod = _load_module()
    alignment = mod.GapAlignment(
        gap_real_iteration=1,
        compared_steps=16,
        matched_steps=16,
        match_rate=1.0,
    )

    verdict = mod.state_evolution_verdict(
        alignment=alignment,
        real_max_context_requests=2,
        sim_max_context_requests=2,
        real_first_trigger_tokens=32928,
        real_first_victim_tokens=32927,
        sim_first_trigger_tokens=32928,
        sim_first_victim_tokens=32928,
        vllm_long_prefill_threshold=0,
        sim_long_prefill_threshold=0,
    )

    assert verdict["prefill_packing"] == "aligned_not_root"
    assert verdict["prefill_completion"] == "one_step_arrival_gap_then_aligned"
    assert verdict["block_boundary"] == "aligned_not_root"
    assert verdict["root_cause"] == "engine_visible_arrival_gap"
    assert verdict["self_sustaining_loop"] == "falsified_as_origin"
    assert verdict["runtime_fix_allowed"] is False


def test_csv_writer_uses_lf_only(tmp_path: Path) -> None:
    mod = _load_module()
    output = tmp_path / "report.csv"
    row = {field: "value" for field in mod.CSV_FIELDS}

    mod.write_csv(output, [row])

    assert b"\r\n" not in output.read_bytes()
