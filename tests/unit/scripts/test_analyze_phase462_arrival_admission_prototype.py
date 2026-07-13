#!/usr/bin/env python3
"""Tests for the Phase462 arrival/admission offline prototype."""

from __future__ import annotations

import importlib.util
import gzip
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase462_arrival_admission_prototype.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_phase462_arrival_admission_prototype", MODULE_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_bt65536_decision_rejects_admission_fix_when_both_sides_schedule_large_cohorts() -> None:
    mod = _load_module()

    verdict = mod.admission_cadence_verdict(
        real_max_context_requests=9,
        sim_max_fresh_admissions=9,
        small_cohort_limit=2,
    )

    assert verdict["verdict"] == "budget_driven_cadence_aligned"
    assert verdict["admission_fix_allowed"] is False
    assert verdict["prototype_scope"] == "arrival_primitive_only"


def test_bt65536_decision_selects_admission_fix_only_for_real_small_sim_large() -> None:
    mod = _load_module()

    verdict = mod.admission_cadence_verdict(
        real_max_context_requests=2,
        sim_max_fresh_admissions=8,
        small_cohort_limit=2,
    )

    assert verdict["verdict"] == "admission_cadence_diverged"
    assert verdict["admission_fix_allowed"] is True
    assert verdict["prototype_scope"] == "arrival_plus_admission"


def test_measured_primitive_fit_recovers_exact_linear_surface() -> None:
    mod = _load_module()
    samples = [
        mod.PrimitiveSample(8_000, 1, 9.5, 1),
        mod.PrimitiveSample(32_000, 1, 33.5, 1),
        mod.PrimitiveSample(256_000, 32, 273.0, 32),
        mod.PrimitiveSample(1_024_000, 32, 1_041.0, 32),
    ]

    fit = mod.fit_measured_primitive(samples)

    assert abs(fit.intercept_ms - 1.0) < 1e-9
    assert abs(fit.ms_per_prompt_token - 0.001) < 1e-12
    assert abs(fit.ms_per_request - 0.5) < 1e-9
    assert fit.weighted_mape == 0.0


def test_load_primitive_samples_uses_one_row_per_tokenizer_batch(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    path = tmp_path / "arrival.jsonl.gz"
    rows = [
        {
            "kind": "tokenizer_batch_enter",
            "scenario": "8k2k",
            "batch_id": "b1",
            "batch_size": 2,
            "batch_start_ns": 1_000_000,
            "trace_ids": ["r0", "r1"],
        },
        {
            "kind": "tokenizer_batch_complete",
            "scenario": "8k2k",
            "batch_id": "b1",
            "batch_complete_ns": 11_000_000,
            "prompt_token_lengths": [8_000, 8_000],
        },
        {
            "kind": "engine_receive",
            "scenario": "8k2k",
            "trace_id": "r0",
            "ts_ns": 21_000_000,
        },
        {
            "kind": "engine_receive",
            "scenario": "8k2k",
            "trace_id": "r1",
            "ts_ns": 23_000_000,
        },
    ]
    with gzip.open(path, "wt", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row) + "\n")

    samples, boundaries = mod.load_primitive_samples([path])

    assert samples == [mod.PrimitiveSample(16_000, 2, 10.0, 2, "8k2k", "b1")]
    assert boundaries[0]["tokenizer_to_engine_ms"] == 11.0


def test_parse_real_iteration_rows_extracts_bt65536_large_context_step() -> None:
    mod = _load_module()
    lines = [
        "Iteration(1): 0 context requests, 0 context tokens, 1 generation requests, 1 generation tokens, iteration elapsed time: 65.15 ms",
        "Iteration(2): 9 context requests, 65535 context tokens, 1 generation requests, 1 generation tokens, iteration elapsed time: 4695.50 ms",
    ]

    rows = mod.parse_iteration_rows(lines)

    assert rows[1]["iteration"] == 2
    assert rows[1]["context_requests"] == 9
    assert rows[1]["context_tokens"] == 65_535


def test_preemption_signature_counts_self_and_repeat_victims() -> None:
    mod = _load_module()

    summary = mod.summarize_preemption_signature(
        [
            {"trigger_req_id": 1, "victim_req_id": 1},
            {"trigger_req_id": 2, "victim_req_id": 1},
            {"trigger_req_id": 3, "victim_req_id": 3},
        ]
    )

    assert summary == {
        "preemptions": 3,
        "unique_victims": 2,
        "repeat_victim_events": 1,
        "self_preemptions": 2,
    }


def test_prototype_gate_requires_all_three_signatures() -> None:
    mod = _load_module()

    passed = mod.prototype_gate(
        visible_step_match=1.0,
        first_16_match=1.0,
        self_preemptions=0,
        repeat_victim_events=0,
        preemptions=10,
        target_preemptions=10,
    )
    blocked = mod.prototype_gate(
        visible_step_match=1.0,
        first_16_match=1.0,
        self_preemptions=1,
        repeat_victim_events=0,
        preemptions=10,
        target_preemptions=10,
    )

    assert passed["passed"] is True
    assert passed["runtime_fix_allowed"] is False
    assert blocked["passed"] is False


def test_csv_writer_uses_lf_only(tmp_path: Path) -> None:
    mod = _load_module()
    output = tmp_path / "report.csv"
    row = {field: "value" for field in mod.CSV_FIELDS}

    mod.write_csv(output, [row])

    assert b"\r\n" not in output.read_bytes()
