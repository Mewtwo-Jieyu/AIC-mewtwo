#!/usr/bin/env python3
"""Tests for Phase462 arrival-observation self-checks."""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts/analyze_phase462_arrival_observation.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase462_arrival_analysis", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rows(scenario: str, prompt_tokens: int) -> list[dict]:
    trace_ids = [f"{scenario}-{idx}" for idx in range(2)]
    return [
        {
            "kind": "runtime_config",
            "scenario": scenario,
            "max_batch_size": 32,
            "batch_wait_timeout_s": 0.002,
        },
        {
            "kind": "tokenizer_batch_enter",
            "scenario": scenario,
            "batch_id": "batch-1",
            "trace_ids": trace_ids,
            "expected_prompt_tokens": [prompt_tokens, prompt_tokens],
            "batch_size": 2,
        },
        {
            "kind": "tokenizer_batch_complete",
            "scenario": scenario,
            "batch_id": "batch-1",
            "trace_ids": trace_ids,
            "prompt_token_lengths": [prompt_tokens, prompt_tokens],
        },
        *[
            {"kind": "engine_receive", "scenario": scenario, "trace_id": trace_id}
            for trace_id in trace_ids
        ],
        {
            "kind": "scheduler_step",
            "scenario": scenario,
            "step": 1,
            "waiting_before": 2,
            "waiting_after": 0,
            "new_context_count": 2,
            "new_context_trace_ids": trace_ids,
        },
    ]


def test_summarize_requires_four_point_pairing_and_runtime_config() -> None:
    mod = _load_module()
    rows = _rows("8k", 8000) + _rows("32k", 32000)

    result = mod.summarize(rows, required_scenarios={"8k", "32k"})

    assert result["passed"] is True
    assert result["paired_trace_ids"] == 4
    assert result["runtime_config_matches"] is True
    assert result["scenarios"] == ["32k", "8k"]


def test_load_rows_supports_gzip(tmp_path: Path) -> None:
    mod = _load_module()
    path = tmp_path / "events.jsonl.gz"
    expected = [{"kind": "runtime_config", "scenario": "8k"}]
    with gzip.open(path, "wt", encoding="utf-8") as output:
        output.write(json.dumps(expected[0]) + "\n")

    assert mod.load_rows([path]) == expected


def test_summarize_fails_when_first_schedule_or_prompt_length_is_missing() -> None:
    mod = _load_module()
    rows = _rows("8k", 8000)
    rows[-1]["new_context_trace_ids"] = []
    rows[2]["prompt_token_lengths"] = [7999, 8000]

    result = mod.summarize(rows, required_scenarios={"8k"})

    assert result["passed"] is False
    assert result["paired_trace_ids"] == 0
    assert result["prompt_length_mismatches"] == 1


def test_summarize_rejects_duplicate_boundary_events() -> None:
    mod = _load_module()
    rows = _rows("8k", 8000)
    rows.append(dict(rows[3]))

    result = mod.summarize(rows, required_scenarios={"8k"})

    assert result["passed"] is False
    assert result["duplicate_engine_receive_trace_ids"] == 1


def test_measure_scenario_reports_four_boundary_latencies_and_admission_spread() -> None:
    mod = _load_module()
    rows = _rows("8k", 8000)
    rows[1].update(
        {
            "batch_start_ns": 10_000_000,
            "queue_enter_ns": [8_000_000, 9_000_000],
        }
    )
    rows[2].update(
        {
            "batch_start_ns": 10_000_000,
            "batch_complete_ns": 14_000_000,
        }
    )
    rows[3]["ts_ns"] = 15_000_000
    rows[4]["ts_ns"] = 16_000_000
    rows[-1].update(
        {
            "ts_ns": 17_000_000,
            "step": 10,
            "new_context_count": 1,
            "new_context_trace_ids": ["8k-0"],
        }
    )
    rows.append(
        {
            "kind": "scheduler_step",
            "scenario": "8k",
            "ts_ns": 20_000_000,
            "step": 11,
            "waiting_before": 1,
            "waiting_after": 0,
            "new_context_count": 1,
            "new_context_trace_ids": ["8k-1"],
        }
    )

    result = mod.measure_scenario(rows, "8k")

    assert result["paired_requests"] == 2
    assert result["batch_size_median"] == 2
    assert result["tokenizer_service_ms_median"] == 4.0
    assert result["queue_wait_ms_median"] == 1.5
    assert result["queue_wait_ms_p90"] == 2.0
    assert result["engine_receive_ms_median"] == 1.5
    assert result["engine_receive_ms_p90"] == 2.0
    assert result["first_schedule_ms_median"] == 3.0
    assert result["first_schedule_ms_p90"] == 4.0
    assert result["total_visible_ms_median"] == 10.0
    assert result["total_visible_ms_p90"] == 11.0
    assert result["scheduled_steps_per_batch_median"] == 2
    assert result["new_context_count_max"] == 1

    groups = mod.summarize_batch_groups(rows, "8k")

    assert groups == [
        {
            "scenario": "8k",
            "batch_size": 2,
            "batch_count": 1,
            "total_prompt_tokens": 16000,
            "tokenizer_service_ms_median": 4.0,
            "tokenizer_service_ms_p90": 4.0,
            "scheduled_steps_median": 2,
            "scheduled_steps_max": 2,
        }
    ]
