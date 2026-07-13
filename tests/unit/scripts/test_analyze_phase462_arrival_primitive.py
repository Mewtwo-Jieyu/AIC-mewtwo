#!/usr/bin/env python3
"""Tests for Phase462 Step2a-3 arrival-primitive audit."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase462_arrival_primitive.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_phase462_arrival_primitive", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_source_spec_identifies_microbatch_queue_not_serial_requests() -> None:
    mod = _load_module()

    spec = mod.RuntimeSourceSpec(
        max_batch_size=32,
        batch_wait_timeout_ms=2.0,
        executor_workers=1,
        engine_drains_before_step=True,
    )

    assert spec.structure == "microbatch_queue"
    assert spec.per_request_serial_formula_allowed is False


def test_observation_scan_requires_both_pipeline_boundaries() -> None:
    mod = _load_module()
    lines = [
        "INFO request_pipeline_start request_id=req-1 ts_ns=100\n",
        "INFO tokenizer_batch_start batch_id=batch-1\n",
        "INFO Added request req-1.\n",
        "INFO engine_core_receive request_id=req-1 ts_ns=150\n",
        "INFO request_pipeline_start request_id=req-2 ts_ns=200\n",
    ]

    inventory = mod.scan_observation_lines(lines)

    assert inventory.pipeline_start_rows == 2
    assert inventory.tokenizer_batch_rows == 1
    assert inventory.post_add_rows == 1
    assert inventory.engine_receive_rows == 1
    assert inventory.complete_request_pairs == 1
    assert inventory.measurement_ready is False


def test_missing_visibility_measurements_blocks_prototype_and_runtime() -> None:
    mod = _load_module()
    source = mod.RuntimeSourceSpec(32, 2.0, 1, True)
    observations = mod.ObservationInventory(
        pipeline_start_rows=0,
        tokenizer_batch_rows=0,
        post_add_rows=0,
        engine_receive_rows=0,
        complete_request_pairs=0,
    )

    verdict = mod.arrival_primitive_verdict(source, observations)

    assert verdict["structure_gate"] == "pass"
    assert verdict["measurement_gate"] == "fail"
    assert verdict["prototype_gate"] == "blocked"
    assert verdict["runtime_fix_allowed"] is False
    assert verdict["visible_step_signature"] == "not_run"
    assert verdict["first_16_steps_signature"] == "not_run"
    assert verdict["preemption_signature"] == "not_run"


def test_complete_boundary_measurement_unlocks_only_offline_prototype() -> None:
    mod = _load_module()
    source = mod.RuntimeSourceSpec(32, 2.0, 1, True)
    observations = mod.scan_observation_lines(
        [
            "request_pipeline_start request_id=req-1 ts_ns=100\n",
            "tokenizer_batch_start batch_id=batch-1\n",
            "Added request req-1.\n",
            "engine_core_receive request_id=req-1 ts_ns=150\n",
        ]
    )

    verdict = mod.arrival_primitive_verdict(source, observations)

    assert verdict["measurement_gate"] == "pass"
    assert verdict["prototype_gate"] == "pending"
    assert verdict["runtime_fix_allowed"] is False


def test_csv_writer_uses_lf_only(tmp_path: Path) -> None:
    mod = _load_module()
    output = tmp_path / "report.csv"
    row = {field: "value" for field in mod.CSV_FIELDS}

    mod.write_csv(output, [row])

    assert b"\r\n" not in output.read_bytes()
