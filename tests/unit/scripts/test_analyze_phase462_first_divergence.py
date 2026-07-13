#!/usr/bin/env python3
"""Tests for Phase462 Step2a first-divergence evidence gate."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase462_first_divergence.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_phase462_first_divergence", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_first_counter_increase_returns_scrape_bracket_not_exact_timestamp() -> None:
    mod = _load_module()
    snapshots = [
        mod.MetricSnapshot(0.0, {"num_preemptions_total": 0.0}),
        mod.MetricSnapshot(2.0, {"num_preemptions_total": 0.0}),
        mod.MetricSnapshot(4.0, {"num_preemptions_total": 3.0}),
    ]

    bracket = mod.first_counter_increase(snapshots, "num_preemptions_total")

    assert bracket.start_s == 2.0
    assert bracket.end_s == 4.0
    assert bracket.delta == 3.0
    assert bracket.precision == "scrape_interval_only"


def test_decision_observability_gate_blocks_without_exact_block_state() -> None:
    mod = _load_module()
    available = {
        "counter_bracket",
        "running_gauge",
        "waiting_gauge",
        "kv_usage_gauge",
        "iteration_composition",
    }

    gate = mod.decision_observability_gate(available)

    assert gate["status"] == "blocked"
    assert set(gate["missing_fields"]) == {
        "decision_timestamp",
        "free_blocks_before",
        "requested_blocks",
        "trigger_request_id",
        "victim_request_id",
        "victim_queue_position",
    }


def test_throughput_increase_worsens_existing_sim_overprediction() -> None:
    mod = _load_module()

    prediction = mod.predict_preemption_fix_direction(
        real_throughput=146.6397006101265,
        sim_throughput=166.0729566780424,
        threshold=1.15,
    )

    assert prediction["direction"] == "worsen"
    assert prediction["threshold_risk"] == "current_pass_at_risk"
    assert math.isclose(
        prediction["max_sim_increase_before_threshold"],
        146.6397006101265 * 1.15 / 166.0729566780424 - 1.0,
    )


def test_throughput_increase_initially_improves_sim_underprediction() -> None:
    mod = _load_module()

    prediction = mod.predict_preemption_fix_direction(
        real_throughput=162.94851622073128,
        sim_throughput=156.48378768131056,
        threshold=1.15,
    )

    assert prediction["direction"] == "improve_until_crossing"
    assert prediction["threshold_risk"] == "current_pass_not_immediately_at_risk"


def test_step2b_scenario_uses_largest_observed_preemption_excess() -> None:
    mod = _load_module()
    panorama = [
        {"scenario": "tp8-8k", "sim_over_real": 1.68},
        {"scenario": "tp8-32k", "sim_over_real": 4.62},
        {"scenario": "dp2-32k", "sim_over_real": 2.43},
    ]

    selected = mod.select_step2b_scenario(panorama)

    assert selected["scenario"] == "tp8-32k"
    assert selected["sim_over_real"] == 4.62


def test_csv_writer_uses_lf_only(tmp_path: Path) -> None:
    mod = _load_module()
    output = tmp_path / "report.csv"
    row = {field: "value" for field in mod.CSV_FIELDS}

    mod.write_csv(output, [row])

    assert b"\r\n" not in output.read_bytes()
