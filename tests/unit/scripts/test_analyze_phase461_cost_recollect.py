#!/usr/bin/env python3
"""Tests for Phase461 Step3 GPU collection analysis."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase461_cost_recollect.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase461_cost_recollect", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _event(
    dp_rank: int,
    ctx: int,
    batch: int,
    busy: float,
    *,
    tp_rank: int | None = None,
) -> dict[str, object]:
    return {
        "schema": "phase446_graph_outer_event_v2",
        "dp_rank": dp_rank,
        "tp_rank": tp_rank,
        "ctx_tokens": ctx,
        "generation_requests": batch,
        "generation_tokens": batch,
        "num_tokens_unpadded": ctx + batch,
        "num_tokens_padded": ctx + batch,
        "cudagraph_mode": "PIECEWISE" if ctx else "FULL",
        "forward_busy_ms": busy,
    }


def test_group_tp_rows_uses_max_busy_and_rejects_mixed_keys() -> None:
    mod = _load_module()
    rows = [
        _event(0, 7990, 10, value, tp_rank=rank)
        for rank, value in enumerate((100.0, 110.0, 105.0, 109.0))
    ]

    grouped = mod.group_tp_steps(rows, tp_width=4)

    assert len(grouped) == 1
    assert grouped[0].bucket_tokens == 8000
    assert grouped[0].forward_busy_ms == 110.0

    bad = rows[:-1] + [_event(0, 7989, 11, 111.0, tp_rank=3)]
    try:
        mod.group_tp_steps(bad, tp_width=4)
    except ValueError as exc:
        assert "mixed TP chunk" in str(exc)
    else:
        raise AssertionError("mixed TP keys must fail")

    try:
        mod.group_tp_steps(rows[:-1], tp_width=4)
    except ValueError as exc:
        assert "explicit TP streams" in str(exc)
    else:
        raise AssertionError("missing explicit TP stream must fail")

    leader_only = [_event(0, 7990, 10, value) for value in (100.0, 110.0)]
    assert len(mod.group_tp_steps(leader_only, tp_width=8)) == 2

    stream_a = [_event(0, 7990, 10, 100.0), _event(0, 0, 10, 20.0)]
    stream_b = [_event(0, 7990, 10, 110.0), _event(0, 0, 10, 21.0)]
    block_flushed = mod.group_tp_steps(stream_a + stream_b, tp_width=2)
    assert [step.forward_busy_ms for step in block_flushed] == [110.0, 21.0]


def test_cross_rank_summary_keeps_real_spike_diagnostic_only() -> None:
    mod = _load_module()
    rows = []
    for value in (10.0, 11.0):
        rows.append(_event(0, 14470, 14, value))
    for value in (80.0, 90.0):
        rows.append(_event(1, 14470, 14, value))

    summary = mod.summarize_cross_rank_cells(mod.group_tp_steps(rows, tp_width=1))
    cell = summary[(14484, 14)]

    assert cell["rank_medians"] == {"0": 10.5, "1": 85.0}
    assert cell["max_over_min"] == 85.0 / 10.5
    assert mod.spike_transition_decision(cell["max_over_min"], busy_over_wall=0.999) == (
        "retain_measured_row_diagnostic_only_until_phase_model"
    )


def test_collection_gate_requires_overhead_events_and_clean_residuals(tmp_path: Path) -> None:
    mod = _load_module()
    root = tmp_path / "point"
    (root / "overhead_on" / "scenario").mkdir(parents=True)
    (root / "overhead_gate.json").write_text(
        json.dumps({"passed": True, "overhead_pct": 0.2}) + "\n",
        encoding="utf-8",
    )
    (root / "overhead_on" / "event_timing.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "overhead_on" / "process_residual_after.txt").write_text("", encoding="utf-8")
    (root / "overhead_on" / "gpu_compute_apps_after.txt").write_text("", encoding="utf-8")

    result = mod.collection_gate(root, "scenario")

    assert result["passed"] is True
    assert result["event_rows"] == 1
    (root / "overhead_on" / "gpu_compute_apps_after.txt").write_text("123\n", encoding="utf-8")
    assert mod.collection_gate(root, "scenario")["passed"] is False


def test_busy_wall_distribution_requires_exact_cell_and_matching_counts() -> None:
    mod = _load_module()
    events = [
        mod.EventStep("0", 6471, 15, 6486, "PIECEWISE", 560.0),
        mod.EventStep("1", 6471, 15, 6486, "PIECEWISE", 4800.0),
    ]
    wall = [
        SimpleNamespace(ctx_tokens=6471, generation_requests=15, elapsed_ms=561.0),
        SimpleNamespace(ctx_tokens=6471, generation_requests=15, elapsed_ms=4802.0),
    ]

    result = mod.summarize_busy_wall(events, wall, bucket_tokens=6486, decode_batch=15)

    assert result["samples"] == 2
    assert result["max_busy_over_wall"] == 4800.0 / 4802.0
