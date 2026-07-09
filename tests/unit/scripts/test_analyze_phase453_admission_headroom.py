#!/usr/bin/env python3
"""Tests for Phase453 admission-headroom analyzer."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase453_admission_headroom.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase453", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _request(rid: str, tokens: int, computed: int = 0) -> dict[str, object]:
    return {
        "request_id": rid,
        "num_tokens": tokens,
        "num_computed_tokens": computed,
        "num_prompt_tokens": 8000,
        "num_preemptions": 0,
        "status": "RUNNING",
    }


def _metrics_line(ts: str, engine0: float, engine1: float) -> str:
    body = "\n".join(
        [
            '# TYPE vllm:request_success_total counter',
            (
                'vllm:request_success_total{engine="0",finished_reason="length",'
                'model_name="kimi-k2.5"} '
                f"{engine0}"
            ),
            (
                'vllm:request_success_total{engine="1",finished_reason="length",'
                'model_name="kimi-k2.5"} '
                f"{engine1}"
            ),
        ]
    )
    return json.dumps({"ts": ts, "body": body})


def test_wait_summary_passes_when_chunk_fit_victims_wait_for_completion_wave() -> None:
    mod = _load_module()
    records = []
    for idx, wait_s in enumerate([20.0, 30.0, 40.0, 50.0, 60.0]):
        rid = f"req-{idx}"
        preempt_ns = int(idx * 1_000_000_000)
        reschedule_ns = int((idx + wait_s) * 1_000_000_000)
        records.extend(
            [
                {
                    "kind": "preempt_decision",
                    "ts_ns": preempt_ns,
                    "scheduler_timestamp": float(idx),
                    "victim": _request(rid, 8000),
                },
                {
                    "kind": "preempt_after_free",
                    "ts_ns": preempt_ns + 1,
                    "scheduler_timestamp": float(idx),
                    "free_blocks_after_free": 500,
                    "victim": _request(rid, 8000),
                },
                {
                    "kind": "victim_reschedule",
                    "ts_ns": reschedule_ns,
                    "scheduler_timestamp": float(idx + wait_s),
                    "free_blocks_after_alloc": 900,
                    "num_new_tokens": 8000,
                    "num_computed_tokens_for_schedule": 0,
                    "running_count": 45,
                    "waiting_count": 2,
                    "request": _request(rid, 8000),
                },
            ]
        )
    success = [
        mod.SuccessDelta(ts_s=idx + wait_s + 0.5, total_delta=2.0, by_engine={0: 1.0, 1: 1.0})
        for idx, wait_s in enumerate([20.0, 30.0, 40.0, 50.0, 60.0])
    ]

    waits = mod.pair_victim_waits(records, success)
    summary = mod.summarize_waits(waits, poll_interval_s=2.0)

    assert summary["verdict"] == "admission_headroom_supported"
    assert summary["chunk_fit_after_free"] == 5
    assert summary["completion_aligned"] == 5
    assert summary["recompute_from_zero"] == 5


def test_report_blocks_when_reschedule_is_immediate() -> None:
    mod = _load_module()
    records = [
        {
            "kind": "preempt_decision",
            "ts_ns": 0,
            "scheduler_timestamp": 0.0,
            "victim": _request("req", 8000),
        },
        {
            "kind": "preempt_after_free",
            "ts_ns": 1,
            "scheduler_timestamp": 0.0,
            "free_blocks_after_free": 500,
            "victim": _request("req", 8000),
        },
        {
            "kind": "victim_reschedule",
            "ts_ns": 500_000_000,
            "scheduler_timestamp": 0.5,
            "free_blocks_after_alloc": 500,
            "num_new_tokens": 8000,
            "num_computed_tokens_for_schedule": 0,
            "running_count": 56,
            "waiting_count": 0,
            "request": _request("req", 8000),
        },
    ]
    success = [mod.SuccessDelta(ts_s=0.5, total_delta=1.0, by_engine={0: 1.0})]

    waits = mod.pair_victim_waits(records, success)
    summary = mod.summarize_waits(waits, poll_interval_s=2.0)

    assert summary["verdict"] == "admission_headroom_not_proven"
    assert summary["immediate_reentry"] == 1


def test_metric_success_parser_reports_poll_interval(tmp_path: Path) -> None:
    mod = _load_module()
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        "\n".join(
            [
                _metrics_line("2026-07-09T00:00:00+00:00", 0, 0),
                _metrics_line("2026-07-09T00:00:02+00:00", 1, 0),
                _metrics_line("2026-07-09T00:00:04+00:00", 1, 3),
            ]
        ),
        encoding="utf-8",
    )

    deltas, poll = mod.parse_success_deltas(metrics)

    assert poll == 2.0
    assert [delta.total_delta for delta in deltas] == [1.0, 3.0]


def test_post_fix_summary_parser_keeps_partial_gate(tmp_path: Path) -> None:
    mod = _load_module()
    csv_path = tmp_path / "phase451d.csv"
    csv_path.write_text(
        "\n".join(
            [
                "section,scenario,side,metric,value,target,status,note",
                (
                    "sim_forensics,K2.5,sim,preemption_events,192,,,"
                    "throughput_tok_s_gpu=155.535661"
                ),
                "sim_category,K2.5,sim,thrash_repeat_victim,120,,,",
                (
                    "decision,K2.5,sim,phase451e_runtime_fix_gate,"
                    "blocked_pending_unique_semantic_fix,"
                    "declared metrics pass,blocked,"
                ),
            ]
        ),
        encoding="utf-8",
    )

    summary = mod.read_post_fix_summary(csv_path)

    assert summary["preemption_events"] == "192"
    assert summary["throughput_tok_s_gpu"] == "155.535661"
    assert summary["runtime_fix_status"] == "blocked"


def test_validate_summary_parser_reports_max_and_dp2_target(tmp_path: Path) -> None:
    mod = _load_module()
    csv_path = tmp_path / "validate.csv"
    csv_path.write_text(
        "\n".join(
            [
                "name,tp,dp,ep,max_bt,real_output_tok_s_gpu,sim_output_tok_s_gpu,error_ratio",
                "K2.5-tp4ep8dp2-8k2k,4,2,8,8000,137.7,156.5,1.136",
                "K2.5-tp8ep8-8k2k,8,1,8,8000,133.5,192.1,1.439",
            ]
        ),
        encoding="utf-8",
    )

    summary = mod.read_validate_summary(csv_path)

    assert summary["max_name"] == "K2.5-tp8ep8-8k2k"
    assert summary["max_error_ratio"] == 1.439
    assert summary["by_name"]["K2.5-tp4ep8dp2-8k2k"]["error_ratio"] == 1.136


if __name__ == "__main__":
    test_wait_summary_passes_when_chunk_fit_victims_wait_for_completion_wave()
    test_report_blocks_when_reschedule_is_immediate()
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_metric_success_parser_reports_poll_interval(Path(tmp))
        test_post_fix_summary_parser_keeps_partial_gate(Path(tmp))
        test_validate_summary_parser_reports_max_and_dp2_target(Path(tmp))
