from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[3] / "scripts" / "analyze_phase446_b2b_ingest.py"
    spec = importlib.util.spec_from_file_location("analyze_phase446_b2b_ingest", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_phase446_marks_peer_phase_and_excludes_peer_prefill_decode_rows():
    phase446 = _load_module()
    steps = [
        phase446.EngineStep("0", 0, 8000, 0, 654.0),
        phase446.EngineStep("1", 0, 0, 1, 680.0),
        phase446.EngineStep("0", 1, 7999, 1, 1131.0),
        phase446.EngineStep("1", 1, 0, 1, 1072.0),
        phase446.EngineStep("0", 2, 0, 64, 41.0, num_tokens_unpadded=64, cudagraph_mode="FULL"),
        phase446.EngineStep("1", 2, 0, 63, 40.0, num_tokens_unpadded=63, cudagraph_mode="FULL"),
    ]

    annotated = phase446.annotate_peer_phase(steps)
    rows = phase446.build_forward_total_rows(annotated, scenario="unit")

    assert annotated[1].peer_phase == "peer_prefill"
    assert annotated[3].peer_phase == "peer_prefill"
    assert annotated[4].peer_phase == "peer_decode"
    assert {(row["phase"], row["bucket_tokens"], row["decode_batch"]) for row in rows} == {
        ("mixed_prefill", 8000, 1),
        ("decode", 64, 64),
        ("decode", 63, 63),
    }
    assert all(row["row_kind"] == "forward_total" for row in rows)
    assert all(row["category"] == "forward_total" for row in rows)


def test_phase446_uses_unpadded_tokens_for_chunked_prefill_steps():
    phase446 = _load_module()
    steps = [
        phase446.EngineStep(
            "0",
            0,
            ctx_tokens=0,
            generation_requests=44,
            forward_busy_ms=1100.0,
            generation_tokens=8000,
            num_tokens_unpadded=8000,
            num_tokens_padded=8000,
            cudagraph_mode="NONE",
        ),
        phase446.EngineStep(
            "1",
            0,
            ctx_tokens=0,
            generation_requests=44,
            forward_busy_ms=38.0,
            generation_tokens=44,
            num_tokens_unpadded=44,
            num_tokens_padded=48,
            cudagraph_mode="FULL",
        ),
    ]

    rows = phase446.build_forward_total_rows(phase446.annotate_peer_phase(steps), scenario="unit")

    assert {(row["phase"], row["bucket_tokens"], row["decode_batch"]) for row in rows} == {
        ("mixed_prefill", 8000, 44),
    }


def test_phase446_decode_rows_require_full_cudagraph_mode():
    phase446 = _load_module()
    steps = [
        phase446.EngineStep(
            "0",
            0,
            ctx_tokens=0,
            generation_requests=44,
            forward_busy_ms=674.0,
            generation_tokens=44,
            num_tokens_unpadded=44,
            num_tokens_padded=48,
            cudagraph_mode="NONE",
        ),
        phase446.EngineStep(
            "1",
            0,
            ctx_tokens=0,
            generation_requests=44,
            forward_busy_ms=675.0,
            generation_tokens=44,
            num_tokens_unpadded=44,
            num_tokens_padded=48,
            cudagraph_mode="NONE",
        ),
        phase446.EngineStep(
            "0",
            1,
            ctx_tokens=0,
            generation_requests=44,
            forward_busy_ms=38.0,
            generation_tokens=44,
            num_tokens_unpadded=44,
            num_tokens_padded=48,
            cudagraph_mode="FULL",
        ),
        phase446.EngineStep(
            "1",
            1,
            ctx_tokens=0,
            generation_requests=44,
            forward_busy_ms=39.0,
            generation_tokens=44,
            num_tokens_unpadded=44,
            num_tokens_padded=48,
            cudagraph_mode="FULL",
        ),
    ]

    rows = phase446.build_forward_total_rows(phase446.annotate_peer_phase(steps), scenario="unit")

    assert len(rows) == 1
    assert rows[0]["phase"] == "decode"
    assert rows[0]["bucket_tokens"] == 44
    assert rows[0]["latency_ms"] == 38.5


def test_phase446_perfdb_export_uses_32k_only_for_mixed_rows():
    phase446 = _load_module()
    rows = [
        {
            "scenario": "K2.5-tp4ep8dp2-8k2k",
            "phase": "decode",
            "row_kind": "forward_total",
            "category": "forward_total",
            "bucket_tokens": 8,
            "decode_batch": 8,
        },
        {
            "scenario": "K2.5-tp4ep8dp2-32k3k",
            "phase": "decode",
            "row_kind": "forward_total",
            "category": "forward_total",
            "bucket_tokens": 8,
            "decode_batch": 8,
        },
        {
            "scenario": "K2.5-tp4ep8dp2-32k3k",
            "phase": "mixed_prefill",
            "row_kind": "forward_total",
            "category": "forward_total",
            "bucket_tokens": 32000,
            "decode_batch": 8,
        },
    ]

    selected = phase446.select_perfdb_rows(rows)

    assert [(row["scenario"], row["phase"], row["bucket_tokens"]) for row in selected] == [
        ("K2.5-tp4ep8dp2-8k2k", "decode", 8),
        ("K2.5-tp4ep8dp2-32k3k", "mixed_prefill", 32000),
    ]


def test_phase446_groups_tp_rows_by_max_busy_per_engine_step():
    phase446 = _load_module()
    records = [
        phase446.EventRecord("0", 8000, 0, 100.0),
        phase446.EventRecord("0", 8000, 0, 110.0),
        phase446.EventRecord("1", 0, 1, 20.0),
        phase446.EventRecord("1", 0, 1, 30.0),
    ]

    grouped = phase446.group_event_steps(records, tp_width=2)

    assert grouped == [
        phase446.EngineStep("0", 0, 8000, 0, 110.0),
        phase446.EngineStep("1", 0, 0, 1, 30.0),
    ]


if __name__ == "__main__":
    test_phase446_marks_peer_phase_and_excludes_peer_prefill_decode_rows()
    test_phase446_uses_unpadded_tokens_for_chunked_prefill_steps()
    test_phase446_decode_rows_require_full_cudagraph_mode()
    test_phase446_perfdb_export_uses_32k_only_for_mixed_rows()
    test_phase446_groups_tp_rows_by_max_busy_per_engine_step()
