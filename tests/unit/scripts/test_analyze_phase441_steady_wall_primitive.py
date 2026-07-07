import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[3] / "scripts" / "analyze_phase441_steady_wall_primitive.py"
    spec = importlib.util.spec_from_file_location("phase441_steady_wall_primitive", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_phase441_decomposes_candidate_gap_without_counting_residual_as_explained():
    phase441 = _load_module()
    row = phase441.AuditPoint(
        bucket_tokens=8000,
        decode_batch=1,
        step_count=2,
        category_sum_ms=100.0,
        non_attn_total_ms=150.0,
        attention_ms=10.0,
        peer_wait_ms=20.0,
        host_gap_upper_ms=15.0,
        attention_error_upper_ms=5.0,
    )

    assert row.candidate_gap_ms == 50.0
    assert row.explained_ms == 40.0
    assert row.unexplained_ms == 10.0
    assert row.explained_share == 0.8


def test_phase441_life_gate_requires_explanation_and_high_batch_convergence():
    phase441 = _load_module()
    rows = [
        phase441.AuditPoint(
            bucket_tokens=8000,
            decode_batch=1,
            step_count=2,
            category_sum_ms=100.0,
            non_attn_total_ms=150.0,
            attention_ms=10.0,
            peer_wait_ms=20.0,
            host_gap_upper_ms=15.0,
            attention_error_upper_ms=5.0,
        ),
        phase441.AuditPoint(
            bucket_tokens=8000,
            decode_batch=45,
            step_count=3,
            category_sum_ms=100.0,
            non_attn_total_ms=112.0,
            attention_ms=10.0,
            peer_wait_ms=6.0,
            host_gap_upper_ms=4.0,
            attention_error_upper_ms=2.0,
        ),
    ]

    summary = phase441.summarize_life_gate(rows)

    assert summary["life_gate"] == "passed"
    assert round(summary["high_batch_max_error_pct"], 6) == 12.0


def test_phase441_life_gate_fails_when_high_batch_does_not_converge():
    phase441 = _load_module()
    rows = [
        phase441.AuditPoint(
            bucket_tokens=8000,
            decode_batch=1,
            step_count=2,
            category_sum_ms=100.0,
            non_attn_total_ms=150.0,
            attention_ms=10.0,
            peer_wait_ms=20.0,
            host_gap_upper_ms=15.0,
            attention_error_upper_ms=5.0,
        ),
        phase441.AuditPoint(
            bucket_tokens=8000,
            decode_batch=55,
            step_count=3,
            category_sum_ms=100.0,
            non_attn_total_ms=120.0,
            attention_ms=10.0,
            peer_wait_ms=10.0,
            host_gap_upper_ms=5.0,
            attention_error_upper_ms=5.0,
        ),
    ]

    summary = phase441.summarize_life_gate(rows)

    assert summary["life_gate"] == "failed"
    assert summary["high_batch_gate"] == "failed"
