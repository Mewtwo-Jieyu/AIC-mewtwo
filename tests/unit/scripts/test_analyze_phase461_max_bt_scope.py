import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase461_max_bt_scope.py"
SPEC = importlib.util.spec_from_file_location("analyze_phase461_max_bt_scope", MODULE_PATH)
analyzer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(analyzer)


def test_phase458_rows_become_step1_baseline():
    rows = [
        {
            "name": "scenario-a",
            "current_real_output_tok_s_gpu": "10.0",
            "current_sim_output_tok_s_gpu": "12.0",
            "current_error_ratio": "1.2",
        }
    ]

    baseline = analyzer.phase458_current_baseline(rows)

    assert baseline == {
        "scenario-a": {
            "name": "scenario-a",
            "real_output_tok_s_gpu": 10.0,
            "sim_output_tok_s_gpu": 12.0,
            "error_ratio": 1.2,
        }
    }


def test_spike_reachability_counts_exact_and_interpolated_hits():
    spike = analyzer.SpikeCell(
        scenario="scenario-a",
        max_bt=32_000,
        bucket_tokens=63,
        decode_batch=9,
    )
    table = {
        61: {8: 1.0, 10: 1.0},
        63: {8: 1.0, 9: 100.0, 10: 1.0},
        64: {8: 1.0, 10: 1.0},
    }
    records = [
        {
            "phase": "mixed_prefill",
            "row_kind": "forward_total",
            "category": "forward_total",
            "max_num_batched_tokens": 32_000,
            "bucket_tokens": 63,
            "decode_batch": 9,
            "hit": True,
        },
        {
            "phase": "mixed_prefill",
            "row_kind": "forward_total",
            "category": "forward_total",
            "max_num_batched_tokens": 32_000,
            "bucket_tokens": 62,
            "decode_batch": 9,
            "hit": True,
        },
        {
            "phase": "mixed_prefill",
            "row_kind": "forward_total",
            "category": "forward_total",
            "max_num_batched_tokens": 65_536,
            "bucket_tokens": 63,
            "decode_batch": 9,
            "hit": True,
        },
    ]

    summary = analyzer.summarize_spike_reachability(records, table, spike)

    assert summary["scope_query_count"] == 2
    assert summary["exact_query_count"] == 1
    assert summary["influence_hit_count"] == 2
    assert summary["reachable"] is True


def test_spike_outside_query_brackets_is_not_reachable():
    spike = analyzer.SpikeCell(
        scenario="scenario-a",
        max_bt=65_536,
        bucket_tokens=14_484,
        decode_batch=14,
    )
    table = {
        9_405: {13: 1.0},
        14_484: {14: 100.0},
        16_078: {15: 1.0},
    }
    records = [
        {
            "phase": "mixed_prefill",
            "row_kind": "forward_total",
            "category": "forward_total",
            "max_num_batched_tokens": 65_536,
            "bucket_tokens": 9_400,
            "decode_batch": 13,
            "hit": False,
        }
    ]

    summary = analyzer.summarize_spike_reachability(records, table, spike)

    assert summary["exact_query_count"] == 0
    assert summary["influence_hit_count"] == 0
    assert summary["reachable"] is False


def test_max_bt_scope_does_not_prove_generic_isl_safety():
    decision = analyzer.evaluate_tp8_candidate_scope(
        candidate_max_bt=8_000,
        candidate_isl=8_000,
        validation_regimes={
            "tp8-8k": (8_000, 8_000),
            "tp8-32k": (32_000, 32_000),
            "tp8-bt65k": (65_536, 8_000),
        },
    )

    assert decision["six_point_collision_count"] == 0
    assert decision["generic_isl_safe"] is False
    assert decision["status"] == "retain_pending_isl_replay"
