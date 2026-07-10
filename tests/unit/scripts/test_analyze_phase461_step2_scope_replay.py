import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase461_step2_scope_replay.py"
SPEC = importlib.util.spec_from_file_location("analyze_phase461_step2_scope_replay", MODULE_PATH)
analyzer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(analyzer)


def test_scope_inventory_keeps_parallel_topologies_separate():
    rows = [
        {
            "topology": "tp8ep8",
            "max_num_batched_tokens": "65536",
            "phase": "mixed_prefill",
            "row_kind": "forward_total",
            "kernel_source": "tp8-run",
            "provenance": "measured",
            "bucket_tokens": "8000",
            "decode_batch": "16",
        },
        {
            "topology": "tp4dp2ep8",
            "max_num_batched_tokens": "65536",
            "phase": "mixed_prefill",
            "row_kind": "forward_total",
            "kernel_source": "dp2-run",
            "provenance": "measured",
            "bucket_tokens": "8000",
            "decode_batch": "16",
        },
    ]

    families = analyzer.inventory_scope_families(rows)

    assert {(row["topology"], row["max_bt"]) for row in families} == {
        ("tp8ep8", 65536),
        ("tp4dp2ep8", 65536),
    }
    assert all(row["rows"] == 1 for row in families)


def test_modal_replay_distinguishes_cost_match_from_phase_dynamics():
    matched = analyzer.classify_modal_replay(
        modal_sim_over_real=0.9998,
        large_step_sim_over_real=0.99,
        rank_spread=1.05,
    )
    phase_gap = analyzer.classify_modal_replay(
        modal_sim_over_real=0.5017,
        large_step_sim_over_real=0.966,
        rank_spread=8.5,
    )

    assert matched == "cost_rows_matched_no_window"
    assert phase_gap == "dp_phase_lockstep_not_scalar_row_gap"


def test_isl_band_is_rejected_when_same_scope_same_isl_already_diverges():
    decision = analyzer.decide_isl_band(
        distinct_observed_isls=1,
        same_run_modal_sim_over_real=0.5017,
        rank_spread=8.5,
    )

    assert decision == "reject_isl_band_no_causal_evidence"


def test_rank_spread_uses_per_rank_medians():
    rows = [
        {"dp_rank": "0", "forward_busy_ms": 10.0},
        {"dp_rank": "0", "forward_busy_ms": 12.0},
        {"dp_rank": "1", "forward_busy_ms": 80.0},
        {"dp_rank": "1", "forward_busy_ms": 100.0},
    ]

    summary = analyzer.summarize_rank_spread(rows)

    assert summary["rank_medians"] == {"0": 11.0, "1": 90.0}
    assert summary["max_over_min"] == 90.0 / 11.0


def test_gpu_manifest_adds_dp2_bt_only_as_diagnostic_repeat():
    manifest = analyzer.build_gpu_manifest(
        dp2_32k_modal_ratio=0.9998,
        dp2_bt_modal_ratio=0.5017,
        dp2_bt_rank_spread=8.5,
        spike_reachable=True,
    )
    by_name = {row["item"]: row for row in manifest}

    assert by_name["dp2_32k3k_mixed"]["decision"] == "skip"
    assert by_name["dp2_bt65536_mixed"]["decision"] == "collect_diagnostic_repeat"
    assert "14484/14" in by_name["dp2_bt65536_mixed"]["coverage"]
