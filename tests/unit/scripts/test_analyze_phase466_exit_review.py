from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase466_exit_review.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_phase466_exit_review", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _real_row(rank: int) -> dict[str, object]:
    return {
        "run_id": "real-test",
        "source": "real",
        "rank_id": rank,
        "rank_scope": "dp_rank",
        "workload_cohort_digest": "a" * 64,
        "progress_window_id": 0,
        "iteration_elapsed_ms": 10.0 + rank,
        "scheduled_prefill_tokens": 64000,
        "scheduled_decode_tokens": 4,
        "prefill_request_count": 2,
        "decode_request_count": 4,
        "running_count": 6,
        "waiting_count": 122,
        "completed_request_count": 0,
    }


def _sim_row() -> dict[str, object]:
    return {
        "run_id": "sim-test",
        "source": "sim",
        "rank_id": 0,
        "rank_scope": "global_simulator",
        "workload_cohort_digest": "a" * 64,
        "progress_window_id": 0,
        "iteration_elapsed_ms": 8.0,
        "scheduled_prefill_tokens": 64000,
        "scheduled_decode_tokens": 4,
        "prefill_request_count": 2,
        "decode_request_count": 4,
        "running_count": 6,
        "waiting_count": 122,
        "completed_request_count": 0,
        "sim_predicted_iteration_ms": 8.0,
        "sim_component_cost_ms": 8.0,
    }


def test_evidence_coverage_keeps_stock_probe_missing_fields_explicit() -> None:
    analysis = _load_module()

    coverage = analysis.evaluate_evidence_coverage(
        [_real_row(0), _real_row(1)], [_sim_row()]
    )

    assert coverage["joined_progress_windows"] == 1
    assert coverage["real_rank_ids"] == [0, 1]
    assert coverage["candidate_coverage"]["dp_rank_synchronization_asymmetry"]["status"] == "EVALUABLE"
    schedule = coverage["candidate_coverage"]["schedule_merged_batch_composition"]
    assert schedule["status"] == "INCONCLUSIVE_MISSING_FIELDS"
    assert "prefill_chunk_token_histogram" in schedule["missing_fields"]
    cost = coverage["candidate_coverage"]["iteration_cost_serving_state_coverage"]
    assert "sim_serving_state_key" in cost["missing_fields"]


def test_route_selection_requires_exactly_one_human_reviewed_pass() -> None:
    analysis = _load_module()
    candidates = list(analysis.CANDIDATES)

    none = analysis.select_route({candidate: "INCONCLUSIVE" for candidate in candidates})
    assert none == {"status": "INCONCLUSIVE", "selected_route": None, "pass_count": 0}

    one = {candidate: "DISPROVED" for candidate in candidates}
    one["dp_rank_synchronization_asymmetry"] = "PASS"
    assert analysis.select_route(one) == {
        "status": "SELECTED",
        "selected_route": "dp_rank_synchronization_asymmetry",
        "pass_count": 1,
    }

    two = dict(one)
    two["iteration_cost_serving_state_coverage"] = "PASS"
    assert analysis.select_route(two) == {
        "status": "INCONCLUSIVE",
        "selected_route": None,
        "pass_count": 2,
    }


def test_evidence_coverage_rejects_wrong_scope_or_unjoinable_digest() -> None:
    analysis = _load_module()
    wrong_scope = _real_row(0)
    wrong_scope["rank_scope"] = "global_simulator"
    with pytest.raises(ValueError, match="real_rank_scope_mismatch"):
        analysis.evaluate_evidence_coverage([wrong_scope], [_sim_row()])

    sim = _sim_row()
    sim["workload_cohort_digest"] = "b" * 64
    with pytest.raises(ValueError, match="workload_cohort_digest_mismatch"):
        analysis.evaluate_evidence_coverage([_real_row(0)], [sim])


def test_candidate_fields_must_exist_on_the_required_source() -> None:
    analysis = _load_module()
    real = _real_row(0)
    real["completed_request_count"] = ""
    sim = _sim_row()
    sim.update(
        {
            "prefill_chunk_token_histogram": "64000",
            "fresh_prefill_tokens": 64000,
            "recompute_prefill_tokens": 0,
            "resume_prefill_tokens": 0,
            "decode_kv_token_sum": 4,
            "cudagraph_mode": "full",
        }
    )

    coverage = analysis.evaluate_evidence_coverage([real], [sim])

    schedule = coverage["candidate_coverage"]["schedule_merged_batch_composition"]
    assert "prefill_chunk_token_histogram" in schedule["missing_real_fields"]
    dp = coverage["candidate_coverage"]["dp_rank_synchronization_asymmetry"]
    assert "completed_request_count" in dp["missing_real_fields"]
    assert dp["status"] == "INCONCLUSIVE_MISSING_FIELDS"
