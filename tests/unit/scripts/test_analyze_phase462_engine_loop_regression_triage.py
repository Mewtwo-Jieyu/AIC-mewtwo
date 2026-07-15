from __future__ import annotations

from types import SimpleNamespace

import scripts.analyze_phase462_engine_loop_regression_triage as analysis


def test_immediate_arrival_exposes_submissions_without_timing_state() -> None:
    layer = analysis.ImmediateArrivalLayer(SimpleNamespace())

    layer.submit_many([((3, 1.5), 8000), ((4, 1.5), 8000)], now_ms=1.5)

    assert layer.drain(1.5) == [(3, 1.5), (4, 1.5)]
    assert layer.drain(1.5) == []
    assert layer.next_event_ms() is None


def test_constant_provenance_has_one_locked_class_per_constant() -> None:
    rows = analysis.constant_provenance_rows()
    names = [row.constant for row in rows]

    assert len(names) == len(set(names))
    assert {row.classification for row in rows} == {
        analysis.VERSION_PROFILE,
        analysis.PHYSICAL_SCALING,
        analysis.SCOPED_MEASUREMENT,
        analysis.REMOVE_FROM_DEFAULT,
    }
    assert all(row.classification in analysis.CLASSIFICATIONS for row in rows)
    assert {
        "null_block_reserve",
        "batch_queue_depth",
        "sampled_computed_placeholder_lifecycle",
        "tokenizer_ms_per_prompt_token",
        "tokenizer_intercept_ms",
        "tokenizer_max_batch_size",
        "initial_workload_exposure",
    }.issubset(names)


def test_split_requires_core_to_explain_at_least_half_of_full_improvement() -> None:
    retained = analysis.assess_core_split(
        baseline_error=1.104,
        core_error=1.050,
        full_error=1.006,
    )
    arrival_driven = analysis.assess_core_split(
        baseline_error=1.104,
        core_error=1.090,
        full_error=1.006,
    )

    assert retained.split_supported is True
    assert retained.core_share_of_full_improvement > 0.5
    assert arrival_driven.split_supported is False
    assert arrival_driven.core_share_of_full_improvement < 0.5


def test_step2c9_parity_fails_when_baseline_predates_null_block() -> None:
    audit = analysis.assess_step2c9_parity(
        baseline_has_null_block=False,
        candidate_has_null_block=True,
        tp_engine_loop_enabled=True,
        dp_engine_loop_enabled=False,
    )

    assert audit.status == "fail_confounded"
    assert audit.only_engine_loop_difference is False
    assert audit.confounds == ("null_block_semantics",)


def test_dynamic_signature_counts_mixed_steps_and_preemption_shape() -> None:
    signature = analysis.summarize_dynamic_signature(
        trace=[
            {"prefill_reqs": 1, "prefill_tokens": 8000, "decode_reqs": 0, "total_tokens": 8000},
            {"prefill_reqs": 1, "prefill_tokens": 16, "decode_reqs": 4, "total_tokens": 20},
            {"prefill_reqs": 0, "prefill_tokens": 0, "decode_reqs": 4, "total_tokens": 4},
        ],
        preemption_events=[
            {"trigger_request_id": 7, "victim_request_id": 7, "recompute_tokens": 9000},
            {"trigger_request_id": 8, "victim_request_id": 7, "recompute_tokens": 9001},
        ],
    )

    assert signature.iterations == 3
    assert signature.mixed_steps == 1
    assert signature.mixed_step_share == 1 / 3
    assert signature.preemptions == 2
    assert signature.self_preemptions == 1
    assert signature.repeat_victim_events == 1
    assert signature.recompute_tokens == 18001
    assert signature.peak_decode_batch == 4
    assert signature.peak_prefill_tokens == 8000


def test_dynamic_signature_preserves_unobserved_event_streams() -> None:
    signature = analysis.summarize_dynamic_signature(
        trace=[],
        preemption_events=[],
        total_iterations=10,
        trace_observed=False,
        preemption_events_observed=False,
    )

    assert signature.mixed_steps is None
    assert signature.mixed_step_share is None
    assert signature.preemptions is None
    assert signature.self_preemptions is None
    assert signature.repeat_victim_events is None
    assert signature.recompute_tokens is None
