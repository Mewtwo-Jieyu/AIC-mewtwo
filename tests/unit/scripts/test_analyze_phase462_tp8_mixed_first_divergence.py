from __future__ import annotations

import scripts.analyze_phase462_tp8_mixed_first_divergence as analysis


def _step(
    iteration: int,
    *,
    new_admits: int,
    budget_before_admit: int,
    next_request_tokens: int,
    stopped_after_partial: bool = False,
) -> analysis.SimStep:
    return analysis.SimStep(
        iteration=iteration,
        prefill_requests=9,
        prefill_tokens=64000,
        decode_requests=8,
        new_admit_ids=tuple(range(new_admits)),
        continued_ids=(),
        recompute_ids=(),
        budget_before_admit=budget_before_admit,
        budget_after_schedule=0,
        next_request_tokens=next_request_tokens,
        max_num_seqs_reached=False,
        block_capacity_reached=False,
        stopped_after_partial_prefill=stopped_after_partial,
    )


def test_verdict_identifies_token_budget_boundary() -> None:
    step = _step(
        12,
        new_admits=8,
        budget_before_admit=64000,
        next_request_tokens=8000,
    )

    verdict = analysis.classify_formation_mechanism(step)

    assert verdict == "token_budget_split"


def test_verdict_identifies_partial_prefill_stop() -> None:
    step = _step(
        12,
        new_admits=1,
        budget_before_admit=65536,
        next_request_tokens=8000,
        stopped_after_partial=True,
    )

    verdict = analysis.classify_formation_mechanism(step)

    assert verdict == "chunked_prefill_partial_stop"


def test_first_divergence_requires_real_macro_state_comparator() -> None:
    sim = [
        _step(10, new_admits=8, budget_before_admit=64000, next_request_tokens=8000),
        _step(11, new_admits=8, budget_before_admit=64000, next_request_tokens=8000),
    ]
    real = [
        analysis.RealStep(2, 9, 65535, 1),
        analysis.RealStep(3, 9, 65527, 2),
    ]

    result = analysis.first_divergence(sim, real)

    assert result.sim_iteration == 10
    assert result.streak_length == 2
    assert result.real_has_same_macro_state is False
    assert result.mechanism == "token_budget_split"


def test_real_eight_wide_transition_moves_root_to_initial_admission_wave() -> None:
    sim = [
        analysis.SimStep(
            iteration=1,
            prefill_requests=9,
            prefill_tokens=65536,
            decode_requests=0,
            new_admit_ids=tuple(range(9)),
            continued_ids=(),
            recompute_ids=(),
            budget_before_admit=65536,
            budget_after_schedule=0,
            next_request_tokens=8000,
            max_num_seqs_reached=False,
            block_capacity_reached=False,
            stopped_after_partial_prefill=True,
        ),
        _step(
            2,
            new_admits=8,
            budget_before_admit=59064,
            next_request_tokens=8000,
            stopped_after_partial=True,
        ),
        _step(
            3,
            new_admits=8,
            budget_before_admit=60584,
            next_request_tokens=8000,
            stopped_after_partial=True,
        ),
    ]
    real = [
        analysis.RealStep(0, 1, 8000, 0),
        analysis.RealStep(1, 0, 0, 1),
        analysis.RealStep(2, 9, 65535, 1),
        analysis.RealStep(3, 9, 65527, 9),
    ]

    result = analysis.first_divergence(sim, real)

    assert result.sim_iteration == 1
    assert result.cluster_start_iteration == 2
    assert result.real_has_equivalent_eight_wide_transition is True
    assert result.mechanism == "initial_admission_wave_phase_offset"
