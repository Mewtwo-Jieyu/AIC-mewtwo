from __future__ import annotations

import scripts.analyze_phase462_tp8_bt65536_residual as analysis


def _step(
    iteration: int,
    wall_ms: float,
    *,
    chunk: str,
    coverage: str,
    admits: int,
) -> analysis.ResidualStep:
    return analysis.ResidualStep(
        iteration=iteration,
        bucket_tokens=65536,
        decode_batch=8,
        wall_ms=wall_ms,
        prefill_tokens=65528,
        recompute_tokens=0,
        chunk_composition=chunk,
        serving_coverage=coverage,
        serving_miss_reasons=(),
        new_admit_count=admits,
    )


def test_chunk_composition_is_derived_from_request_state() -> None:
    parts = [
        analysis.PrefillPart("fresh", "complete", 8000),
        analysis.PrefillPart("continued", "partial", 4000),
    ]

    assert analysis.chunk_composition(parts) == "continued_partial+fresh_complete"


def test_three_dimensions_each_conserve_residual_wall() -> None:
    steps = [
        _step(1, 10.0, chunk="fresh_partial", coverage="all_hit", admits=1),
        _step(2, 20.0, chunk="continued_partial", coverage="all_miss", admits=0),
        _step(4, 30.0, chunk="fresh_complete", coverage="partial_hit", admits=2),
    ]

    result = analysis.decompose_residual_steps(steps)

    assert result.total_wall_ms == 60.0
    assert sum(result.chunk_wall_ms.values()) == 60.0
    assert sum(result.coverage_wall_ms.values()) == 60.0
    assert sum(result.admit_count_wall_ms.values()) == 60.0
    assert sum(result.streak_length_wall_ms.values()) == 60.0
    assert result.streak_length_wall_ms == {2: 30.0, 1: 30.0}


def test_serving_coverage_uses_exact_audit_hits() -> None:
    assert analysis.serving_coverage([]) == ("scope_disabled", ())
    assert analysis.serving_coverage(
        [analysis.AuditHit(True, ""), analysis.AuditHit(True, "")]
    ) == ("all_hit", ())
    assert analysis.serving_coverage(
        [analysis.AuditHit(True, ""), analysis.AuditHit(False, "outside_scope")]
    ) == ("partial_hit", ("outside_scope",))
    assert analysis.serving_coverage(
        [analysis.AuditHit(False, "missing"), analysis.AuditHit(False, "missing")]
    ) == ("all_miss", ("missing",))


def test_serving_cache_key_matches_runtime_kv_bucketing() -> None:
    left = analysis.serving_cache_key(100, 2, 8000, 8, 1001)
    right = analysis.serving_cache_key(100, 2, 8000, 8, 1023)

    assert left == right
