from pathlib import Path

import pytest

from scripts import analyze_phase462_step3f as analysis


TP_LINE = (
    "(EngineCore pid=35102) INFO 07-09 16:53:00 [core.py:359] "
    "Iteration(2): 9 context requests, 65535 context tokens, "
    "1 generation requests, 1 generation tokens, "
    "iteration elapsed time: 4695.50 ms"
)
DP_LINE = (
    "(EngineCore_DP1 pid=37032) INFO 07-09 17:13:55 [core.py:359] "
    "Iteration(6): 1 context requests, 6471 context tokens, "
    "15 generation requests, 15 generation tokens, "
    "iteration elapsed time: 561.97 ms"
)


def test_parse_iteration_line_extracts_tp_and_dp_rank() -> None:
    tp = analysis.parse_iteration_line(TP_LINE)
    dp = analysis.parse_iteration_line(DP_LINE)

    assert tp is not None
    assert (tp.dp_rank, tp.iteration, tp.context_tokens, tp.elapsed_ms) == (
        0,
        2,
        65535,
        4695.50,
    )
    assert dp is not None
    assert (dp.dp_rank, dp.context_requests, dp.generation_requests) == (1, 1, 15)


def test_read_iteration_records_rejects_iteration_schema_drift(tmp_path: Path) -> None:
    log = tmp_path / "serve.log"
    log.write_text(
        "(EngineCore pid=1) INFO [core.py:359] Iteration(1): changed schema\n",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="iteration_log_schema_drift"):
        analysis.read_iteration_records(log)


def test_same_cell_spread_keeps_rank_conditioned_latency() -> None:
    rows = [
        analysis.IterationRecord(0, 10, 1, 6471, 15, 15, 4791.13),
        analysis.IterationRecord(1, 11, 1, 6471, 15, 15, 562.35),
        analysis.IterationRecord(1, 12, 1, 6471, 15, 15, 561.59),
    ]

    summary = analysis.same_cell_spread(
        rows,
        bucket_tokens=6486,
        decode_batch=15,
    )

    assert summary.rank_samples == {0: 1, 1: 2}
    assert summary.rank_medians == {0: 4791.13, 1: 561.97}
    assert summary.max_over_min == pytest.approx(8.525597451821271)


def test_existing_fields_cannot_run_exact_composition_or_cost_judgement() -> None:
    audit = analysis.audit_existing_fields()

    assert audit["coarse_spread_reproducible"] is True
    assert audit["composition_attribution_executable"] is False
    assert audit["exact_undercharge_judgement_executable"] is False
    assert audit["missing_fields"] == (
        "context_chunk_tokens_multiset",
        "context_state_token_counts",
        "decode_kv_token_sum",
        "cudagraph_mode",
    )


def test_periodic_sampling_is_rejected_for_single_sample_target_cell() -> None:
    design = analysis.lightweight_v2_design(
        total_steps=75_277,
        mixed_steps=100,
        minimum_target_rank_samples=1,
    )

    assert design["periodic_k"] is None
    assert design["trigger"] == "mixed_steps_only"
    assert design["emitted_fraction"] == pytest.approx(100 / 75_277)
    assert design["emitted_rows"] == 100
