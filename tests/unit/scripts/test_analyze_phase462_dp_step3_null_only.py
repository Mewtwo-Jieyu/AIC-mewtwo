from __future__ import annotations

from pathlib import Path

import scripts.analyze_phase462_dp_step3_null_only as analysis


def _step(rank: int, iteration: int, prefill: int, decode: int) -> dict[str, int]:
    return {
        "replica_id": rank,
        "local_iter": iteration,
        "prefill_tokens": prefill,
        "decode_reqs": decode,
    }


def test_phase_gate_rejects_a_score_pass_with_the_wrong_rank_signature() -> None:
    real = [
        _step(0, 0, 65536, 0),
        _step(1, 0, 8000, 0),
        _step(0, 1, 62464, 8),
        _step(1, 1, 0, 1),
    ]
    candidate = [
        _step(0, 0, 65536, 0),
        _step(1, 0, 65536, 0),
        _step(0, 1, 62464, 8),
        _step(1, 1, 62464, 8),
    ]

    real_summary = analysis.summarize_paired_rank_phase(real)
    candidate_summary = analysis.summarize_paired_rank_phase(candidate)
    decision = analysis.candidate_decision(
        candidate_error=1.095,
        real_summary=real_summary,
        candidate_summary=candidate_summary,
    )

    assert real_summary.active_exact_fraction == 0.0
    assert candidate_summary.active_exact_fraction == 1.0
    assert decision == "score_pass_phase_fail_do_not_adopt"


def test_sim_only_bucket_recheck_weights_wall_and_recompute() -> None:
    steps = [
        analysis.MixedStep(100, 4, 10.0, 96, 0),
        analysis.MixedStep(200, 8, 30.0, 192, 96),
        analysis.MixedStep(300, 8, 60.0, 292, 146),
    ]

    result = analysis.summarize_sim_only_bucket_residual(
        steps,
        real_cells={(100, 4)},
    )

    assert result.sim_only_wall_share == 0.9
    assert result.recompute_associated_wall_share == 1.0
    assert result.recompute_token_share == (96 + 146) / (192 + 292)


def test_report_keeps_runtime_and_perfdb_closed(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    rows = [
        analysis.ReportRow(
            "dp_candidate",
            analysis.DP2_BT,
            "candidate_error_ratio",
            "1.095046",
            "score_pass_phase_fail",
            "diagnostic only",
        )
    ]

    analysis.write_report(
        report,
        rows,
        decision="logging_only_dp_route_visibility_required",
    )

    text = report.read_text(encoding="utf-8")
    assert "不得进入默认路径" in text
    assert "runtime" in text
    assert "PerfDB" in text
    assert "Default AIC=No-Go" in text
