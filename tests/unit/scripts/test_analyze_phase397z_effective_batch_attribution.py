from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import analyze_phase397z_effective_batch_attribution as phase397z


def _budget_row(
    name: str,
    *,
    tp: int,
    dp: int,
    ep: int,
    max_bt: int,
    real: float,
    sim: float,
    avg_decode: float,
    steady_iters: int,
    steady_ms: float,
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        tp=tp,
        dp=dp,
        ep=ep,
        max_bt=max_bt,
        real_output_tok_s_gpu=real,
        sim_output_tok_s_gpu=sim,
        error_ratio=max(sim / real, real / sim),
        rank=1,
        real_rank=1,
        avg_prefill_reqs_per_iter=1.0,
        avg_decode_reqs_per_iter=avg_decode,
        avg_tokens_per_iter=avg_decode + 1.0,
        peak_prefill_reqs_per_iter=1.0,
        peak_decode_reqs_per_iter=avg_decode,
        peak_tokens_per_iter=avg_decode + 1.0,
        steady_state_iterations=steady_iters,
        steady_state_time_ms=steady_ms,
        paired_baseline_name=name,
        paired_baseline_max_bt=max_bt,
        paired_baseline_sim_output_tok_s_gpu=sim,
        paired_baseline_error_ratio=max(sim / real, real / sim),
        paired_baseline_rank=1,
        paired_baseline_avg_tokens_per_iter=avg_decode + 1.0,
        max_bt_vs_paired_baseline=1.0,
        sim_vs_paired_baseline_ratio=1.0,
        avg_tokens_vs_paired_baseline_ratio=1.0,
        steady_state_time_vs_paired_baseline_ratio=1.0,
        diagnostic_only=True,
        valid_for_default=False,
        perf_database=False,
    )


def test_effective_batch_decomposition_reconstructs_throughput_ratio() -> None:
    factors = phase397z.decompose_throughput_ratio(
        sim_effective_decode_batch=125.181996,
        real_effective_decode_batch=42.074119,
        sim_iter_ms=55.110,
        real_iter_ms=39.3855,
    )

    assert factors["effective_batch_ratio"] == pytest.approx(2.975, rel=1e-3)
    assert factors["iter_latency_factor"] == pytest.approx(0.715, rel=1e-3)
    assert factors["reconstructed_throughput_ratio"] == pytest.approx(2.126, rel=1e-3)


def test_phase397z_rows_identify_effective_batch_as_main_driver() -> None:
    budget_rows = [
        _budget_row(
            "K2.5-tp8ep8-8k2k",
            tp=8,
            dp=1,
            ep=8,
            max_bt=8000,
            real=133.528000,
            sim=283.919323,
            avg_decode=125.181996,
            steady_iters=3874,
            steady_ms=218117.419278,
        )
    ]
    real_decode_anchors = {
        "tp8": phase397z.RealDecodeAnchor(
            source_scenario="K2.5-tp8ep8-8k2k",
            tier="A",
            real_decode_iter_ms=39.3855,
            provenance="phase397l_op_breakdown_exact",
        )
    }

    rows = phase397z.build_phase397z_rows(
        budget_rows=budget_rows,
        real_decode_anchors=real_decode_anchors,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["tier"] == "A"
    assert row["real_effective_decode_batch_per_dp"] == pytest.approx(42.072536, rel=1e-6)
    assert row["sim_effective_decode_batch_per_dp"] == pytest.approx(125.181996, rel=1e-6)
    assert row["effective_batch_ratio"] == pytest.approx(2.975385, rel=1e-6)
    assert row["sim_to_real_iteration_count_factor"] == pytest.approx(0.336091, rel=1e-6)
    assert row["dominant_driver"] == "effective_batch_overestimate"
    assert row["scheduler_localization"] == "decode_reserved_before_prefill_and_closed_loop_replacement"
    assert row["default_readiness"] == "No-Go"


def test_phase397z_writer_rejects_default_upgrade(tmp_path: Path) -> None:
    row = phase397z._base_row("scenario")
    row["valid_for_default"] = True

    with pytest.raises(ValueError, match="valid_for_default"):
        phase397z.write_phase397z_csv(tmp_path / "bad.csv", [row])


def test_phase397z_markdown_reports_tier_boundary_and_no_go() -> None:
    row = phase397z._base_row("scenario")
    row.update(
        {
            "scenario": "K2.5-tp8ep8-8k2k",
            "tier": "A",
            "throughput_ratio_sim_over_real": 2.126291,
            "effective_batch_ratio": 2.975275,
            "iter_latency_factor": 0.714648,
            "sim_effective_decode_batch_per_dp": 125.181996,
            "real_effective_decode_batch_per_dp": 42.072536,
            "sim_to_real_iteration_count_factor": 0.336091,
            "sim_to_real_wall_factor": 0.470311,
            "sim_decode_occupancy": 0.977984,
            "real_decode_occupancy": 0.328692,
            "steady_state_iterations": 3874,
            "steady_state_time_ms": 218117.419278,
            "dominant_driver": "effective_batch_overestimate",
        }
    )

    md = phase397z.render_phase397z_md([row])

    assert "Default AIC remains No-Go" in md
    assert "effective batch" in md
    assert "Tier A" in md
