from __future__ import annotations

from pathlib import Path

import pytest

from scripts import analyze_phase397x_multi_config_attribution as phase397x


def test_phase397x_classifies_throughput_overprediction() -> None:
    assert phase397x.classify_direction(sim_output=300.0, real_output=100.0) == "sim_overpredicts_throughput"
    assert phase397x.symmetric_error_ratio(sim_output=300.0, real_output=100.0) == pytest.approx(3.0)


def test_phase397x_buckets_generation_ops_into_structural_categories() -> None:
    latency = {
        "generation_attention": 10.0,
        "generation_moe": 2.0,
        "generation_moe_pre_dispatch": 1.0,
        "generation_ar_1": 3.0,
        "generation_q_b_proj_gemm": 4.0,
        "generation_add_norm_1": 0.5,
    }

    categories = phase397x.categorize_latency_dict(latency)

    assert categories["attention"] == pytest.approx(10.0)
    assert categories["moe"] == pytest.approx(2.0)
    assert categories["comm"] == pytest.approx(4.0)
    assert categories["gemm"] == pytest.approx(4.0)
    assert categories["other"] == pytest.approx(0.5)


def test_phase397x_writer_rejects_default_upgrade(tmp_path: Path) -> None:
    rows = [phase397x._base_row("scenario_summary")]
    rows[0]["valid_for_default"] = "true"

    with pytest.raises(ValueError, match="valid_for_default"):
        phase397x.write_phase397x_csv(tmp_path / "bad.csv", rows)


def test_phase397x_summary_reports_active_gate_failure() -> None:
    rows = [
        {
            **phase397x._base_row("scenario_summary"),
            "scenario": "worst",
            "real_output_tok_s_gpu": "100.000000",
            "sim_output_tok_s_gpu": "357.000000",
            "throughput_ratio_sim_over_real": "3.570000",
            "error_ratio": "3.570000",
            "context_attention_excluded_ms_per_iter": "1.000000",
            "direction": "sim_overpredicts_throughput",
            "tier": "B",
            "real_per_op_available": "false",
            "verdict": "active_gate_failed",
        },
        {
            **phase397x._base_row("scenario_summary"),
            "scenario": "best",
            "real_output_tok_s_gpu": "100.000000",
            "sim_output_tok_s_gpu": "212.000000",
            "throughput_ratio_sim_over_real": "2.120000",
            "error_ratio": "2.120000",
            "context_attention_excluded_ms_per_iter": "0.000000",
            "direction": "sim_overpredicts_throughput",
            "tier": "B",
            "real_per_op_available": "false",
            "verdict": "active_gate_failed",
        },
    ]

    md = phase397x.render_phase397x_md(rows)

    assert "max=3.57x" in md
    assert "Default AIC remains No-Go" in md
    assert "Tier B" in md
