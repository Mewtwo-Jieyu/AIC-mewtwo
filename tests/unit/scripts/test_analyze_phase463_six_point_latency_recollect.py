from __future__ import annotations

import importlib.util
import csv
import json
import sys
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).resolve().parents[3] / "scripts" / "analyze_phase463_six_point_latency_recollect.py"
    spec = importlib.util.spec_from_file_location("analyze_phase463_six_point_latency_recollect", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _metric_lines(*, itl_count: int = 6, scale: int = 1) -> str:
    rows = [
        f'vllm:prompt_tokens_total{{engine="0",model_name="kimi-k2.5"}} {16 * scale}',
        f'vllm:generation_tokens_total{{engine="0",model_name="kimi-k2.5"}} {8 * scale}',
        f'vllm:request_success_total{{engine="0",finished_reason="length",model_name="kimi-k2.5"}} {2 * scale}',
    ]
    histograms = {
        "time_to_first_token_seconds": (2, 1.2, [("0.5", 1), ("1.0", 2), ("+Inf", 2)]),
        "inter_token_latency_seconds": (
            itl_count,
            0.6,
            [("0.05", 2), ("0.1", itl_count), ("+Inf", itl_count)],
        ),
        "request_time_per_output_token_seconds": (
            2,
            0.2,
            [("0.1", 2), ("+Inf", 2)],
        ),
        "e2e_request_latency_seconds": (2, 2.6, [("1.0", 1), ("2.0", 2), ("+Inf", 2)]),
    }
    for name, (count, total, buckets) in histograms.items():
        for le, value in buckets:
            rows.append(
                f'vllm:{name}_bucket{{engine="0",le="{le}",model_name="kimi-k2.5"}} {value * scale}'
            )
        rows.append(f'vllm:{name}_count{{engine="0",model_name="kimi-k2.5"}} {count * scale}')
        rows.append(f'vllm:{name}_sum{{engine="0",model_name="kimi-k2.5"}} {total * scale}')
    return "\n".join(rows) + "\n"


def test_service_metrics_use_exact_sum_count_and_enforce_token_counts() -> None:
    phase463 = _load_module()

    summary = phase463.summarize_service_metrics(
        before_text=_metric_lines(scale=0),
        after_text=_metric_lines(),
        expected_requests=2,
        expected_prompt_tokens=16,
        expected_generation_tokens=8,
        output_len=4,
    )

    assert summary["request_success"] == 2
    assert summary["ttft_mean_ms"] == pytest.approx(600.0)
    assert summary["tpot_mean_ms"] == pytest.approx(100.0)
    assert summary["itl_mean_ms"] == pytest.approx(100.0)
    assert summary["e2e_mean_ms"] == pytest.approx(1300.0)
    assert summary["itl_count"] == 6
    assert summary["ttft_p90_ms"] == pytest.approx(900.0)


def test_service_metrics_reject_wrong_itl_count() -> None:
    phase463 = _load_module()

    with pytest.raises(ValueError, match="itl_count_mismatch"):
        phase463.summarize_service_metrics(
            before_text=_metric_lines(scale=0),
            after_text=_metric_lines(itl_count=5),
            expected_requests=2,
            expected_prompt_tokens=16,
            expected_generation_tokens=8,
            output_len=4,
        )


def test_sampling_continuity_rejects_failed_or_missing_samples(tmp_path: Path) -> None:
    phase463 = _load_module()
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"ts": "2026-07-16T00:00:00+00:00", "status": 200},
                {"ts": "2026-07-16T00:00:02+00:00", "status": 200},
                {"ts": "2026-07-16T00:00:04+00:00", "status": 200},
            ]
        )
        + "\n"
    )

    assert phase463.validate_metrics_continuity(metrics, max_gap_s=5.0)["sample_count"] == 3

    metrics.write_text(
        json.dumps({"ts": "2026-07-16T00:00:00+00:00", "status": 200})
        + "\n"
        + json.dumps({"ts": "2026-07-16T00:00:10+00:00", "status": 500})
        + "\n"
    )
    with pytest.raises(ValueError, match="metrics_sampling_failed"):
        phase463.validate_metrics_continuity(metrics, max_gap_s=5.0)


def test_iteration_and_gpu_summaries_are_strict(tmp_path: Path) -> None:
    phase463 = _load_module()
    serve_log = tmp_path / "serve.log"
    serve_log.write_text(
        "\n".join(
            [
                "(EngineCore_DP0 pid=1) Iteration(1): 1 context requests, 8000 context tokens, 0 generation requests, 0 generation tokens, iteration elapsed time: 100.0 ms",
                "(EngineCore_DP1 pid=2) Iteration(2): 0 context requests, 0 context tokens, 4 generation requests, 4 generation tokens, iteration elapsed time: 20.0 ms",
            ]
        )
        + "\n"
    )
    gpu_csv = tmp_path / "gpu.csv"
    gpu_csv.write_text(
        "\n".join(
            [
                "2026/07/16 00:00:00.000, 0, 50, 400, 1000",
                "2026/07/16 00:00:00.003, 1, 60, 420, 1100",
                "2026/07/16 00:00:02.000, 0, 70, 440, 1200",
                "2026/07/16 00:00:02.004, 1, 80, 460, 1300",
            ]
        )
        + "\n"
    )

    iteration = phase463.summarize_iterations(serve_log)
    gpu = phase463.summarize_gpu_telemetry(gpu_csv, expected_gpus=2, max_gap_s=5.0)

    assert iteration["iteration_count"] == 2
    assert iteration["context_tokens"] == 8000
    assert iteration["generation_tokens"] == 4
    assert gpu["sample_count"] == 2
    assert gpu["gpu_util_mean_pct"] == pytest.approx(65.0)
    assert gpu["power_mean_w"] == pytest.approx(430.0)
    assert gpu["memory_max_mib"] == 1300


def test_canary_gate_uses_symmetric_throughput_delta() -> None:
    phase463 = _load_module()

    passed = phase463.canary_gate(nonstream_output_tok_s=100.0, stream_output_tok_s=98.5)
    assert passed["delta_pct"] == pytest.approx(1.5)
    assert passed["passed"] is True

    with pytest.raises(ValueError, match="stream_canary_throughput_delta"):
        phase463.canary_gate(nonstream_output_tok_s=100.0, stream_output_tok_s=97.0)


def test_analyze_scenario_validates_the_complete_artifact(tmp_path: Path) -> None:
    phase463 = _load_module()
    (tmp_path / "meta.json").write_text(
        json.dumps(
            {
                "name": "unit",
                "role": "formal",
                "stream": True,
                "bench_num_prompts": 2,
                "batch_size": 2,
                "tp": 2,
                "dp": 1,
                "world_size": 2,
                "isl": 8,
                "osl": 4,
                "max_num_batched_tokens": 8,
            }
        )
        + "\n"
    )
    (tmp_path / "bench_result.json").write_text(
        json.dumps(
            {
                "stream": True,
                "ok_requests": 2,
                "failed_requests": 0,
                "total_prompt_tokens": 16,
                "total_completion_tokens": 8,
                "total_tokens": 24,
                "output_tok_s": 80.0,
                "total_tok_s": 240.0,
                "mean_latency_ms": 1400.0,
                "p50_latency_ms": 1300.0,
                "p90_latency_ms": 1700.0,
                "p99_latency_ms": 1790.0,
                "mean_ttft_ms": 610.0,
                "p50_ttft_ms": 600.0,
                "p90_ttft_ms": 700.0,
                "p99_ttft_ms": 720.0,
                "mean_tpot_ms": 105.0,
                "p50_tpot_ms": 100.0,
                "p90_tpot_ms": 120.0,
                "p99_tpot_ms": 125.0,
            }
        )
        + "\n"
    )
    (tmp_path / "metrics_before.prom").write_text(_metric_lines(scale=0))
    (tmp_path / "metrics_after.prom").write_text(_metric_lines())
    (tmp_path / "metrics.jsonl").write_text(
        json.dumps({"ts": "2026-07-16T00:00:00+00:00", "status": 200})
        + "\n"
        + json.dumps({"ts": "2026-07-16T00:00:02+00:00", "status": 200})
        + "\n"
    )
    (tmp_path / "serve.log").write_text(
        "(EngineCore pid=1) Iteration(1): 1 context requests, 16 context tokens, 2 generation requests, 2 generation tokens, iteration elapsed time: 10.0 ms\n"
    )
    (tmp_path / "gpu.csv").write_text(
        "\n".join(
            [
                "2026/07/16 00:00:00.000, 0, 50, 400, 1000",
                "2026/07/16 00:00:00.003, 1, 60, 420, 1100",
                "2026/07/16 00:00:02.000, 0, 70, 440, 1200",
                "2026/07/16 00:00:02.004, 1, 80, 460, 1300",
            ]
        )
        + "\n"
    )

    result = phase463.analyze_scenario(tmp_path)

    assert result["scenario"] == "unit"
    assert result["real_output_tok_s_gpu"] == 40.0
    assert result["service_ttft_mean_ms"] == pytest.approx(600.0)
    assert result["client_ttft_mean_ms"] == pytest.approx(610.0)
    assert result["metrics_sample_count"] == 2


def test_comparison_row_gates_only_throughput() -> None:
    phase463 = _load_module()
    real = {
        "scenario": "unit",
        "role": "formal",
        "real_output_tok_s_gpu": 100.0,
        "service_ttft_mean_ms": 200.0,
        "service_tpot_mean_ms": 20.0,
        "service_e2e_mean_ms": 400.0,
        "gpu_util_mean_pct": 95.0,
        "power_mean_w": 500.0,
        "iteration_count": 10,
        "context_requests": 2,
        "generation_requests": 80,
    }
    sim = {
        "output_tok_s_gpu": 110.0,
        "ttft_ms": 300.0,
        "tpot_ms": 10.0,
        "e2e_ms": 350.0,
        "avg_prefill_reqs_per_iter": 0.2,
        "avg_decode_reqs_per_iter": 8.0,
    }

    row = phase463.build_comparison_row(real, sim, frozen_real_output_tok_s_gpu=90.0)

    assert row["throughput_abs_error_pct"] == pytest.approx(10.0)
    assert row["throughput_gate"] == "pass"
    assert row["ttft_sim_over_real"] == pytest.approx(1.5)
    assert row["tpot_sim_over_real"] == pytest.approx(0.5)
    assert row["e2e_sim_over_real"] == pytest.approx(0.875)
    assert row["latency_gate"] == "baseline_only"
    assert row["frozen_real_drift_pct"] == pytest.approx(100.0 / 90.0 * 100.0 - 100.0)


def test_diagnostic_comparison_row_has_no_frozen_baseline_or_gate() -> None:
    phase463 = _load_module()
    real = {
        "scenario": "diagnostic",
        "role": "diagnostic",
        "real_output_tok_s_gpu": 100.0,
        "service_ttft_mean_ms": 200.0,
        "service_tpot_mean_ms": 20.0,
        "service_e2e_mean_ms": 400.0,
        "iteration_count": 10,
        "context_requests": 2,
        "generation_requests": 80,
    }
    sim = {
        "output_tok_s_gpu": 110.0,
        "ttft_ms": 300.0,
        "tpot_ms": 10.0,
        "e2e_ms": 350.0,
        "avg_prefill_reqs_per_iter": 0.2,
        "avg_decode_reqs_per_iter": 8.0,
    }

    row = phase463.build_comparison_row(
        real,
        sim,
        frozen_real_output_tok_s_gpu=None,
    )

    assert row["throughput_gate"] == "diagnostic"
    assert row["frozen_real_output_tok_s_gpu"] is None
    assert row["frozen_real_drift_pct"] is None


def test_write_comparison_outputs_preserves_no_go_boundary(tmp_path: Path) -> None:
    phase463 = _load_module()
    row = {
        "scenario": "unit",
        "role": "formal",
        "real_output_tok_s_gpu": 100.0,
        "sim_output_tok_s_gpu": 110.0,
        "throughput_sim_over_real": 1.1,
        "throughput_abs_error_pct": 10.0,
        "throughput_gate": "pass",
        "service_ttft_mean_ms": 200.0,
        "sim_ttft_ms": 300.0,
        "ttft_sim_over_real": 1.5,
        "service_tpot_mean_ms": 20.0,
        "sim_tpot_ms": 10.0,
        "tpot_sim_over_real": 0.5,
        "service_e2e_mean_ms": 400.0,
        "sim_e2e_ms": 350.0,
        "e2e_sim_over_real": 0.875,
        "gpu_util_mean_pct": 95.0,
        "power_mean_w": 500.0,
        "real_avg_context_reqs_per_logged_iteration": 0.2,
        "real_avg_generation_reqs_per_logged_iteration": 8.0,
        "sim_avg_prefill_reqs_per_iter": 0.2,
        "sim_avg_decode_reqs_per_iter": 8.0,
        "frozen_real_output_tok_s_gpu": 90.0,
        "frozen_real_drift_pct": 11.111,
        "latency_gate": "baseline_only",
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
        "default_readiness": "No-Go",
    }
    csv_path = tmp_path / "result.csv"
    md_path = tmp_path / "result.md"

    phase463.write_comparison_outputs(
        [row],
        csv_path=csv_path,
        md_path=md_path,
        artifact_root=tmp_path,
        canary_delta_pct=0.1,
    )

    with csv_path.open(newline="", encoding="utf-8") as f:
        written = list(csv.DictReader(f))
    assert written[0]["default_readiness"] == "No-Go"
    assert written[0]["latency_gate"] == "baseline_only"
    assert b"\r" not in csv_path.read_bytes()
    assert "0.1000%" in md_path.read_text(encoding="utf-8")
    assert "Default AIC 仍为 No-Go" in md_path.read_text(encoding="utf-8")
