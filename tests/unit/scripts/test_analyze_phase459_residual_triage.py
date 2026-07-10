import gzip
import json

from scripts.analyze_phase459_residual_triage import (
    IterationStep,
    classify_bt_residual,
    classify_bt_wall_replay,
    decompose_tp8_32k_gap,
    load_serving_state_regimes,
    parse_iteration_steps,
    parse_metrics_profile,
)


def test_parse_iteration_steps_supports_tp8_and_dp2(tmp_path) -> None:
    path = tmp_path / "serve.log"
    path.write_text(
        "\n".join(
            [
                "(EngineCore pid=11) INFO Iteration(3): 2 context requests, 31998 context tokens, 2 generation requests, 2 generation tokens, iteration elapsed time: 2418.12 ms",
                "(EngineCore_DP1 pid=12) INFO Iteration(4): 0 context requests, 0 context tokens, 16 generation requests, 16 generation tokens, iteration elapsed time: 16.57 ms",
            ]
        ),
        encoding="utf-8",
    )

    steps = parse_iteration_steps(path)

    assert [(step.engine, step.iteration) for step in steps] == [("0", 3), ("1", 4)]
    assert steps[0].ctx_tokens == 31_998
    assert steps[1].generation_requests == 16


def test_parse_metrics_profile_reads_gzip_counters_and_gauges(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl.gz"
    bodies = [
        "\n".join(
            [
                'vllm:num_requests_running{engine="0"} 10',
                'vllm:num_requests_waiting{engine="0"} 3',
                'vllm:gpu_cache_usage_perc{engine="0"} 0.5',
                'vllm:num_preemptions_total{engine="0"} 2',
                'vllm:prompt_tokens_recomputed_total{engine="0"} 0',
                'vllm:request_success_total{engine="0",finished_reason="length"} 4',
            ]
        ),
        "\n".join(
            [
                'vllm:num_requests_running{engine="0"} 14',
                'vllm:num_requests_waiting{engine="0"} 1',
                'vllm:gpu_cache_usage_perc{engine="0"} 0.9',
                'vllm:num_preemptions_total{engine="0"} 7',
                'vllm:prompt_tokens_recomputed_total{engine="0"} 32000',
                'vllm:request_success_total{engine="0",finished_reason="length"} 12',
            ]
        ),
    ]
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for body in bodies:
            f.write(json.dumps({"status": 200, "body": body}) + "\n")

    profile = parse_metrics_profile(path)

    assert profile.preemptions == 5
    assert profile.recomputed_prompt_tokens == 32_000
    assert profile.successful_requests == 8
    assert profile.running_p50 == 12
    assert profile.kv_usage_p90 > 0.8


def test_tp8_32k_gap_decomposition_prefers_cost_replay_that_closes_gap() -> None:
    result = decompose_tp8_32k_gap(
        real_tput=50.0,
        sim_tput=35.0,
        cost_replay_factor=1.42,
        useful_prompt_tokens=16_384_000,
        real_recompute_tokens=0,
        sim_recompute_tokens=6_250_000,
        real_preemptions=42,
        sim_preemptions=194,
        mixed_step_real_over_sim=1.89,
        decode_step_sim_over_real=1.86,
    )

    assert result["verdict"] == "decode_cost_overcharge_dominates"
    assert result["cost_replay_residual"] < 1.10
    assert result["prompt_work_factor"] > 1.35
    assert result["preemption_role"] == "secondary_unresolved_composition_error"


def test_bt_residual_requires_undercharge_and_coverage_miss() -> None:
    step = IterationStep("0", 1, 8, 65_520, 16, 16, 4700.0)
    result = classify_bt_residual(
        step=step,
        sim_step_ms=3000.0,
        audit_counts={"bucket_above_range": 4, "hit": 0},
    )

    assert result["real_over_sim_step_ratio"] > 1.5
    assert result["coverage_miss_count"] == 4
    assert result["verdict"] == "large_mixed_step_undercharged_with_coverage_miss"


def test_bt_wall_replay_flags_inaccurate_regime_hit() -> None:
    result = classify_bt_wall_replay(
        observed_sim_wall_ratio=0.857,
        cost_replay_factor=0.886,
        modal_mixed_sim_over_real=0.50,
        modal_audit_counts={"hit": 1},
    )

    assert result["residual_after_cost_replay"] < 1.05
    assert result["verdict"] == "mixed_step_undercharge_regime_hit_mismatch"


def test_load_serving_state_regimes_reports_mixed_max_bt_values(tmp_path) -> None:
    path = tmp_path / "serving.csv"
    path.write_text(
        "framework,version,device,model,topology,phase,row_kind,category,kernel_source,max_num_batched_tokens,bucket_tokens,decode_batch,hidden_size,topk,moe_ep_size,quant_runtime,latency,provenance\n"
        "VLLM,0.19,H200,kimi,tp4dp2ep8,mixed_prefill,forward_total,forward_total,p1,8000,8000,1,7168,8,8,q,1.0,a\n"
        "VLLM,0.19,H200,kimi,tp4dp2ep8,mixed_prefill,forward_total,forward_total,p2,65536,65536,1,7168,8,8,q,2.0,b\n",
        encoding="utf-8",
    )

    assert load_serving_state_regimes(path, "tp4dp2ep8") == {8_000, 65_536}
