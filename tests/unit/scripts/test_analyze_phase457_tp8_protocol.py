import json

from scripts.analyze_phase457_tp8_protocol import (
    build_rows,
    load_bench_output_tok_s_gpu,
    parse_preemption_delta,
)


def _row(rows, metric):
    return next(row for row in rows if row["metric"] == metric)


def test_phase457_bench_loader_reports_per_gpu_output(tmp_path) -> None:
    bench = tmp_path / "bench_result.json"
    bench.write_text(
        json.dumps(
            {
                "num_prompts": 512,
                "max_concurrency": 128,
                "wall_s": 10.0,
                "output_tok_s": 800.0,
            }
        ),
        encoding="utf-8",
    )

    loaded = load_bench_output_tok_s_gpu(bench, num_gpus=8)

    assert loaded["output_tok_s_gpu"] == 100.0
    assert loaded["num_prompts"] == 512


def test_phase457_preemption_delta_parser_uses_prometheus_body(tmp_path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    bodies = [
        'vllm:num_preemptions_total{engine="0",model_name="m"} 2.0\n',
        'vllm:num_preemptions_total{engine="0",model_name="m"} 5.0\n',
    ]
    metrics.write_text(
        "\n".join(json.dumps({"body": body}) for body in bodies) + "\n",
        encoding="utf-8",
    )

    assert parse_preemption_delta(metrics) == {"0": 3.0}


def test_phase457_protocol_projection_skips_model_fix_when_n512_proxy_passes(tmp_path) -> None:
    phase456_csv = tmp_path / "phase456.csv"
    phase456_csv.write_text(
        "\n".join(
            [
                "section,scenario,metric,value,target,status,source,note",
                "real_fingerprint,K2.5-tp8ep8-8k2k,mixed_decode_batch,51,,,events,count=1",
                "real_fingerprint,K2.5-tp8ep8-8k2k,decode_batch,59,,,events,count=1",
                "real_profile,K2.5-tp8ep8-8k2k,running_p50,56,,,serve,p10=52; p90=64",
                "capacity_counterfactual,K2.5-tp8ep8-8k2k,current_capacity_error_ratio,1.244,<=1.15,fail,,sim=166.117; real=133.528; peak_decode=67; avg_decode=54.576",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = build_rows(
        validate_row={
            "name": "K2.5-tp8ep8-8k2k",
            "real_output_tok_s_gpu": 133.528,
            "sim_output_tok_s_gpu": 166.1170248292807,
            "error_ratio": 1.2440613566389125,
        },
        off_bench={
            "num_prompts": 512,
            "max_concurrency": 128,
            "wall_s": 874.0,
            "output_tok_s": 1171.0122826772538,
            "output_tok_s_gpu": 146.37653533465672,
        },
        on_bench={
            "num_prompts": 512,
            "max_concurrency": 128,
            "wall_s": 875.0,
            "output_tok_s": 1170.5273652241867,
            "output_tok_s_gpu": 146.31592065302334,
        },
        preemption_delta={"0": 0.0},
        sim_preemption={"preemptions": 0.0, "recompute_tokens": 0.0, "throughput_tok_s_gpu": 166.0},
        phase456_csv=phase456_csv,
    )

    assert _row(rows, "error_vs_n128_reference")["status"] == "fail"
    assert _row(rows, "error_vs_n512_proxy_overhead_off")["status"] == "pass"
    assert _row(rows, "sim_avg_decode_batch")["status"] == "pass"
    assert _row(rows, "phase457_verdict")["value"] == (
        "tp8_protocol_unification_dominates_recollect_vanilla_n512"
    )


def test_phase457_protocol_projection_records_secondary_preemption_residual(tmp_path) -> None:
    phase456_csv = tmp_path / "phase456.csv"
    phase456_csv.write_text(
        "\n".join(
            [
                "section,scenario,metric,value,target,status,source,note",
                "real_fingerprint,K2.5-tp8ep8-8k2k,mixed_decode_batch,51,,,events,count=1",
                "real_fingerprint,K2.5-tp8ep8-8k2k,decode_batch,59,,,events,count=1",
                "real_profile,K2.5-tp8ep8-8k2k,running_p50,56,,,serve,p10=52; p90=64",
                "capacity_counterfactual,K2.5-tp8ep8-8k2k,current_capacity_error_ratio,1.244,<=1.15,fail,,sim=166.117; real=133.528; peak_decode=67; avg_decode=54.576",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = build_rows(
        validate_row={
            "name": "K2.5-tp8ep8-8k2k",
            "real_output_tok_s_gpu": 133.528,
            "sim_output_tok_s_gpu": 166.1170248292807,
            "error_ratio": 1.2440613566389125,
        },
        off_bench={
            "num_prompts": 512,
            "max_concurrency": 128,
            "wall_s": 874.0,
            "output_tok_s": 1171.0122826772538,
            "output_tok_s_gpu": 146.37653533465672,
        },
        on_bench={
            "num_prompts": 512,
            "max_concurrency": 128,
            "wall_s": 875.0,
            "output_tok_s": 1170.5273652241867,
            "output_tok_s_gpu": 146.31592065302334,
        },
        preemption_delta={"0": 110.0},
        sim_preemption={"preemptions": 189.0, "recompute_tokens": 1_588_814.0, "throughput_tok_s_gpu": 169.156},
        phase456_csv=phase456_csv,
    )

    assert _row(rows, "sim_preemptions")["status"] == "warn"
    assert _row(rows, "phase457_verdict")["value"] == (
        "tp8_protocol_unification_closes_gate_with_secondary_preemption_residual"
    )
