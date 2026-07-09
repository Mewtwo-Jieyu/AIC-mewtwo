import json

from scripts.analyze_phase458_n512_unify import (
    build_acceptance_rows,
    load_bench_summary,
    parse_kv_cache_tokens,
)


def _row(rows, scenario, metric):
    return next(
        row for row in rows if row["scenario"] == scenario and row["metric"] == metric
    )


def test_phase458_bench_loader_reports_per_gpu_output(tmp_path) -> None:
    bench = tmp_path / "bench_result.json"
    bench.write_text(
        json.dumps(
            {
                "num_prompts": 512,
                "max_concurrency": 128,
                "ok_requests": 512,
                "failed_requests": 0,
                "wall_s": 10.0,
                "output_tok_s": 800.0,
            }
        ),
        encoding="utf-8",
    )

    loaded = load_bench_summary(bench, num_gpus=8)

    assert loaded["output_tok_s_gpu"] == 100.0
    assert loaded["ok_requests"] == 512


def test_phase458_parse_kv_cache_tokens(tmp_path) -> None:
    serve_log = tmp_path / "serve.log"
    serve_log.write_text(
        "INFO [kv_cache_utils.py:1319] GPU KV cache size: 546,160 tokens\n",
        encoding="utf-8",
    )

    assert parse_kv_cache_tokens(serve_log) == 546_160


def test_phase458_acceptance_rows_mark_gate_result() -> None:
    rows = build_acceptance_rows(
        [
            {
                "name": "K2.5-tp8ep8-8k2k",
                "current_error_ratio": "1.13",
                "classification": "improved",
            },
            {
                "name": "K2.5-tp8ep8-32k3k",
                "current_error_ratio": "1.04",
                "classification": "unchanged",
            },
        ]
    )

    assert _row(rows, "K2.5-tp8ep8-8k2k", "current_error_ratio")["status"] == "pass"
    assert _row(rows, "MULTI_CONFIG", "phase458_gate")["value"] == "pass"


def test_phase458_acceptance_rows_fail_when_any_point_exceeds_target() -> None:
    rows = build_acceptance_rows(
        [
            {
                "name": "K2.5-tp8ep8-8k2k",
                "current_error_ratio": "1.16",
                "classification": "regressed",
            }
        ]
    )

    assert _row(rows, "K2.5-tp8ep8-8k2k", "current_error_ratio")["status"] == "fail"
    assert _row(rows, "MULTI_CONFIG", "phase458_gate")["value"] == "fail"
