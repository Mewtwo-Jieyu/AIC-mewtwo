import json
from pathlib import Path

import pytest

import scripts.analyze_phase423_bt65536_recollect as phase423


def _write_phase423_raw(root: Path) -> None:
    scenario_dir = root / phase423.SCENARIO
    scenario_dir.mkdir(parents=True)
    (scenario_dir / "meta.json").write_text(
        json.dumps(
            {
                "name": phase423.SCENARIO,
                "phase": "phase423",
                "tp": 4,
                "dp": 2,
                "ep": 8,
                "isl": 8000,
                "osl": 2000,
                "max_num_batched_tokens": 65536,
                "batch_size": 128,
                "world_size": 8,
                "prefix_caching": False,
                "prompt_variant_mode": "rotating",
                "gpu_memory_utilization": 0.8,
                "max_model_len": 262144,
                "max_num_seqs": 256,
                "enable_logging_iteration_details": True,
                "cudagraph_metrics": True,
            }
        )
    )
    (scenario_dir / "serve.log").write_text(
        "\n".join(
            [
                "INFO non-default args: {'max_model_len': 262144, 'data_parallel_size': 2, "
                "'gpu_memory_utilization': 0.8, 'enable_prefix_caching': False, "
                "'cudagraph_metrics': True, 'enable_logging_iteration_details': True, "
                "'max_num_batched_tokens': 65536, 'max_num_seqs': 256}",
                "ValueError: To serve at least one request with the models's max seq len "
                "(262144), (17.16 GiB KV cache is needed, which is larger than the "
                "available KV cache memory (8.62 GiB). Based on the available memory, "
                "the estimated maximum model length is 131648.",
            ]
        )
    )
    (scenario_dir / "metrics.jsonl").write_text("")
    (root / "gpu_compute_apps_after.txt").write_text("")
    (root / "process_residual_after.txt").write_text("")
    (root / "outer.log").write_text("service_exited_before_ready\n")


def test_phase423_summarizes_kv_init_failure_without_reference_replacement(tmp_path: Path) -> None:
    _write_phase423_raw(tmp_path)

    rows = phase423.build_phase423_rows(raw_root=tmp_path)

    summary = next(row for row in rows if row["row_type"] == "summary")
    failure = next(row for row in rows if row["row_type"] == "failure")
    config = next(row for row in rows if row["row_type"] == "config")

    assert config["prompt_variant_mode"] == "rotating"
    assert config["prefix_caching"] == "false"
    assert failure["needed_kv_cache_gib"] == "17.160000"
    assert failure["available_kv_cache_gib"] == "8.620000"
    assert failure["estimated_max_model_len"] == "131648"
    assert summary["collection_status"] == "service_init_failed"
    assert summary["mechanism_verdict"] == "clean_recollect_blocked_by_kv_init_capacity"
    assert summary["reference_action"] == "no_reference_replacement"
    assert summary["ab_action"] == "not_run_no_bench_result"
    assert summary["large_step_trace_action"] == "not_run_service_not_ready"
    assert summary["default_readiness"] == "No-Go"


def test_phase423_writer_rejects_default_readiness(tmp_path: Path) -> None:
    _write_phase423_raw(tmp_path)
    rows = phase423.build_phase423_rows(raw_root=tmp_path)
    rows[0]["valid_for_default"] = "true"

    with pytest.raises(ValueError, match="valid_for_default"):
        phase423.write_phase423_csv(tmp_path / "bad.csv", rows)
