from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "export_phase466_cb_sim_iterations.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "export_phase466_cb_sim_iterations", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_normalize_rows_uses_rank_scope_and_progress_windows() -> None:
    export = _load_module()
    rows = [
        SimpleNamespace(
            iter_index=1,
            prefill_requests=2,
            prefill_tokens=64000,
            decode_batch_size=4,
            iter_lat_ms=10.0,
            clock_ms=10.0,
            ctx_non_attn_ms=2.0,
            ctx_attn_ms=3.0,
            gen_non_attn_ms=4.0,
            gen_attn_ms=1.0,
            running_requests=6,
            waiting_requests=122,
            completed_requests=0,
        ),
        SimpleNamespace(
            iter_index=2,
            prefill_requests=0,
            prefill_tokens=0,
            decode_batch_size=128,
            iter_lat_ms=5.0,
            clock_ms=15.0,
            ctx_non_attn_ms=0.0,
            ctx_attn_ms=0.0,
            gen_non_attn_ms=3.0,
            gen_attn_ms=2.0,
            running_requests=128,
            waiting_requests=0,
            completed_requests=128,
        ),
    ]

    normalized = export.normalize_rows(
        rows,
        run_id="sim-test",
        workload_cohort_digest="a" * 64,
        expected_num_prompts=128,
    )

    assert normalized[0]["source"] == "sim"
    assert normalized[0]["rank_id"] == 0
    assert normalized[0]["rank_scope"] == "dp_rank"
    assert normalized[0]["progress_start_tokens"] == 0
    assert normalized[0]["progress_end_tokens"] == 64004
    assert normalized[0]["progress_window_id"] == 0
    assert normalized[1]["progress_window_id"] == 0
    assert normalized[1]["scheduled_decode_tokens"] == 128
    assert normalized[1]["iteration_start_offset_ms"] == 10.0
    assert normalized[1]["iteration_end_offset_ms"] == 15.0
    assert normalized[0]["sim_component_cost_ms"] == 10.0
    assert "preemptions" not in normalized[0]
    assert "prefill_chunk_token_histogram" not in normalized[0]
    assert "decode_kv_token_sum" not in normalized[0]


def test_normalize_rows_rejects_nonmonotonic_simulator_clock() -> None:
    export = _load_module()
    rows = [
        SimpleNamespace(
            iter_index=1,
            prefill_requests=1,
            prefill_tokens=1,
            decode_batch_size=0,
            iter_lat_ms=2.0,
            clock_ms=1.0,
            ctx_non_attn_ms=1.0,
            ctx_attn_ms=1.0,
            gen_non_attn_ms=0.0,
            gen_attn_ms=0.0,
            running_requests=1,
            waiting_requests=0,
            completed_requests=0,
        )
    ]

    try:
        export.normalize_rows(
            rows,
            run_id="sim-test",
            workload_cohort_digest="a" * 64,
            expected_num_prompts=1,
        )
    except ValueError as exc:
        assert str(exc) == "simulator_clock_mismatch:1"
    else:
        raise AssertionError("expected simulator clock validation failure")


def test_normalize_rows_rejects_incomplete_simulation() -> None:
    export = _load_module()
    row = SimpleNamespace(
        iter_index=1,
        prefill_requests=1,
        prefill_tokens=1,
        decode_batch_size=0,
        iter_lat_ms=1.0,
        clock_ms=1.0,
        ctx_non_attn_ms=1.0,
        ctx_attn_ms=0.0,
        gen_non_attn_ms=0.0,
        gen_attn_ms=0.0,
        running_requests=1,
        waiting_requests=0,
        completed_requests=0,
    )

    with pytest.raises(ValueError, match="simulator_incomplete_requests"):
        export.normalize_rows(
            [row],
            run_id="sim-test",
            workload_cohort_digest="a" * 64,
            expected_num_prompts=1,
        )


def test_multi_replica_trace_preserves_rank_local_progress() -> None:
    export = _load_module()
    trace = [
        {
            "replica_id": 0,
            "local_iter": 1,
            "start_ms": 0.0,
            "end_ms": 10.0,
            "prefill_reqs": 1,
            "prefill_tokens": 32000,
            "decode_reqs": 0,
            "total_tokens": 32000,
        },
        {
            "replica_id": 1,
            "local_iter": 1,
            "start_ms": 0.0,
            "end_ms": 10.0,
            "prefill_reqs": 1,
            "prefill_tokens": 31000,
            "decode_reqs": 1,
            "total_tokens": 31001,
        },
    ]

    rows = export.normalize_multi_replica_trace(
        trace,
        run_id="sim-dp2",
        workload_cohort_digest="a" * 64,
        expected_dp=2,
    )

    assert [row["rank_id"] for row in rows] == [0, 1]
    assert all(row["rank_scope"] == "dp_rank" for row in rows)
    assert rows[0]["progress_start_tokens"] == 0
    assert rows[1]["progress_start_tokens"] == 0
    assert rows[1]["scheduled_decode_tokens"] == 1


def test_configure_diagnose_locks_exact_vllm_database() -> None:
    export = _load_module()
    diagnose = SimpleNamespace(BACKEND="vllm", SYSTEM="h200_sxm", DB_VERSION="0.12.0")

    export.configure_diagnose(diagnose)

    assert diagnose.DB_VERSION == "0.19.0"

    with pytest.raises(ValueError, match="unexpected_simulator_backend"):
        export.configure_diagnose(
            SimpleNamespace(BACKEND="trtllm", SYSTEM="h200_sxm", DB_VERSION="0.19.0")
        )


def test_dp_trace_rejects_legacy_single_replica_backend_path() -> None:
    export = _load_module()

    with pytest.raises(
        ValueError, match="unsupported_dp_trace_backend_legacy_single_replica"
    ):
        export.validate_scenario_exportability(
            {
                "scenario": "K2.5-tp4ep8dp2-8k2k-bt65536",
                "dp": 2,
                "input_len": 8000,
                "max_num_batched_tokens": 65536,
            }
        )
