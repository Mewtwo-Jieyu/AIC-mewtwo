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


def test_normalize_rows_uses_global_simulator_scope_and_progress_windows() -> None:
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
            completed_requests=0,
        ),
    ]

    normalized = export.normalize_rows(
        rows,
        run_id="sim-test",
        workload_cohort_digest="a" * 64,
    )

    assert normalized[0]["source"] == "sim"
    assert normalized[0]["rank_id"] == 0
    assert normalized[0]["rank_scope"] == "global_simulator"
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
        export.normalize_rows(rows, run_id="sim-test", workload_cohort_digest="a" * 64)
    except ValueError as exc:
        assert str(exc) == "simulator_clock_mismatch:1"
    else:
        raise AssertionError("expected simulator clock validation failure")


def test_configure_diagnose_locks_exact_vllm_database() -> None:
    export = _load_module()
    diagnose = SimpleNamespace(BACKEND="vllm", SYSTEM="h200_sxm", DB_VERSION="0.12.0")

    export.configure_diagnose(diagnose)

    assert diagnose.DB_VERSION == "0.19.0"

    with pytest.raises(ValueError, match="unexpected_simulator_backend"):
        export.configure_diagnose(
            SimpleNamespace(BACKEND="trtllm", SYSTEM="h200_sxm", DB_VERSION="0.19.0")
        )
