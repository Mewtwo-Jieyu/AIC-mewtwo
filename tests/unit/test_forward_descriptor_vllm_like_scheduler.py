from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pytest

from aiconfigurator.sdk.backends.cb_simulator.forward_descriptor import (
    compare_scheduler_aligned_descriptors,
    vllm_like_scheduler_aligned_descriptors,
)


def _load_diagnose_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / (
        "diagnose_cb_iter_latency.py"
    )
    spec = importlib.util.spec_from_file_location("diagnose_cb_iter_latency", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rows(**overrides):
    kwargs = {
        "source": "cb_sim_vllm_like",
        "scenario": "10k2k_b32_bt8192",
        "isl": 10000,
        "osl": 2000,
        "concurrency": 32,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 256,
        "tp": 4,
        "dp": 2,
        "moe_tp": 1,
        "moe_ep": 8,
        "block_size": 16,
    }
    kwargs.update(overrides)
    return vllm_like_scheduler_aligned_descriptors(**kwargs)


def _short_rows(**overrides):
    kwargs = {
        "source": "cb_sim_vllm_like",
        "scenario": "3k3k_b128_bt8192",
        "isl": 3000,
        "osl": 3000,
        "concurrency": 128,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 256,
        "tp": 4,
        "dp": 2,
        "moe_tp": 1,
        "moe_ep": 8,
        "block_size": 16,
    }
    kwargs.update(overrides)
    return vllm_like_scheduler_aligned_descriptors(**kwargs)


def _long_decode_rows(**overrides):
    kwargs = {
        "source": "cb_sim_vllm_like",
        "scenario": "32k1k_b16_bt8192",
        "isl": 32000,
        "osl": 1000,
        "concurrency": 16,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 256,
        "tp": 4,
        "dp": 2,
        "moe_tp": 1,
        "moe_ep": 8,
        "block_size": 16,
    }
    kwargs.update(overrides)
    return vllm_like_scheduler_aligned_descriptors(**kwargs)


def _index(rows):
    return {(row.alignment_key, row.dp_rank): row for row in rows}


def test_vllm_like_scheduler_matches_phase62_first_evidence_shape() -> None:
    rows = _rows()
    index = _index(rows)

    assert len(rows) == 4003
    assert Counter(row.dp_rank for row in rows) == Counter({0: 2001, 1: 2002})
    assert Counter(row.phase for row in rows) == Counter(
        {"prefill": 4, "mixed": 1, "pure_decode": 3998}
    )

    dp0_step1 = index[("engine_dp:0:step:1", 0)]
    assert dp0_step1.phase == "prefill"
    assert dp0_step1.scheduled_context_tokens == 2048
    assert dp0_step1.scheduled_context_reqs == 16
    assert dp0_step1.forward_regime == "NONE:2048"

    dp1_step1 = index[("engine_dp:1:step:1", 1)]
    assert dp1_step1.phase == "prefill"
    assert dp1_step1.scheduled_context_tokens == 1808
    assert dp1_step1.scheduled_context_reqs == 1
    assert dp1_step1.forward_regime == "NONE:1808"

    dp1_step2 = index[("engine_dp:1:step:2", 1)]
    assert dp1_step2.phase == "mixed"
    assert dp1_step2.scheduled_context_tokens == 240
    assert dp1_step2.scheduled_decode_tokens == 1
    assert dp1_step2.scheduled_context_reqs == 15
    assert dp1_step2.scheduled_decode_reqs == 1
    assert dp1_step2.forward_token_count == 248
    assert dp1_step2.forward_regime == "NONE:248"

    dp1_tail = index[("engine_dp:1:step:2001", 1)]
    assert dp1_tail.phase == "pure_decode"
    assert dp1_tail.scheduled_decode_tokens == 15
    assert dp1_tail.scheduled_decode_reqs == 15
    assert dp1_tail.forward_token_count == 16
    assert dp1_tail.forward_regime == "FULL:16"


def test_vllm_like_scheduler_matches_phase69_short_isl_holdout_shape() -> None:
    rows = _short_rows()
    index = _index(rows)

    assert len(rows) == 6002
    assert Counter(row.dp_rank for row in rows) == Counter({0: 3002, 1: 3000})
    assert Counter(row.phase for row in rows) == Counter(
        {"prefill": 2, "mixed": 1, "pure_decode": 5999}
    )

    dp0_step0 = index[("engine_dp:0:step:0", 0)]
    assert dp0_step0.phase == "prefill"
    assert dp0_step0.scheduled_context_tokens == 3000
    assert dp0_step0.scheduled_context_reqs == 1
    assert dp0_step0.forward_regime == "NONE:3000"

    dp0_step1 = index[("engine_dp:0:step:1", 0)]
    assert dp0_step1.phase == "pure_decode"
    assert dp0_step1.scheduled_decode_tokens == 1
    assert dp0_step1.scheduled_decode_reqs == 1
    assert dp0_step1.forward_regime == "NONE:1"

    dp0_step2 = index[("engine_dp:0:step:2", 0)]
    assert dp0_step2.phase == "mixed"
    assert dp0_step2.scheduled_context_tokens == 496
    assert dp0_step2.scheduled_decode_tokens == 1
    assert dp0_step2.scheduled_context_reqs == 62
    assert dp0_step2.scheduled_decode_reqs == 1
    assert dp0_step2.forward_regime == "PIECEWISE:512"

    dp1_step0 = index[("engine_dp:1:step:0", 1)]
    assert dp1_step0.phase == "prefill"
    assert dp1_step0.scheduled_context_tokens == 3512
    assert dp1_step0.scheduled_context_reqs == 65
    assert dp1_step0.forward_regime == "NONE:3512"

    dp1_step1 = index[("engine_dp:1:step:1", 1)]
    assert dp1_step1.phase == "pure_decode"
    assert dp1_step1.scheduled_decode_tokens == 65
    assert dp1_step1.scheduled_decode_reqs == 65
    assert dp1_step1.forward_regime == "PIECEWISE:512"

    dp1_step2 = index[("engine_dp:1:step:2", 1)]
    assert dp1_step2.phase == "pure_decode"
    assert dp1_step2.scheduled_decode_tokens == 65
    assert dp1_step2.scheduled_decode_reqs == 65
    assert dp1_step2.forward_regime == "FULL:72"

    dp0_tail_0 = index[("engine_dp:0:step:3000", 0)]
    assert dp0_tail_0.scheduled_decode_tokens == 62
    assert dp0_tail_0.forward_regime == "FULL:72"

    dp0_tail_1 = index[("engine_dp:0:step:3001", 0)]
    assert dp0_tail_1.scheduled_decode_tokens == 62
    assert dp0_tail_1.forward_regime == "FULL:64"

    dp1_tail = index[("engine_dp:1:step:2999", 1)]
    assert dp1_tail.scheduled_decode_tokens == 65
    assert dp1_tail.forward_regime == "FULL:72"


def test_vllm_like_scheduler_matches_phase71_long_decode_holdout_shape() -> None:
    rows = _long_decode_rows()
    index = _index(rows)

    assert len(rows) == 2006
    assert Counter(row.dp_rank for row in rows) == Counter({0: 1003, 1: 1003})
    assert Counter(row.phase for row in rows) == Counter(
        {"prefill": 8, "pure_decode": 1998}
    )

    for dp_rank in (0, 1):
        for step in (0, 1, 2):
            row = index[(f"engine_dp:{dp_rank}:step:{step}", dp_rank)]
            assert row.phase == "prefill"
            assert row.scheduled_context_tokens == 8192
            assert row.scheduled_context_reqs == 1
            assert row.forward_regime == "NONE:8192"

        step3 = index[(f"engine_dp:{dp_rank}:step:3", dp_rank)]
        assert step3.phase == "prefill"
        assert step3.scheduled_context_tokens == 7536
        assert step3.scheduled_context_reqs == 8
        assert step3.forward_regime == "NONE:7536"

    dp0_step4 = index[("engine_dp:0:step:4", 0)]
    assert dp0_step4.phase == "pure_decode"
    assert dp0_step4.scheduled_decode_tokens == 8
    assert dp0_step4.scheduled_decode_reqs == 8
    assert dp0_step4.forward_regime == "FULL:8"

    dp1_step4 = index[("engine_dp:1:step:4", 1)]
    assert dp1_step4.phase == "pure_decode"
    assert dp1_step4.scheduled_decode_tokens == 8
    assert dp1_step4.scheduled_decode_reqs == 8
    assert dp1_step4.forward_regime == "NONE:8"

    for dp_rank in (0, 1):
        tail = index[(f"engine_dp:{dp_rank}:step:1002", dp_rank)]
        assert tail.phase == "pure_decode"
        assert tail.scheduled_decode_tokens == 8
        assert tail.scheduled_decode_reqs == 8
        assert tail.forward_regime == "FULL:8"


def test_vllm_like_scheduler_emits_only_descriptor_fields() -> None:
    row = _index(_rows())[("engine_dp:1:step:2", 1)]
    values = asdict(row)

    assert row.valid_for_default is False
    assert row.perf_database is False
    assert row.diagnostic_only is True
    assert "AIC_UNSET" not in row.forward_regime
    assert "AIC_UNSET" not in row.cudagraph_runtime_mode
    for forbidden in (
        "latency_ms",
        "duration_ms",
        "residual_ms",
        "profiled_cuda_time_ms",
        "nccl_trace_line_count",
        "sync_wait_ms",
        "throughput_ratio",
    ):
        assert forbidden not in values


def test_vllm_like_scheduler_can_strict_compare_identical_key_set() -> None:
    rows = _rows()
    compare_rows = compare_scheduler_aligned_descriptors(rows, rows)

    assert len(compare_rows) == 4003
    assert all(row.same_phase == 1 for row in compare_rows)
    assert all(row.same_scheduled_total_tokens == 1 for row in compare_rows)
    assert all(row.same_scheduled_total_reqs == 1 for row in compare_rows)
    assert all(row.same_forward_token_count == 1 for row in compare_rows)
    assert all(row.same_forward_regime == 1 for row in compare_rows)
    assert all(row.same_cudagraph_runtime_mode == 1 for row in compare_rows)


def test_vllm_like_scheduler_short_isl_can_strict_compare_identical_key_set() -> None:
    rows = _short_rows()
    compare_rows = compare_scheduler_aligned_descriptors(rows, rows)

    assert len(compare_rows) == 6002
    assert all(row.same_phase == 1 for row in compare_rows)
    assert all(row.same_scheduled_total_tokens == 1 for row in compare_rows)
    assert all(row.same_scheduled_total_reqs == 1 for row in compare_rows)
    assert all(row.same_forward_token_count == 1 for row in compare_rows)
    assert all(row.same_forward_regime == 1 for row in compare_rows)
    assert all(row.same_cudagraph_runtime_mode == 1 for row in compare_rows)


def test_vllm_like_scheduler_long_decode_can_strict_compare_identical_key_set() -> None:
    rows = _long_decode_rows()
    compare_rows = compare_scheduler_aligned_descriptors(rows, rows)

    assert len(compare_rows) == 2006
    assert all(row.same_phase == 1 for row in compare_rows)
    assert all(row.same_scheduled_total_tokens == 1 for row in compare_rows)
    assert all(row.same_scheduled_total_reqs == 1 for row in compare_rows)
    assert all(row.same_forward_token_count == 1 for row in compare_rows)
    assert all(row.same_forward_regime == 1 for row in compare_rows)
    assert all(row.same_cudagraph_runtime_mode == 1 for row in compare_rows)


def test_vllm_like_scheduler_fail_fast_for_unsupported_shapes() -> None:
    with pytest.raises(ValueError, match="divisible by dp"):
        _rows(concurrency=33)

    with pytest.raises(ValueError, match="short-ISL"):
        _rows(isl=8192)

    with pytest.raises(ValueError, match="short-ISL"):
        _short_rows(concurrency=64)

    with pytest.raises(ValueError, match=">=2 reqs per DP"):
        _rows(concurrency=2)

    with pytest.raises(ValueError, match="exceed token budget"):
        _rows(max_num_batched_tokens=2048)

    with pytest.raises(ValueError, match="32k1k"):
        _long_decode_rows(concurrency=32)


def test_diagnose_builder_uses_vllm_like_scheduler_without_trace_collection() -> None:
    module = _load_diagnose_module()
    args = argparse.Namespace(
        isl=10000,
        osl=2000,
        concurrency=32,
        tp=4,
        dp=2,
        moe_tp=1,
        moe_ep=8,
        max_num_batched_tokens=8192,
        max_num_seqs=256,
        num_requests=0,
        warmup_requests=0,
        long_prefill_token_threshold=0,
        num_gpu_blocks=0,
        block_size=16,
        overlap_factor=0.0,
        per_iteration_overhead_ms=0.0,
    )

    rows = module.build_cb_vllm_like_scheduler_aligned_descriptors(args)

    assert len(rows) == 4003
    assert rows[0].source == "cb_sim_vllm_like"
    assert rows[0].scenario == "10000x2000_b32"
    assert rows[0].alignment_key == "engine_dp:0:step:0"
