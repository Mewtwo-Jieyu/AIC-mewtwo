from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path

import pytest

from aiconfigurator.sdk.backends.cb_simulator.forward_descriptor import (
    compare_scheduler_aligned_descriptors,
    scheduler_aligned_descriptor_from_scheduled,
    scheduler_aligned_descriptor_from_csv_row,
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


def _base_kwargs() -> dict[str, object]:
    return {
        "source": "cb_sim",
        "scenario": "10k2k_b32_bt8192",
        "dp_rank": 1,
        "engine_step_id": 7,
        "phase": "mixed",
        "scheduled_context_tokens": 225,
        "scheduled_decode_tokens": 16,
        "scheduled_context_reqs": 15,
        "scheduled_decode_reqs": 1,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 32,
        "forward_token_count": 248,
        "cudagraph_runtime_mode": "NONE",
        "topology_key": "tp4dp2moetp1ep8",
        "tp": 4,
        "dp": 2,
        "moe_tp": 1,
        "moe_ep": 8,
    }


def _descriptor(**overrides: object):
    kwargs = _base_kwargs()
    kwargs.update(overrides)
    return scheduler_aligned_descriptor_from_scheduled(**kwargs)


def test_aligned_descriptor_flattens_alignment_and_scheduler_fields() -> None:
    descriptor = _descriptor()

    values = asdict(descriptor)
    assert values == {
        "alignment_key": "engine_dp:1:step:7",
        "alignment_key_type": "engine_core_dp_step",
        "engine_step_id": 7,
        "dp_rank": 1,
        "source": "cb_sim",
        "scenario": "10k2k_b32_bt8192",
        "iteration": 7,
        "phase": "mixed",
        "scheduled_context_tokens": 225,
        "scheduled_decode_tokens": 16,
        "scheduled_total_tokens": 241,
        "scheduled_context_reqs": 15,
        "scheduled_decode_reqs": 1,
        "scheduled_total_reqs": 16,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 32,
        "forward_token_count": 248,
        "forward_regime": "NONE:248",
        "cudagraph_runtime_mode": "NONE",
        "topology_key": "tp4dp2moetp1ep8",
        "tp": 4,
        "dp": 2,
        "moe_tp": 1,
        "moe_ep": 8,
        "valid_for_default": False,
        "perf_database": False,
        "diagnostic_only": True,
    }
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


def test_aligned_descriptor_rejects_invalid_alignment_fields() -> None:
    with pytest.raises(ValueError, match="engine_step_id"):
        _descriptor(engine_step_id=-1)

    with pytest.raises(ValueError, match="dp_rank"):
        _descriptor(dp_rank=-1)

    with pytest.raises(ValueError, match="dp_rank"):
        _descriptor(dp_rank=2)


def test_aligned_descriptor_keeps_iteration_equal_to_engine_step_id() -> None:
    descriptor = _descriptor(engine_step_id=0, dp_rank=0)

    assert descriptor.alignment_key == "engine_dp:0:step:0"
    assert descriptor.iteration == 0
    assert descriptor.engine_step_id == 0


def test_aligned_descriptor_inherits_scheduler_validation() -> None:
    with pytest.raises(ValueError, match="phase"):
        _descriptor(phase="decode")

    with pytest.raises(ValueError, match="forward_token_count"):
        _descriptor(forward_token_count=240)

    with pytest.raises(ValueError, match="topology_key"):
        _descriptor(topology_key="tp4dp2moetp4ep8")


def test_cb_sim_alignment_builder_uses_dp_local_ordinal_not_iter_index() -> None:
    module = _load_diagnose_module()
    rows = [
        module.CBIterationTraceRow(
            iter_index=10,
            phase_type="prefill",
            trace_type="scheduled",
            prefill_requests=1,
            prefill_tokens=8192,
            decode_batch_size=0,
            decode_avg_kv_len=0,
            iter_lat_ms=1.0,
            ctx_non_attn_ms=0.0,
            ctx_attn_ms=0.0,
            gen_non_attn_ms=0.0,
            gen_attn_ms=0.0,
            clock_ms=1.0,
            completed_requests=0,
            running_requests=1,
            waiting_requests=31,
            in_steady_state=0,
        ),
        module.CBIterationTraceRow(
            iter_index=99,
            phase_type="mixed",
            trace_type="scheduled",
            prefill_requests=15,
            prefill_tokens=225,
            decode_batch_size=16,
            decode_avg_kv_len=10000,
            iter_lat_ms=2.0,
            ctx_non_attn_ms=0.0,
            ctx_attn_ms=0.0,
            gen_non_attn_ms=0.0,
            gen_attn_ms=0.0,
            clock_ms=3.0,
            completed_requests=0,
            running_requests=16,
            waiting_requests=0,
            in_steady_state=1,
        ),
    ]
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
        num_requests=96,
        warmup_requests=32,
        long_prefill_token_threshold=0,
        num_gpu_blocks=0,
        block_size=16,
        overlap_factor=1.0,
        per_iteration_overhead_ms=0.0,
    )

    descriptors = module.build_cb_scheduler_aligned_descriptors(
        rows,
        args,
        dp_rank=1,
    )

    assert [row.engine_step_id for row in descriptors] == [0, 1]
    assert [row.iteration for row in descriptors] == [0, 1]
    assert [row.alignment_key for row in descriptors] == [
        "engine_dp:1:step:0",
        "engine_dp:1:step:1",
    ]


def test_diagnose_parser_reads_scheduler_aligned_descriptor_csv(tmp_path: Path) -> None:
    module = _load_diagnose_module()
    descriptor = _descriptor(source="vllm_scheduler", engine_step_id=2, dp_rank=1)
    csv_path = tmp_path / "vllm_aligned.csv"
    values = asdict(descriptor)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(values))
        writer.writeheader()
        writer.writerow(values)

    rows = module.parse_scheduler_aligned_descriptors(csv_path)

    assert rows == [descriptor]


def test_compare_scheduler_aligned_descriptors_joins_only_by_alignment_key() -> None:
    cb_row = _descriptor(
        source="cb_sim",
        engine_step_id=2,
        dp_rank=1,
        forward_token_count=241,
        cudagraph_runtime_mode="AIC_UNSET",
    )
    vllm_row = _descriptor(
        source="vllm_scheduler",
        engine_step_id=2,
        dp_rank=1,
        forward_token_count=248,
        cudagraph_runtime_mode="NONE",
    )

    rows = compare_scheduler_aligned_descriptors([cb_row], [vllm_row])

    assert len(rows) == 1
    row = rows[0]
    assert row.alignment_key == "engine_dp:1:step:2"
    assert row.alignment_key_type == "engine_core_dp_step"
    assert row.dp_rank == 1
    assert row.phase == "mixed"
    assert row.cb_engine_step_id == 2
    assert row.vllm_engine_step_id == 2
    assert row.scheduled_total_token_delta == 0
    assert row.scheduled_total_req_delta == 0
    assert row.forward_token_delta == 7
    assert row.same_scheduled_total_tokens == 1
    assert row.same_scheduled_total_reqs == 1
    assert row.same_forward_token_count == 0
    assert row.same_forward_regime == 0
    assert row.same_cudagraph_runtime_mode == 0
    assert row.same_topology_key == 1
    assert row.valid_for_default is False
    assert row.perf_database is False
    assert row.diagnostic_only is True
    for forbidden in (
        "latency_ms",
        "duration_ms",
        "residual_ms",
        "profiled_cuda_time_ms",
        "nccl_trace_line_count",
        "sync_wait_ms",
        "throughput_ratio",
    ):
        assert forbidden not in asdict(row)


def test_compare_scheduler_aligned_descriptors_rejects_missing_or_duplicate_keys() -> None:
    cb_row = _descriptor(source="cb_sim", engine_step_id=2, dp_rank=1)
    vllm_row = _descriptor(source="vllm_scheduler", engine_step_id=3, dp_rank=1)

    with pytest.raises(ValueError, match="missing vllm alignment keys"):
        compare_scheduler_aligned_descriptors([cb_row], [vllm_row])

    with pytest.raises(ValueError, match="missing cb alignment keys"):
        compare_scheduler_aligned_descriptors([cb_row], [cb_row, vllm_row])

    with pytest.raises(ValueError, match="duplicate cb alignment key"):
        compare_scheduler_aligned_descriptors([cb_row, cb_row], [cb_row])

    with pytest.raises(ValueError, match="duplicate vllm alignment key"):
        compare_scheduler_aligned_descriptors([cb_row], [cb_row, cb_row])


def test_compare_scheduler_aligned_descriptors_rejects_non_engine_core_key_type() -> None:
    cb_row = _descriptor(source="cb_sim", engine_step_id=2, dp_rank=1)
    bad_vllm_row = replace(
        _descriptor(source="vllm_scheduler", engine_step_id=2, dp_rank=1),
        alignment_key_type="phase_ordinal",
    )

    with pytest.raises(ValueError, match="alignment_key_type"):
        compare_scheduler_aligned_descriptors([cb_row], [bad_vllm_row])


def test_scheduler_aligned_descriptor_from_csv_row_rebuilds_and_validates_row() -> None:
    descriptor = _descriptor(source="vllm_scheduler", engine_step_id=2, dp_rank=1)
    raw = {key: str(value) for key, value in asdict(descriptor).items()}

    parsed = scheduler_aligned_descriptor_from_csv_row(raw)

    assert parsed == descriptor


def test_scheduler_aligned_descriptor_from_csv_row_rejects_forbidden_or_bad_flags() -> None:
    descriptor = _descriptor(source="vllm_scheduler", engine_step_id=2, dp_rank=1)
    raw = {key: str(value) for key, value in asdict(descriptor).items()}
    raw["latency_ms"] = "1.0"

    with pytest.raises(ValueError, match="forbidden field"):
        scheduler_aligned_descriptor_from_csv_row(raw)

    raw = {key: str(value) for key, value in asdict(descriptor).items()}
    raw["valid_for_default"] = "true"
    with pytest.raises(ValueError, match="valid_for_default"):
        scheduler_aligned_descriptor_from_csv_row(raw)
