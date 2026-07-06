#!/usr/bin/env python3
"""Validate CB simulator against real vLLM benchmark data.

Compares CB sim predictions vs B1/B1b baseline vs real measurements.

Usage:
    python scripts/validate_cb_simulator.py

Acceptance source: vLLM 0.19 H200 SXM x8 MULTI_CONFIG points.
Legacy 0.17 tp16 throughput/TTFT data is printed but skipped from acceptance.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aiconfigurator.sdk import common
from aiconfigurator.sdk.backends.cb_simulator import CBSimConfig, CBSimulator
from aiconfigurator.sdk.backends.cb_simulator.forward_descriptor import (
    compiled_body_runtime_key_from_nccl_summary_csv,
    descriptor_from_scheduled,
    make_topology_key,
    moe_source_runtime_key_from_loaded_weight_boundary_row,
    runtime_shape_key_from_scheduled,
    scheduler_aligned_descriptor_from_scheduled,
    scheduler_runtime_descriptor_from_scheduled,
)
from aiconfigurator.sdk.backends.vllm_backend import VLLMBackend
from aiconfigurator.sdk.config import ModelConfig, RuntimeConfig
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.perf_database import PerfDatabase

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = "moonshotai/Kimi-K2.5"
SYSTEM = "h200_sxm"
BACKEND = "vllm"
VALIDATION_DB_VERSION = "0.19.0"
TP = 16
NUM_GPUS = 16
DEFAULT_EP8_PER_ITERATION_OVERHEAD_MS = 0.0
THROUGHPUT_GATE_STATUS = "legacy_skip"
TTFT_GATE_STATUS = "legacy_skip"
THROUGHPUT_MAX_ACCEPTANCE = 1.4989592822599629
MULTI_CONFIG_MAX_ACCEPTANCE = 1.50
TTFT_MAX_ACCEPTANCE = 1.790056498134038


@dataclass
class BenchmarkPoint:
    name: str
    isl: int
    osl: int
    batch_size: int
    real_output_tok_s_gpu: float
    real_total_tok_s_gpu: float | None = None
    real_ttft_ms: float | None = None


@dataclass
class MultiConfigPoint:
    name: str
    isl: int
    osl: int
    batch_size: int
    max_num_batched_tokens: int
    tp: int
    dp: int
    moe_tp: int
    moe_ep: int
    real_total_tok_s_gpu: float

    @property
    def real_output_tok_s_gpu(self) -> float:
        return self.real_total_tok_s_gpu * self.osl / (self.isl + self.osl)


@dataclass(frozen=True)
class MultiConfigKVCapacity:
    scenario: str
    kv_cache_tokens: int
    block_size: int
    num_gpu_blocks: int
    max_num_seqs: int
    serve_log: str
    serve_log_line_numbers: tuple[int, ...]
    override_num_gpu_blocks: int
    override_line_numbers: tuple[int, ...]
    capacity_scope: str = "per_engine"


@dataclass(frozen=True)
class MultiConfigTopKRow:
    name: str
    tp: int
    dp: int
    ep: int
    max_bt: int
    real_output_tok_s_gpu: float
    sim_output_tok_s_gpu: float
    error_ratio: float
    rank: int
    real_rank: int
    rank_delta: int
    diagnostic_only: bool
    valid_for_default: bool
    perf_database: bool


@dataclass(frozen=True)
class MultiConfigBudgetBreakdownRow:
    name: str
    tp: int
    dp: int
    ep: int
    max_bt: int
    real_output_tok_s_gpu: float
    sim_output_tok_s_gpu: float
    error_ratio: float
    rank: int
    real_rank: int
    avg_prefill_reqs_per_iter: float
    avg_decode_reqs_per_iter: float
    avg_tokens_per_iter: float
    peak_prefill_reqs_per_iter: float
    peak_decode_reqs_per_iter: float
    peak_tokens_per_iter: float
    steady_state_iterations: int
    steady_state_time_ms: float
    paired_baseline_name: str
    paired_baseline_max_bt: int
    paired_baseline_sim_output_tok_s_gpu: float
    paired_baseline_error_ratio: float
    paired_baseline_rank: int
    paired_baseline_avg_tokens_per_iter: float
    max_bt_vs_paired_baseline: float
    sim_vs_paired_baseline_ratio: float
    avg_tokens_vs_paired_baseline_ratio: float
    steady_state_time_vs_paired_baseline_ratio: float
    diagnostic_only: bool
    valid_for_default: bool
    perf_database: bool


@dataclass(frozen=True)
class ValidationResult:
    overlap_factor: float
    per_iteration_overhead_ms: float
    ep8_per_iteration_overhead_ms: float
    throughput_max: float
    throughput_mean: float
    multi_config_max: float
    multi_config_mean: float
    ttft_max: float
    ttft_mean: float


THROUGHPUT_DATA = [
    BenchmarkPoint("3k-3k b=128", 3000, 3000, 128, 115.9, 254.2),
    BenchmarkPoint("8k-2k b=256", 8000, 2000, 256, 117.6, 575.4),
    BenchmarkPoint("10k-2k b=32", 10000, 2000, 32, 49.8, 267.6),
    BenchmarkPoint("10k-3k b=128", 10000, 3000, 128, 89.5, 386.9),
    BenchmarkPoint("16k-2k b=32", 16000, 2000, 32, 41.7, 352.7),
    BenchmarkPoint("30k-3k b=8", 30000, 3000, 8, 18.2, 179.2),
    BenchmarkPoint("32k-1k b=16", 32000, 1000, 16, 17.8, 559.9),
]

TTFT_DATA = [
    BenchmarkPoint("30k-3k b=4", 30000, 3000, 4, 0.0, real_ttft_ms=1231.0),
    BenchmarkPoint("30k-3k b=8", 30000, 3000, 8, 0.0, real_ttft_ms=1823.0),
    BenchmarkPoint("20k-5k b=4", 20000, 5000, 4, 0.0, real_ttft_ms=1326.0),
    BenchmarkPoint("20k-5k b=8", 20000, 5000, 8, 0.0, real_ttft_ms=1649.0),
    BenchmarkPoint("16k-2k b=16", 16000, 2000, 16, 0.0, real_ttft_ms=782.0),
    BenchmarkPoint("16k-2k b=32", 16000, 2000, 32, 0.0, real_ttft_ms=814.0),
]

MULTI_CONFIG_DATA = [
    MultiConfigPoint(
        "K2.5-tp8ep8-8k2k",
        8000,
        2000,
        128,
        max_num_batched_tokens=8000,
        tp=8,
        dp=1,
        moe_tp=1,
        moe_ep=8,
        real_total_tok_s_gpu=667.64,
    ),
    MultiConfigPoint(
        "K2.5-tp8ep8-32k3k",
        32000,
        3000,
        128,
        max_num_batched_tokens=32000,
        tp=8,
        dp=1,
        moe_tp=1,
        moe_ep=8,
        real_total_tok_s_gpu=612.14,
    ),
    MultiConfigPoint(
        "K2.5-tp4ep8dp2-8k2k",
        8000,
        2000,
        128,
        max_num_batched_tokens=8000,
        tp=4,
        dp=2,
        moe_tp=1,
        moe_ep=8,
        real_total_tok_s_gpu=688.58,
    ),
    MultiConfigPoint(
        "K2.5-tp4ep8dp2-32k3k",
        32000,
        3000,
        128,
        max_num_batched_tokens=32000,
        tp=4,
        dp=2,
        moe_tp=1,
        moe_ep=8,
        real_total_tok_s_gpu=621.57,
    ),
    MultiConfigPoint(
        "K2.5-tp8ep8-8k2k-bt65536",
        8000,
        2000,
        128,
        max_num_batched_tokens=65536,
        tp=8,
        dp=1,
        moe_tp=1,
        moe_ep=8,
        real_total_tok_s_gpu=692.35,
    ),
    MultiConfigPoint(
        "K2.5-tp4ep8dp2-8k2k-bt65536",
        8000,
        2000,
        128,
        max_num_batched_tokens=65536,
        tp=4,
        dp=2,
        moe_tp=1,
        moe_ep=8,
        real_total_tok_s_gpu=569.5509309402136,
    ),
]

KV_CACHE_BLOCK_SIZE = 16
PHASE397K_KV_CAPACITY_BY_SCENARIO = {
    "K2.5-tp8ep8-8k2k": MultiConfigKVCapacity(
        scenario="K2.5-tp8ep8-8k2k",
        kv_cache_tokens=760_160,
        block_size=KV_CACHE_BLOCK_SIZE,
        num_gpu_blocks=47_510,
        max_num_seqs=256,
        serve_log="docs/iter_gap_investigation/phase397k_measured_0190/K2.5-tp8ep8-8k2k/serve.log",
        serve_log_line_numbers=(202,),
        override_num_gpu_blocks=512,
        override_line_numbers=(161, 163, 165, 167, 168, 171, 173, 175),
    ),
    "K2.5-tp8ep8-32k3k": MultiConfigKVCapacity(
        scenario="K2.5-tp8ep8-32k3k",
        kv_cache_tokens=675_216,
        block_size=KV_CACHE_BLOCK_SIZE,
        num_gpu_blocks=42_201,
        max_num_seqs=256,
        serve_log="docs/iter_gap_investigation/phase397k_measured_0190/K2.5-tp8ep8-32k3k/serve.log",
        serve_log_line_numbers=(195,),
        override_num_gpu_blocks=512,
        override_line_numbers=(154, 156, 158, 159, 162, 163, 166, 168),
    ),
    "K2.5-tp4ep8dp2-8k2k": MultiConfigKVCapacity(
        scenario="K2.5-tp4ep8dp2-8k2k",
        kv_cache_tokens=672_128,
        block_size=KV_CACHE_BLOCK_SIZE,
        num_gpu_blocks=42_008,
        max_num_seqs=256,
        serve_log="docs/iter_gap_investigation/phase397k_measured_0190/K2.5-tp4ep8dp2-8k2k/serve.log",
        serve_log_line_numbers=(210, 213),
        override_num_gpu_blocks=512,
        override_line_numbers=(170, 172, 174, 175, 176, 180, 182, 184),
    ),
    "K2.5-tp4ep8dp2-32k3k": MultiConfigKVCapacity(
        scenario="K2.5-tp4ep8dp2-32k3k",
        kv_cache_tokens=320_800,
        block_size=KV_CACHE_BLOCK_SIZE,
        num_gpu_blocks=20_050,
        max_num_seqs=256,
        serve_log="docs/iter_gap_investigation/phase397k_measured_0190/K2.5-tp4ep8dp2-32k3k/serve.log",
        serve_log_line_numbers=(204, 212),
        override_num_gpu_blocks=512,
        override_line_numbers=(165, 167, 168, 169, 173, 174, 177, 179),
    ),
    "K2.5-tp8ep8-8k2k-bt65536": MultiConfigKVCapacity(
        scenario="K2.5-tp8ep8-8k2k-bt65536",
        kv_cache_tokens=343_552,
        block_size=KV_CACHE_BLOCK_SIZE,
        num_gpu_blocks=21_472,
        max_num_seqs=256,
        serve_log="docs/iter_gap_investigation/phase397k_measured_0190/K2.5-tp8ep8-8k2k-bt65536/serve.log",
        serve_log_line_numbers=(195,),
        override_num_gpu_blocks=512,
        override_line_numbers=(154, 155, 156, 160, 161, 164, 166, 167),
    ),
    "K2.5-tp4ep8dp2-8k2k-bt65536": MultiConfigKVCapacity(
        scenario="K2.5-tp4ep8dp2-8k2k-bt65536",
        kv_cache_tokens=131_664,
        block_size=KV_CACHE_BLOCK_SIZE,
        num_gpu_blocks=8_229,
        max_num_seqs=256,
        serve_log="docs/iter_gap_investigation/phase424_bt65536_recollect/K2.5-tp4ep8dp2-8k2k-bt65536/serve.log",
        serve_log_line_numbers=(203, 206),
        override_num_gpu_blocks=512,
        override_line_numbers=(163, 164, 167, 169, 171, 172, 175, 177),
    ),
}


def _multi_config_kv_capacity(point: MultiConfigPoint) -> MultiConfigKVCapacity:
    try:
        capacity = PHASE397K_KV_CAPACITY_BY_SCENARIO[point.name]
    except KeyError as exc:
        raise ValueError(f"missing Phase397k KV capacity for {point.name}") from exc
    if capacity.block_size != KV_CACHE_BLOCK_SIZE:
        raise ValueError(f"unexpected KV block size for {point.name}: {capacity.block_size}")
    if capacity.kv_cache_tokens != capacity.num_gpu_blocks * capacity.block_size:
        raise ValueError(f"KV tokens must equal blocks * block_size for {point.name}")
    return capacity


def _multi_config_num_gpu_blocks(point: MultiConfigPoint) -> int:
    return _multi_config_kv_capacity(point).num_gpu_blocks


def _multi_config_max_num_seqs(point: MultiConfigPoint) -> int:
    return _multi_config_kv_capacity(point).max_num_seqs


def _write_rows(path: Path, rows: list[object]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def run_experimental_forward_descriptor(out_csv: Path) -> None:
    topology_key = make_topology_key(TP, 1, TP, 1)
    rows = []
    for point in THROUGHPUT_DATA:
        scenario = point.name.replace(" ", "_")
        rows.append(
            descriptor_from_scheduled(
                source="validate_input_shape",
                scenario=scenario,
                iteration=0,
                phase="prefill",
                scheduled_context_tokens=point.isl,
                scheduled_decode_tokens=0,
                topology_key=topology_key,
                tp=TP,
                dp=1,
                moe_tp=TP,
                moe_ep=1,
                max_num_batched_tokens=point.isl,
                max_num_seqs=256,
            )
        )
        rows.append(
            descriptor_from_scheduled(
                source="validate_input_shape",
                scenario=scenario,
                iteration=1,
                phase="pure_decode",
                scheduled_context_tokens=0,
                scheduled_decode_tokens=point.batch_size,
                topology_key=topology_key,
                tp=TP,
                dp=1,
                moe_tp=TP,
                moe_ep=1,
                max_num_batched_tokens=point.isl,
                max_num_seqs=256,
            )
        )
    _write_rows(out_csv, rows)
    print(f"wrote experimental forward descriptors: {out_csv}")


def run_experimental_runtime_shape_key(out_csv: Path) -> None:
    topology_key = make_topology_key(TP, 1, TP, 1)
    rows = []
    for point in THROUGHPUT_DATA:
        scenario = point.name.replace(" ", "_")
        rows.append(
            runtime_shape_key_from_scheduled(
                source="validate_input_shape",
                scenario=scenario,
                iteration=0,
                phase="prefill",
                scheduled_context_tokens=point.isl,
                scheduled_decode_tokens=0,
                topology_key=topology_key,
                tp=TP,
                dp=1,
                moe_tp=TP,
                moe_ep=1,
            )
        )
        rows.append(
            runtime_shape_key_from_scheduled(
                source="validate_input_shape",
                scenario=scenario,
                iteration=1,
                phase="pure_decode",
                scheduled_context_tokens=0,
                scheduled_decode_tokens=point.batch_size,
                topology_key=topology_key,
                tp=TP,
                dp=1,
                moe_tp=TP,
                moe_ep=1,
            )
        )
    _write_rows(out_csv, rows)
    print(f"wrote experimental runtime shape keys: {out_csv}")


def run_experimental_scheduler_descriptor(out_csv: Path) -> None:
    topology_key = make_topology_key(TP, 1, TP, 1)
    rows = []
    for point in THROUGHPUT_DATA:
        scenario = point.name.replace(" ", "_")
        rows.append(
            scheduler_runtime_descriptor_from_scheduled(
                source="validate_input_shape",
                scenario=scenario,
                iteration=0,
                phase="prefill",
                scheduled_context_tokens=point.isl,
                scheduled_decode_tokens=0,
                scheduled_context_reqs=1,
                scheduled_decode_reqs=0,
                max_num_batched_tokens=point.isl,
                max_num_seqs=256,
                forward_token_count=point.isl,
                cudagraph_runtime_mode="AIC_UNSET",
                topology_key=topology_key,
                tp=TP,
                dp=1,
                moe_tp=TP,
                moe_ep=1,
            )
        )
        rows.append(
            scheduler_runtime_descriptor_from_scheduled(
                source="validate_input_shape",
                scenario=scenario,
                iteration=1,
                phase="pure_decode",
                scheduled_context_tokens=0,
                scheduled_decode_tokens=point.batch_size,
                scheduled_context_reqs=0,
                scheduled_decode_reqs=point.batch_size,
                max_num_batched_tokens=point.isl,
                max_num_seqs=256,
                forward_token_count=point.batch_size,
                cudagraph_runtime_mode="AIC_UNSET",
                topology_key=topology_key,
                tp=TP,
                dp=1,
                moe_tp=TP,
                moe_ep=1,
            )
        )
    _write_rows(out_csv, rows)
    print(f"wrote experimental scheduler descriptors: {out_csv}")


def run_experimental_scheduler_alignment_descriptor(out_csv: Path) -> None:
    topology_key = make_topology_key(TP, 1, TP, 1)
    rows = []
    for point in THROUGHPUT_DATA:
        scenario = point.name.replace(" ", "_")
        rows.append(
            scheduler_aligned_descriptor_from_scheduled(
                source="validate_input_shape",
                scenario=scenario,
                dp_rank=0,
                engine_step_id=0,
                phase="prefill",
                scheduled_context_tokens=point.isl,
                scheduled_decode_tokens=0,
                scheduled_context_reqs=1,
                scheduled_decode_reqs=0,
                max_num_batched_tokens=point.isl,
                max_num_seqs=256,
                forward_token_count=point.isl,
                cudagraph_runtime_mode="AIC_UNSET",
                topology_key=topology_key,
                tp=TP,
                dp=1,
                moe_tp=TP,
                moe_ep=1,
            )
        )
        rows.append(
            scheduler_aligned_descriptor_from_scheduled(
                source="validate_input_shape",
                scenario=scenario,
                dp_rank=0,
                engine_step_id=1,
                phase="pure_decode",
                scheduled_context_tokens=0,
                scheduled_decode_tokens=point.batch_size,
                scheduled_context_reqs=0,
                scheduled_decode_reqs=point.batch_size,
                max_num_batched_tokens=point.isl,
                max_num_seqs=256,
                forward_token_count=point.batch_size,
                cudagraph_runtime_mode="AIC_UNSET",
                topology_key=topology_key,
                tp=TP,
                dp=1,
                moe_tp=TP,
                moe_ep=1,
            )
        )
    _write_rows(out_csv, rows)
    print(f"wrote experimental scheduler alignment descriptors: {out_csv}")


def run_experimental_compiled_body_key(out_csv: Path, nccl_summary_csv: Path) -> None:
    key = compiled_body_runtime_key_from_nccl_summary_csv(
        nccl_summary_csv,
        source="phase39_nccl_trace",
        scenario="10k2k_b32_bt8192",
        phase="mixed",
        topology_key=make_topology_key(4, 2, 1, 8),
        tp=4,
        dp=2,
        ep=8,
        world_size=8,
        forward_regime="NONE:248",
        tokens_padded=248,
        tokens_actual=241,
        cudagraph_runtime_mode="NONE",
        moe_module="DeepseekV2MoE",
        moe_kernel="wna16",
        moe_hidden=7168,
        moe_intermediate=2048,
        moe_experts=384,
        moe_topk=8,
        moe_dtype="bfloat16",
        tuning_config_loaded=False,
        fallback=True,
    )
    _write_rows(out_csv, [key])
    print(f"wrote experimental compiled-body runtime key: {out_csv}")


def run_experimental_moe_source_key(out_csv: Path) -> None:
    key = moe_source_runtime_key_from_loaded_weight_boundary_row(
        {
            "source": "phase82_loaded_weight_boundary",
            "scenario": "kimi_k25_h200_tp4dp2ep8",
            "runtime_backend": "vllm",
            "vllm_version": "0.19.0",
            "model_family": "kimi_k25",
            "module_class": "DeepseekV2MoE",
            "experts_class": "SharedFusedMoE",
            "hidden_size": 7168,
            "moe_intermediate_size": 2048,
            "n_routed_experts": 384,
            "local_experts": 48,
            "global_experts": 384,
            "topk": 8,
            "n_shared_experts": 1,
            "moe_method": "CompressedTensorsWNA16MarlinMoEMethod",
            "kernel_backend": "wna16_marlin",
            "group_size": 32,
            "num_bits": 4,
            "dtype": "bfloat16",
            "tp_size": 4,
            "dp_size": 2,
            "ep_size": 8,
            "world_size": 8,
            "rank": 0,
            "device": "cuda:0",
            "tuning_config_loaded": False,
            "moe_config_fallback": True,
            "moe_tuning_config_file": "",
            "loaded_weight": True,
            "random_weight": False,
            "timing": False,
            "valid_for_default": False,
            "perf_database": False,
            "diagnostic_only": True,
        }
    )
    _write_rows(out_csv, [key])
    print(f"wrote experimental MoE source runtime key: {out_csv}")


def _abs_error(predicted: float, real: float) -> float:
    """Symmetric absolute error ratio: max(p/r, r/p)."""
    if predicted <= 0 or real <= 0:
        return float("inf")
    ratio = predicted / real
    return max(ratio, 1 / ratio)


@lru_cache(maxsize=None)
def _load_model_and_db(
    tp: int = TP,
    dp: int = 1,
    moe_tp: int | None = None,
    moe_ep: int = 1,
) -> tuple:
    """Load model and database with same pattern as fit_cb_factor.py."""
    model_config = ModelConfig(
        tp_size=tp,
        pp_size=1,
        moe_tp_size=moe_tp if moe_tp is not None else tp,
        moe_ep_size=moe_ep,
        attention_dp_size=dp,
    )
    model = get_model(MODEL_PATH, model_config, backend_name=BACKEND)

    systems_root = str(
        Path(__file__).resolve().parent.parent / "src" / "aiconfigurator" / "systems"
    )
    db = PerfDatabase(
        system=SYSTEM,
        backend=BACKEND,
        version=VALIDATION_DB_VERSION,
        systems_root=systems_root,
    )
    backend = VLLMBackend()
    return model, db, backend


def _make_cb_config(
    isl: int,
    concurrency: int,
    long_prefill_token_threshold: int = 0,
    overlap_factor: float = 1.0,
    per_iteration_overhead_ms: float = 0.0,
    max_num_batched_tokens: int | None = None,
    num_gpu_blocks: int = 0,
    max_num_seqs: int = 256,
) -> CBSimConfig:
    """Create CBSimConfig aligned with run_agg default chunk budget."""
    # Need enough requests for meaningful steady-state measurement.
    # At minimum 3x concurrency to avoid requests running out.
    num_requests = max(200, concurrency * 3)
    warmup_requests = max(50, concurrency)
    return CBSimConfig(
        max_num_batched_tokens=max_num_batched_tokens or isl,
        num_requests=num_requests,
        warmup_requests=warmup_requests,
        long_prefill_token_threshold=long_prefill_token_threshold,
        num_gpu_blocks=num_gpu_blocks,
        max_num_seqs=max_num_seqs,
        block_size=KV_CACHE_BLOCK_SIZE,
        overlap_factor=overlap_factor,
        per_iteration_overhead_ms=per_iteration_overhead_ms,
    )


def _assert_multi_config_cb_config(point: MultiConfigPoint, config: CBSimConfig) -> None:
    expected_blocks = _multi_config_num_gpu_blocks(point)
    expected_max_num_seqs = _multi_config_max_num_seqs(point)
    if config.max_num_batched_tokens != point.max_num_batched_tokens:
        raise AssertionError(
            f"{point.name}: max_num_batched_tokens={config.max_num_batched_tokens} "
            f"!= {point.max_num_batched_tokens}"
        )
    if config.num_gpu_blocks != expected_blocks:
        raise AssertionError(f"{point.name}: num_gpu_blocks={config.num_gpu_blocks} != {expected_blocks}")
    if config.block_size != KV_CACHE_BLOCK_SIZE:
        raise AssertionError(f"{point.name}: block_size={config.block_size} != {KV_CACHE_BLOCK_SIZE}")
    if config.max_num_seqs != expected_max_num_seqs:
        raise AssertionError(f"{point.name}: max_num_seqs={config.max_num_seqs} != {expected_max_num_seqs}")


def _make_multi_config_cb_config(
    point: MultiConfigPoint,
    *,
    overlap_factor: float,
    ep8_per_iteration_overhead_ms: float,
) -> CBSimConfig:
    config = _make_cb_config(
        point.isl,
        point.batch_size,
        max_num_batched_tokens=point.max_num_batched_tokens,
        overlap_factor=overlap_factor,
        per_iteration_overhead_ms=ep8_per_iteration_overhead_ms,
        num_gpu_blocks=_multi_config_num_gpu_blocks(point),
        max_num_seqs=_multi_config_max_num_seqs(point),
    )
    _assert_multi_config_cb_config(point, config)
    return config


def _rank_by_output(values: dict[str, float]) -> dict[str, int]:
    ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    return {name: rank for rank, (name, _) in enumerate(ordered, start=1)}


def _run_multi_config_diagnostic_raw(
    overlap_factor: float,
    ep8_per_iteration_overhead_ms: float,
) -> list[tuple[MultiConfigPoint, float, float, dict]]:
    backend = VLLMBackend()
    loaded: dict[tuple[int, int, int, int], tuple] = {}
    raw_rows: list[tuple[MultiConfigPoint, float, float, dict]] = []
    for pt in MULTI_CONFIG_DATA:
        key = (pt.tp, pt.dp, pt.moe_tp, pt.moe_ep)
        if key not in loaded:
            model, db, _ = _load_model_and_db(
                tp=pt.tp,
                dp=pt.dp,
                moe_tp=pt.moe_tp,
                moe_ep=pt.moe_ep,
            )
            loaded[key] = (model, db)
        model, db = loaded[key]

        cb_config = _make_multi_config_cb_config(
            pt,
            overlap_factor=overlap_factor,
            ep8_per_iteration_overhead_ms=ep8_per_iteration_overhead_ms,
        )
        cb_summary = backend.run_agg(
            model,
            db,
            RuntimeConfig(batch_size=pt.batch_size, isl=pt.isl, osl=pt.osl),
            ctx_tokens=pt.max_num_batched_tokens,
            database_mode=common.DatabaseMode.HYBRID,
            method="cb_sim",
            cb_config=cb_config,
        )
        cb_dict = cb_summary.get_result_dict()
        sim_gpu = cb_dict["tokens/s/gpu"] if cb_dict else 0.0
        raw_rows.append((pt, pt.real_output_tok_s_gpu, sim_gpu, cb_summary.get_per_ops_data()))
    return raw_rows


def run_diagnostic_multi_config_topk(
    overlap_factor: float = 0.0,
    ep8_per_iteration_overhead_ms: float = DEFAULT_EP8_PER_ITERATION_OVERHEAD_MS,
    verbose: bool = True,
) -> list[MultiConfigTopKRow]:
    """Run budget-aware diagnostic Top-K rows without changing validation gates."""
    raw_rows = _run_multi_config_diagnostic_raw(
        overlap_factor,
        ep8_per_iteration_overhead_ms,
    )
    real_ranks = _rank_by_output({pt.name: real for pt, real, _, _ in raw_rows})
    sim_ranks = _rank_by_output({pt.name: sim for pt, _, sim, _ in raw_rows})
    rows = [
        MultiConfigTopKRow(
            name=pt.name,
            tp=pt.tp,
            dp=pt.dp,
            ep=pt.moe_ep,
            max_bt=pt.max_num_batched_tokens,
            real_output_tok_s_gpu=real,
            sim_output_tok_s_gpu=sim,
            error_ratio=_abs_error(sim, real),
            rank=sim_ranks[pt.name],
            real_rank=real_ranks[pt.name],
            rank_delta=sim_ranks[pt.name] - real_ranks[pt.name],
            diagnostic_only=True,
            valid_for_default=False,
            perf_database=False,
        )
        for pt, real, sim, _ in raw_rows
    ]
    rows.sort(key=lambda row: (row.rank, row.real_rank, row.name))

    if verbose:
        print()
        print("=" * 116)
        print("DIAGNOSTIC MULTI-CONFIG TOP-K (budget-aware, not default AIC)")
        print(
            f"{'Rank':>4} {'RealR':>5} {'Delta':>5} {'Scenario':<34} "
            f"{'tp':>3} {'dp':>3} {'ep':>3} {'max_bt':>7} "
            f"{'RealOut':>8} {'CB-Sim':>8} {'Error':>8}"
        )
        print("-" * 116)
        for row in rows:
            print(
                f"{row.rank:>4} {row.real_rank:>5} {row.rank_delta:>5} "
                f"{row.name:<34} {row.tp:>3} {row.dp:>3} {row.ep:>3} "
                f"{row.max_bt:>7} {row.real_output_tok_s_gpu:>8.1f} "
                f"{row.sim_output_tok_s_gpu:>8.1f} {row.error_ratio:>7.2f}x"
            )
        print("-" * 116)
        print("diagnostic_only=true valid_for_default=false perf_database=false")
    return rows


def _require_metric(mapping: dict, key: str, scenario: str) -> float:
    if key not in mapping:
        raise ValueError(f"missing cb_sim_scheduling.{key} for {scenario}")
    return float(mapping[key])


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("inf")
    return numerator / denominator


def run_diagnostic_multi_config_budget_breakdown(
    overlap_factor: float = 0.0,
    ep8_per_iteration_overhead_ms: float = DEFAULT_EP8_PER_ITERATION_OVERHEAD_MS,
    verbose: bool = True,
) -> list[MultiConfigBudgetBreakdownRow]:
    """Emit budget/scheduler root-cause rows without changing validation gates."""
    raw_rows = _run_multi_config_diagnostic_raw(
        overlap_factor,
        ep8_per_iteration_overhead_ms,
    )
    real_ranks = _rank_by_output({pt.name: real for pt, real, _, _ in raw_rows})
    sim_ranks = _rank_by_output({pt.name: sim for pt, _, sim, _ in raw_rows})

    interim: list[dict] = []
    baseline_by_shape: dict[tuple[int, int, int, int, int, int, int], dict] = {}
    for pt, real, sim, per_ops in raw_rows:
        scheduling = per_ops.get("cb_sim_scheduling")
        if not isinstance(scheduling, dict):
            raise ValueError(f"missing cb_sim_scheduling for {pt.name}")
        row = {
            "shape_key": (pt.isl, pt.osl, pt.batch_size, pt.tp, pt.dp, pt.moe_tp, pt.moe_ep),
            "name": pt.name,
            "tp": pt.tp,
            "dp": pt.dp,
            "ep": pt.moe_ep,
            "max_bt": pt.max_num_batched_tokens,
            "isl": pt.isl,
            "real_output_tok_s_gpu": real,
            "sim_output_tok_s_gpu": sim,
            "error_ratio": _abs_error(sim, real),
            "rank": sim_ranks[pt.name],
            "real_rank": real_ranks[pt.name],
            "avg_prefill_reqs_per_iter": _require_metric(
                scheduling,
                "avg_prefill_reqs_per_iter",
                pt.name,
            ),
            "avg_decode_reqs_per_iter": _require_metric(
                scheduling,
                "avg_decode_reqs_per_iter",
                pt.name,
            ),
            "avg_tokens_per_iter": _require_metric(
                scheduling,
                "avg_tokens_per_iter",
                pt.name,
            ),
            "peak_prefill_reqs_per_iter": _require_metric(
                scheduling,
                "peak_prefill_reqs_per_iter",
                pt.name,
            ),
            "peak_decode_reqs_per_iter": _require_metric(
                scheduling,
                "peak_decode_reqs_per_iter",
                pt.name,
            ),
            "peak_tokens_per_iter": _require_metric(
                scheduling,
                "peak_tokens_per_iter",
                pt.name,
            ),
            "steady_state_iterations": int(
                _require_metric(scheduling, "steady_state_iterations", pt.name)
            ),
            "steady_state_time_ms": _require_metric(
                scheduling,
                "steady_state_time_ms",
                pt.name,
            ),
        }
        interim.append(row)
        if pt.max_num_batched_tokens == pt.isl:
            baseline_by_shape[row["shape_key"]] = row

    rows: list[MultiConfigBudgetBreakdownRow] = []
    for row in interim:
        baseline = baseline_by_shape.get(row["shape_key"])
        if baseline is None:
            raise ValueError(f"missing max_bt=isl paired baseline for {row['name']}")
        rows.append(
            MultiConfigBudgetBreakdownRow(
                name=row["name"],
                tp=row["tp"],
                dp=row["dp"],
                ep=row["ep"],
                max_bt=row["max_bt"],
                real_output_tok_s_gpu=row["real_output_tok_s_gpu"],
                sim_output_tok_s_gpu=row["sim_output_tok_s_gpu"],
                error_ratio=row["error_ratio"],
                rank=row["rank"],
                real_rank=row["real_rank"],
                avg_prefill_reqs_per_iter=row["avg_prefill_reqs_per_iter"],
                avg_decode_reqs_per_iter=row["avg_decode_reqs_per_iter"],
                avg_tokens_per_iter=row["avg_tokens_per_iter"],
                peak_prefill_reqs_per_iter=row["peak_prefill_reqs_per_iter"],
                peak_decode_reqs_per_iter=row["peak_decode_reqs_per_iter"],
                peak_tokens_per_iter=row["peak_tokens_per_iter"],
                steady_state_iterations=row["steady_state_iterations"],
                steady_state_time_ms=row["steady_state_time_ms"],
                paired_baseline_name=baseline["name"],
                paired_baseline_max_bt=baseline["max_bt"],
                paired_baseline_sim_output_tok_s_gpu=baseline["sim_output_tok_s_gpu"],
                paired_baseline_error_ratio=baseline["error_ratio"],
                paired_baseline_rank=baseline["rank"],
                paired_baseline_avg_tokens_per_iter=baseline["avg_tokens_per_iter"],
                max_bt_vs_paired_baseline=_safe_ratio(row["max_bt"], baseline["max_bt"]),
                sim_vs_paired_baseline_ratio=_safe_ratio(
                    row["sim_output_tok_s_gpu"],
                    baseline["sim_output_tok_s_gpu"],
                ),
                avg_tokens_vs_paired_baseline_ratio=_safe_ratio(
                    row["avg_tokens_per_iter"],
                    baseline["avg_tokens_per_iter"],
                ),
                steady_state_time_vs_paired_baseline_ratio=_safe_ratio(
                    row["steady_state_time_ms"],
                    baseline["steady_state_time_ms"],
                ),
                diagnostic_only=True,
                valid_for_default=False,
                perf_database=False,
            )
        )
    rows.sort(key=lambda item: (item.rank, item.real_rank, item.name))

    if verbose:
        print()
        print("=" * 132)
        print("DIAGNOSTIC MULTI-CONFIG BUDGET BREAKDOWN (not default AIC)")
        print(
            f"{'Rank':>4} {'RealR':>5} {'Scenario':<34} {'tp':>3} {'dp':>3} "
            f"{'ep':>3} {'max_bt':>7} {'PairBT':>7} {'RealOut':>8} "
            f"{'CB-Sim':>8} {'Error':>8} {'AvgTok':>8} {'PeakTok':>8} "
            f"{'SteadyI':>8} {'SteadyMs':>9}"
        )
        print("-" * 132)
        for row in rows:
            print(
                f"{row.rank:>4} {row.real_rank:>5} {row.name:<34} "
                f"{row.tp:>3} {row.dp:>3} {row.ep:>3} {row.max_bt:>7} "
                f"{row.paired_baseline_max_bt:>7} "
                f"{row.real_output_tok_s_gpu:>8.1f} "
                f"{row.sim_output_tok_s_gpu:>8.1f} {row.error_ratio:>7.2f}x "
                f"{row.avg_tokens_per_iter:>8.1f} {row.peak_tokens_per_iter:>8.1f} "
                f"{row.steady_state_iterations:>8} {row.steady_state_time_ms:>9.1f}"
            )
        print("-" * 132)
        print("diagnostic_only=true valid_for_default=false perf_database=false")
    return rows


def _run_multi_config_validation(
    overlap_factor: float,
    per_iteration_overhead_ms: float,
    ep8_per_iteration_overhead_ms: float,
    verbose: bool = True,
) -> list[float]:
    if verbose:
        print()
        print("=" * 104)
        print("MULTI-CONFIG THROUGHPUT (tok/s/GPU, output-only compare)")
        print("CSV total tok/s/GPU is converted by output = total * OSL / (ISL + OSL)")
        print(
            f"{'Scenario':<32} {'tp':>3} {'dp':>3} {'ep':>3} {'RealOut':>8} "
            f"{'RealTot':>8} {'CB-Sim':>8} {'Sim/Out':>10} {'Source':>14}"
        )
        print("-" * 104)

    errs: list[float] = []
    backend = VLLMBackend()
    loaded: dict[tuple[int, int, int, int], tuple] = {}
    for pt in MULTI_CONFIG_DATA:
        key = (pt.tp, pt.dp, pt.moe_tp, pt.moe_ep)
        if key not in loaded:
            model, db, _ = _load_model_and_db(
                tp=pt.tp,
                dp=pt.dp,
                moe_tp=pt.moe_tp,
                moe_ep=pt.moe_ep,
            )
            loaded[key] = (model, db)
        model, db = loaded[key]

        cb_config = _make_multi_config_cb_config(
            pt,
            overlap_factor=overlap_factor,
            ep8_per_iteration_overhead_ms=ep8_per_iteration_overhead_ms,
        )
        cb_summary = backend.run_agg(
            model,
            db,
            RuntimeConfig(batch_size=pt.batch_size, isl=pt.isl, osl=pt.osl),
            ctx_tokens=pt.max_num_batched_tokens,
            database_mode=common.DatabaseMode.HYBRID,
            method="cb_sim",
            cb_config=cb_config,
        )
        cb_dict = cb_summary.get_result_dict()
        per_ops = cb_summary.get_per_ops_data()
        sim_gpu = cb_dict["tokens/s/gpu"] if cb_dict else 0.0
        real_output = pt.real_output_tok_s_gpu
        ratio = sim_gpu / real_output if real_output > 0 else 0.0
        errs.append(_abs_error(sim_gpu, real_output))
        boundary = per_ops.get("cb_sim_boundary", {})
        source = boundary.get("throughput_source", "unknown")

        if verbose:
            print(
                f"{pt.name:<32} {pt.tp:>3} {pt.dp:>3} {pt.moe_ep:>3} "
                f"{real_output:>8.1f} {pt.real_total_tok_s_gpu:>8.1f} "
                f"{sim_gpu:>8.1f} {ratio:>9.2f}x {source:>14}"
            )

    if verbose:
        print("-" * 104)
        print(f"MULTI-CONFIG: max={max(errs):.2f}x mean={np.mean(errs):.2f}x")
    return errs


def run_validation(
    overlap_factor: float = 0.0,
    per_iteration_overhead_ms: float = 0.0,
    ep8_per_iteration_overhead_ms: float = DEFAULT_EP8_PER_ITERATION_OVERHEAD_MS,
    verbose: bool = True,
) -> ValidationResult:
    model, db, backend = _load_model_and_db()
    model_max_len = max(int(getattr(model, "_context_length", 0)), 0)
    threshold = int(model_max_len * 0.04) if model_max_len > 0 else 0

    # --- Throughput ---
    if verbose:
        print("=" * 90)
        print("THROUGHPUT (tok/s/GPU, output-only compare)")
        print(f"overlap_factor = {overlap_factor}")
        print(f"per_iteration_overhead_ms = {per_iteration_overhead_ms}")
        print(f"ep8_per_iteration_overhead_ms = {ep8_per_iteration_overhead_ms}")
        print(
            f"{'Scenario':<22} {'RealOut':>8} {'RealTot':>8} {'CB-Sim':>8} "
            f"{'B1':>8} {'Sim/Out':>10} {'B1/Out':>10}"
        )
        print("-" * 100)

    sim_errs: list[float] = []
    b1_errs: list[float] = []

    for pt in THROUGHPUT_DATA:
        # CB sim: validate the same public metric surface exposed by run_agg().
        cb_config = _make_cb_config(
            pt.isl,
            pt.batch_size,
            overlap_factor=overlap_factor,
            per_iteration_overhead_ms=per_iteration_overhead_ms,
        )
        cb_summary = backend.run_agg(
            model, db,
            RuntimeConfig(batch_size=pt.batch_size, isl=pt.isl, osl=pt.osl),
            ctx_tokens=pt.isl,
            database_mode=common.DatabaseMode.HYBRID,
            method="cb_sim",
            cb_config=cb_config,
        )
        cb_dict = cb_summary.get_result_dict()
        sim_gpu = cb_dict["tokens/s/gpu"] if cb_dict else 0

        # B1 baseline is display-only; parameter sweep skips it.
        if verbose:
            try:
                b1_summary = backend.run_agg(
                    model, db,
                    RuntimeConfig(batch_size=pt.batch_size, isl=pt.isl, osl=pt.osl),
                    ctx_tokens=pt.isl,
                    database_mode=common.DatabaseMode.HYBRID,
                )
                b1_dict = b1_summary.get_result_dict()
                b1_gpu = b1_dict["tokens/s/gpu"] if b1_dict else 0
            except Exception as e:
                logger.warning("B1 failed for %s: %s", pt.name, e)
                b1_gpu = 0
        else:
            b1_gpu = 0

        sim_r = (
            sim_gpu / pt.real_output_tok_s_gpu if pt.real_output_tok_s_gpu > 0 else 0
        )
        b1_r = (
            b1_gpu / pt.real_output_tok_s_gpu if pt.real_output_tok_s_gpu > 0 else 0
        )
        sim_errs.append(_abs_error(sim_gpu, pt.real_output_tok_s_gpu))
        if verbose:
            b1_errs.append(_abs_error(b1_gpu, pt.real_output_tok_s_gpu))

        real_total = pt.real_total_tok_s_gpu if pt.real_total_tok_s_gpu is not None else 0.0
        if verbose:
            print(
                f"{pt.name:<22} {pt.real_output_tok_s_gpu:>8.1f} {real_total:>8.1f} "
                f"{sim_gpu:>8.1f} {b1_gpu:>8.1f} {sim_r:>9.2f}x {b1_r:>9.2f}x"
            )

    if verbose:
        print("-" * 100)
        print(f"CB-Sim: max={max(sim_errs):.2f}x mean={np.mean(sim_errs):.2f}x | "
              f"B1: max={max(b1_errs):.2f}x mean={np.mean(b1_errs):.2f}x")

    # --- TTFT ---
    if verbose:
        print()
        print("=" * 90)
        print("TTFT (ms)")
        print(f"long_prefill_token_threshold compare = {threshold}")
        print(
            f"{'Scenario':<22} {'Real':>8} {'CB-Sim':>8} "
            f"{'CB+Thresh':>10} {'Sim/Real':>10} {'Thr/Real':>10}"
        )
        print("-" * 90)

    ttft_errs: list[float] = []
    ttft_thresh_errs: list[float] = []
    for pt in TTFT_DATA:
        base_config = _make_cb_config(
            pt.isl,
            pt.batch_size,
            overlap_factor=overlap_factor,
            per_iteration_overhead_ms=per_iteration_overhead_ms,
        )
        base_sim = CBSimulator(backend, model, db, base_config)
        base_result = base_sim.run(
            isl=pt.isl, osl=pt.osl, concurrency=pt.batch_size, num_gpus=1,
        )
        thresh_config = _make_cb_config(
            pt.isl,
            pt.batch_size,
            long_prefill_token_threshold=threshold,
            overlap_factor=overlap_factor,
            per_iteration_overhead_ms=per_iteration_overhead_ms,
        )
        thresh_sim = CBSimulator(backend, model, db, thresh_config)
        thresh_result = thresh_sim.run(
            isl=pt.isl, osl=pt.osl, concurrency=pt.batch_size, num_gpus=1,
        )
        ratio = base_result.mean_ttft_ms / pt.real_ttft_ms if pt.real_ttft_ms else 0
        thresh_ratio = (
            thresh_result.mean_ttft_ms / pt.real_ttft_ms if pt.real_ttft_ms else 0
        )
        ttft_errs.append(_abs_error(base_result.mean_ttft_ms, pt.real_ttft_ms))
        ttft_thresh_errs.append(
            _abs_error(thresh_result.mean_ttft_ms, pt.real_ttft_ms),
        )
        if verbose:
            print(
                f"{pt.name:<22} {pt.real_ttft_ms:>8.1f} {base_result.mean_ttft_ms:>8.1f} "
                f"{thresh_result.mean_ttft_ms:>10.1f} {ratio:>9.2f}x {thresh_ratio:>9.2f}x"
            )

    if verbose:
        print("-" * 90)
        print(
            f"TTFT: base max={max(ttft_errs):.2f}x mean={np.mean(ttft_errs):.2f}x | "
            f"threshold max={max(ttft_thresh_errs):.2f}x mean={np.mean(ttft_thresh_errs):.2f}x"
        )

    multi_errs = _run_multi_config_validation(
        overlap_factor,
        per_iteration_overhead_ms,
        ep8_per_iteration_overhead_ms,
        verbose=verbose,
    )

    # --- Summary ---
    multi_ok = max(multi_errs) <= MULTI_CONFIG_MAX_ACCEPTANCE
    if verbose:
        print()
        print("=" * 90)
        print("ACCEPTANCE CRITERIA:")
        print(
            f"  Throughput legacy 0.17 surface: SKIP "
            f"({THROUGHPUT_GATE_STATUS}; max={max(sim_errs):.2f}x)"
        )
        print(
            f"  Multi-config max error <= {MULTI_CONFIG_MAX_ACCEPTANCE:.2f}x: "
            f"{'PASS' if multi_ok else 'FAIL'} ({max(multi_errs):.2f}x)"
        )
        print(
            f"  TTFT legacy 0.17 surface:       SKIP "
            f"({TTFT_GATE_STATUS}; max={max(ttft_thresh_errs):.2f}x)"
        )

    return ValidationResult(
        overlap_factor=overlap_factor,
        per_iteration_overhead_ms=per_iteration_overhead_ms,
        ep8_per_iteration_overhead_ms=ep8_per_iteration_overhead_ms,
        throughput_max=max(sim_errs),
        throughput_mean=float(np.mean(sim_errs)),
        multi_config_max=max(multi_errs),
        multi_config_mean=float(np.mean(multi_errs)),
        ttft_max=max(ttft_thresh_errs),
        ttft_mean=float(np.mean(ttft_thresh_errs)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate CB simulator against real vLLM benchmark data."
    )
    parser.add_argument("--overlap-factor", type=float, default=0.0)
    parser.add_argument("--per-iteration-overhead-ms", type=float, default=0.0)
    parser.add_argument(
        "--ep8-per-iteration-overhead-ms",
        type=float,
        default=DEFAULT_EP8_PER_ITERATION_OVERHEAD_MS,
    )
    parser.add_argument(
        "--diagnostic-multi-config-topk",
        action="store_true",
        help=(
            "Emit budget-aware multi-config Top-K diagnostics only. "
            "This does not change acceptance gates or default cb_sim behavior."
        ),
    )
    parser.add_argument(
        "--diagnostic-multi-config-topk-out",
        type=Path,
        default=None,
        help="Optional CSV path for budget-aware Top-K diagnostic rows.",
    )
    parser.add_argument(
        "--diagnostic-multi-config-budget-breakdown",
        action="store_true",
        help=(
            "Emit budget/scheduler breakdown diagnostics only. "
            "This does not change acceptance gates or default cb_sim behavior."
        ),
    )
    parser.add_argument(
        "--diagnostic-multi-config-budget-breakdown-out",
        type=Path,
        default=None,
        help="Optional CSV path for budget/scheduler breakdown diagnostic rows.",
    )
    parser.add_argument(
        "--experimental-forward-descriptor",
        action="store_true",
        help=(
            "Emit vLLM forward descriptor schema diagnostics only. "
            "This does not change the default cb_sim validation path."
        ),
    )
    parser.add_argument(
        "--experimental-forward-descriptor-out",
        type=Path,
        default=Path(
            "docs/iter_gap_investigation/phase8_forward_descriptor/"
            "validate_forward_descriptor_inputs.csv"
        ),
    )
    parser.add_argument(
        "--experimental-runtime-shape-key",
        action="store_true",
        help=(
            "Emit vLLM runtime shape key schema diagnostics only. "
            "This does not change the default cb_sim validation path."
        ),
    )
    parser.add_argument(
        "--experimental-runtime-shape-key-out",
        type=Path,
        default=Path(
            "docs/iter_gap_investigation/phase10_runtime_shape_key/"
            "validate_runtime_shape_keys.csv"
        ),
    )
    parser.add_argument(
        "--experimental-scheduler-descriptor",
        action="store_true",
        help=(
            "Emit vLLM scheduler/runtime descriptor diagnostics only. "
            "This does not change the default cb_sim validation path."
        ),
    )
    parser.add_argument(
        "--experimental-scheduler-descriptor-out",
        type=Path,
        default=Path(
            "docs/iter_gap_investigation/phase58_scheduler_descriptor/"
            "validate_scheduler_descriptors.csv"
        ),
    )
    parser.add_argument(
        "--experimental-scheduler-alignment-descriptor",
        action="store_true",
        help=(
            "Emit DP-aware scheduler/runtime alignment descriptors only. "
            "This does not change the default cb_sim validation path."
        ),
    )
    parser.add_argument(
        "--experimental-scheduler-alignment-descriptor-out",
        type=Path,
        default=Path(
            "docs/iter_gap_investigation/phase63_scheduler_alignment_descriptor/"
            "validate_scheduler_alignment_descriptors.csv"
        ),
    )
    parser.add_argument(
        "--experimental-compiled-body-key",
        action="store_true",
        help=(
            "Emit vLLM compiled-body runtime key diagnostics only. "
            "This does not change the default cb_sim validation path."
        ),
    )
    parser.add_argument(
        "--experimental-compiled-body-key-out",
        type=Path,
        default=Path(
            "docs/iter_gap_investigation/phase41_compiled_body_runtime_key/"
            "validate_compiled_body_runtime_key.csv"
        ),
    )
    parser.add_argument(
        "--experimental-compiled-body-key-nccl-summary",
        type=Path,
        default=Path(
            "docs/iter_gap_investigation/phase39_nccl_trace_10k2k_b32_bt8192/"
            "nccl_trace_phase39_summary.csv"
        ),
        help="Phase 39 NCCL trace summary CSV used only for candidate flags.",
    )
    parser.add_argument(
        "--experimental-moe-source-key",
        action="store_true",
        help=(
            "Emit vLLM loaded-weight MoE source key diagnostics only. "
            "This does not change the default cb_sim validation path."
        ),
    )
    parser.add_argument(
        "--experimental-moe-source-key-out",
        type=Path,
        default=Path(
            "docs/iter_gap_investigation/phase83_moe_source_runtime_key/"
            "validate_moe_source_runtime_key.csv"
        ),
    )
    args = parser.parse_args()

    if args.diagnostic_multi_config_topk:
        rows = run_diagnostic_multi_config_topk(
            overlap_factor=args.overlap_factor,
            ep8_per_iteration_overhead_ms=args.ep8_per_iteration_overhead_ms,
        )
        if args.diagnostic_multi_config_topk_out is not None:
            _write_rows(args.diagnostic_multi_config_topk_out, rows)
        return

    if args.diagnostic_multi_config_budget_breakdown:
        rows = run_diagnostic_multi_config_budget_breakdown(
            overlap_factor=args.overlap_factor,
            ep8_per_iteration_overhead_ms=args.ep8_per_iteration_overhead_ms,
        )
        if args.diagnostic_multi_config_budget_breakdown_out is not None:
            _write_rows(args.diagnostic_multi_config_budget_breakdown_out, rows)
        return

    if args.experimental_forward_descriptor:
        run_experimental_forward_descriptor(args.experimental_forward_descriptor_out)
        return

    if args.experimental_runtime_shape_key:
        run_experimental_runtime_shape_key(args.experimental_runtime_shape_key_out)
        return

    if args.experimental_scheduler_descriptor:
        run_experimental_scheduler_descriptor(
            args.experimental_scheduler_descriptor_out
        )
        return

    if args.experimental_scheduler_alignment_descriptor:
        run_experimental_scheduler_alignment_descriptor(
            args.experimental_scheduler_alignment_descriptor_out
        )
        return

    if args.experimental_compiled_body_key:
        run_experimental_compiled_body_key(
            args.experimental_compiled_body_key_out,
            args.experimental_compiled_body_key_nccl_summary,
        )
        return

    if args.experimental_moe_source_key:
        run_experimental_moe_source_key(args.experimental_moe_source_key_out)
        return

    run_validation(
        overlap_factor=args.overlap_factor,
        per_iteration_overhead_ms=args.per_iteration_overhead_ms,
        ep8_per_iteration_overhead_ms=args.ep8_per_iteration_overhead_ms,
    )


if __name__ == "__main__":
    main()
