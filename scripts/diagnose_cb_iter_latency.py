#!/usr/bin/env python3
"""Diagnose CB-sim iteration latency against vLLM iteration logs."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aiconfigurator.sdk.backends.cb_simulator import CBSimConfig
from aiconfigurator.sdk.backends.cb_simulator.datatypes import Request, RequestState
from aiconfigurator.sdk.backends.cb_simulator.forward_descriptor import (
    RuntimeShapeSubkeyDistributionRow,
    VLLMCompiledBodyRuntimeKey,
    VLLMForwardDescriptor,
    VLLMRuntimeShapeKey,
    VLLMSchedulerAlignedDescriptor,
    VLLMSchedulerRuntimeDescriptor,
    attention_subkey_from_runtime_shape_key,
    compiled_body_runtime_key_from_nccl_summary_csv,
    compare_forward_descriptors,
    compare_runtime_shape_keys,
    compare_scheduler_aligned_descriptors,
    descriptor_from_runtime_shape_summary,
    descriptor_from_scheduled,
    forward_wrapper_subkey_from_runtime_shape_key,
    kv_subkey_from_runtime_shape_key,
    make_topology_key,
    moe_source_runtime_key_from_loaded_weight_boundary_row,
    runtime_shape_key_from_rank_row,
    runtime_shape_key_from_scheduled,
    scheduler_aligned_descriptor_from_scheduled,
    scheduler_aligned_descriptor_from_csv_row,
    scheduler_runtime_descriptor_from_scheduled,
    vllm_like_scheduler_aligned_descriptors,
)
from aiconfigurator.sdk.backends.cb_simulator.simulator import CBSimulator
from aiconfigurator.sdk.backends.vllm_backend import VLLMBackend
from aiconfigurator.sdk.config import ModelConfig
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.perf_database import PerfDatabase

MODEL_PATH = "moonshotai/Kimi-K2.5"
SYSTEM = "h200_sxm"
BACKEND = "vllm"
DB_VERSION = "0.12.0"
DEFAULT_TP = 16

_VLLM_ITER_RE = re.compile(
    r"(?:(?P<engine_id>EngineCore_DP\d+).*?INFO\s+"
    r"(?P<date>\d\d-\d\d)\s+(?P<timestamp>\d\d:\d\d:\d\d).*?)?"
    r"Iteration\((?P<iter_index>\d+)\):\s*"
    r"(?P<context_requests>\d+)\s+context requests,\s*"
    r"(?P<context_tokens>\d+)\s+context tokens,\s*"
    r"(?P<generation_requests>\d+)\s+generation requests,\s*"
    r"(?P<generation_tokens>\d+)\s+generation tokens,\s*"
    r"iteration elapsed time:\s*(?P<iter_lat_ms>\d+(?:\.\d+)?)\s*ms"
)
_PROM_METRIC_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)

_PROM_METRIC_FIELDS = {
    "vllm:iteration_tokens_total_count": "iteration_count",
    "vllm:iteration_tokens_total_sum": "iteration_tokens",
    "vllm:request_prefill_time_seconds_count": "prefill_time_count",
    "vllm:request_prefill_time_seconds_sum": "prefill_time_s",
    "vllm:request_decode_time_seconds_count": "decode_time_count",
    "vllm:request_decode_time_seconds_sum": "decode_time_s",
    "vllm:prompt_tokens_total": "prompt_tokens",
    "vllm:generation_tokens_total": "generation_tokens",
    "vllm:request_success_total": "request_success",
    "vllm:num_preemptions_total": "preemptions",
}


@dataclass(frozen=True)
class CBIterationTraceRow:
    iter_index: int
    phase_type: str
    trace_type: str
    prefill_requests: int
    prefill_tokens: int
    decode_batch_size: int
    decode_avg_kv_len: int
    iter_lat_ms: float
    ctx_non_attn_ms: float
    ctx_attn_ms: float
    gen_non_attn_ms: float
    gen_attn_ms: float
    clock_ms: float
    completed_requests: int
    running_requests: int
    waiting_requests: int
    in_steady_state: int
    iter_overhead_ms: float = 0.0


@dataclass(frozen=True)
class VLLMIterationTraceRow:
    iter_index: int
    phase_type: str
    context_requests: int
    context_tokens: int
    generation_requests: int
    generation_tokens: int
    iter_lat_ms: float
    engine_id: str = ""
    timestamp: str = ""
    line_no: int = 0


@dataclass(frozen=True)
class PhaseLatencySummary:
    source: str
    phase_type: str
    num_iters: int
    mean_lat_ms: float
    p50_lat_ms: float
    p99_lat_ms: float
    mean_prefill_tokens: float
    mean_decode_batch_size: float
    mean_decode_avg_kv_len: float


@dataclass(frozen=True)
class PhaseCompareRow:
    phase_type: str
    ordinal_in_phase: int
    cb_iter_index: int
    cb_trace_type: str
    cb_iter_lat_ms: float
    vllm_iter_index: int
    vllm_iter_lat_ms: float
    overhead_ms: float
    vllm_context_tokens: int = 0
    vllm_generation_requests: int = 0


@dataclass(frozen=True)
class VLLMDecodeBinSummary:
    decode_bs_bin: str
    num_iters: int
    mean_lat_ms: float
    p50_lat_ms: float
    p99_lat_ms: float
    mean_context_tokens: float
    mean_generation_requests: float


@dataclass(frozen=True)
class PhaseCompareSummary:
    phase_type: str
    cb_iters: int
    vllm_iters: int
    aligned_pairs: int
    cb_mean_lat_ms: float
    vllm_mean_lat_ms: float
    overhead_mean_ms: float
    overhead_p50_ms: float
    overhead_p99_ms: float


@dataclass(frozen=True)
class MetricsSnapshot:
    iteration_count: float = 0.0
    iteration_tokens: float = 0.0
    prefill_time_count: float = 0.0
    prefill_time_s: float = 0.0
    decode_time_count: float = 0.0
    decode_time_s: float = 0.0
    prompt_tokens: float = 0.0
    generation_tokens: float = 0.0
    request_success: float = 0.0
    preemptions: float = 0.0


@dataclass(frozen=True)
class MetricsDeltaSummary:
    iter_count_delta: float
    iteration_tokens_delta: float
    avg_tokens_per_iter: float
    benchmark_wall_ms: float
    avg_iter_lat_ms: float
    prefill_time_count_delta: float
    avg_prefill_time_ms: float
    decode_time_count_delta: float
    avg_decode_time_ms: float
    prompt_tokens_delta: float
    generation_tokens_delta: float
    request_success_delta: float
    preemptions_delta: float


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    isl: int
    osl: int
    concurrency: int
    real_output_tok_s_gpu: float


@dataclass(frozen=True)
class CBScenarioSummary:
    name: str
    isl: int
    osl: int
    concurrency: int
    real_output_tok_s_gpu: float
    steady_output_tok_s_gpu: float
    steady_pred_real_ratio: float
    full_output_tok_s_gpu: float
    full_pred_real_ratio: float
    completed_requests: int
    total_iterations: int
    total_wall_ms: float
    prefill_iter_pct: float
    mixed_iter_pct: float
    pure_decode_iter_pct: float
    prefill_time_pct: float
    mixed_time_pct: float
    pure_decode_time_pct: float
    pure_decode_mean_bs: float
    pure_decode_mean_lat_ms: float
    pure_decode_p99_lat_ms: float
    pure_decode_mean_kv: float


@dataclass(frozen=True)
class SweepResult:
    overlap_factor: float
    per_iteration_overhead_ms: float
    throughput_max: float
    throughput_mean: float
    multi_config_max: float
    multi_config_mean: float
    ttft_max: float
    ttft_mean: float
    acceptance_score: float
    passed: int


VALIDATE_THROUGHPUT_SCENARIOS = [
    ScenarioSpec("3k-3k b=128", 3000, 3000, 128, 115.9),
    ScenarioSpec("8k-2k b=256", 8000, 2000, 256, 117.6),
    ScenarioSpec("10k-2k b=32", 10000, 2000, 32, 49.8),
    ScenarioSpec("10k-3k b=128", 10000, 3000, 128, 89.5),
    ScenarioSpec("16k-2k b=32", 16000, 2000, 32, 41.7),
    ScenarioSpec("30k-3k b=8", 30000, 3000, 8, 18.2),
    ScenarioSpec("32k-1k b=16", 32000, 1000, 16, 17.8),
]


def _load_model_and_db(
    tp: int,
    dp: int = 1,
    moe_tp: int | None = None,
    moe_ep: int = 1,
) -> tuple:
    model_config = ModelConfig(
        tp_size=tp,
        pp_size=1,
        attention_dp_size=dp,
        moe_tp_size=moe_tp if moe_tp is not None else tp,
        moe_ep_size=moe_ep,
    )
    model = get_model(MODEL_PATH, model_config, backend_name=BACKEND)
    systems_root = str(
        Path(__file__).resolve().parent.parent / "src" / "aiconfigurator" / "systems"
    )
    db = PerfDatabase(
        system=SYSTEM,
        backend=BACKEND,
        version=DB_VERSION,
        systems_root=systems_root,
    )
    backend = VLLMBackend()
    return model, db, backend


def _phase_type(*, prefill_tokens: int, decode_batch_size: int) -> str:
    if prefill_tokens > 0 and decode_batch_size > 0:
        return "mixed"
    if prefill_tokens > 0:
        return "prefill"
    return "pure_decode"


def _numpy_stat(values: list[float], fn: str) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=float)
    if fn == "mean":
        return float(np.mean(arr))
    if fn == "p50":
        return float(np.percentile(arr, 50))
    if fn == "p99":
        return float(np.percentile(arr, 99))
    raise ValueError(f"unknown stat {fn}")


def _write_rows(path: Path, rows: Iterable[object]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    delimiter = "," if path.suffix.lower() == ".csv" else "\t"
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _scenario_name(args: argparse.Namespace) -> str:
    return f"{args.isl}x{args.osl}_b{args.concurrency}"


def _topology_key_from_args(args: argparse.Namespace) -> str:
    moe_tp = args.moe_tp if args.moe_tp is not None else args.tp
    return make_topology_key(args.tp, args.dp, moe_tp, args.moe_ep)


def _parse_float_list(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("empty float list")
    return values


def _load_validate_module():
    script_path = Path(__file__).resolve().with_name("validate_cb_simulator.py")
    spec = importlib.util.spec_from_file_location("validate_cb_simulator", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_alpha_overhead_sweep(args: argparse.Namespace) -> list[SweepResult]:
    validate = _load_validate_module()
    rows: list[SweepResult] = []
    for alpha in _parse_float_list(args.alpha_values):
        for overhead_ms in _parse_float_list(args.overhead_values):
            result = validate.run_validation(
                overlap_factor=alpha,
                per_iteration_overhead_ms=overhead_ms,
                ep8_per_iteration_overhead_ms=args.ep8_per_iteration_overhead_ms,
                verbose=False,
            )
            passed = (
                result.throughput_max <= validate.THROUGHPUT_MAX_ACCEPTANCE
                and result.multi_config_max <= validate.MULTI_CONFIG_MAX_ACCEPTANCE
                and result.ttft_max <= validate.TTFT_MAX_ACCEPTANCE
            )
            rows.append(
                SweepResult(
                    overlap_factor=alpha,
                    per_iteration_overhead_ms=overhead_ms,
                    throughput_max=result.throughput_max,
                    throughput_mean=result.throughput_mean,
                    multi_config_max=result.multi_config_max,
                    multi_config_mean=result.multi_config_mean,
                    ttft_max=result.ttft_max,
                    ttft_mean=result.ttft_mean,
                    acceptance_score=max(
                        result.throughput_max / validate.THROUGHPUT_MAX_ACCEPTANCE,
                        result.multi_config_max / validate.MULTI_CONFIG_MAX_ACCEPTANCE,
                        result.ttft_max / validate.TTFT_MAX_ACCEPTANCE,
                    ),
                    passed=int(passed),
                )
            )
    return sorted(rows, key=lambda row: row.acceptance_score)


def parse_vllm_iteration_log(log_path: Path) -> list[VLLMIterationTraceRow]:
    rows: list[VLLMIterationTraceRow] = []
    for line_no, raw_line in enumerate(log_path.read_text().splitlines(), start=1):
        match = _VLLM_ITER_RE.search(raw_line)
        if match is None:
            continue
        context_tokens = int(match.group("context_tokens"))
        generation_requests = int(match.group("generation_requests"))
        engine_id = match.group("engine_id") or ""
        if engine_id.startswith("EngineCore_"):
            engine_id = engine_id.removeprefix("EngineCore_")
        rows.append(
            VLLMIterationTraceRow(
                iter_index=int(match.group("iter_index")),
                phase_type=_phase_type(
                    prefill_tokens=context_tokens,
                    decode_batch_size=generation_requests,
                ),
                context_requests=int(match.group("context_requests")),
                context_tokens=context_tokens,
                generation_requests=generation_requests,
                generation_tokens=int(match.group("generation_tokens")),
                iter_lat_ms=float(match.group("iter_lat_ms")),
                engine_id=engine_id,
                timestamp=match.group("timestamp") or "",
                line_no=line_no,
            )
        )
    return rows


def filter_vllm_rows(
    rows: list[VLLMIterationTraceRow],
    *,
    engine_id: str = "",
    start_time: str = "",
    end_time: str = "",
    min_decode_bs: int = 0,
    max_decode_bs: int = 0,
) -> list[VLLMIterationTraceRow]:
    filtered = rows
    if engine_id:
        normalized_engine = engine_id.removeprefix("EngineCore_")
        filtered = [row for row in filtered if row.engine_id == normalized_engine]
    if start_time:
        filtered = [
            row for row in filtered
            if row.timestamp and row.timestamp >= start_time
        ]
    if end_time:
        filtered = [
            row for row in filtered
            if row.timestamp and row.timestamp <= end_time
        ]
    if min_decode_bs > 0:
        filtered = [
            row for row in filtered
            if row.generation_requests >= min_decode_bs
        ]
    if max_decode_bs > 0:
        filtered = [
            row for row in filtered
            if row.generation_requests <= max_decode_bs
        ]
    return filtered


def parse_metrics_snapshot(path: Path) -> MetricsSnapshot:
    values = {field: 0.0 for field in MetricsSnapshot.__dataclass_fields__}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PROM_METRIC_RE.match(line)
        if match is None:
            continue
        field = _PROM_METRIC_FIELDS.get(match.group("name"))
        if field is None:
            continue
        values[field] += float(match.group("value"))
    return MetricsSnapshot(**values)


def compare_metrics_snapshots(
    before: MetricsSnapshot,
    after: MetricsSnapshot,
    benchmark_wall_ms: float = 0.0,
) -> MetricsDeltaSummary:
    iter_count_delta = after.iteration_count - before.iteration_count
    iteration_tokens_delta = after.iteration_tokens - before.iteration_tokens
    prefill_time_count_delta = (
        after.prefill_time_count - before.prefill_time_count
    )
    prefill_time_s_delta = after.prefill_time_s - before.prefill_time_s
    decode_time_count_delta = after.decode_time_count - before.decode_time_count
    decode_time_s_delta = after.decode_time_s - before.decode_time_s

    avg_tokens_per_iter = (
        iteration_tokens_delta / iter_count_delta
        if iter_count_delta > 0
        else 0.0
    )
    avg_iter_lat_ms = (
        benchmark_wall_ms / iter_count_delta
        if benchmark_wall_ms > 0 and iter_count_delta > 0
        else 0.0
    )
    avg_prefill_time_ms = (
        prefill_time_s_delta * 1000.0 / prefill_time_count_delta
        if prefill_time_count_delta > 0
        else 0.0
    )
    avg_decode_time_ms = (
        decode_time_s_delta * 1000.0 / decode_time_count_delta
        if decode_time_count_delta > 0
        else 0.0
    )

    return MetricsDeltaSummary(
        iter_count_delta=iter_count_delta,
        iteration_tokens_delta=iteration_tokens_delta,
        avg_tokens_per_iter=avg_tokens_per_iter,
        benchmark_wall_ms=benchmark_wall_ms,
        avg_iter_lat_ms=avg_iter_lat_ms,
        prefill_time_count_delta=prefill_time_count_delta,
        avg_prefill_time_ms=avg_prefill_time_ms,
        decode_time_count_delta=decode_time_count_delta,
        avg_decode_time_ms=avg_decode_time_ms,
        prompt_tokens_delta=after.prompt_tokens - before.prompt_tokens,
        generation_tokens_delta=after.generation_tokens - before.generation_tokens,
        request_success_delta=after.request_success - before.request_success,
        preemptions_delta=after.preemptions - before.preemptions,
    )


def summarize_cb_rows(rows: list[CBIterationTraceRow]) -> list[PhaseLatencySummary]:
    summaries: list[PhaseLatencySummary] = []
    for phase_type in ("prefill", "mixed", "pure_decode"):
        phase_rows = [row for row in rows if row.phase_type == phase_type]
        if not phase_rows:
            continue
        summaries.append(
            PhaseLatencySummary(
                source="cb_sim",
                phase_type=phase_type,
                num_iters=len(phase_rows),
                mean_lat_ms=_numpy_stat([row.iter_lat_ms for row in phase_rows], "mean"),
                p50_lat_ms=_numpy_stat([row.iter_lat_ms for row in phase_rows], "p50"),
                p99_lat_ms=_numpy_stat([row.iter_lat_ms for row in phase_rows], "p99"),
                mean_prefill_tokens=_numpy_stat(
                    [float(row.prefill_tokens) for row in phase_rows],
                    "mean",
                ),
                mean_decode_batch_size=_numpy_stat(
                    [float(row.decode_batch_size) for row in phase_rows],
                    "mean",
                ),
                mean_decode_avg_kv_len=_numpy_stat(
                    [float(row.decode_avg_kv_len) for row in phase_rows],
                    "mean",
                ),
            )
        )
    return summaries


def summarize_vllm_rows(rows: list[VLLMIterationTraceRow]) -> list[PhaseLatencySummary]:
    summaries: list[PhaseLatencySummary] = []
    for phase_type in ("prefill", "mixed", "pure_decode"):
        phase_rows = [row for row in rows if row.phase_type == phase_type]
        if not phase_rows:
            continue
        summaries.append(
            PhaseLatencySummary(
                source="vllm",
                phase_type=phase_type,
                num_iters=len(phase_rows),
                mean_lat_ms=_numpy_stat([row.iter_lat_ms for row in phase_rows], "mean"),
                p50_lat_ms=_numpy_stat([row.iter_lat_ms for row in phase_rows], "p50"),
                p99_lat_ms=_numpy_stat([row.iter_lat_ms for row in phase_rows], "p99"),
                mean_prefill_tokens=_numpy_stat(
                    [float(row.context_tokens) for row in phase_rows],
                    "mean",
                ),
                mean_decode_batch_size=_numpy_stat(
                    [float(row.generation_requests) for row in phase_rows],
                    "mean",
                ),
                mean_decode_avg_kv_len=0.0,
            )
        )
    return summaries


def _decode_bs_bin(decode_bs: int) -> str:
    if decode_bs <= 0:
        return "0"
    if decode_bs <= 8:
        return "1-8"
    if decode_bs <= 16:
        return "9-16"
    if decode_bs <= 32:
        return "17-32"
    if decode_bs <= 64:
        return "33-64"
    if decode_bs <= 128:
        return "65-128"
    return ">128"


def summarize_vllm_decode_bins(
    rows: list[VLLMIterationTraceRow],
) -> list[VLLMDecodeBinSummary]:
    summaries: list[VLLMDecodeBinSummary] = []
    for bin_name in ("0", "1-8", "9-16", "17-32", "33-64", "65-128", ">128"):
        bin_rows = [
            row for row in rows if _decode_bs_bin(row.generation_requests) == bin_name
        ]
        if not bin_rows:
            continue
        summaries.append(
            VLLMDecodeBinSummary(
                decode_bs_bin=bin_name,
                num_iters=len(bin_rows),
                mean_lat_ms=_numpy_stat([row.iter_lat_ms for row in bin_rows], "mean"),
                p50_lat_ms=_numpy_stat([row.iter_lat_ms for row in bin_rows], "p50"),
                p99_lat_ms=_numpy_stat([row.iter_lat_ms for row in bin_rows], "p99"),
                mean_context_tokens=_numpy_stat(
                    [float(row.context_tokens) for row in bin_rows],
                    "mean",
                ),
                mean_generation_requests=_numpy_stat(
                    [float(row.generation_requests) for row in bin_rows],
                    "mean",
                ),
            )
        )
    return summaries


def summarize_cb_scenario(
    *,
    spec: ScenarioSpec,
    rows: list[CBIterationTraceRow],
    tp: int,
    steady_output_tok_s_gpu: float,
) -> CBScenarioSummary:
    total_iters = len(rows)
    total_wall_ms = rows[-1].clock_ms if rows else 0.0
    completed_requests = rows[-1].completed_requests if rows else 0
    total_output_tokens = completed_requests * max(spec.osl - 1, 0)
    full_output_tok_s_gpu = (
        total_output_tokens / (total_wall_ms / 1000.0) / max(tp, 1)
        if total_wall_ms > 0
        else 0.0
    )
    full_pred_real_ratio = (
        full_output_tok_s_gpu / spec.real_output_tok_s_gpu
        if spec.real_output_tok_s_gpu > 0
        else 0.0
    )
    steady_pred_real_ratio = (
        steady_output_tok_s_gpu / spec.real_output_tok_s_gpu
        if spec.real_output_tok_s_gpu > 0
        else 0.0
    )

    def _phase_rows(phase_type: str) -> list[CBIterationTraceRow]:
        return [row for row in rows if row.phase_type == phase_type]

    def _pct(value: float, total: float) -> float:
        return value * 100.0 / total if total > 0 else 0.0

    prefill = _phase_rows("prefill")
    mixed = _phase_rows("mixed")
    pure_decode = _phase_rows("pure_decode")
    prefill_time = sum(row.iter_lat_ms for row in prefill)
    mixed_time = sum(row.iter_lat_ms for row in mixed)
    pure_decode_time = sum(row.iter_lat_ms for row in pure_decode)

    return CBScenarioSummary(
        name=spec.name,
        isl=spec.isl,
        osl=spec.osl,
        concurrency=spec.concurrency,
        real_output_tok_s_gpu=spec.real_output_tok_s_gpu,
        steady_output_tok_s_gpu=steady_output_tok_s_gpu,
        steady_pred_real_ratio=steady_pred_real_ratio,
        full_output_tok_s_gpu=full_output_tok_s_gpu,
        full_pred_real_ratio=full_pred_real_ratio,
        completed_requests=completed_requests,
        total_iterations=total_iters,
        total_wall_ms=total_wall_ms,
        prefill_iter_pct=_pct(len(prefill), total_iters),
        mixed_iter_pct=_pct(len(mixed), total_iters),
        pure_decode_iter_pct=_pct(len(pure_decode), total_iters),
        prefill_time_pct=_pct(prefill_time, total_wall_ms),
        mixed_time_pct=_pct(mixed_time, total_wall_ms),
        pure_decode_time_pct=_pct(pure_decode_time, total_wall_ms),
        pure_decode_mean_bs=_numpy_stat(
            [float(row.decode_batch_size) for row in pure_decode],
            "mean",
        ),
        pure_decode_mean_lat_ms=_numpy_stat(
            [row.iter_lat_ms for row in pure_decode],
            "mean",
        ),
        pure_decode_p99_lat_ms=_numpy_stat(
            [row.iter_lat_ms for row in pure_decode],
            "p99",
        ),
        pure_decode_mean_kv=_numpy_stat(
            [float(row.decode_avg_kv_len) for row in pure_decode],
            "mean",
        ),
    )


def compare_phasewise(
    cb_rows: list[CBIterationTraceRow],
    vllm_rows: list[VLLMIterationTraceRow],
) -> tuple[list[PhaseCompareRow], list[PhaseCompareSummary]]:
    compare_rows: list[PhaseCompareRow] = []
    compare_summaries: list[PhaseCompareSummary] = []
    for phase_type in ("prefill", "mixed", "pure_decode"):
        cb_phase = [row for row in cb_rows if row.phase_type == phase_type]
        vllm_phase = [row for row in vllm_rows if row.phase_type == phase_type]
        aligned = min(len(cb_phase), len(vllm_phase))
        overheads: list[float] = []
        for ordinal, (cb_row, vllm_row) in enumerate(
            zip(cb_phase[:aligned], vllm_phase[:aligned], strict=False),
            start=1,
        ):
            overhead = vllm_row.iter_lat_ms - cb_row.iter_lat_ms
            overheads.append(overhead)
            compare_rows.append(
                PhaseCompareRow(
                    phase_type=phase_type,
                    ordinal_in_phase=ordinal,
                    cb_iter_index=cb_row.iter_index,
                    cb_trace_type=cb_row.trace_type,
                    cb_iter_lat_ms=cb_row.iter_lat_ms,
                    vllm_iter_index=vllm_row.iter_index,
                    vllm_iter_lat_ms=vllm_row.iter_lat_ms,
                    overhead_ms=overhead,
                    vllm_context_tokens=vllm_row.context_tokens,
                    vllm_generation_requests=vllm_row.generation_requests,
                )
            )
        compare_summaries.append(
            PhaseCompareSummary(
                phase_type=phase_type,
                cb_iters=len(cb_phase),
                vllm_iters=len(vllm_phase),
                aligned_pairs=aligned,
                cb_mean_lat_ms=_numpy_stat([row.iter_lat_ms for row in cb_phase], "mean"),
                vllm_mean_lat_ms=_numpy_stat([row.iter_lat_ms for row in vllm_phase], "mean"),
                overhead_mean_ms=_numpy_stat(overheads, "mean"),
                overhead_p50_ms=_numpy_stat(overheads, "p50"),
                overhead_p99_ms=_numpy_stat(overheads, "p99"),
            )
        )
    return compare_rows, compare_summaries


def build_cb_forward_descriptors(
    rows: list[CBIterationTraceRow],
    args: argparse.Namespace,
) -> list[VLLMForwardDescriptor]:
    topology_key = _topology_key_from_args(args)
    moe_tp = args.moe_tp if args.moe_tp is not None else args.tp
    cfg = _make_cb_config(args)
    return [
        descriptor_from_scheduled(
            source="cb_sim",
            scenario=_scenario_name(args),
            iteration=row.iter_index,
            phase=row.phase_type,
            scheduled_context_tokens=row.prefill_tokens,
            scheduled_decode_tokens=row.decode_batch_size,
            topology_key=topology_key,
            tp=args.tp,
            dp=args.dp,
            moe_tp=moe_tp,
            moe_ep=args.moe_ep,
            max_num_batched_tokens=cfg.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            decode_avg_kv_len=row.decode_avg_kv_len,
        )
        for row in rows
    ]


def build_cb_scheduler_descriptors(
    rows: list[CBIterationTraceRow],
    args: argparse.Namespace,
) -> list[VLLMSchedulerRuntimeDescriptor]:
    topology_key = _topology_key_from_args(args)
    moe_tp = args.moe_tp if args.moe_tp is not None else args.tp
    cfg = _make_cb_config(args)
    return [
        scheduler_runtime_descriptor_from_scheduled(
            source="cb_sim",
            scenario=_scenario_name(args),
            iteration=row.iter_index,
            phase=row.phase_type,
            scheduled_context_tokens=row.prefill_tokens,
            scheduled_decode_tokens=row.decode_batch_size,
            scheduled_context_reqs=row.prefill_requests,
            scheduled_decode_reqs=row.decode_batch_size,
            max_num_batched_tokens=cfg.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            forward_token_count=row.prefill_tokens + row.decode_batch_size,
            cudagraph_runtime_mode="AIC_UNSET",
            topology_key=topology_key,
            tp=args.tp,
            dp=args.dp,
            moe_tp=moe_tp,
            moe_ep=args.moe_ep,
        )
        for row in rows
    ]


def build_cb_scheduler_aligned_descriptors(
    rows: list[CBIterationTraceRow],
    args: argparse.Namespace,
    *,
    dp_rank: int,
) -> list[VLLMSchedulerAlignedDescriptor]:
    topology_key = _topology_key_from_args(args)
    moe_tp = args.moe_tp if args.moe_tp is not None else args.tp
    cfg = _make_cb_config(args)
    return [
        scheduler_aligned_descriptor_from_scheduled(
            source="cb_sim",
            scenario=_scenario_name(args),
            dp_rank=dp_rank,
            engine_step_id=engine_step_id,
            phase=row.phase_type,
            scheduled_context_tokens=row.prefill_tokens,
            scheduled_decode_tokens=row.decode_batch_size,
            scheduled_context_reqs=row.prefill_requests,
            scheduled_decode_reqs=row.decode_batch_size,
            max_num_batched_tokens=cfg.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            forward_token_count=row.prefill_tokens + row.decode_batch_size,
            cudagraph_runtime_mode="AIC_UNSET",
            topology_key=topology_key,
            tp=args.tp,
            dp=args.dp,
            moe_tp=moe_tp,
            moe_ep=args.moe_ep,
        )
        for engine_step_id, row in enumerate(rows)
    ]


def parse_scheduler_aligned_descriptors(
    path: Path,
) -> list[VLLMSchedulerAlignedDescriptor]:
    with path.open(newline="") as f:
        return [
            scheduler_aligned_descriptor_from_csv_row(row)
            for row in csv.DictReader(f)
        ]


def build_cb_vllm_like_scheduler_aligned_descriptors(
    args: argparse.Namespace,
) -> list[VLLMSchedulerAlignedDescriptor]:
    topology_moe_tp = args.moe_tp if args.moe_tp is not None else args.tp
    return vllm_like_scheduler_aligned_descriptors(
        source="cb_sim_vllm_like",
        scenario=_scenario_name(args),
        isl=args.isl,
        osl=args.osl,
        concurrency=args.concurrency,
        max_num_batched_tokens=_make_cb_config(args).max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        tp=args.tp,
        dp=args.dp,
        moe_tp=topology_moe_tp,
        moe_ep=args.moe_ep,
        block_size=args.block_size,
    )


def parse_runtime_shape_descriptors(
    path: Path,
    args: argparse.Namespace,
) -> list[VLLMForwardDescriptor]:
    topology_key = _topology_key_from_args(args)
    moe_tp = args.moe_tp if args.moe_tp is not None else args.tp
    cfg = _make_cb_config(args)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return [
            descriptor_from_runtime_shape_summary(
                row,
                source="vllm_runtime_shape",
                scenario=_scenario_name(args),
                topology_key=topology_key,
                tp=args.tp,
                dp=args.dp,
                moe_tp=moe_tp,
                moe_ep=args.moe_ep,
                max_num_batched_tokens=cfg.max_num_batched_tokens,
                max_num_seqs=args.max_num_seqs,
            )
            for row in reader
        ]


def build_cb_runtime_shape_keys(
    rows: list[CBIterationTraceRow],
    args: argparse.Namespace,
) -> list[VLLMRuntimeShapeKey]:
    topology_key = _topology_key_from_args(args)
    moe_tp = args.moe_tp if args.moe_tp is not None else args.tp
    return [
        runtime_shape_key_from_scheduled(
            source="cb_sim",
            scenario=_scenario_name(args),
            iteration=row.iter_index,
            phase=row.phase_type,
            scheduled_context_tokens=row.prefill_tokens,
            scheduled_decode_tokens=row.decode_batch_size,
            topology_key=topology_key,
            tp=args.tp,
            dp=args.dp,
            moe_tp=moe_tp,
            moe_ep=args.moe_ep,
        )
        for row in rows
    ]


def parse_runtime_shape_keys(
    path: Path,
    args: argparse.Namespace,
) -> list[VLLMRuntimeShapeKey]:
    topology_key = _topology_key_from_args(args)
    moe_tp = args.moe_tp if args.moe_tp is not None else args.tp
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = [
            runtime_shape_key_from_rank_row(
                row,
                source="vllm_runtime_shape",
                scenario=_scenario_name(args),
                topology_key=topology_key,
                tp=args.tp,
                dp=args.dp,
                moe_tp=moe_tp,
                moe_ep=args.moe_ep,
            )
            for row in reader
            if int(float(row["rank"])) == args.runtime_shape_key_vllm_rank
        ]
    if not rows:
        raise ValueError(
            "no vLLM runtime shape key rows for rank "
            f"{args.runtime_shape_key_vllm_rank}"
        )
    return rows


def build_vllm_runtime_shape_key_distribution(
    path: Path,
    args: argparse.Namespace,
) -> list[RuntimeShapeSubkeyDistributionRow]:
    topology_key = _topology_key_from_args(args)
    moe_tp = args.moe_tp if args.moe_tp is not None else args.tp
    grouped: dict[
        tuple[str, object],
        dict[str, object],
    ] = {}
    with path.open(newline="") as f:
        for raw in csv.DictReader(f):
            key = runtime_shape_key_from_rank_row(
                raw,
                source="vllm_runtime_shape",
                scenario=_scenario_name(args),
                topology_key=topology_key,
                tp=args.tp,
                dp=args.dp,
                moe_tp=moe_tp,
                moe_ep=args.moe_ep,
            )
            rank = int(float(raw["rank"]))
            for key_type, subkey in (
                ("attention", attention_subkey_from_runtime_shape_key(key)),
                ("kv", kv_subkey_from_runtime_shape_key(key)),
                ("forward_wrapper", forward_wrapper_subkey_from_runtime_shape_key(key)),
            ):
                group_key = (key_type, subkey)
                if group_key not in grouped:
                    grouped[group_key] = {"count": 0, "iterations": set(), "ranks": set()}
                grouped[group_key]["count"] = int(grouped[group_key]["count"]) + 1
                grouped[group_key]["iterations"].add(key.iteration)
                grouped[group_key]["ranks"].add(rank)

    rows: list[RuntimeShapeSubkeyDistributionRow] = []
    for (key_type, subkey), stats in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            getattr(item[0][1], "phase"),
            str(item[0][1]),
        ),
    ):
        rows.append(
            RuntimeShapeSubkeyDistributionRow(
                key_type=key_type,
                scenario=_scenario_name(args),
                phase=getattr(subkey, "phase"),
                topology_key=getattr(subkey, "topology_key"),
                rank_rows=int(stats["count"]),
                iterations=len(stats["iterations"]),
                ranks=",".join(str(rank) for rank in sorted(stats["ranks"])),
                cudagraph_runtime_mode=str(
                    getattr(subkey, "cudagraph_runtime_mode", "")
                ),
                attention_actual_tokens=str(
                    getattr(subkey, "attention_actual_tokens", "")
                ),
                attention_max_query_len=str(
                    getattr(subkey, "attention_max_query_len", "")
                ),
                slot_mapping_tokens=str(getattr(subkey, "slot_mapping_tokens", "")),
                block_table_shape=str(getattr(subkey, "block_table_shape", "")),
                forward_context_tokens=str(
                    getattr(subkey, "forward_context_tokens", "")
                ),
                forward_token_count=str(getattr(subkey, "forward_token_count", "")),
            )
        )
    return rows


def build_compiled_body_runtime_key(
    args: argparse.Namespace,
) -> VLLMCompiledBodyRuntimeKey:
    moe_tp = args.moe_tp if args.moe_tp is not None else args.tp
    if (args.tp, args.dp, moe_tp, args.moe_ep) != (4, 2, 1, 8):
        raise ValueError(
            "--experimental-compiled-body-key currently requires "
            "tp=4, dp=2, moe_tp=1, moe_ep=8 to match Phase 39 evidence"
        )
    return compiled_body_runtime_key_from_nccl_summary_csv(
        args.compiled_body_key_nccl_summary,
        source="phase39_nccl_trace",
        scenario=_scenario_name(args),
        phase="mixed",
        topology_key=_topology_key_from_args(args),
        tp=args.tp,
        dp=args.dp,
        ep=args.moe_ep,
        world_size=args.tp * args.dp,
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


def build_moe_source_runtime_key(args: argparse.Namespace):
    moe_tp = args.moe_tp if args.moe_tp is not None else args.tp
    if (args.tp, args.dp, moe_tp, args.moe_ep) != (4, 2, 1, 8):
        raise ValueError(
            "--experimental-moe-source-key currently requires "
            "tp=4, dp=2, moe_tp=1, moe_ep=8 to match Phase 82 evidence"
        )
    return moe_source_runtime_key_from_loaded_weight_boundary_row(
        {
            "source": "phase82_loaded_weight_boundary",
            "scenario": _scenario_name(args),
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
            "tp_size": args.tp,
            "dp_size": args.dp,
            "ep_size": args.moe_ep,
            "world_size": args.tp * args.dp,
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


def _make_cb_config(args: argparse.Namespace) -> CBSimConfig:
    num_requests = args.num_requests
    if num_requests <= 0:
        num_requests = max(200, args.concurrency * 3)
    warmup_requests = args.warmup_requests
    if warmup_requests <= 0:
        warmup_requests = max(50, args.concurrency)
    max_num_batched_tokens = args.max_num_batched_tokens
    if max_num_batched_tokens <= 0:
        max_num_batched_tokens = args.isl
    return CBSimConfig(
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        num_requests=num_requests,
        warmup_requests=warmup_requests,
        long_prefill_token_threshold=args.long_prefill_token_threshold,
        num_gpu_blocks=args.num_gpu_blocks,
        block_size=args.block_size,
        overlap_factor=args.overlap_factor,
        per_iteration_overhead_ms=args.per_iteration_overhead_ms,
    )


def collect_cb_iteration_trace(args: argparse.Namespace) -> list[CBIterationTraceRow]:
    model, db, backend = _load_model_and_db(
        tp=args.tp,
        dp=args.dp,
        moe_tp=args.moe_tp,
        moe_ep=args.moe_ep,
    )
    cfg = _make_cb_config(args)
    sim = CBSimulator(backend, model, db, cfg)
    latency_calc = sim._create_latency_calc(prefix=args.prefix)

    waiting: list[Request] = []
    running: list[Request] = []
    completed: list[Request] = []
    rows: list[CBIterationTraceRow] = []
    clock_ms = 0.0
    next_id = 0
    total_iters = 0

    for _ in range(min(args.concurrency, cfg.num_requests)):
        waiting.append(
            Request(
                request_id=next_id,
                isl=args.isl,
                osl=args.osl,
                arrival_time_ms=0.0,
            )
        )
        next_id += 1

    max_iters = cfg.num_requests * (
        args.osl + args.isl // cfg.max_num_batched_tokens + 10
    )
    while len(completed) < cfg.num_requests and total_iters < max_iters:
        has_external_supply = bool(waiting) or next_id < cfg.num_requests
        schedule = sim._scheduler.schedule(waiting, running)
        if schedule.is_empty:
            break

        avg_kv = (
            int(np.mean([r.kv_cache_len for r in schedule.decode_reqs]))
            if schedule.decode_reqs
            else 0
        )
        iter_lat = latency_calc.compute(
            prefill_tokens=schedule.total_prefill_tokens,
            prefill_batch_size=len(schedule.prefill_reqs),
            prefill_seq_len=args.isl,
            decode_batch_size=len(schedule.decode_reqs),
            decode_avg_kv_len=avg_kv,
        )
        breakdown = latency_calc.get_last_breakdown()
        assert breakdown is not None

        in_steady_state = (
            len(completed) >= cfg.warmup_requests and has_external_supply
        )
        clock_ms += iter_lat
        total_iters += 1

        for req in schedule.prefill_reqs:
            tokens = schedule.prefill_tokens[req.request_id]
            if req.state in {RequestState.WAITING, RequestState.PREEMPTED}:
                req.state = RequestState.PREFILLING
                if req.prefill_start_ms < 0:
                    req.prefill_start_ms = clock_ms - iter_lat
                if req in waiting:
                    waiting.remove(req)
                running.append(req)

            req.prefill_tokens_remaining -= tokens
            if req.prefill_tokens_remaining <= 0:
                req.prefill_tokens_remaining = 0
                if req.first_token_ms < 0:
                    req.first_token_ms = clock_ms
                req.state = RequestState.DECODING

        newly_done: list[Request] = []
        for req in schedule.decode_reqs:
            req.generated_tokens += 1
            if req.generated_tokens >= req.osl - 1:
                req.state = RequestState.DONE
                req.finish_ms = clock_ms
                newly_done.append(req)

        for req in newly_done:
            running.remove(req)
            completed.append(req)
            if next_id < cfg.num_requests:
                waiting.append(
                    Request(
                        request_id=next_id,
                        isl=args.isl,
                        osl=args.osl,
                        arrival_time_ms=clock_ms,
                    )
                )
                next_id += 1

        rows.append(
            CBIterationTraceRow(
                iter_index=total_iters,
                phase_type=_phase_type(
                    prefill_tokens=schedule.total_prefill_tokens,
                    decode_batch_size=len(schedule.decode_reqs),
                ),
                trace_type="scheduled",
                prefill_requests=len(schedule.prefill_reqs),
                prefill_tokens=schedule.total_prefill_tokens,
                decode_batch_size=len(schedule.decode_reqs),
                decode_avg_kv_len=avg_kv,
                iter_lat_ms=breakdown.total_ms,
                ctx_non_attn_ms=breakdown.context_non_attention_ms,
                ctx_attn_ms=breakdown.context_attention_ms,
                gen_non_attn_ms=breakdown.generation_non_attention_ms,
                gen_attn_ms=breakdown.generation_attention_ms,
                clock_ms=clock_ms,
                completed_requests=len(completed),
                running_requests=len(running),
                waiting_requests=len(waiting),
                in_steady_state=int(in_steady_state),
                iter_overhead_ms=breakdown.iteration_overhead_ms,
            )
        )

        if (
            not waiting
            and all(r.state == RequestState.DECODING for r in running)
            and running
        ):
            min_remaining = min(
                r.osl - 1 - r.generated_tokens for r in running
            )
            skip_iters = max(0, min_remaining - 1)
            if skip_iters > 0:
                if args.expand_skips:
                    for offset in range(skip_iters):
                        skip_avg_kv = avg_kv + 1 + offset
                        skip_iter_lat = latency_calc.compute(
                            prefill_tokens=0,
                            prefill_batch_size=0,
                            prefill_seq_len=1,
                            decode_batch_size=len(running),
                            decode_avg_kv_len=skip_avg_kv,
                        )
                        skip_breakdown = latency_calc.get_last_breakdown()
                        assert skip_breakdown is not None
                        clock_ms += skip_iter_lat
                        total_iters += 1
                        rows.append(
                            CBIterationTraceRow(
                                iter_index=total_iters,
                                phase_type="pure_decode",
                                trace_type="skip_expanded",
                                prefill_requests=0,
                                prefill_tokens=0,
                                decode_batch_size=len(running),
                                decode_avg_kv_len=skip_avg_kv,
                                iter_lat_ms=skip_breakdown.total_ms,
                                ctx_non_attn_ms=skip_breakdown.context_non_attention_ms,
                                ctx_attn_ms=skip_breakdown.context_attention_ms,
                                gen_non_attn_ms=skip_breakdown.generation_non_attention_ms,
                                gen_attn_ms=skip_breakdown.generation_attention_ms,
                                clock_ms=clock_ms,
                                completed_requests=len(completed),
                                running_requests=len(running),
                                waiting_requests=0,
                                in_steady_state=int(in_steady_state),
                                iter_overhead_ms=skip_breakdown.iteration_overhead_ms,
                            )
                        )
                    for req in running:
                        req.generated_tokens += skip_iters
                else:
                    skip_lat = sim._estimate_decode_skip_latency(
                        latency_calc=latency_calc,
                        decode_batch_size=len(running),
                        start_avg_kv_len=avg_kv + 1,
                        first_iter_latency_ms=iter_lat,
                        skip_iters=skip_iters,
                    )
                    clock_ms += skip_lat
                    total_iters += skip_iters
                    rows.append(
                        CBIterationTraceRow(
                            iter_index=total_iters,
                            phase_type="pure_decode",
                            trace_type="skip_aggregate",
                            prefill_requests=0,
                            prefill_tokens=0,
                            decode_batch_size=len(running),
                            decode_avg_kv_len=avg_kv + 1,
                            iter_lat_ms=skip_lat,
                            ctx_non_attn_ms=0.0,
                            ctx_attn_ms=0.0,
                            gen_non_attn_ms=0.0,
                            gen_attn_ms=0.0,
                            clock_ms=clock_ms,
                            completed_requests=len(completed),
                            running_requests=len(running),
                            waiting_requests=0,
                            in_steady_state=int(in_steady_state),
                            iter_overhead_ms=cfg.per_iteration_overhead_ms * skip_iters,
                        )
                    )
                    for req in running:
                        req.generated_tokens += skip_iters

    return rows


def run_cb_sim(args: argparse.Namespace):
    model, db, backend = _load_model_and_db(
        tp=args.tp,
        dp=args.dp,
        moe_tp=args.moe_tp,
        moe_ep=args.moe_ep,
    )
    cfg = _make_cb_config(args)
    sim = CBSimulator(backend, model, db, cfg)
    return sim.run(
        isl=args.isl,
        osl=args.osl,
        concurrency=args.concurrency,
        prefix=args.prefix,
        num_gpus=args.tp,
    )


def _print_phase_summaries(title: str, rows: list[PhaseLatencySummary]) -> None:
    print(title)
    print(
        f"{'source':<8} {'phase':<12} {'iters':>8} {'mean_ms':>10} {'p50_ms':>10} "
        f"{'p99_ms':>10} {'mean_prefill_tok':>18} {'mean_decode_bs':>16} {'mean_kv':>12}"
    )
    print("-" * 120)
    for row in rows:
        print(
            f"{row.source:<8} {row.phase_type:<12} {row.num_iters:>8d} "
            f"{row.mean_lat_ms:>10.2f} {row.p50_lat_ms:>10.2f} {row.p99_lat_ms:>10.2f} "
            f"{row.mean_prefill_tokens:>18.2f} {row.mean_decode_batch_size:>16.2f} "
            f"{row.mean_decode_avg_kv_len:>12.2f}"
        )
    print()


def _print_compare_summaries(rows: list[PhaseCompareSummary]) -> None:
    print("cb_sim vs vLLM phase-wise alignment")
    print(
        f"{'phase':<12} {'cb_iters':>8} {'vllm_iters':>10} {'aligned':>8} "
        f"{'cb_mean_ms':>12} {'real_mean_ms':>14} {'overhead_mean':>14} "
        f"{'overhead_p50':>14} {'overhead_p99':>14}"
    )
    print("-" * 126)
    for row in rows:
        print(
            f"{row.phase_type:<12} {row.cb_iters:>8d} {row.vllm_iters:>10d} "
            f"{row.aligned_pairs:>8d} {row.cb_mean_lat_ms:>12.2f} "
            f"{row.vllm_mean_lat_ms:>14.2f} {row.overhead_mean_ms:>14.2f} "
            f"{row.overhead_p50_ms:>14.2f} {row.overhead_p99_ms:>14.2f}"
        )
    print()


def _print_metrics_delta(row: MetricsDeltaSummary) -> None:
    print("vLLM metrics delta summary")
    print(
        f"{'iter_delta':>12} {'tok_delta':>12} {'tok/iter':>10} "
        f"{'wall_ms':>12} {'iter_lat_ms':>12} {'prefill_ms':>12} "
        f"{'decode_ms':>12} {'prompt_tok':>12} {'gen_tok':>12} "
        f"{'success':>10} {'preempt':>10}"
    )
    print("-" * 136)
    print(
        f"{row.iter_count_delta:>12.0f} {row.iteration_tokens_delta:>12.0f} "
        f"{row.avg_tokens_per_iter:>10.2f} {row.benchmark_wall_ms:>12.2f} "
        f"{row.avg_iter_lat_ms:>12.2f} {row.avg_prefill_time_ms:>12.2f} "
        f"{row.avg_decode_time_ms:>12.2f} {row.prompt_tokens_delta:>12.0f} "
        f"{row.generation_tokens_delta:>12.0f} {row.request_success_delta:>10.0f} "
        f"{row.preemptions_delta:>10.0f}"
    )
    print()


def _print_decode_bin_summaries(rows: list[VLLMDecodeBinSummary]) -> None:
    print("vLLM decode-batch latency summary")
    print(
        f"{'decode_bs':<10} {'iters':>8} {'mean_ms':>10} {'p50_ms':>10} "
        f"{'p99_ms':>10} {'mean_ctx_tok':>14} {'mean_gen_req':>14}"
    )
    print("-" * 86)
    for row in rows:
        print(
            f"{row.decode_bs_bin:<10} {row.num_iters:>8d} "
            f"{row.mean_lat_ms:>10.2f} {row.p50_lat_ms:>10.2f} "
            f"{row.p99_lat_ms:>10.2f} {row.mean_context_tokens:>14.2f} "
            f"{row.mean_generation_requests:>14.2f}"
        )
    print()


def _print_scenario_summaries(rows: list[CBScenarioSummary]) -> None:
    print("cb_sim validate-scenario summary")
    print(
        f"{'scenario':<16} {'bs':>5} {'real':>8} {'steady':>8} {'s/r':>8} "
        f"{'full':>8} {'f/r':>8} {'done':>6} {'iters':>8} {'wall_s':>10} "
        f"{'mix_i%':>8} {'dec_i%':>8} "
        f"{'mix_t%':>8} {'dec_t%':>8} {'dec_bs':>8} {'dec_ms':>8}"
    )
    print("-" * 148)
    for row in rows:
        print(
            f"{row.name:<16} {row.concurrency:>5d} {row.real_output_tok_s_gpu:>8.1f} "
            f"{row.steady_output_tok_s_gpu:>8.1f} {row.steady_pred_real_ratio:>7.2f}x "
            f"{row.full_output_tok_s_gpu:>8.1f} {row.full_pred_real_ratio:>7.2f}x "
            f"{row.completed_requests:>6d} {row.total_iterations:>8d} "
            f"{row.total_wall_ms / 1000.0:>10.2f} "
            f"{row.mixed_iter_pct:>8.1f} {row.pure_decode_iter_pct:>8.1f} "
            f"{row.mixed_time_pct:>8.1f} {row.pure_decode_time_pct:>8.1f} "
            f"{row.pure_decode_mean_bs:>8.1f} {row.pure_decode_mean_lat_ms:>8.2f}"
        )
    print()


def _print_sweep_results(rows: list[SweepResult]) -> None:
    print("alpha/overhead validation sweep")
    print(
        f"{'alpha':>6} {'overhead':>9} {'thr_max':>8} {'multi_max':>10} "
        f"{'ttft_max':>9} {'score':>8} {'pass':>5}"
    )
    print("-" * 64)
    for row in rows:
        print(
            f"{row.overlap_factor:>6.2f} {row.per_iteration_overhead_ms:>9.2f} "
            f"{row.throughput_max:>8.2f} {row.multi_config_max:>10.2f} "
            f"{row.ttft_max:>9.2f} {row.acceptance_score:>8.2f} "
            f"{'yes' if row.passed else 'no':>5}"
        )
    print()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose cb_sim iteration latency and compare with vLLM logs."
    )
    parser.add_argument("--isl", type=int, default=3000)
    parser.add_argument("--osl", type=int, default=3000)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--tp", type=int, default=DEFAULT_TP)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--moe-tp", type=int)
    parser.add_argument("--moe-ep", type=int, default=1)
    parser.add_argument("--prefix", type=int, default=0)
    parser.add_argument("--max-num-batched-tokens", type=int, default=0)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--num-requests", type=int, default=0)
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument("--long-prefill-token-threshold", type=int, default=0)
    parser.add_argument("--num-gpu-blocks", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--overlap-factor", type=float, default=0.0)
    parser.add_argument("--per-iteration-overhead-ms", type=float, default=0.0)
    parser.add_argument("--ep8-per-iteration-overhead-ms", type=float, default=90.0)
    parser.add_argument(
        "--experimental-forward-descriptor",
        action="store_true",
        help=(
            "Emit vLLM forward descriptor diagnostics only. "
            "This does not change cb_sim latency."
        ),
    )
    parser.add_argument(
        "--vllm-runtime-shape-summary",
        type=Path,
        help="Phase 7 runtime-shape iteration_summary.csv for descriptor compare.",
    )
    parser.add_argument("--forward-descriptor-out", type=Path)
    parser.add_argument("--forward-descriptor-compare-out", type=Path)
    parser.add_argument(
        "--experimental-runtime-shape-key",
        action="store_true",
        help=(
            "Emit vLLM runtime shape key diagnostics only. "
            "This does not change cb_sim latency."
        ),
    )
    parser.add_argument(
        "--vllm-runtime-shape-rank-rows",
        type=Path,
        help="Phase 10 runtime_shape_rank_rows.csv for runtime shape key compare.",
    )
    parser.add_argument("--runtime-shape-key-out", type=Path)
    parser.add_argument("--runtime-shape-key-compare-out", type=Path)
    parser.add_argument("--runtime-shape-key-distribution-out", type=Path)
    parser.add_argument("--runtime-shape-key-vllm-rank", type=int, default=0)
    parser.add_argument(
        "--experimental-scheduler-descriptor",
        action="store_true",
        help=(
            "Emit scheduler/runtime descriptor diagnostics only. "
            "This does not change cb_sim latency."
        ),
    )
    parser.add_argument("--scheduler-descriptor-out", type=Path)
    parser.add_argument(
        "--experimental-scheduler-alignment-descriptor",
        action="store_true",
        help=(
            "Emit DP-aware scheduler alignment descriptors only. "
            "This does not change cb_sim latency."
        ),
    )
    parser.add_argument("--scheduler-alignment-descriptor-out", type=Path)
    parser.add_argument("--scheduler-alignment-dp-rank", type=int)
    parser.add_argument("--scheduler-alignment-vllm-csv", type=Path)
    parser.add_argument("--scheduler-alignment-compare-out", type=Path)
    parser.add_argument(
        "--experimental-vllm-like-scheduler-descriptor",
        action="store_true",
        help=(
            "Emit an experimental vLLM-like DP scheduler descriptor. "
            "This does not change cb_sim latency."
        ),
    )
    parser.add_argument("--vllm-like-scheduler-descriptor-out", type=Path)
    parser.add_argument(
        "--experimental-compiled-body-key",
        action="store_true",
        help=(
            "Emit vLLM compiled-body runtime key diagnostics only. "
            "This does not change cb_sim latency."
        ),
    )
    parser.add_argument("--compiled-body-key-out", type=Path)
    parser.add_argument(
        "--compiled-body-key-nccl-summary",
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
            "This does not change cb_sim latency."
        ),
    )
    parser.add_argument("--moe-source-key-out", type=Path)
    parser.add_argument(
        "--sweep-alpha-overhead",
        action="store_true",
        help="Run validate_cb_simulator.py across alpha/overhead candidates.",
    )
    parser.add_argument(
        "--alpha-values",
        default="0.0,0.2,0.4,0.6,0.8,1.0",
        help="Comma-separated overlap_factor candidates.",
    )
    parser.add_argument(
        "--overhead-values",
        default="0,2,4,6,8",
        help="Comma-separated per-iteration overhead candidates in ms.",
    )
    parser.add_argument("--sweep-out", type=Path)
    parser.add_argument(
        "--expand-skips",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Expand pure-decode skip optimization into per-iteration rows.",
    )
    parser.add_argument("--vllm-log", type=Path)
    parser.add_argument("--vllm-engine-id", default="")
    parser.add_argument("--vllm-start-time", default="")
    parser.add_argument("--vllm-end-time", default="")
    parser.add_argument("--vllm-min-decode-bs", type=int, default=0)
    parser.add_argument("--vllm-max-decode-bs", type=int, default=0)
    parser.add_argument("--metrics-before", type=Path)
    parser.add_argument("--metrics-after", type=Path)
    parser.add_argument(
        "--benchmark-wall-ms",
        type=float,
        default=0.0,
        help="Benchmark wall time between metrics snapshots, in milliseconds.",
    )
    parser.add_argument("--cb-trace-out", type=Path)
    parser.add_argument("--vllm-trace-out", type=Path)
    parser.add_argument("--compare-out", type=Path)
    parser.add_argument("--metrics-delta-out", type=Path)
    parser.add_argument(
        "--validate-scenarios",
        action="store_true",
        help="Run cb_sim traces for the seven validate_cb_simulator throughput scenarios.",
    )
    parser.add_argument("--scenario-summary-out", type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.sweep_alpha_overhead:
        rows = run_alpha_overhead_sweep(args)
        _print_sweep_results(rows)
        if args.sweep_out is not None:
            _write_rows(args.sweep_out, rows)
            print(f"wrote sweep: {args.sweep_out}")
        return

    if args.experimental_forward_descriptor:
        if (
            args.forward_descriptor_out is None
            and args.forward_descriptor_compare_out is None
        ):
            raise ValueError(
                "--experimental-forward-descriptor requires "
                "--forward-descriptor-out or --forward-descriptor-compare-out"
            )
        if (
            args.forward_descriptor_compare_out is not None
            and args.vllm_runtime_shape_summary is None
        ):
            raise ValueError(
                "--forward-descriptor-compare-out requires "
                "--vllm-runtime-shape-summary"
            )

    if args.experimental_runtime_shape_key:
        if (
            args.runtime_shape_key_out is None
            and args.runtime_shape_key_compare_out is None
            and args.runtime_shape_key_distribution_out is None
        ):
            raise ValueError(
                "--experimental-runtime-shape-key requires "
                "--runtime-shape-key-out, --runtime-shape-key-compare-out, "
                "or --runtime-shape-key-distribution-out"
            )
        if (
            args.runtime_shape_key_compare_out is not None
            and args.vllm_runtime_shape_rank_rows is None
        ):
            raise ValueError(
                "--runtime-shape-key-compare-out requires "
                "--vllm-runtime-shape-rank-rows"
            )
        if (
            args.runtime_shape_key_distribution_out is not None
            and args.vllm_runtime_shape_rank_rows is None
        ):
            raise ValueError(
                "--runtime-shape-key-distribution-out requires "
                "--vllm-runtime-shape-rank-rows"
            )

    if args.experimental_scheduler_descriptor:
        if args.scheduler_descriptor_out is None:
            raise ValueError(
                "--experimental-scheduler-descriptor requires "
                "--scheduler-descriptor-out"
            )

    if args.experimental_scheduler_alignment_descriptor:
        if args.scheduler_alignment_descriptor_out is None:
            raise ValueError(
                "--experimental-scheduler-alignment-descriptor requires "
                "--scheduler-alignment-descriptor-out"
            )
        if (
            args.scheduler_alignment_compare_out is not None
            and args.scheduler_alignment_vllm_csv is None
        ):
            raise ValueError(
                "--scheduler-alignment-compare-out requires "
                "--scheduler-alignment-vllm-csv"
            )
        if args.scheduler_alignment_dp_rank is None and args.dp > 1:
            raise ValueError(
                "--experimental-scheduler-alignment-descriptor with dp > 1 "
                "requires --scheduler-alignment-dp-rank"
            )

    if args.experimental_vllm_like_scheduler_descriptor:
        if args.vllm_like_scheduler_descriptor_out is None:
            raise ValueError(
                "--experimental-vllm-like-scheduler-descriptor requires "
                "--vllm-like-scheduler-descriptor-out"
            )
        if (
            args.scheduler_alignment_compare_out is not None
            and args.scheduler_alignment_vllm_csv is None
        ):
            raise ValueError(
                "--scheduler-alignment-compare-out requires "
                "--scheduler-alignment-vllm-csv"
            )
        rows = build_cb_vllm_like_scheduler_aligned_descriptors(args)
        _write_rows(args.vllm_like_scheduler_descriptor_out, rows)
        print(
            "wrote vLLM-like scheduler descriptors: "
            f"{args.vllm_like_scheduler_descriptor_out}"
        )
        if args.scheduler_alignment_compare_out is not None:
            vllm_rows = parse_scheduler_aligned_descriptors(
                args.scheduler_alignment_vllm_csv
            )
            compare_rows = compare_scheduler_aligned_descriptors(rows, vllm_rows)
            _write_rows(args.scheduler_alignment_compare_out, compare_rows)
            print(
                "wrote vLLM-like scheduler strict compare: "
                f"{args.scheduler_alignment_compare_out}"
            )
        return

    if args.experimental_compiled_body_key:
        if args.compiled_body_key_out is None:
            raise ValueError(
                "--experimental-compiled-body-key requires --compiled-body-key-out"
            )
        key = build_compiled_body_runtime_key(args)
        _write_rows(args.compiled_body_key_out, [key])
        print(f"wrote compiled-body runtime key: {args.compiled_body_key_out}")
        return

    if args.experimental_moe_source_key:
        if args.moe_source_key_out is None:
            raise ValueError(
                "--experimental-moe-source-key requires --moe-source-key-out"
            )
        key = build_moe_source_runtime_key(args)
        _write_rows(args.moe_source_key_out, [key])
        print(f"wrote MoE source runtime key: {args.moe_source_key_out}")
        return

    if (
        args.experimental_runtime_shape_key
        and args.runtime_shape_key_distribution_out is not None
        and args.runtime_shape_key_out is None
        and args.runtime_shape_key_compare_out is None
    ):
        distribution_rows = build_vllm_runtime_shape_key_distribution(
            args.vllm_runtime_shape_rank_rows,
            args,
        )
        _write_rows(args.runtime_shape_key_distribution_out, distribution_rows)
        print(
            "wrote vLLM runtime shape key distribution: "
            f"{args.runtime_shape_key_distribution_out}"
        )
        return

    if args.validate_scenarios:
        summaries: list[CBScenarioSummary] = []
        for spec in VALIDATE_THROUGHPUT_SCENARIOS:
            scenario_args = argparse.Namespace(**vars(args))
            scenario_args.isl = spec.isl
            scenario_args.osl = spec.osl
            scenario_args.concurrency = spec.concurrency
            scenario_args.num_requests = max(200, spec.concurrency * 3)
            scenario_args.warmup_requests = max(50, spec.concurrency)
            scenario_args.max_num_batched_tokens = spec.isl
            rows = collect_cb_iteration_trace(scenario_args)
            result = run_cb_sim(scenario_args)
            summaries.append(
                summarize_cb_scenario(
                    spec=spec,
                    rows=rows,
                    tp=args.tp,
                    steady_output_tok_s_gpu=result.throughput_tok_s_gpu,
                )
            )
        _print_scenario_summaries(summaries)
        if args.scenario_summary_out is not None:
            _write_rows(args.scenario_summary_out, summaries)
            print(f"wrote scenario summary: {args.scenario_summary_out}")
        return

    cb_rows = collect_cb_iteration_trace(args)
    cb_summaries = summarize_cb_rows(cb_rows)

    print(
        f"cb_sim trace complete: isl={args.isl}, osl={args.osl}, "
        f"concurrency={args.concurrency}, tp={args.tp}, dp={args.dp}, "
        f"moe_tp={args.moe_tp if args.moe_tp is not None else args.tp}, "
        f"moe_ep={args.moe_ep}, expand_skips={args.expand_skips}"
    )
    print(f"cb_sim iterations: {len(cb_rows)}")
    print()
    _print_phase_summaries("cb_sim phase summary", cb_summaries)

    if args.cb_trace_out is not None:
        _write_rows(args.cb_trace_out, cb_rows)
        print(f"wrote cb trace: {args.cb_trace_out}")

    if args.experimental_forward_descriptor:
        cb_descriptors = build_cb_forward_descriptors(cb_rows, args)
        if args.forward_descriptor_out is not None:
            _write_rows(args.forward_descriptor_out, cb_descriptors)
            print(f"wrote cb forward descriptors: {args.forward_descriptor_out}")
        if args.forward_descriptor_compare_out is not None:
            vllm_descriptors = parse_runtime_shape_descriptors(
                args.vllm_runtime_shape_summary,
                args,
            )
            compare_rows = compare_forward_descriptors(
                cb_descriptors,
                vllm_descriptors,
            )
            _write_rows(args.forward_descriptor_compare_out, compare_rows)
            print(
                "wrote forward descriptor compare: "
                f"{args.forward_descriptor_compare_out}"
            )

    if args.experimental_runtime_shape_key:
        cb_keys = build_cb_runtime_shape_keys(cb_rows, args)
        if args.runtime_shape_key_out is not None:
            _write_rows(args.runtime_shape_key_out, cb_keys)
            print(f"wrote cb runtime shape keys: {args.runtime_shape_key_out}")
        if args.runtime_shape_key_compare_out is not None:
            vllm_keys = parse_runtime_shape_keys(
                args.vllm_runtime_shape_rank_rows,
                args,
            )
            compare_rows = compare_runtime_shape_keys(cb_keys, vllm_keys)
            _write_rows(args.runtime_shape_key_compare_out, compare_rows)
            print(
                "wrote runtime shape key compare: "
                f"{args.runtime_shape_key_compare_out}"
            )
        if args.runtime_shape_key_distribution_out is not None:
            distribution_rows = build_vllm_runtime_shape_key_distribution(
                args.vllm_runtime_shape_rank_rows,
                args,
            )
            _write_rows(args.runtime_shape_key_distribution_out, distribution_rows)
            print(
                "wrote vLLM runtime shape key distribution: "
                f"{args.runtime_shape_key_distribution_out}"
            )

    if args.experimental_scheduler_descriptor:
        scheduler_descriptors = build_cb_scheduler_descriptors(cb_rows, args)
        _write_rows(args.scheduler_descriptor_out, scheduler_descriptors)
        print(f"wrote scheduler descriptors: {args.scheduler_descriptor_out}")

    if args.experimental_scheduler_alignment_descriptor:
        dp_rank = (
            args.scheduler_alignment_dp_rank
            if args.scheduler_alignment_dp_rank is not None
            else 0
        )
        scheduler_alignment_descriptors = build_cb_scheduler_aligned_descriptors(
            cb_rows,
            args,
            dp_rank=dp_rank,
        )
        _write_rows(
            args.scheduler_alignment_descriptor_out,
            scheduler_alignment_descriptors,
        )
        print(
            "wrote scheduler alignment descriptors: "
            f"{args.scheduler_alignment_descriptor_out}"
        )
        if args.scheduler_alignment_compare_out is not None:
            vllm_alignment_descriptors = parse_scheduler_aligned_descriptors(
                args.scheduler_alignment_vllm_csv
            )
            compare_rows = compare_scheduler_aligned_descriptors(
                scheduler_alignment_descriptors,
                vllm_alignment_descriptors,
            )
            _write_rows(args.scheduler_alignment_compare_out, compare_rows)
            print(
                "wrote scheduler alignment compare: "
                f"{args.scheduler_alignment_compare_out}"
            )

    if args.metrics_before is not None or args.metrics_after is not None:
        if args.metrics_before is None or args.metrics_after is None:
            raise ValueError("--metrics-before and --metrics-after must be provided together")
        before = parse_metrics_snapshot(args.metrics_before)
        after = parse_metrics_snapshot(args.metrics_after)
        metrics_delta = compare_metrics_snapshots(
            before,
            after,
            benchmark_wall_ms=args.benchmark_wall_ms,
        )
        _print_metrics_delta(metrics_delta)
        if args.metrics_delta_out is not None:
            _write_rows(args.metrics_delta_out, [metrics_delta])
            print(f"wrote metrics delta: {args.metrics_delta_out}")

    if args.vllm_log is None:
        return

    vllm_rows = parse_vllm_iteration_log(args.vllm_log)
    vllm_rows = filter_vllm_rows(
        vllm_rows,
        engine_id=args.vllm_engine_id,
        start_time=args.vllm_start_time,
        end_time=args.vllm_end_time,
        min_decode_bs=args.vllm_min_decode_bs,
        max_decode_bs=args.vllm_max_decode_bs,
    )
    print(f"vLLM iterations after filters: {len(vllm_rows)}")
    vllm_summaries = summarize_vllm_rows(vllm_rows)
    _print_phase_summaries("vLLM phase summary", vllm_summaries)
    _print_decode_bin_summaries(summarize_vllm_decode_bins(vllm_rows))

    if args.vllm_trace_out is not None:
        _write_rows(args.vllm_trace_out, vllm_rows)
        print(f"wrote vLLM trace: {args.vllm_trace_out}")

    compare_rows, compare_summaries = compare_phasewise(cb_rows, vllm_rows)
    _print_compare_summaries(compare_summaries)

    if args.compare_out is not None:
        _write_rows(args.compare_out, compare_rows)
        print(f"wrote compare rows: {args.compare_out}")


if __name__ == "__main__":
    main()
