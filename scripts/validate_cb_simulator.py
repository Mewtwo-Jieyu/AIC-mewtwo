#!/usr/bin/env python3
"""Validate CB simulator against real vLLM benchmark data.

Compares CB sim predictions vs B1/B1b baseline vs real measurements.

Usage:
    python scripts/validate_cb_simulator.py

Data source: vllm h200 kimi 实测数据-整理版 0.17.md
Config: Kimi-K2.5, vLLM 0.17, H200 SXM x16, tp=16 dp=1
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
    runtime_shape_key_from_scheduled,
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
TP = 16
NUM_GPUS = 16


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
    tp: int
    dp: int
    moe_tp: int
    moe_ep: int
    real_total_tok_s_gpu: float

    @property
    def real_output_tok_s_gpu(self) -> float:
        return self.real_total_tok_s_gpu * self.osl / (self.isl + self.osl)


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
        tp=4,
        dp=2,
        moe_tp=1,
        moe_ep=8,
        real_total_tok_s_gpu=779.76,
    ),
]


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
        system=SYSTEM, backend=BACKEND, version="0.12.0",
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
) -> CBSimConfig:
    """Create CBSimConfig aligned with run_agg default chunk budget."""
    # Need enough requests for meaningful steady-state measurement.
    # At minimum 3x concurrency to avoid requests running out.
    num_requests = max(200, concurrency * 3)
    warmup_requests = max(50, concurrency)
    return CBSimConfig(
        max_num_batched_tokens=isl,
        num_requests=num_requests,
        warmup_requests=warmup_requests,
        long_prefill_token_threshold=long_prefill_token_threshold,
        overlap_factor=overlap_factor,
        per_iteration_overhead_ms=per_iteration_overhead_ms,
    )


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

        cb_config = _make_cb_config(
            pt.isl,
            pt.batch_size,
            overlap_factor=overlap_factor,
            per_iteration_overhead_ms=ep8_per_iteration_overhead_ms,
        )
        cb_summary = backend.run_agg(
            model,
            db,
            RuntimeConfig(batch_size=pt.batch_size, isl=pt.isl, osl=pt.osl),
            ctx_tokens=pt.isl,
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
    ep8_per_iteration_overhead_ms: float = 90.0,
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
    thr_ok = max(sim_errs) <= 1.5
    ttft_ok = max(ttft_thresh_errs) <= 2.0
    multi_ok = max(multi_errs) <= 1.5
    if verbose:
        print()
        print("=" * 90)
        print("ACCEPTANCE CRITERIA:")
        print(f"  Throughput max error <= 1.5x: {'PASS' if thr_ok else 'FAIL'} ({max(sim_errs):.2f}x)")
        print(f"  Multi-config max error <= 1.5x: {'PASS' if multi_ok else 'FAIL'} ({max(multi_errs):.2f}x)")
        print(f"  TTFT max error <= 2.0x:       {'PASS' if ttft_ok else 'FAIL'} ({max(ttft_thresh_errs):.2f}x)")

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
    parser.add_argument("--ep8-per-iteration-overhead-ms", type=float, default=90.0)
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
    args = parser.parse_args()

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

    if args.experimental_compiled_body_key:
        run_experimental_compiled_body_key(
            args.experimental_compiled_body_key_out,
            args.experimental_compiled_body_key_nccl_summary,
        )
        return

    run_validation(
        overlap_factor=args.overlap_factor,
        per_iteration_overhead_ms=args.per_iteration_overhead_ms,
        ep8_per_iteration_overhead_ms=args.ep8_per_iteration_overhead_ms,
    )


if __name__ == "__main__":
    main()
