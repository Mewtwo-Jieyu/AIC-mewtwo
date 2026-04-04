#!/usr/bin/env python3
"""Validate CB simulator against real vLLM benchmark data.

Compares CB sim predictions vs B1/B1b baseline vs real measurements.

Usage:
    python scripts/validate_cb_simulator.py

Data source: vllm h200 kimi 实测数据-整理版 0.17.md
Config: Kimi-K2.5, vLLM 0.17, H200 SXM x16, tp=16 dp=1
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aiconfigurator.sdk import common
from aiconfigurator.sdk.backends.cb_simulator import CBSimConfig, CBSimulator
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
    real_tok_s_gpu: float
    real_ttft_ms: float | None = None


THROUGHPUT_DATA = [
    BenchmarkPoint("3k-3k b=128", 3000, 3000, 128, 254.2),
    BenchmarkPoint("8k-2k b=256", 8000, 2000, 256, 575.4),
    BenchmarkPoint("10k-2k b=32", 10000, 2000, 32, 267.6),
    BenchmarkPoint("10k-3k b=128", 10000, 3000, 128, 386.9),
    BenchmarkPoint("16k-2k b=32", 16000, 2000, 32, 352.7),
    BenchmarkPoint("30k-3k b=8", 30000, 3000, 8, 179.2),
    BenchmarkPoint("32k-1k b=16", 32000, 1000, 16, 559.9),
]

TTFT_DATA = [
    BenchmarkPoint("30k-3k b=4", 30000, 3000, 4, 0, 1231.0),
    BenchmarkPoint("30k-3k b=8", 30000, 3000, 8, 0, 1823.0),
    BenchmarkPoint("20k-5k b=4", 20000, 5000, 4, 0, 1326.0),
    BenchmarkPoint("20k-5k b=8", 20000, 5000, 8, 0, 1649.0),
    BenchmarkPoint("16k-2k b=16", 16000, 2000, 16, 0, 782.0),
    BenchmarkPoint("16k-2k b=32", 16000, 2000, 32, 0, 814.0),
]


def _abs_error(predicted: float, real: float) -> float:
    """Symmetric absolute error ratio: max(p/r, r/p)."""
    if predicted <= 0 or real <= 0:
        return float("inf")
    ratio = predicted / real
    return max(ratio, 1 / ratio)


def _load_model_and_db() -> tuple:
    """Load model and database with same pattern as fit_cb_factor.py."""
    model_config = ModelConfig(tp_size=TP, pp_size=1, moe_tp_size=TP, moe_ep_size=1)
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


def main() -> None:
    model, db, backend = _load_model_and_db()
    cb_config = CBSimConfig(num_requests=100, warmup_requests=30)

    # --- Throughput ---
    print("=" * 90)
    print("THROUGHPUT (tok/s/GPU)")
    print(f"{'Scenario':<22} {'Real':>8} {'CB-Sim':>8} {'B1':>8} {'Sim/Real':>10} {'B1/Real':>10}")
    print("-" * 90)

    sim_errs: list[float] = []
    b1_errs: list[float] = []

    for pt in THROUGHPUT_DATA:
        # CB sim
        sim = CBSimulator(backend, model, db, cb_config)
        r = sim.run(isl=pt.isl, osl=pt.osl, concurrency=pt.batch_size, num_gpus=NUM_GPUS)
        sim_gpu = r.throughput_tok_s_gpu

        # B1 baseline (with correction factors enabled)
        ctx_tokens = pt.isl
        try:
            b1_summary = backend.run_agg(
                model, db,
                RuntimeConfig(batch_size=pt.batch_size, isl=pt.isl, osl=pt.osl),
                ctx_tokens=ctx_tokens,
                database_mode=common.DatabaseMode.HYBRID,
            )
            b1_dict = b1_summary.get_result_dict()
            b1_gpu = b1_dict["tokens/s/gpu"] if b1_dict else 0
        except Exception as e:
            logger.warning("B1 failed for %s: %s", pt.name, e)
            b1_gpu = 0

        sim_r = sim_gpu / pt.real_tok_s_gpu if pt.real_tok_s_gpu > 0 else 0
        b1_r = b1_gpu / pt.real_tok_s_gpu if pt.real_tok_s_gpu > 0 else 0
        sim_errs.append(_abs_error(sim_gpu, pt.real_tok_s_gpu))
        b1_errs.append(_abs_error(b1_gpu, pt.real_tok_s_gpu))

        print(f"{pt.name:<22} {pt.real_tok_s_gpu:>8.1f} {sim_gpu:>8.1f} {b1_gpu:>8.1f} {sim_r:>9.2f}x {b1_r:>9.2f}x")

    print("-" * 90)
    print(f"CB-Sim: max={max(sim_errs):.2f}x mean={np.mean(sim_errs):.2f}x | "
          f"B1: max={max(b1_errs):.2f}x mean={np.mean(b1_errs):.2f}x")

    # --- TTFT ---
    print()
    print("=" * 90)
    print("TTFT (ms)")
    print(f"{'Scenario':<22} {'Real':>8} {'CB-Sim':>8} {'Sim/Real':>10}")
    print("-" * 90)

    ttft_errs: list[float] = []
    for pt in TTFT_DATA:
        sim = CBSimulator(backend, model, db, cb_config)
        r = sim.run(isl=pt.isl, osl=pt.osl, concurrency=pt.batch_size, num_gpus=NUM_GPUS)
        ratio = r.mean_ttft_ms / pt.real_ttft_ms if pt.real_ttft_ms else 0
        ttft_errs.append(_abs_error(r.mean_ttft_ms, pt.real_ttft_ms))
        print(f"{pt.name:<22} {pt.real_ttft_ms:>8.1f} {r.mean_ttft_ms:>8.1f} {ratio:>9.2f}x")

    print("-" * 90)
    print(f"TTFT: max={max(ttft_errs):.2f}x mean={np.mean(ttft_errs):.2f}x")

    # --- Summary ---
    print()
    print("=" * 90)
    print("ACCEPTANCE CRITERIA:")
    thr_ok = max(sim_errs) <= 1.5
    ttft_ok = max(ttft_errs) <= 2.0
    print(f"  Throughput max error <= 1.5x: {'PASS' if thr_ok else 'FAIL'} ({max(sim_errs):.2f}x)")
    print(f"  TTFT max error <= 2.0x:       {'PASS' if ttft_ok else 'FAIL'} ({max(ttft_errs):.2f}x)")


if __name__ == "__main__":
    main()
