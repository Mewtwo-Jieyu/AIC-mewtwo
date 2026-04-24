#!/usr/bin/env python3
"""Inspect CB iteration latency decomposition for a few representative cases."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aiconfigurator.sdk.config import ModelConfig
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.perf_database import PerfDatabase
from aiconfigurator.sdk.backends.vllm_backend import VLLMBackend
from aiconfigurator.sdk.backends.cb_simulator.iteration_latency import IterationLatencyCalculator

MODEL_PATH = "moonshotai/Kimi-K2.5"
SYSTEM = "h200_sxm"
BACKEND = "vllm"
TP = 16


def _load():
    model_config = ModelConfig(tp_size=TP, pp_size=1, moe_tp_size=TP, moe_ep_size=1)
    model = get_model(MODEL_PATH, model_config, backend_name=BACKEND)
    systems_root = str(Path(__file__).resolve().parent.parent / "src" / "aiconfigurator" / "systems")
    db = PerfDatabase(system=SYSTEM, backend=BACKEND, version="0.12.0", systems_root=systems_root)
    backend = VLLMBackend()
    return model, db, backend


def main() -> None:
    model, db, backend = _load()
    calc = IterationLatencyCalculator(backend=backend, model=model, database=db)

    cases = [
        ("prefill-only chunk", 8192, 1, 30000, 0, 0),
        ("mixed moderate", 8192, 1, 16000, 16, 16384),
        ("mixed decode-heavy", 2048, 1, 8000, 128, 8192),
        ("decode-only", 0, 0, 0, 128, 8192),
    ]

    print("=" * 100)
    print("CB ITERATION LATENCY BREAKDOWN")
    print(f"{'Case':<18} {'Total':>10} {'CtxNon':>10} {'CtxAttn':>10} {'GenNon':>10} {'GenAttn':>10}")
    print("-" * 100)
    for name, prefill_tokens, prefill_bs, prefill_seq_len, decode_bs, decode_kv_len in cases:
        total = calc.compute(
            prefill_tokens=prefill_tokens,
            prefill_batch_size=prefill_bs,
            prefill_seq_len=max(prefill_seq_len, 1),
            decode_batch_size=decode_bs,
            decode_avg_kv_len=decode_kv_len,
        )
        breakdown = calc.get_last_breakdown()
        assert breakdown is not None
        print(
            f"{name:<18} "
            f"{total:>10.1f} "
            f"{breakdown.context_non_attention_ms:>10.1f} "
            f"{breakdown.context_attention_ms:>10.1f} "
            f"{breakdown.generation_non_attention_ms:>10.1f} "
            f"{breakdown.generation_attention_ms:>10.1f}"
        )


if __name__ == "__main__":
    main()
