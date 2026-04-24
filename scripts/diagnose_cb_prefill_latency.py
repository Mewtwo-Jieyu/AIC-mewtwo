#!/usr/bin/env python3
"""Inspect prefill-side static_ctx latency for representative CB chunk shapes."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aiconfigurator.sdk.backends.vllm_backend import VLLMBackend
from aiconfigurator.sdk.config import ModelConfig, RuntimeConfig
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.perf_database import PerfDatabase

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


def _print_breakdown(label: str, data: dict[str, float]) -> None:
    total = sum(data.values())
    print(f"\n=== {label} total={total:.3f} ms ===")
    for name, latency in sorted(data.items(), key=lambda item: item[1], reverse=True)[:12]:
        print(f"{name:<40} {latency:>10.3f} ms")


def main() -> None:
    model, db, backend = _load()

    ctx_cases = [
        ("ctx_bs1_8192", RuntimeConfig(batch_size=1, beam_width=1, isl=8192, osl=1, prefix=0)),
        ("ctx_bs1_30000", RuntimeConfig(batch_size=1, beam_width=1, isl=30000, osl=1, prefix=0)),
        ("ctx_bs128_8192", RuntimeConfig(batch_size=128, beam_width=1, isl=8192, osl=1, prefix=0)),
    ]
    gen_cases = [
        ("gen_bs16_kv16k", RuntimeConfig(batch_size=16, beam_width=1, isl=16384, osl=2, prefix=0)),
        ("gen_bs128_kv8k", RuntimeConfig(batch_size=128, beam_width=1, isl=8192, osl=2, prefix=0)),
    ]

    print("=" * 100)
    print("CB PREFILL/GEN STATIC BREAKDOWN")
    print("=" * 100)

    for label, runtime in ctx_cases:
        summary = backend.run_static(model, db, runtime, mode="static_ctx")
        _print_breakdown(label, summary.get_context_latency_dict())

    for label, runtime in gen_cases:
        summary = backend.run_static(model, db, runtime, mode="static_gen")
        _print_breakdown(label, summary.get_generation_latency_dict())


if __name__ == "__main__":
    main()
