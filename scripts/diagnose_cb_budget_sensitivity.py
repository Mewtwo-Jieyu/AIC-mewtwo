#!/usr/bin/env python3
"""Compare CB simulator behavior under different token-budget assumptions."""
from __future__ import annotations

from dataclasses import dataclass

from validate_cb_simulator import THROUGHPUT_DATA, _load_model_and_db

from aiconfigurator.sdk.backends.cb_simulator import CBSimConfig, CBSimulator


@dataclass
class BudgetCase:
    label: str
    max_num_batched_tokens: int


def _run_case(pt, budget: int, model, db, backend):
    cfg = CBSimConfig(
        max_num_batched_tokens=budget,
        max_num_seqs=256,
        num_requests=max(200, pt.batch_size * 3),
        warmup_requests=max(50, pt.batch_size),
    )
    result = CBSimulator(backend, model, db, cfg).run(
        isl=pt.isl, osl=pt.osl, concurrency=pt.batch_size, num_gpus=1,
    )
    return result


def main() -> None:
    model, db, backend = _load_model_and_db()
    selected = {"10k-2k b=32", "16k-2k b=32", "30k-3k b=8", "32k-1k b=16"}
    budget_modes = [
        BudgetCase("8192", 8192),
        BudgetCase("ctx_tokens", 0),
    ]

    print("=" * 130)
    print("CB BUDGET SENSITIVITY")
    print("=" * 130)
    print(
        f"{'Scenario':<18} {'Budget':<10} {'Tok/s/GPU':>10} {'TTFT':>10} "
        f"{'AvgPrefill':>10} {'AvgDecode':>10} {'AvgTokens':>10} "
        f"{'PeakTokens':>10} {'SteadyIters':>11}"
    )
    print("-" * 130)

    for pt in THROUGHPUT_DATA:
        if pt.name not in selected:
            continue
        for mode in budget_modes:
            budget = pt.isl if mode.max_num_batched_tokens == 0 else mode.max_num_batched_tokens
            r = _run_case(pt, budget, model, db, backend)
            print(
                f"{pt.name:<18} {mode.label:<10} {r.throughput_tok_s_gpu:>10.1f} {r.mean_ttft_ms:>10.1f} "
                f"{r.avg_prefill_reqs_per_iter:>10.2f} {r.avg_decode_reqs_per_iter:>10.2f} "
                f"{r.avg_tokens_per_iter:>10.1f} {r.peak_tokens_per_iter:>10d} "
                f"{r.steady_state_iterations:>11d}"
            )
        print("-" * 130)


if __name__ == "__main__":
    main()
