#!/usr/bin/env python3
"""Inspect steady-state iteration composition for selected CB scenarios."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from validate_cb_simulator import THROUGHPUT_DATA, _load_model_and_db

from aiconfigurator.sdk.backends.cb_simulator import CBSimConfig
from aiconfigurator.sdk.backends.cb_simulator.datatypes import Request, RequestState
from aiconfigurator.sdk.backends.cb_simulator.simulator import CBSimulator


@dataclass
class IterAccumulator:
    count: int = 0
    prefill_reqs: float = 0.0
    prefill_tokens: float = 0.0
    decode_reqs: float = 0.0
    total_tokens: float = 0.0
    avg_kv: float = 0.0
    total_ms: float = 0.0
    ctx_non_ms: float = 0.0
    ctx_attn_ms: float = 0.0
    gen_non_ms: float = 0.0
    gen_attn_ms: float = 0.0

    def add(
        self,
        *,
        repeats: int,
        prefill_reqs: int,
        prefill_tokens: int,
        decode_reqs: int,
        total_tokens: int,
        avg_kv: int,
        total_ms: float,
        ctx_non_ms: float,
        ctx_attn_ms: float,
        gen_non_ms: float,
        gen_attn_ms: float,
    ) -> None:
        self.count += repeats
        self.prefill_reqs += prefill_reqs * repeats
        self.prefill_tokens += prefill_tokens * repeats
        self.decode_reqs += decode_reqs * repeats
        self.total_tokens += total_tokens * repeats
        self.avg_kv += avg_kv * repeats
        self.total_ms += total_ms * repeats
        self.ctx_non_ms += ctx_non_ms * repeats
        self.ctx_attn_ms += ctx_attn_ms * repeats
        self.gen_non_ms += gen_non_ms * repeats
        self.gen_attn_ms += gen_attn_ms * repeats

    def mean_dict(self) -> dict[str, float]:
        n = max(self.count, 1)
        return {
            "prefill_reqs": self.prefill_reqs / n,
            "prefill_tokens": self.prefill_tokens / n,
            "decode_reqs": self.decode_reqs / n,
            "total_tokens": self.total_tokens / n,
            "avg_kv": self.avg_kv / n,
            "total_ms": self.total_ms / n,
            "ctx_non_ms": self.ctx_non_ms / n,
            "ctx_attn_ms": self.ctx_attn_ms / n,
            "gen_non_ms": self.gen_non_ms / n,
            "gen_attn_ms": self.gen_attn_ms / n,
        }


def _trace_scenario(pt) -> tuple:
    model, db, backend = _load_model_and_db()
    cfg = CBSimConfig(
        max_num_batched_tokens=pt.isl,
        max_num_seqs=256,
        num_requests=max(200, pt.batch_size * 3),
        warmup_requests=max(50, pt.batch_size),
    )
    sim = CBSimulator(backend, model, db, cfg)
    latency_calc = sim._create_latency_calc(prefix=0)

    waiting: list[Request] = []
    running: list[Request] = []
    completed: list[Request] = []
    acc = IterAccumulator()
    clock_ms = 0.0
    next_id = 0
    total_iters = 0
    steady_output_tokens = 0
    steady_time_ms = 0.0

    for _ in range(min(pt.batch_size, cfg.num_requests)):
        waiting.append(Request(request_id=next_id, isl=pt.isl, osl=pt.osl, arrival_time_ms=0.0))
        next_id += 1

    max_iters = cfg.num_requests * (pt.osl + pt.isl // cfg.max_num_batched_tokens + 10)

    while len(completed) < cfg.num_requests and total_iters < max_iters:
        schedule = sim._scheduler.schedule(waiting, running)
        if schedule.is_empty:
            break

        avg_kv = int(np.mean([r.kv_cache_len for r in schedule.decode_reqs])) if schedule.decode_reqs else 0
        iter_lat = latency_calc.compute(
            prefill_tokens=schedule.total_prefill_tokens,
            prefill_batch_size=len(schedule.prefill_reqs),
            prefill_seq_len=pt.isl,
            decode_batch_size=len(schedule.decode_reqs),
            decode_avg_kv_len=avg_kv,
        )
        breakdown = latency_calc.get_last_breakdown()
        in_steady_state = len(completed) >= cfg.warmup_requests

        if in_steady_state and breakdown is not None:
            acc.add(
                repeats=1,
                prefill_reqs=len(schedule.prefill_reqs),
                prefill_tokens=schedule.total_prefill_tokens,
                decode_reqs=len(schedule.decode_reqs),
                total_tokens=schedule.total_tokens,
                avg_kv=avg_kv,
                total_ms=breakdown.total_ms,
                ctx_non_ms=breakdown.context_non_attention_ms,
                ctx_attn_ms=breakdown.context_attention_ms,
                gen_non_ms=breakdown.generation_non_attention_ms,
                gen_attn_ms=breakdown.generation_attention_ms,
            )

        clock_ms += iter_lat
        total_iters += 1
        if in_steady_state:
            steady_time_ms += iter_lat

        for req in schedule.prefill_reqs:
            tokens = schedule.prefill_tokens[req.request_id]
            if req.state == RequestState.WAITING:
                req.state = RequestState.PREFILLING
                if req in waiting:
                    waiting.remove(req)
                running.append(req)
            req.prefill_tokens_remaining -= tokens
            if req.prefill_tokens_remaining <= 0:
                req.prefill_tokens_remaining = 0
                req.first_token_ms = clock_ms
                req.state = RequestState.DECODING

        newly_done: list[Request] = []
        for req in schedule.decode_reqs:
            req.generated_tokens += 1
            if in_steady_state:
                steady_output_tokens += 1
            if req.generated_tokens >= req.osl - 1:
                req.state = RequestState.DONE
                req.finish_ms = clock_ms
                newly_done.append(req)

        for req in newly_done:
            running.remove(req)
            completed.append(req)
            if next_id < cfg.num_requests:
                waiting.append(Request(request_id=next_id, isl=pt.isl, osl=pt.osl, arrival_time_ms=clock_ms))
                next_id += 1

        if (not waiting and all(r.state == RequestState.DECODING for r in running) and running):
            min_remaining = min(r.osl - 1 - r.generated_tokens for r in running)
            skip = max(0, min_remaining - 1)
            if skip > 0:
                skip_lat = iter_lat * skip
                clock_ms += skip_lat
                total_iters += skip
                if in_steady_state:
                    steady_time_ms += skip_lat
                    steady_output_tokens += len(running) * skip
                if in_steady_state and breakdown is not None:
                    acc.add(
                        repeats=skip,
                        prefill_reqs=0,
                        prefill_tokens=0,
                        decode_reqs=len(running),
                        total_tokens=len(running),
                        avg_kv=avg_kv,
                        total_ms=breakdown.total_ms,
                        ctx_non_ms=breakdown.context_non_attention_ms,
                        ctx_attn_ms=breakdown.context_attention_ms,
                        gen_non_ms=breakdown.generation_non_attention_ms,
                        gen_attn_ms=breakdown.generation_attention_ms,
                    )
                for req in running:
                    req.generated_tokens += skip

    throughput = steady_output_tokens / (steady_time_ms / 1000.0)
    return throughput, acc.mean_dict()


def main() -> None:
    selected = {"3k-3k b=128", "16k-2k b=32", "30k-3k b=8", "32k-1k b=16"}
    print("=" * 150)
    print("CB STEADY-STATE DIAGNOSIS")
    print("=" * 150)
    print(
        f"{'Scenario':<18} {'Tok/s':>9} {'RealOut':>9} {'PrefReq':>8} {'PrefTok':>9} "
        f"{'DecReq':>8} {'AvgKV':>8} {'IterMs':>8} {'CtxNon':>8} {'CtxAttn':>8} "
        f"{'GenNon':>8} {'GenAttn':>8}"
    )
    print("-" * 150)
    for pt in THROUGHPUT_DATA:
        if pt.name not in selected:
            continue
        throughput, summary = _trace_scenario(pt)
        print(
            f"{pt.name:<18} {throughput:>9.1f} {pt.real_output_tok_s_gpu:>9.1f} "
            f"{summary['prefill_reqs']:>8.2f} {summary['prefill_tokens']:>9.1f} "
            f"{summary['decode_reqs']:>8.2f} {summary['avg_kv']:>8.1f} {summary['total_ms']:>8.1f} "
            f"{summary['ctx_non_ms']:>8.1f} {summary['ctx_attn_ms']:>8.1f} "
            f"{summary['gen_non_ms']:>8.1f} {summary['gen_attn_ms']:>8.1f}"
        )


if __name__ == "__main__":
    main()
