#!/usr/bin/env python3
"""Phase397l: per-op decode-iteration breakdown from vLLM torch profiler tables.

Parses the per-rank ``profiler_out_<rank>.txt`` key_averages() tables produced by
the phase397l collector, classifies every GPU kernel into a structural category
(attention / moe_expert_gemm / proj_gemm / allreduce / allgather / all2all /
moe_aux / other), and reports the per-decode-iteration wall time and the
per-category GPU time. The goal is to ground-truth (1) the real EP communication
cost that the cb_sim ``ep8_per_iteration_overhead_ms=90.0`` constant stands in
for, and (2) the serial-sum-vs-wall overlap that makes the real decode iteration
much faster than cb_sim's fully-serial kernel sum.

Report only. Does not touch the perf DB, the simulator, or any gate constant.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

# Structural categories, matched in order (first hit wins). Patterns are matched
# against the lowercased kernel name.
CATEGORY_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("attention", ("flash", "flashattn", "unified_mla", "mla_attention",
                   "concat_and_cache_mla", "sparse_attn", "get_scheduler_metadata",
                   "prepare_varlen", "paged", "compute_slot_mapping")),
    ("moe_expert_gemm", ("marlin_moe", "marlin<", "fused_moe", "moe_wna16")),
    ("moe_aux", ("moe::grouped_topk", "moe_align_block", "count_and_sort_expert",
                 "act_and_mul", "moe_forward_share", "topk", "moe::", "silu")),
    ("allreduce", ("all_reduce", "allreduce", "multimem_all_reduce", "reduce_scatter",
                   "reducescatter", "cross_device_reduce", "cross_device")),
    ("allgather", ("allgather", "all_gather", "_all_gather_base")),
    ("all2all", ("all_to_all", "alltoall", "dispatch", "combine", "nvshmem", "pplx",
                 "deepep", "ep_")),
    ("proj_gemm", ("nvjet", "cublas", "splitkreduce", "cutlass_", "gemm", "aten::mm",
                   "cutlass::device_kernel<cutlass")),
    ("norm_elementwise", ("rmsnorm", "rms_norm", "rsqrt", "layernorm", "add_mean",
                          "vectorized_elementwise", "unrolled_elementwise",
                          "elementwise_kernel", "triton_poi_fused", "triton_red_fused")),
]

# Rows that are the decode-step markers / profiler bookkeeping, not GPU kernels.
STEP_MARKER = "profilerstep"
DECODE_MARKER_RE = re.compile(r"execute_context_0\(0\)_generation_(\d+)")


def _to_ms(tok: str) -> float:
    tok = tok.strip()
    if tok.endswith("us"):
        return float(tok[:-2]) / 1000.0
    if tok.endswith("ms"):
        return float(tok[:-2])
    if tok.endswith("s"):
        return float(tok[:-1]) * 1000.0
    return 0.0


def _classify(name: str) -> str:
    low = name.lower()
    for cat, pats in CATEGORY_PATTERNS:
        for p in pats:
            if p in low:
                return cat
    return "other"


def parse_table(path: Path) -> dict:
    """Return {n_steps, decode_bs, wall_ms_per_iter, kernels:[(name,self_cuda_ms,calls)]}."""
    rows: list[tuple[str, float, int]] = []
    decode_bs = 0
    # The step markers (ProfilerStep* / execute_context..generation) appear more
    # than once due to torch's CPU/CUDA aggregation; the row with the largest
    # CPU-total is the real per-step wall region, and its call count is the true
    # number of profiled decode iterations. Track that row.
    best_marker_cpu_total = 0.0
    n_steps = 0
    step_cpu_total_ms = 0.0
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.startswith("---") or raw.startswith("Self "):
            continue
        if raw.lstrip().startswith("Name") and "Self CUDA" in raw:
            continue
        cols = re.split(r"\s{2,}", raw.strip())
        if len(cols) < 11:
            continue
        name = " ".join(cols[:-10]).strip()
        nums = cols[-10:]
        try:
            self_cuda_ms = _to_ms(nums[5])
            cuda_total_ms = _to_ms(nums[7])
            calls = int(nums[9])
            cpu_total_ms = _to_ms(nums[3])
        except (ValueError, IndexError):
            continue
        low = name.lower()
        m = DECODE_MARKER_RE.search(name)
        is_marker = low.startswith(STEP_MARKER) or m is not None
        if is_marker:
            if m is not None:
                decode_bs = int(m.group(1))
            if calls and cpu_total_ms > best_marker_cpu_total:
                best_marker_cpu_total = cpu_total_ms
                n_steps = calls
                step_cpu_total_ms = cpu_total_ms / calls
            continue
        if self_cuda_ms > 0:
            rows.append((name, self_cuda_ms, calls))
    return {
        "n_steps": n_steps,
        "decode_bs": decode_bs,
        "wall_ms_per_iter": step_cpu_total_ms,
        "kernels": rows,
    }


def analyze_rank(path: Path) -> dict:
    t = parse_table(path)
    n = t["n_steps"] or 1
    cats: dict[str, float] = {}
    gpu_busy_ms = 0.0
    unclassified: list[tuple[str, float]] = []
    for name, self_ms, _calls in t["kernels"]:
        cat = _classify(name)
        per_iter = self_ms / n
        cats[cat] = cats.get(cat, 0.0) + per_iter
        gpu_busy_ms += per_iter
        if cat == "other":
            unclassified.append((name, per_iter))
    unclassified.sort(key=lambda x: -x[1])
    return {
        "rank": path.stem.split("_")[-1],
        "n_steps": t["n_steps"],
        "decode_bs": t["decode_bs"],
        "wall_ms_per_iter": t["wall_ms_per_iter"],
        "gpu_busy_ms_per_iter": gpu_busy_ms,
        "cats": cats,
        "unclassified_top": unclassified[:8],
    }


CAT_ORDER = ["attention", "moe_expert_gemm", "proj_gemm", "moe_aux", "allreduce",
             "allgather", "all2all", "norm_elementwise", "other"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prof-dir", type=Path, required=True,
                    help="directory with profiler_out_<rank>.txt files")
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    files = sorted(args.prof_dir.glob("profiler_out_*.txt"))
    if not files:
        raise SystemExit(f"no profiler_out_*.txt under {args.prof_dir}")

    ranks = [analyze_rank(f) for f in files]
    n_ranks = len(ranks)

    # Average per-category across ranks (ranks are near-symmetric on tp).
    agg: dict[str, float] = {c: 0.0 for c in CAT_ORDER}
    for r in ranks:
        for c, v in r["cats"].items():
            agg[c] = agg.get(c, 0.0) + v
    agg = {c: v / n_ranks for c, v in agg.items()}
    gpu_busy = sum(agg.values())
    wall = sum(r["wall_ms_per_iter"] for r in ranks) / n_ranks
    n_steps = ranks[0]["n_steps"]
    decode_bs = max(r["decode_bs"] for r in ranks)
    comm = agg.get("allreduce", 0) + agg.get("allgather", 0) + agg.get("all2all", 0)

    print(f"=== phase397l per-op decode breakdown {args.label} ===")
    print(f"ranks={n_ranks} n_profiled_steps={n_steps} decode_bs={decode_bs}")
    print(f"decode wall ms/iter (step cpu-total avg): {wall:8.3f}")
    print(f"GPU-busy ms/iter (sum self-cuda):         {gpu_busy:8.3f}")
    print(f"overlap = GPU-busy/wall:                  {gpu_busy / wall if wall else 0:8.3f}")
    print(f"comm (allreduce+allgather+all2all) ms/iter: {comm:7.3f}  ({100*comm/gpu_busy:4.1f}% of gpu-busy)")
    print()
    hdr = f"{'category':20s} {'ms/iter':>10s} {'% gpu-busy':>11s}"
    print(hdr); print("-" * len(hdr))
    for c in CAT_ORDER:
        v = agg.get(c, 0.0)
        if v <= 0:
            continue
        print(f"{c:20s} {v:>10.3f} {100*v/gpu_busy:>10.1f}%")
    print("-" * len(hdr))
    print(f"{'TOTAL gpu-busy':20s} {gpu_busy:>10.3f} {100.0:>10.1f}%")

    # Unclassified sanity (rank0).
    if ranks[0]["unclassified_top"]:
        print("\ntop 'other' kernels (rank0, ms/iter):")
        for name, v in ranks[0]["unclassified_top"]:
            if v > 0.02:
                print(f"  {v:7.3f}  {name[:70]}")

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["category", "ms_per_iter", "pct_gpu_busy"])
            for c in CAT_ORDER:
                v = agg.get(c, 0.0)
                if v > 0:
                    w.writerow([c, round(v, 4), round(100 * v / gpu_busy, 2)])
            w.writerow(["TOTAL_gpu_busy", round(gpu_busy, 4), 100.0])
            w.writerow(["decode_wall_ms_per_iter", round(wall, 4), ""])
            w.writerow(["comm_ms_per_iter", round(comm, 4), round(100 * comm / gpu_busy, 2)])
            w.writerow(["n_profiled_steps", n_steps, ""])
            w.writerow(["decode_bs", decode_bs, ""])
        print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
