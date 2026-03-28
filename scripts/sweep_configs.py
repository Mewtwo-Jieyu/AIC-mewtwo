#!/usr/bin/env python3
"""
Batch-evaluate parallel configurations and report latency/throughput metrics.

Uses the same config enumeration as `aiconfigurator cli default`, but outputs
ALL evaluated configurations (not just Pareto-optimal ones).

Usage:
  python scripts/sweep_configs.py \
    --model-path moonshotai/Kimi-K2.5 \
    --total-gpus 16 \
    --system h200_sxm \
    --backend vllm \
    --database-mode HYBRID \
    --isl 8192 --osl 2048 \
    --output sweep_results.csv
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from aiconfigurator.cli.main import build_default_task_configs
from aiconfigurator.sdk.task import TaskRunner

# Columns to display (in order), all from ColumnsAgg.
DISPLAY_COLUMNS = [
    "tp",
    "pp",
    "dp",
    "moe_tp",
    "moe_ep",
    "num_total_gpus",
    "bs",
    "concurrency",
    "request_rate",
    "memory",
    "ttft",
    "tpot",
    "request_latency",
    "tokens/s",
    "tokens/s/gpu",
    "tokens/s/user",
]

RENAME_MAP = {
    "num_total_gpus": "GPUs",
    "ttft": "TTFT_ms",
    "tpot": "TPOT_ms",
    "request_latency": "req_latency_ms",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep parallel configurations and report latency/throughput metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model-path", required=True, help="HuggingFace model path or local path")
    p.add_argument("--total-gpus", type=int, required=True, help="Total number of GPUs")
    p.add_argument("--system", required=True, help="System name (e.g. h200_sxm, gb200)")
    p.add_argument("--backend", default="trtllm", help="Backend name (default: trtllm)")
    p.add_argument("--database-mode", default="SILICON", help="Database mode (default: SILICON)")
    p.add_argument("--isl", type=int, default=4000, help="Input sequence length (default: 4000)")
    p.add_argument("--osl", type=int, default=1000, help="Output sequence length (default: 1000)")
    p.add_argument("--output", "-o", default=None, help="Output CSV path")
    p.add_argument(
        "--sort-by",
        default="tokens/s/gpu",
        help="Column to sort results by (default: tokens/s/gpu)",
    )

    # Override search space lists (applied on top of SDK defaults)
    p.add_argument("--tp", type=int, nargs="+", default=None, help="Override TP sizes (e.g. --tp 1 2 4 8 16)")
    p.add_argument("--pp", type=int, nargs="+", default=None, help="Override PP sizes")
    p.add_argument("--dp", type=int, nargs="+", default=None, help="Override DP sizes")
    p.add_argument("--moe-tp", type=int, nargs="+", default=None, help="Override MoE TP sizes")
    p.add_argument("--moe-ep", type=int, nargs="+", default=None, help="Override MoE EP sizes")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    print(
        f"Running sweep: model={args.model_path}, system={args.system}, "
        f"backend={args.backend}, total_gpus={args.total_gpus}, "
        f"isl={args.isl}, osl={args.osl}, database_mode={args.database_mode}\n"
    )

    # Build task configs (same logic as `aiconfigurator cli default`)
    task_configs = build_default_task_configs(
        model_path=args.model_path,
        total_gpus=args.total_gpus,
        system=args.system,
        backend=args.backend,
        database_mode=args.database_mode,
        isl=args.isl,
        osl=args.osl,
        ttft=999999,
        tpot=999999,
    )

    # Run the agg task directly to get the full results (before Pareto filtering)
    agg_task = task_configs.get("agg")
    if agg_task is None:
        # When backend='auto', keys are like 'agg_vllm'
        agg_keys = [k for k in task_configs if k.startswith("agg")]
        if not agg_keys:
            print("No agg task config found.")
            sys.exit(1)
        agg_task = task_configs[agg_keys[0]]

    # Patch search space lists if user provided overrides
    wc = agg_task.config.worker_config
    if args.tp is not None:
        wc.tp_list = sorted(set(args.tp))
    if args.pp is not None:
        wc.pp_list = sorted(set(args.pp))
    if args.dp is not None:
        wc.dp_list = sorted(set(args.dp))
    if args.moe_tp is not None:
        wc.moe_tp_list = sorted(set(args.moe_tp))
    if args.moe_ep is not None:
        wc.moe_ep_list = sorted(set(args.moe_ep))

    # Ensure num_gpu_per_worker covers any new GPU counts from overrides
    max_needed = max(wc.tp_list) * max(wc.pp_list) * max(wc.dp_list)
    existing = set(wc.num_gpu_per_worker)
    for v in wc.tp_list:
        for p in wc.pp_list:
            for d in wc.dp_list:
                existing.add(v * p * d)
    wc.num_gpu_per_worker = sorted(existing)

    runner = TaskRunner()
    result = runner.run_agg(agg_task.config)

    if result is None:
        print("TaskRunner.run_agg() returned None (check logs for errors).")
        sys.exit(1)

    df = result.get("pareto_df")
    if df is None or df.empty:
        print("No valid configurations found (all OOM or errored).")
        sys.exit(1)

    # Select and rename columns for display
    cols = [c for c in DISPLAY_COLUMNS if c in df.columns]
    df_out = df[cols].copy()
    df_out.rename(columns=RENAME_MAP, inplace=True)

    # Sort
    sort_col = RENAME_MAP.get(args.sort_by, args.sort_by)
    if sort_col in df_out.columns:
        df_out.sort_values(by=sort_col, ascending=False, inplace=True)

    # Round numeric columns
    for col in df_out.select_dtypes(include="number").columns:
        df_out[col] = df_out[col].round(3)

    print(f"Found {len(df_out)} evaluated configurations:\n")
    print(df_out.to_string(index=False))

    if args.output:
        df_out.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
