# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""phase397m: collect 4-bit (W4A16 / wna16 marlin) MoE latency for the Kimi-K2.5
routed-expert shape and append rows to the vLLM 0.19.0 moe_perf table.

Real K2.5 serving quantizes routed experts with compressed-tensors pack-quantized
int4 (num_bits=4, group_size=32, symmetric) and dispatches to `marlin_moe_wna16`.
cb_sim previously defaulted K2.5 MoE to float16 (no quant_algo in config), causing
a ~5x over-prediction of generation_moe. This script produces the int4_wo table so
cb_sim can query the real kernel latency.

Writes to a node-local temp file (default /tmp) to avoid NFS lock issues; the caller
then strips to the canonical 15 columns and appends to the tracked moe_perf.txt.

Usage:
  python3 collector/vllm/run_phase397m_moe4bit.py --out /tmp/moe4bit_k25.txt
"""
import argparse
import os
import sys

# Ensure the repo root (for `collector.*`) and this dir (for `collect_moe`) are
# importable. collector/vllm is not a package, so import collect_moe directly.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_REPO_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from collect_moe import run_moe_torch  # noqa: E402

# K2.5 routed-expert shape (config: n_routed_experts=384, topk=8,
# hidden_size=7168, moe_intermediate_size=2048). Decode uses moe_tp=1, moe_ep=8.
K25_HIDDEN = 7168
K25_INTER = 2048
K25_TOPK = 8
K25_NUM_EXPERTS = 384
K25_MOE_TP = 1
K25_MOE_EP = 8

# Match the num_tokens grid used by the existing float16 K2.5 rows so cb_sim can
# interpolate over the full context+generation range without extrapolation.
NUM_TOKENS_GRID = [
    1, 2, 4, 8, 16, 32, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384,
    512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384,
    20480, 32768, 65536,
]

# DeepSeekModel (K2.5's cb_sim class) uses power_law alpha 1.01. Collect 1.2 too
# to mirror the existing float16 coverage.
POWER_LAW_ALPHAS = [1.01, 1.2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/phase397m_moe4bit_k25.txt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--moe-ep", type=int, default=K25_MOE_EP)
    parser.add_argument(
        "--alphas",
        default=",".join(str(a) for a in POWER_LAW_ALPHAS),
        help="comma-separated power_law alphas",
    )
    parser.add_argument(
        "--max-num-tokens",
        type=int,
        default=65536,
        help="skip grid points above this (guard against OOM at large batch)",
    )
    args = parser.parse_args()

    # Fresh temp file so log_perf writes its own header (and optional power cols).
    if os.path.exists(args.out):
        os.remove(args.out)

    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    grid = [n for n in NUM_TOKENS_GRID if n <= args.max_num_tokens]

    print(f"[phase397m] collecting int4_wo MoE: ep={args.moe_ep} alphas={alphas}")
    print(f"[phase397m] num_tokens grid: {grid}")
    print(f"[phase397m] out={args.out}")

    for alpha in alphas:
        print(f"\n===== power_law_alpha={alpha} =====")
        run_moe_torch(
            "int4_wo",
            grid,
            K25_HIDDEN,
            K25_INTER,
            K25_TOPK,
            K25_NUM_EXPERTS,
            K25_MOE_TP,
            args.moe_ep,
            "moonshotai/Kimi-K2.5",
            args.out,
            distributed="power_law",
            power_law_alpha=alpha,
            device=args.device,
        )

    print(f"\n[phase397m] done -> {args.out}")


if __name__ == "__main__":
    main()
