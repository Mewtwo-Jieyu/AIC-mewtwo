# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase397u: probe marlin local-expert and block-size levers.

This is a diagnostic runner only. It writes one-row perf snippets under a
caller-provided directory and never mutates the tracked perf database.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_REPO_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from collect_moe import run_moe_torch  # noqa: E402

K25_HIDDEN = 7168
K25_INTER = 2048
K25_TOPK = 8
K25_NUM_EXPERTS = 384
K25_MOE_TP = 1
K25_MOE_EP = 8
K25_MODEL = "moonshotai/Kimi-K2.5"
ANCHOR_NUM_TOKENS = 128
ANCHOR_ALPHA = 1.01

SHOTS = {
    "local48_direct": "power_law_local48_direct",
    "force_block64": "power_law_force_block64",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/phase397u_marlin_config"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--shot",
        choices=tuple(SHOTS),
        default="",
        help="run a single shot; default runs all Phase397u shots",
    )
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected = [args.shot] if args.shot else list(SHOTS)

    for shot in selected:
        distributed = SHOTS[shot]
        out_path = args.out_dir / f"{shot}.txt"
        if out_path.exists():
            out_path.unlink()
        print(f"[phase397u] shot={shot} distributed={distributed} out={out_path}")
        run_moe_torch(
            "int4_wo",
            [ANCHOR_NUM_TOKENS],
            K25_HIDDEN,
            K25_INTER,
            K25_TOPK,
            K25_NUM_EXPERTS,
            K25_MOE_TP,
            K25_MOE_EP,
            K25_MODEL,
            str(out_path),
            distributed=distributed,
            power_law_alpha=ANCHOR_ALPHA,
            device=args.device,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
