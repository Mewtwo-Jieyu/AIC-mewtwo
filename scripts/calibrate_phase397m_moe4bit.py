#!/usr/bin/env python3
"""phase397m: deprecated profiler-anchor for the 4-bit MoE micro-benchmark table.

Deprecated by Phase397v: int4_wo MoE now uses a calibrated-SOL path in
``PerfDatabase.query_moe`` instead of writing micro-benchmark rows multiplied by
``k``. Keep this script only as historical provenance for the old table path.

Why calibration is needed
--------------------------
``collector/vllm/collect_moe.py`` measures ``marlin_moe_wna16`` for the K2.5
routed-expert shape with a synthetic ``power_law`` router. Even with CUDA-graph
timing (no launch overhead) the micro-benchmark over-predicts decode MoE latency
by ~3.2x versus the profiled reality:

  * micro-bench (power_law, ep8, bs=128): ~0.51 ms / MoE layer
  * profiled ``marlin_moe_wna16`` self-CUDA (phase397l tp8ep8-8k2k, bs=128):
        expert GEMM 8.020 ms + moe_aux 1.509 ms = 9.529 ms / iter over 60 MoE
        layers = ~0.159 ms / layer

The gap is a *routing-distribution* effect: marlin decode MoE is weight-load bound,
so cost scales with the number of DISTINCT active local experts. The synthetic
power_law router activates ~all 48 local experts (ep8 -> 384/8), while K2.5's real
grouped-topk router concentrates tokens onto fewer experts per rank. The wall-clock
ground truth (36.85 ms/iter total decode) confirms MoE must be ~9.5 ms, not ~30 ms.

Rather than invent a constant, we anchor the micro-benchmark *shape* to the
*profiled* magnitude at the decode operating point (bs=128). The single scale factor
is derived entirely from the phase397l measurement and replaces the fictional 90 ms
ep8 overhead with a measured value.

Scope / caveats
---------------
* Anchored to the tp8ep8-8k2k decode point (the phase397m primary target).
* A single multiplicative factor assumes the micro-bench/real ratio is ~constant vs
  num_tokens. It is most accurate in the decode range; large-num_tokens (context/
  prefill) MoE may need its own anchor (left to a later phase). K2.5 decode only
  queries this table near bs, so this is acceptable for decode convergence.
* tp4dp2ep8 has a separate DP num_tokens modeling residual (out of scope here).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

# --- profiler anchor (phase397l tp8ep8-8k2k, bs=128) ---
MEASURED_MOE_MS_PER_ITER = 9.529  # moe_expert_gemm 8.020 + moe_aux 1.509
NUM_MOE_LAYERS = 60  # K2.5: 61 layers, first_k_dense_replace=1 -> 60 MoE layers
ANCHOR_NUM_TOKENS = 128  # decode operating point (bs=128)
ANCHOR_DISTRIBUTION = "power_law_1.01"  # DeepSeekModel MoE workload_distribution

RAW_KERNEL_SOURCE = "vllm_marlin_moe_wna16"
CAL_KERNEL_SOURCE = "vllm_marlin_moe_wna16_profcal"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=Path("/tmp/phase397m_moe4bit_full.txt"))
    ap.add_argument("--out", type=Path, default=Path("/tmp/phase397m_moe4bit_calibrated.txt"))
    args = ap.parse_args()

    with args.raw.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"no rows in {args.raw}")

    anchor = [
        r for r in rows
        if int(r["num_tokens"]) == ANCHOR_NUM_TOKENS and r["distribution"] == ANCHOR_DISTRIBUTION
    ]
    if not anchor:
        raise SystemExit(f"no anchor row (num_tokens={ANCHOR_NUM_TOKENS}, {ANCHOR_DISTRIBUTION})")
    raw_anchor_ms = float(anchor[0]["latency"])

    target_per_layer_ms = MEASURED_MOE_MS_PER_ITER / NUM_MOE_LAYERS
    k = target_per_layer_ms / raw_anchor_ms

    print(f"[calibrate] raw(bs={ANCHOR_NUM_TOKENS}, {ANCHOR_DISTRIBUTION}) = {raw_anchor_ms:.5f} ms/layer")
    print(f"[calibrate] profiled target per layer      = {target_per_layer_ms:.5f} ms/layer "
          f"({MEASURED_MOE_MS_PER_ITER} ms / {NUM_MOE_LAYERS} layers)")
    print(f"[calibrate] scale factor k                 = {k:.5f}  (micro-bench over-predicts {1/k:.2f}x)")

    fieldnames = list(rows[0].keys())
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = dict(r)
            out["latency"] = f"{float(r['latency']) * k:.10g}"
            out["kernel_source"] = CAL_KERNEL_SOURCE
            w.writerow(out)
    print(f"[calibrate] wrote {sum(1 for _ in rows)} calibrated rows -> {args.out}")


if __name__ == "__main__":
    main()
