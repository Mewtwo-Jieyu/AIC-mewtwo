# Phase397v int4_wo SOL Calibration

Phase397v switched from the failed structural marlin reproduction path to a profiler-anchored calibrated roofline candidate for `int4_wo` MoE.

## Calibration

| Item | Value |
|---|---:|
| Phase397l serve anchor | `8.0201 ms / 60 layers = 0.133668 ms/layer` |
| SOL ep8 at `num_tokens=128,tp1,ep8` | `0.220474 ms/layer` |
| SOL math | `0.011400 ms/layer` |
| SOL mem | `0.220474 ms/layer` |
| Calibration `C_int4` | `0.606277` |

The anchor is memory-bound. `SOL_FULL` remains raw roofline; the calibrated path only applies to `h200_sxm + vllm + 0.19.0 + int4_wo` normal MoE lookup.

## Local Unit Gate

| Check | Result |
|---|---|
| ep8 anchor reproduction | `0.133668 ms/layer` |
| tp16/ep1 physical SOL scaling | passed |
| raw `SOL_FULL` components preserved | passed |

## Validate Gate

`scripts/validate_cb_simulator.py` was repointed to `vllm 0.19.0` for the gate run. The gate did not pass.

| Section | Result |
|---|---|
| Throughput | `max=2.62x mean=2.29x` |
| TTFT | `base max=6.09x mean=4.79x`; threshold path `max=6.30x mean=5.00x` |
| Multi-config | Phase397w fallback removes the exact bucket crash, but the active 0.19-real surface still fails at `max=3.57x mean=2.58x` |

The failure is not an int4_wo scalar fitting issue. The 0.19 migration also exposes broader MULTI_CONFIG over-prediction after exact misses fall back to structural modeling.

## Verdict

Phase397v calibrated-SOL is a valid local model candidate, but the full validate migration is still blocked. Do not claim validate green, do not change thresholds, and do not open Default AIC.

Next decision: investigate the 0.19 MULTI_CONFIG over-prediction directly. Do not restart the exact-bucket widening route; Phase397w has rejected it.
