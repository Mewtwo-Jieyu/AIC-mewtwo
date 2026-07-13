# Phase462 Step 1: dynamics-family triage

## Verdict

Common-root verdict: `preemption_recompute_common_root_supported`. Sim-only giant-bucket mixed steps account for 42.1% of simulated mixed wall; 85.4% of that wall occurs in steps carrying recompute prefill tokens.

Unresolved MLA anomaly/scenario pairs: 0. Active anomaly candidates audited: 16/16.

This phase is report-only: runtime, PerfDB, validation gate, and Default AIC were not changed.

## Preemption panorama

| Scenario | Real | Sim | Sim/real |
|---|---:|---:|---:|
| K2.5-tp8ep8-8k2k | 111 | 186 | 1.68x |
| K2.5-tp8ep8-32k3k | 42 | 194 | 4.62x |
| K2.5-tp4ep8dp2-8k2k | 101 | 192 | 1.90x |
| K2.5-tp4ep8dp2-32k3k | 56 | 136 | 2.43x |
| K2.5-tp8ep8-8k2k-bt65536 | 117 | 219 | 1.87x |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 121 | 277 | 2.29x |

## PerfDB anomaly audit

A candidate is reachable only when its exact row coordinate is part of the runtime three-dimensional interpolation stencil. Static shape-range overlap is not counted.

No MLA anomaly candidate remains unresolved in the three failing scenarios.

Reachable but bounded below the observed residual:

| Scenario | Candidate | Influence calls | Bound |
|---|---|---:|---|
| K2.5-tp8ep8-8k2k-bt65536 | context_mla_perf.txt:1059 | 1 | heads=8;batch=1;seq=2048;deviation=5.277491;queries=372;excess_upper_ms=37.876390;mixed_wall_share=0.000284872;observed_residual=0.235 |

## Next gate

Step 2 route: `preemption_first_divergence`. No fix is authorized by this report alone; first-divergence evidence and vLLM 0.19 source agreement remain mandatory.

`diagnostic_only=true`, `valid_for_default=false`, `perf_database=false`, Default AIC No-Go.
