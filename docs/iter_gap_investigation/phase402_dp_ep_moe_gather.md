# Phase402 DP+EP MoE Gather Attribution

## Verdict

Phase402 rejects the proposed root cause: the current DP2 runtime already charges `generation_moe` with gathered MoE tokens. `run_agg` splits the global batch to 64 per replica, then `MoE.query` multiplies by `attention_dp_size=2`, so the effective MoE token key is 128.

Forcing a local-only MoE@64 counterfactual does not explain the DP2 residual. The current DP2 worst error is 1.85x; the forced MoE@64 counterfactual is 1.85x. With the Phase397v int4_wo calibrated-SOL path, MoE@64 and MoE@128 are nearly identical because decode is weight-load bound.

## Scenario Summary

| scenario | dp | clean tok/s/gpu | current tok/s/gpu | current ratio | MoE@64 ms | MoE@128 ms | forced MoE@64 ratio | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| K2.5-tp8ep8-8k2k | 1 | 137.386464 | 169.141051 | 1.231133 | 8.153768 | 8.153768 | 1.231133 | control_no_dp_gather_delta |
| K2.5-tp8ep8-32k3k | 1 | 50.099161 | 39.087433 | 1.281720 | 8.153768 | 8.153768 | 1.281720 | control_no_dp_gather_delta |
| K2.5-tp4ep8dp2-8k2k | 2 | 151.681886 | 271.958961 | 1.792956 | 8.148719 | 8.153768 | 1.793168 | rejected_current_runtime_already_gathers_moe_tokens |
| K2.5-tp4ep8dp2-32k3k | 2 | 50.064056 | 92.590110 | 1.849433 | 8.148719 | 8.153768 | 1.849822 | rejected_current_runtime_already_gathers_moe_tokens |

## DP Imbalance

The DP0/DP1 72/56 split is kept as a secondary effect. Its max-vs-mean ratio is 1.125 and max-vs-min ratio is 1.286, which is too small to explain the remaining DP2 1.8x throughput gap.

## Boundary

- runtime_modified=false
- perf_database=false
- valid_for_default=false
- diagnostic_only=true
- default_readiness=No-Go

## Next

Phase403 should not implement a MoE gather-token fix as the next runtime change. The remaining DP2 residual should be attributed to another mechanism, such as DP stats visibility, communication, or per-op composition under the Phase400 clean baseline.
