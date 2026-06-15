# Phase264: Phase Mix / Decode Batch Diagnostic

## Decision

| Item | Result |
|---|---|
| Mechanism hypothesis | phase_mix_decode_batch_diagnostic |
| Default AIC | No-Go |
| PerfDatabase | No-Go |
| GPU benchmark | No-Go in Phase264 |
| Boundary flags | diagnostic_only=true valid_for_default=false perf_database=false |

## Family Summary

| Topology / shape | Role | Scenario | max_bt | phase mix prefill/mixed/decode | pure decode p99/max | mixed p99/max | output ratio | rank sum p99/max fill |
|---|---|---|---:|---:|---:|---:|---:|---:|
| tp8_dp1_ep8 / 4k2k | control | tp8ep8-4k2k-bt4000 | 4000 | 1/1/2000 | 128.000000/128 | 2033.000000/2033 | 1.000000 | 0.032000/1.000000 |
| tp8_dp1_ep8 / 4k2k | holdout | tp8ep8-4k2k-bt65536 | 65536 | 1/2/2000 | 128.000000/128 | 1553.000000/1553 | 0.989048 | 0.001953/0.061035 |
| tp4_dp2_ep8 / 4k2k | control | tp4dp2ep8-4k2k-bt4000 | 4000 | 1/2/1999 | 128.000000/128 | 1793.000000/1793 | 1.000000 | 0.016000/1.000000 |
| tp4_dp2_ep8 / 4k2k | holdout | tp4dp2ep8-4k2k-bt65536 | 65536 | 1/1/2000 | 128.000000/128 | 1793.000000/1793 | 1.095889 | 0.000977/0.062866 |
| tp8_dp1_ep8 / 12k2k | control | tp8ep8-12k2k-bt12000 | 12000 | 1/4/2003 | 128.000000/128 | 1041.000000/1041 | 1.000000 | 0.010667/1.000000 |
| tp8_dp1_ep8 / 12k2k | holdout | tp8ep8-12k2k-bt65536 | 65536 | 1/3/2000 | 128.000000/128 | 1265.000000/1265 | 0.967741 | 0.001953/0.183105 |
| tp4_dp2_ep8 / 12k2k | control | tp4dp2ep8-12k2k-bt12000 | 12000 | 1/2/1999 | 128.000000/128 | 1208.000000/1208 | 1.000000 | 0.005333/1.000000 |
| tp4_dp2_ep8 / 12k2k | holdout | tp4dp2ep8-12k2k-bt65536 | 65536 | 1/2/2000 | 128.000000/128 | 1778.000000/1778 | 1.339504 | 0.000977/0.183105 |

## Interpretation

Phase264 keeps the Phase258 actual scheduled token boundary and adds phase mix and decode batch readouts.
The analyzer classifies each iteration after rank-sum aggregation, so tp4_dp2_ep8 can have different DP payload phases without being reduced to the first raw worker row.
The high-budget rows still keep rank-sum p99/max fill far below the aggregate configured budget, so max_num_batched_tokens remains a ceiling rather than a direct linear scheduler cost.
The next diagnostic question is whether throughput spread is better explained by phase mix and decode batch shape than by the configured budget ceiling.
This remains diagnostic-only evidence. Do not wire it into VLLMBackend.run_agg, default AIC, or PerfDatabase.
