# Phase206: Holdout Default Readiness Audit

default_readiness=No-Go

## Decision

| Item | Result |
|---|---|
| Default AIC | No-Go |
| VLLMBackend.run_agg | Unchanged |
| PerfDatabase | No-Go |
| GPU benchmark | No-Go |
| Allowed use | diagnostic-only lookup |

## Readiness Gates

| Gate | Result |
|---|---|
| Holdout pair count | 4 |
| Evidence sufficiency | Only 4 holdout pairs; default No-Go |
| Steady penalty diagnosis | true |
| Shape behavior differs | true; no global constant |
| tp4dp2 12k2k below 1 | true; raw multiplier disallowed |
| Exact-key candidate | diagnostic-only lookup |

## Pair Evidence

| topology_key | shape_key | control_bt | holdout_bt | clean_high_over_control | steady_state_time_ratio |
|---|---|---:|---:|---:|---:|
| tp8_dp1_ep8 | isl4000_osl2000_batch128 | 4000 | 65536 | 1.012212 | 3.688569 |
| tp4_dp2_ep8 | isl4000_osl2000_batch128 | 4000 | 65536 | 1.133187 | 6.840699 |
| tp8_dp1_ep8 | isl12000_osl2000_batch128 | 12000 | 65536 | 1.074279 | 2.460281 |
| tp4_dp2_ep8 | isl12000_osl2000_batch128 | 12000 | 65536 | 0.993809 | 4.198380 |

## Reasons

- Only 4 holdout pairs are available, so default AIC is No-Go.
- Large steady-state ratios with clean ratios near 1 support a diagnostic penalty hypothesis.
- 4k2k and 12k2k behavior differs, so there is no global constant.
- tp4dp2 12k2k is below 1, so a raw multiplier is not allowed.
- Phase200 remains a diagnostic-only lookup, not default model logic.
