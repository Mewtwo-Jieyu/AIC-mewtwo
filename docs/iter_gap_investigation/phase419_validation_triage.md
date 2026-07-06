# Phase419 Validation Triage

Verdict: `fix_regression_bt65536_underprediction`.

Default AIC: `No-Go`.

## Summary

- Max before error: `6.573003x`.
- Max after error: `7.524468x`.
- Improved configs: `4`.
- Regressed configs: `2`.
- Max after-error config: `K2.5-tp4ep8dp2-8k2k-bt65536` (`7.524468x`, `bt65536_underprediction_regression`).
- Max config before/after: `6.573003x` -> `7.524468x`.
- Next phase target: `phase420_trace_bt65536_capacity_or_ep_fallback_branch`.

## Config Table

| Scenario | Before sim/real | After sim/real | Before error | After error | Class | Mechanism |
|---|---:|---:|---:|---:|---|---|
| K2.5-tp4ep8dp2-32k3k | 1.737886 | 1.734090 | 1.737886 | 1.734090 | improved | not_regressed |
| K2.5-tp4ep8dp2-8k2k | 2.794785 | 2.553139 | 2.794785 | 2.553139 | improved | not_regressed |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 0.152137 | 0.132900 | 6.573003 | 7.524468 | regressed | bt65536_underprediction_regression |
| K2.5-tp8ep8-32k3k | 1.216957 | 1.068795 | 1.216957 | 1.068795 | improved | not_regressed |
| K2.5-tp8ep8-8k2k | 1.492161 | 1.250461 | 1.492161 | 1.250461 | improved | not_regressed |
| K2.5-tp8ep8-8k2k-bt65536 | 1.026471 | 0.840071 | 1.026471 | 1.190376 | regressed | bt65536_underprediction_regression |

## Guardrails

- `diagnostic_only=true`.
- `valid_for_default=false`.
- `perf_database=false`.
- `runtime_modified=false`.
- `gpu_allowed=false`, `ssh_allowed=false`.
