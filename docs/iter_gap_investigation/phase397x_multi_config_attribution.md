# Phase397x Multi-Config Attribution

Phase397x is offline and report-only. It does not change runtime, DB tables, gates, or thresholds.

## Summary

| Item | Result |
|---|---|
| Active gate | 0.19-real 8-card MULTI_CONFIG x6 |
| Direction | all six scenarios overpredict throughput |
| Error | max=3.57x mean=2.57x |
| Worst scenario | `K2.5-tp4ep8dp2-32k3k` |
| Default AIC | Default AIC remains No-Go |

## Scenario Results

| Scenario | Tier | Real tok/s/GPU | Sim tok/s/GPU | Sim/real | Error | Direction |
|---|---:|---:|---:|---:|---:|---|
| K2.5-tp4ep8dp2-32k3k | B | 53.3 | 190.1 | 3.57x | 3.57x | sim_overpredicts_throughput |
| K2.5-tp4ep8dp2-8k2k | A | 137.7 | 396.6 | 2.88x | 2.88x | sim_overpredicts_throughput |
| K2.5-tp4ep8dp2-8k2k-bt65536 | B | 156.0 | 394.4 | 2.53x | 2.53x | sim_overpredicts_throughput |
| K2.5-tp8ep8-8k2k | A | 133.5 | 292.9 | 2.19x | 2.19x | sim_overpredicts_throughput |
| K2.5-tp8ep8-32k3k | B | 52.5 | 113.8 | 2.17x | 2.17x | sim_overpredicts_throughput |
| K2.5-tp8ep8-8k2k-bt65536 | B | 138.5 | 289.6 | 2.09x | 2.09x | sim_overpredicts_throughput |

## Attribution

Tier A has decode per-op profiles only. It can explain decode composition, not full prefill/mixed request cost.

| Scenario | Category | Sim decode ms | Real decode ms | Sim/real latency | Direction |
|---|---|---:|---:|---:|---|
| K2.5-tp8ep8-8k2k | attention | 31.89 | 16.90 | 1.89x | sim_overcharges_latency |
| K2.5-tp8ep8-8k2k | comm | 6.40 | 6.18 | 1.04x | sim_overcharges_latency |
| K2.5-tp8ep8-8k2k | gemm | 2.87 | 4.04 | 0.71x | sim_undercharges_latency |
| K2.5-tp8ep8-8k2k | moe | 8.66 | 9.53 | 0.91x | sim_undercharges_latency |
| K2.5-tp8ep8-8k2k | other | 0.79 | 2.74 | 0.29x | sim_undercharges_latency |
| K2.5-tp4ep8dp2-8k2k | attention | 30.05 | 13.65 | 2.20x | sim_overcharges_latency |
| K2.5-tp4ep8dp2-8k2k | comm | 6.63 | 12.00 | 0.55x | sim_undercharges_latency |
| K2.5-tp4ep8dp2-8k2k | gemm | 3.57 | 6.17 | 0.58x | sim_undercharges_latency |
| K2.5-tp4ep8dp2-8k2k | moe | 8.67 | 15.36 | 0.56x | sim_undercharges_latency |
| K2.5-tp4ep8dp2-8k2k | other | 0.79 | 2.25 | 0.35x | sim_undercharges_latency |

Strongest actionable Tier A decode undercharge: `K2.5-tp4ep8dp2-8k2k` `comm` sim/real latency = 0.55x.
Decode attention is not the aggregate undercharge: Tier A attention is overcharged in both available profiles.
The largest modeled mixed-step exclusion is context attention in `K2.5-tp4ep8dp2-32k3k`: 17.26 ms/iter equivalent.

## Boundaries

- Tier B rows have no real per-op profile and stay aggregate-only.
- No GPU or SSH was used.
- No DB row, runtime path, threshold, or Default AIC gate was changed.
