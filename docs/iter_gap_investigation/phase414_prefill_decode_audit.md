# Phase414 Prefill Decode Audit

Phase414 复用 Phase412 N=512 trace 与 Phase413 steady 窗口，只做离线计费审计；不改 runtime、PerfDatabase 或 gate。

## Verdict

- verdict: `prefill_charge_gap_dominates`
- Phase415 target: `audit_mixed_prefill_charge_components`
- consistency gate: `passed`
- reconstructed penalty: `1.339864` vs target `1.357625`

## Audit A: Mixed Prefill

| prefill steps | real mean ms | sim mean ms | real/sim | gap ms | gen reqs mean |
|---:|---:|---:|---:|---:|---:|
| 332 | 4718.954849 | 1157.464898 | 4.076974 | 1182414.663744 | 6.451807 |

## Audit B: Decode KV Growth

| bins | real first ms | real last ms | sim first ms | sim last ms | real slope | sim slope | slope ratio | gap ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 27.813512 | 27.473771 | 19.481817 | 20.045884 | -0.339741 | 0.564068 | -0.602306 | 39.343076 |

## Reconstruction

| target missing ms | reconstructed missing ms | error pct |
|---:|---:|---:|
| 1244249.221128 | 1182454.006820 | 1.308264 |

## Boundary

- GPU/SSH: not used in Phase414.
- Runtime/PerfDatabase/gate: not modified.
- Phase405 penalty: not read.
- Default AIC: No-Go.
