# Phase128: Diagnostic Use Design

## Decision

| Item | Result |
|---|---|
| Phase128 scope | Diagnostic use-design only |
| Remote full capture | No-Go |
| Source of truth | `docs/iter_gap_investigation/phase127_diagnostic_evidence_manifest.csv` |
| Default AIC | No-Go |
| PerfDatabase | No-Go |
| Time model | No-Go |
| Benchmark performance claim | No-Go |
| Interpolation / extrapolation | No-Go |

## Reference Contract

| Field | Downstream use |
|---|---|
| `tokens_actual` | Exact MoE activation bucket identifier |
| `occurrence_count` | Exact accepted marker count from capture2 drain |
| `source` | Evidence provenance; must be `capture2_drain` |

Downstream design may cite only these fields from the Phase127 manifest unless a later phase defines a new evidence gate.

## Accepted Buckets

| tokens_actual | occurrence_count | source |
|---:|---:|---|
| 1 | 6960 | `capture2_drain` |
| 15 | 240 | `capture2_drain` |
| 16 | 959280 | `capture2_drain` |
| 241 | 240 | `capture2_drain` |
| 1808 | 240 | `capture2_drain` |
| 2048 | 240 | `capture2_drain` |
| 8192 | 480 | `capture2_drain` |

Total marker rows: `967680`.

## Allowed Use

| Use | Decision |
|---|---|
| State that these 7 buckets have diagnostic exact-shape evidence | Allowed |
| Compare downstream proposed bucket lists against Phase127 manifest | Allowed |
| Explain Phase117 exact-key coverage gap for these buckets | Allowed |
| Design a future evidence gate that starts from these buckets | Allowed |

## Forbidden Use

| Use | Decision |
|---|---|
| Default AIC integration | No-Go |
| PerfDatabase write | No-Go |
| Time model input | No-Go |
| Benchmark metric based performance claim | No-Go |
| Phase117 interpolation | No-Go |
| Phase117 extrapolation | No-Go |
| New remote capture without a separate plan | No-Go |

## Next Gate

Any move toward default model behavior, performance tables, or timing evidence must start a new phase. That phase must define its own source, schema, acceptance criteria, cleanup checks, and Go/No-Go decision before any remote run or model-path change.
