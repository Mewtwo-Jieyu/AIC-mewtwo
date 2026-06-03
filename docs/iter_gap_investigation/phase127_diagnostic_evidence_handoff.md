# Phase127: Diagnostic Evidence Handoff

## Decision

| Item | Result |
|---|---|
| Phase127 scope | Handoff / use-design only |
| Remote full capture | No-Go |
| Accepted evidence | Phase126 capture2 drain-stable exact-shape marker evidence |
| Rejected evidence | Phase126 capture1 exact-count mismatch artifact |
| Benchmark metric use | No-Go |
| Default AIC | No-Go |
| PerfDatabase | No-Go |
| Interpolation / extrapolation | No-Go |

## Evidence Sources

| Source | Status | Path |
|---|---|---|
| Capture1 blocker dir | Rejected for exact-count use | `/mnt/nvme1n1/ml_research/jieyu/aic/docs/iter_gap_investigation/phase126_moe_activation_10k2k_b32_bt8192_p18000_capture1` |
| Capture1 blocker doc | Local blocker record | `docs/iter_gap_investigation/phase126_exact_shape_capture_blocker.md` |
| Capture1 blocker review | Root cause record | `docs/iter_gap_investigation/phase126_bucket1_blocker_review.md` |
| Capture2 accepted dir | Accepted diagnostic exact-shape source | `/mnt/nvme1n1/ml_research/jieyu/aic/docs/iter_gap_investigation/phase126_moe_activation_10k2k_b32_bt8192_p18000_capture2_drain` |
| Capture2 result doc | Local acceptance summary | `docs/iter_gap_investigation/phase126_capture2_drain_capture_plan.md` |
| Phase127 manifest | Local lightweight handoff manifest | `docs/iter_gap_investigation/phase127_diagnostic_evidence_manifest.csv` |

Large remote CSV and log files stay on the target machine. Phase127 only records their accepted summary and the lightweight bucket manifest.

## Accepted Manifest

| tokens_actual | occurrence_count | source | Use |
|---:|---:|---|---|
| 1 | 6960 | `capture2_drain` | diagnostic exact-shape evidence |
| 15 | 240 | `capture2_drain` | diagnostic exact-shape evidence |
| 16 | 959280 | `capture2_drain` | diagnostic exact-shape evidence |
| 241 | 240 | `capture2_drain` | diagnostic exact-shape evidence |
| 1808 | 240 | `capture2_drain` | diagnostic exact-shape evidence |
| 2048 | 240 | `capture2_drain` | diagnostic exact-shape evidence |
| 8192 | 480 | `capture2_drain` | diagnostic exact-shape evidence |

Total marker rows: `967680`.

## Allowed Use

| Use | Decision |
|---|---|
| Exact bucket coverage handoff | Allowed |
| Phase125 manifest consistency check | Allowed |
| Future diagnostic use-design | Allowed |
| Explaining why Phase117 exact keys do not cover these buckets | Allowed |

## Forbidden Use

| Use | Decision |
|---|---|
| Default AIC model path | No-Go |
| PerfDatabase row | No-Go |
| Time model input | No-Go |
| Phase117 interpolation | No-Go |
| Phase117 extrapolation | No-Go |
| Benchmark-derived performance claim | No-Go |

## Handoff Contract

Any downstream Phase127 or Phase128 design may cite `phase127_diagnostic_evidence_manifest.csv` only as diagnostic exact-shape evidence. If a downstream step needs performance data, model integration, or default AIC behavior, it must start a new phase with a separate plan and fresh acceptance gates.
