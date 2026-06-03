# Phase126: Go / No-Go

## Decision

| Item | Result |
|---|---|
| Runner local preflight | Go |
| Target container patch dry-run | Go |
| Exact-shape diagnostic capture design | Go |
| Traffic generation | Go |
| Benchmark metric use | No-Go |
| Full exact-shape capture run | No-Go after capture1 mismatch |
| Bucket1 blocker review | Completed |
| Drain-boundary design | Local Go |
| Drain runner target patch-dry-run | Go |
| Capture2 drain plan | Accepted |
| Capture2 drain full capture | Completed / Accepted |
| Phase127 diagnostic handoff | Go |
| Phase128 diagnostic use-design | Go |
| Next remote full capture | No-Go unless separately planned |
| Phase127 remote full capture | No-Go |
| Phase128 remote full capture | No-Go |
| Default AIC | No-Go |
| PerfDatabase | No-Go |
| Interpolation / extrapolation | No-Go |

## Entry Conditions

| Gate | Requirement |
|---|---|
| Phase125 manifest | Accepted exact-shape manifest |
| Phase125 guard | Accepted forbidden-field guard |
| Phase126 runner | Local checks pass |
| Target dry-run | PASS |
| Target buckets | `1 / 15 / 16 / 241 / 1808 / 2048 / 8192` |
| Capture1 blocker dir | `/mnt/nvme1n1/ml_research/jieyu/aic/docs/iter_gap_investigation/phase126_moe_activation_10k2k_b32_bt8192_p18000_capture1` |
| Capture2 accepted dir | `/mnt/nvme1n1/ml_research/jieyu/aic/docs/iter_gap_investigation/phase126_moe_activation_10k2k_b32_bt8192_p18000_capture2_drain` |
| Parser | Use `scripts/analyze_vllm_moe_activation_phase123.py` unchanged |
| Capture1 blocker | `phase126_exact_shape_capture_blocker.md` |
| Bucket1 review | `phase126_bucket1_blocker_review.md` |
| Drain-boundary design | `phase126_drain_boundary_design.md` |
| Drain runner preflight | PASS |
| Capture2 plan | `phase126_capture2_drain_capture_plan.md` |
| Capture2 result | Accepted drain-stable exact-shape evidence |
| Phase127 handoff | `phase127_diagnostic_evidence_handoff.md` |
| Phase127 manifest | `phase127_diagnostic_evidence_manifest.csv` |
| Phase128 use-design | `phase128_diagnostic_use_design.md` |

## Stop Rules

| Stop rule | Action |
|---|---|
| Config SHA mismatch | Stop |
| Config fallback | Discard artifact |
| Random weight | Discard artifact |
| Marker before readiness | Discard artifact |
| Missing marker rows | Stop |
| Bucket/count mismatch | Stop; capture1 hit this rule |
| Undefined drain boundary | No-Go |
| Drain timeout | Stop; do not parse artifact as accepted |
| Capture2 already attempted | No-Go |
| Unplanned full capture after capture2 | No-Go |
| Bench metric use as evidence | No-Go |
| Capture2 use as performance evidence | No-Go |
| Phase127 manifest use as performance evidence | No-Go |
| Default AIC request | No-Go |

## Next Work

Use Phase128 only to define downstream references to Phase127 diagnostic exact-shape evidence. Do not run another full capture without a new plan. Do not write a model table, do not connect default AIC, and do not use Phase117 rows to predict missing buckets.
