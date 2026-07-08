# Phase445-B Per-Layer Prefill Recheck

- Verdict: `prefill_gap_is_busy_charge_not_wall_only`.
- High-confidence mixed prefill buckets: `3`.
- Busy-gap share: `0.994`; wall-gap share: `0.006`.
- The current evidence is whole-forward event timing, not per-layer timing. It can reject a wall-only explanation, but it cannot split the busy excess by layer without a lower-level map.

## Summary
| metric | value |
| --- | --- |
| verdict | prefill_gap_is_busy_charge_not_wall_only |
| mixed_high_bucket_count | 3 |
| busy_gap_total_ms | 2154.894164 |
| wall_gap_total_ms | 13.729852 |
| busy_gap_share | 0.993669 |
| wall_gap_share | 0.006331 |
| busy_over_sim_bucket_count | 3 |
| wall_over_busy_bucket_count | 0 |
| mixed_or_inconclusive_bucket_count | 0 |

## Largest Mixed-Prefill Buckets
| ctx_tokens | decode_batch | event_count | sim_ms | busy_ms | wall_ms | busy_minus_sim | wall_minus_busy | candidate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7959 | 41 | 408 | 413.019601 | 1131.856384 | 1136.245000 | 718.836783 | 4.388616 | busy_over_sim |
| 7957 | 43 | 264 | 413.361055 | 1131.437927 | 1136.110000 | 718.076872 | 4.672073 | busy_over_sim |
| 7958 | 42 | 556 | 413.190328 | 1131.170837 | 1135.840000 | 717.980509 | 4.669163 | busy_over_sim |

## Next Gate

Do not model the 8k prefill residual as a generic host or DP wall term. The high-confidence mixed prefill buckets show the forward-busy event is already materially above cb_sim charge. The next valid gate is either a lower-overhead B2b busy measurement with finer attribution, or a category/layer mapping that explains which part of the busy charge is missing.
