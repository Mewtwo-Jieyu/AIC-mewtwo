# Phase 72 Scheduler Regime Taxonomy

## Conclusion

The vLLM-like scheduler generator now has evidence for three scheduler regimes, but only two are implemented. `32k1k_b16` must be treated as a third regime, not a tweak to the existing mixed-row rules.

| regime | evidence case | observed shape | generator status |
|---|---|---|---|
| chunked-prefill with mixed bridge | `10k2k_b32 bt8192 tp4dp2ep8` | prefill chunks lead into a mixed row such as `240+1 / NONE:248` | implemented |
| short-ISL leader/follower with mixed bridge | `3k3k_b128 bt8192 tp4dp2ep8` | leader prefill, follower decode pressure, mixed row such as `496+1 / PIECEWISE:512` | implemented as narrow holdout |
| multi-chunk prefill then pure decode | `32k1k_b16 bt8192 tp4dp2ep8` | four prefill chunks per DP, then pure decode; mixed row count is `0` | design only |

## Regime Boundaries

| boundary | rule |
|---|---|
| alignment key | all regimes use `engine_dp:<dp_rank>:step:<engine_step_id>` |
| compare key | only `(alignment_key, dp_rank)` is valid |
| phase ordinal | not allowed as a join key |
| intersection-only compare | not allowed |
| missing rows | fail-fast |
| unsupported shape | fail-fast |
| latency fields | forbidden |

## Why `32k1k_b16` Is Separate

| observed fact | implication |
|---|---|
| `prefill=8`, `mixed=0`, `pure_decode=1998` | there is no mixed bridge to model |
| DP0 and DP1 both have `1003` rows | the shape is symmetric in row count |
| each DP has `8192,8192,8192,7536` prefill chunks | prefill is fully drained before decode |
| decode is `0+8` until step `1002` | decode request split is stable after prefill |
| DP1 first decode is `NONE:8`, then `FULL:8` | first decode mode is observed evidence, not a general rule |

## Modeling Boundary

This taxonomy is only about scheduler input semantics. It does not create latency, does not change default `cb_sim`, and does not write `PerfDatabase`, `run_static`, or `IterationLatencyCalculator`.
