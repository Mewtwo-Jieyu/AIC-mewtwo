# Phase116 MoE WNA16 Experimental Fit Report

## Decision

Phase116 completed a local diagnostic fit audit. The constant plus token-linear form is useful for error inspection, but it is not strong enough to become a default AIC latency model.

| Item | Result |
|---|---|
| Input | `phase114_shape_trend_summary.csv` |
| Shapes | `128`, `248`, `512`, `1024` |
| Target | `cuda_event_ms_mean` |
| Sanity | `wall_ms_mean` trend only |
| Input source | `synthetic_random_hidden_states` |
| Output use | `diagnostic_error_audit_only` |
| Default AIC | No-Go |

## Fit Result

| Field | Value |
|---|---:|
| `diagnostic_intercept_ms` | 0.936879342785 |
| `diagnostic_slope_ms_per_token` | 0.00132581849612 |
| `wall_sanity_intercept_ms` | 0.953326257955 |
| `wall_sanity_slope_ms_per_token` | 0.00132803535306 |
| CUDA event trend | Monotonic |
| Wall trend | Monotonic |

## Holdout Error

| Holdout | Predicted ms | Observed ms | Abs error ms | Relative error percent |
|---:|---:|---:|---:|---:|
| 128 | 1.17480622631 | 1.04053000286 | 0.134276223448 | 12.9045989139 |
| 248 | 1.23404959101 | 1.32156693339 | 0.0875173423855 | 6.6222406277 |
| 512 | 1.60526602248 | 1.6465929985 | 0.0413269760206 | 2.50984767081 |
| 1024 | 2.44373463516 | 2.27379240096 | 0.169942234199 | 7.47395558748 |
| 1024 extrapolation check | 2.44373463516 | 2.27379240096 | 0.169942234199 | 7.47395558748 |

## Interpretation

The linear form captures the broad monotonic trend, but the small-shape holdout error is too large to treat the formula as a robust model. The safer next step is an experimental exact-key perf-table prototype, where missing token shapes fail fast instead of extrapolating.

## Boundaries

| Boundary | Status |
|---|---|
| `diagnostic_only` | true |
| `valid_for_default` | false |
| `perf_database` | false |
| Production activation distribution | Not proven |
| Default `cb_sim` integration | Not allowed |
