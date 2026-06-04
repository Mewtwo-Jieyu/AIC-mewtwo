# Phase154: Budget-Aware Multi-Config Top-K

## Decision

| Item | Result |
|---|---|
| Scope | Local validation tooling |
| Default AIC | No-Go |
| PerfDatabase | No-Go |
| Remote benchmark | No-Go |
| Model formula change | No-Go |

Phase154 fixes the validation input shape for multi-config / Top-K diagnostics. It does not make the cb_sim model more accurate by itself.

## Budget Contract

| Scenario class | `max_num_batched_tokens` |
|---|---:|
| non-`bt65536` multi-config rows | `isl` |
| `bt65536` multi-config rows | `65536` |

The baseline `run_validation()` acceptance gate stays unchanged. The budget-aware path is diagnostic-only and must not change the default validation thresholds.

## Diagnostic Top-K Command

```bash
PYTHONPATH=src python scripts/validate_cb_simulator.py --diagnostic-multi-config-topk
```

Optional CSV output:

```bash
PYTHONPATH=src python scripts/validate_cb_simulator.py \
  --diagnostic-multi-config-topk \
  --diagnostic-multi-config-topk-out /private/tmp/phase154_topk.csv
```

## Output Fields

| Field | Meaning |
|---|---|
| `tp` / `dp` / `ep` | parallel topology |
| `max_bt` | budget used by cb_sim for this row |
| `real_output_tok_s_gpu` | real output throughput |
| `sim_output_tok_s_gpu` | cb_sim output throughput |
| `error_ratio` | symmetric error ratio |
| `rank` / `real_rank` / `rank_delta` | simulated vs real ordering |
| `diagnostic_only` | fixed `true` |
| `valid_for_default` | fixed `false` |
| `perf_database` | fixed `false` |

## Boundary

| Forbidden use | Reason |
|---|---|
| Default AIC integration | This is a validation view, not accepted model evidence |
| PerfDatabase write | No clean timing source is created |
| Phase127 bucket timing claim | Exact-shape buckets remain descriptor evidence until a separate timing gate |
| Interpolation / extrapolation | Budget-aware rows only describe observed benchmark rows |

If the diagnostic Top-K table exposes a new ordering problem, the next phase must decide a keyed model item separately. Phase154 only fixes the validation input shape.
