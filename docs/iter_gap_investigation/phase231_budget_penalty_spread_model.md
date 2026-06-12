# Phase231: Budget Penalty Spread Model

## Decision

| Item | Result |
|---|---|
| Best in-sample form | topology_plus_shape |
| Best LOOCV form | global_mean |
| Default AIC | No-Go |
| Recommended boundary | diagnostic_only_exact_key |
| PerfDatabase | No-Go |
| GPU benchmark | No-Go in Phase231 |
| Boundary flags | diagnostic_only=true valid_for_default=false perf_database=false |

## Model Errors

| Model | Parameters | Mean abs error | Max abs error | Max relative error | LOOCV mean abs error | LOOCV max abs error |
|---|---:|---:|---:|---:|---:|---:|
| global_mean | 1 | 0.063593 | 0.106103 | 0.090881 | 0.076312 | 0.127323 |
| topology_mean | 2 | 0.051337 | 0.104355 | 0.105005 | 0.077005 | 0.156533 |
| shape_mean | 3 | 0.063594 | 0.090058 | 0.091209 | 0.127187 | 0.180115 |
| topology_plus_shape | 4 | 0.051337 | 0.077005 | 0.077485 | 0.127880 | 0.156791 |

## Interpretation

The six-row evidence family is still diagnostic-only.
The complex model fits in-sample better, but the generalization evidence is insufficient.
The best_loocv_form remains a warning against promoting the in-sample fit into default AIC.
Default AIC remains No-Go; the next safe boundary is exact-key diagnostic evidence, not VLLMBackend.run_agg or PerfDatabase wiring.
