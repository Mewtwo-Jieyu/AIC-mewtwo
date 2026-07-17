# Phase463 six-point latency recollect

- Remote artifact: `/mnt/shared-storage-user/zhaojieyu/backup/aic/phase463_six_point_latency_recollect_263a969`
- Streaming canary throughput delta: `0.0122%` (limit `2.0000%`)
- Formal throughput gate: `3/6` passed; failed: `K2.5-tp4ep8dp2-32k3k;K2.5-tp8ep8-8k2k-bt65536;K2.5-tp4ep8dp2-8k2k-bt65536`.
- Maximum frozen-real throughput drift: `1.2050%`.
- Throughput gate: simulator/real absolute error must be at most `15%`.
- Latency truth: vLLM Prometheus histogram `_sum/_count`; streaming client timing is cross-check evidence.
- Latency gate: `baseline_only`; TTFT, TPOT, and E2E are comparison evidence, not readiness gates. Percentiles remain in the CSV.
- Evidence boundary: `diagnostic_only=true`, `valid_for_default=false`, `perf_database=false`.
- Default AIC 仍为 No-Go.

## Throughput

| Scenario | Real output tok/s/GPU | Sim output tok/s/GPU | Absolute error | Gate | Frozen drift |
|---|---:|---:|---:|---|---:|
| K2.5-tp8ep8-8k2k | 146.1113 | 165.9845 | 13.60% | pass | -0.360% |
| K2.5-tp8ep8-32k3k | 50.0801 | 49.5027 | 1.15% | pass | -0.063% |
| K2.5-tp4ep8dp2-8k2k | 164.3446 | 163.0730 | 0.77% | pass | 0.857% |
| K2.5-tp4ep8dp2-32k3k | 52.6355 | 64.0492 | 21.68% | fail | -1.205% |
| K2.5-tp8ep8-8k2k-bt65536 | 124.2006 | 155.8699 | 25.50% | fail | 0.148% |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 110.1960 | 127.4641 | 15.67% | fail | 0.636% |
| K2.5-tp4ep8dp2-32k3k-c64-diagnostic | 53.4844 | 64.0492 | 19.75% | diagnostic | n/a |

## Mean latency

| Scenario | TTFT real/sim/ratio (ms) | TPOT real/sim/ratio (ms) | E2E real/sim/ratio (ms) |
|---|---:|---:|---:|
| K2.5-tp8ep8-8k2k | 105573.42 / 95635.50 / 0.906x | 48.72 / 43.41 / 0.891x | 202966.76 / 182406.85 / 0.899x |
| K2.5-tp8ep8-32k3k | 756839.83 / 877331.74 / 1.159x | 31.27 / 33.43 / 1.069x | 850633.17 / 977576.67 / 1.149x |
| K2.5-tp4ep8dp2-8k2k | 42886.67 / 39622.88 / 0.924x | 69.95 / 72.81 / 1.041x | 182721.52 / 185164.15 / 1.013x |
| K2.5-tp4ep8dp2-32k3k | 689454.70 / 561265.59 / 0.814x | 40.62 / 34.92 / 0.859x | 811288.87 / 665981.52 / 0.821x |
| K2.5-tp8ep8-8k2k-bt65536 | 165474.44 / 138575.02 / 0.837x | 34.24 / 29.97 / 0.875x | 233926.87 / 198476.24 / 0.848x |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 200142.37 / 171586.55 / 0.857x | 28.20 / 28.15 / 0.998x | 256523.58 / 227865.82 / 0.888x |
| K2.5-tp4ep8dp2-32k3k-c64-diagnostic | 307809.85 / 250963.53 / 0.815x | 39.96 / 34.92 / 0.874x | 427638.33 / 355679.46 / 0.832x |

## Runtime diagnostics

| Scenario | GPU util mean | Power mean (W) | Real context/gen reqs per logged iter | Sim prefill/decode reqs per iter |
|---|---:|---:|---:|---:|
| K2.5-tp8ep8-8k2k | 99.38% | 539.51 | 0.0565 / 56.5464 | 0.0751 / 54.5763 |
| K2.5-tp8ep8-32k3k | 99.80% | 577.40 | 0.0087 / 13.1134 | 0.0104 / 13.2297 |
| K2.5-tp4ep8dp2-8k2k | 98.40% | 490.22 | 0.0431 / 42.9938 | 0.0593 / 43.3681 |
| K2.5-tp4ep8dp2-32k3k | 98.49% | 534.12 | 0.0059 / 8.8186 | 0.0071 / 8.8214 |
| K2.5-tp8ep8-8k2k-bt65536 | 99.55% | 506.58 | 0.0204 / 36.5205 | 0.0262 / 34.8837 |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 97.85% | 457.45 | 0.0071 / 13.4961 | 0.0109 / 14.1679 |
| K2.5-tp4ep8dp2-32k3k-c64-diagnostic | 98.50% | 540.67 | 0.0059 / 8.8187 | 0.0071 / 8.8214 |
