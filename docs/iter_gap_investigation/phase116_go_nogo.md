# Phase116 Go/No-Go

## Decision

Phase116 is Go for Phase117 experimental exact-key perf-table prototype design. It is No-Go for a formula-based default AIC model.

| Question | Decision |
|---|---|
| Can the local fit script run on diagnostic data? | Yes |
| Can `a + b * tokens` be used as default latency? | No |
| Can Phase117 design an experimental table prototype? | Yes |
| Can Phase117 write `PerfDatabase`? | No |
| Can Phase117 change `run_static` or `IterationLatencyCalculator`? | No |

## Reason

| Check | Result |
|---|---|
| CSV gate | PASS |
| Safety flags | PASS |
| Trend sanity | PASS |
| Leave-one-shape-out max relative error | 12.9045989139 percent |
| 1024 extrapolation relative error | 7.47395558748 percent |
| Synthetic input | Still a hard limitation |

## Phase117 Entry

| Allowed | Not allowed |
|---|---|
| Experimental exact-key lookup table design | Default AIC integration |
| Key definition for model/config/topology/tokens | Global empirical factor |
| Fail-fast behavior for missing token shape | Residual bucket |
| Local prototype over Phase116 CSV only | Profiler/NCCL/sync/fallback/random-weight data |

## Stop Conditions

| Condition | Action |
|---|---|
| Need default latency path | Stop |
| Need extrapolation outside measured token keys | Stop or add diagnostic holdout first |
| Need production activation claim | Stop and collect real activation evidence first |
| Need to hide `synthetic_random_hidden_states` | Reject the artifact |
