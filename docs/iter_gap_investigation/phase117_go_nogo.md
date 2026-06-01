# Phase117 Go/No-Go

## Decision

Phase117 is Go for Phase118 experimental query API design. It remains No-Go for default AIC.

| Gate | Result |
|---|---|
| Table generation | PASS |
| Exact token keys | PASS: `128`, `248`, `512`, `1024` |
| Missing token behavior | PASS: `256` and `768` fail-fast |
| Safety flags | PASS: `diagnostic_only=true`, `valid_for_default=false`, `perf_database=false` |
| Forbidden headers | PASS |
| Default AIC | No-Go |

## Phase118 Entry

| Allowed | Not allowed |
|---|---|
| Experimental query API over CSV | `PerfDatabase` write |
| Exact-key lookup helper | `run_static` change |
| Fail-fast missing key tests | `IterationLatencyCalculator` change |
| Explicit `experimental_table_prototype_only` metadata | Interpolation or extrapolation |

## Stop Conditions

| Condition | Action |
|---|---|
| Query asks for unmeasured token shape | Fail-fast |
| Query changes model/config/topology | Fail-fast |
| User asks for default AIC integration | Stop |
| Need production activation claim | Stop and collect real activation evidence first |
| Need formula fallback | Stop |
