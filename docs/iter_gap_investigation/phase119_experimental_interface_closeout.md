# Phase119 Experimental Interface Closeout

## Decision

Phase116-118 close out the MoE WNA16 experimental interface. The usable interface is an exact-key diagnostic table plus a full-key query API. This is not a default AIC latency model.

| Component | Status | Boundary |
|---|---|---|
| Phase116 fit audit | Keep | Error audit only, not a formula |
| Phase117 exact-key table | Keep | Experimental diagnostic table |
| Phase118 query API | Keep | Full exact-key lookup only |
| Default AIC latency | No-Go | No `PerfDatabase`, `run_static`, or `IterationLatencyCalculator` integration |

## Interface Shape

| Layer | File | Rule |
|---|---|---|
| Fit audit | `scripts/fit_moe_wna16_diagnostic_phase116.py` | Computes diagnostic linear risk metrics only |
| Table builder | `scripts/build_moe_wna16_experimental_table_phase117.py` | Emits one row per measured token shape |
| Query API | `scripts/query_moe_wna16_experimental_table_phase118.py` | Requires every key field to match |
| Query example | `phase118_query_key_tokens128.json` | Demonstrates a valid full key |
| Query result | `phase118_query_result_tokens128.csv` | Demonstrates one exact hit |

## Hard Boundaries

| Topic | Decision |
|---|---|
| Measured tokens | `128`, `248`, `512`, `1024` only |
| Missing token | Fail-fast |
| Changed model/config/topology/input field | Fail-fast |
| `synthetic_random_hidden_states` | Must remain explicit |
| Interpolation | Forbidden |
| Extrapolation | Forbidden |
| Formula fallback | Forbidden |
| Default AIC | No-Go |

## Closeout Result

Phase119 can enter a staging dry-run plan in Phase120. The dry-run must use a whitelist and must not stage unrelated research artifacts, raw logs, runner scripts, or historical profiler/residual/NCCL trace outputs.
