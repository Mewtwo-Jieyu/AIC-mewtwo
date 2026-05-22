# Phase 72 32k1k Scheduler Rule Design

## Conclusion

The `32k1k_b16` rule should be a narrow evidence-backed branch. It should exactly describe the Phase71 artifact first, then Phase73 can decide whether to implement it.

| field | value |
|---|---|
| scenario | `32k1k_b16` |
| `isl` | `32000` |
| `osl` | `1000` |
| `concurrency` | `16` |
| `max_num_batched_tokens` | `8192` |
| topology | `tp4dp2ep8` |
| DP rows | DP0 `1003`, DP1 `1003` |
| mixed rows | `0` |

## Proposed Rule

| step range | DP0 descriptor | DP1 descriptor |
|---|---|---|
| `0` | `prefill 8192+0 / NONE:8192` | `prefill 8192+0 / NONE:8192` |
| `1` | `prefill 8192+0 / NONE:8192` | `prefill 8192+0 / NONE:8192` |
| `2` | `prefill 8192+0 / NONE:8192` | `prefill 8192+0 / NONE:8192` |
| `3` | `prefill 7536+0 / NONE:7536` | `prefill 7536+0 / NONE:7536` |
| `4` | `pure_decode 0+8 / FULL:8` | `pure_decode 0+8 / NONE:8` |
| `5..1002` | `pure_decode 0+8 / FULL:8` | `pure_decode 0+8 / FULL:8` |

## Implementation Guardrails For Phase73

| guardrail | requirement |
|---|---|
| shape gate | only accept `isl=32000`, `osl=1000`, `concurrency=16`, `bt=8192`, `dp=2` |
| DP split | do not infer from global ordinal; emit DP-local rows |
| mixed | must stay `0`; do not create fake mixed rows |
| graph mode | preserve DP1 step4 `NONE:8`; do not normalize it to `FULL:8` |
| row count | strict output must be `2006` rows |
| compare | strict join on `(alignment_key, dp_rank)` only |
| unsupported variants | fail-fast |

## What This Rule Does Not Prove

| non-claim | reason |
|---|---|
| generic long-ISL behavior | only one `32k1k_b16` holdout exists |
| budget generalization | only `bt8192` is observed |
| topology generalization | only `tp4dp2ep8` is observed |
| latency model | descriptor shape is not timing evidence |
| default `cb_sim` readiness | this remains experimental-only |

## Phase73 Test Targets

| target | expected |
|---|---|
| generated rows | `2006` |
| DP0 rows | `1003` |
| DP1 rows | `1003` |
| prefill rows | `8` |
| pure decode rows | `1998` |
| mixed rows | `0` |
| strict compare | all token/request/forward/regime/cudagraph same flags are `1` |
| deltas | all token/request/forward deltas are `0` |
