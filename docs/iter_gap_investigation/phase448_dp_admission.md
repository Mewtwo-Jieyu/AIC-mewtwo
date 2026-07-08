# Phase448 DP admission

结论: 不合入 runtime lockstep。多 replica admission 诊断路径已经补上,但 8k2k 指纹门未过:sim mixed decode batch p50 仍是 62,真实容差是 34-52。

| check | result | decision |
|---|---:|---|
| 8k2k mixed decode batch p50 | 62 vs 34-52 | fail |
| 8k2k bucket p50 | 8000 vs 8000 | pass |
| 32k3k mixed decode batch p50 | 8 vs 6-9 | pass |
| 32k3k bucket p50 | 32000 vs 67-32000 | pass |
| lockstep allowed | false | stop before runtime |

## What Changed

| area | status |
|---|---|
| DP admission router | added as diagnostic component: `score = waiting * 4 + running`, 100ms stale refresh, optimistic waiting increment |
| multi replica simulator loop | added as diagnostic method with independent scheduler/clock/KV per replica |
| default `run_agg` path | unchanged; failed multi-replica default wiring was withdrawn |
| Phase448 analyzer | added `phase448_dp_admission_fingerprint.{csv,md}` |

## Why Stop

The diagnostic multi-replica path makes 32k3k composition land in the real band, but not 8k2k. A temporary default wiring also made `dp2-8k2k` worse in validate (2.35x), so it was withdrawn. After withdrawal, default validate is back to the floor:

| scenario | ratio |
|---|---:|
| tp8ep8-8k2k | 1.25x |
| tp8ep8-32k3k | 1.07x |
| tp4ep8dp2-8k2k | 2.12x |
| tp4ep8dp2-32k3k | 1.09x |
| tp8ep8-8k2k-bt65536 | 1.12x |
| tp4ep8dp2-8k2k-bt65536 | 1.23x |

MULTI_CONFIG remains No-Go: max 2.12x.

## Next Target

The missing mechanism is narrower now: 8k2k still schedules too many decode requests before the mixed prefill chunk. The next phase should inspect why vLLM produces mixed steps around `(total_tokens=8000, decode_batch=41-43)` while the diagnostic sim still lands near 62. Do not add lockstep or serving-state rows until that scheduler composition gate passes.
