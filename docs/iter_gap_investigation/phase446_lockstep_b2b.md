# Phase446 Lockstep + B2b Closeout

结论: `forward_total` 入库有效,但 runtime lockstep 不能落。94k 实测序列的 max 耦合能重建墙钟;当前 simulator 生成的是两个完全对称的 DP replica,取 max 后不变,所以不能解释 `tp4ep8dp2-8k2k` 的 2.12x。

## Gate Results

| gate | result | note |
|---|---:|---|
| Step0 max-coupled replay | pass | steady window error 0.576%, all windows <=10% |
| B2b overhead gate | pass | overhead_on did not slow throughput; 94,132 event rows |
| 32k short event run | pass | 360,244 event rows |
| forward_total ingest | partial pass | 32k mixed rows useful; 8k high-batch mixed still out of range |
| runtime lockstep | fail | symmetric replica schedules produce identical per-GPU throughput |
| MULTI_CONFIG gate | fail | max ratio 2.12x > 1.50x |

## Validate A/B

Baseline is the Phase438/442 floor table. Current table is after Phase446 `forward_total` rows, with runtime lockstep withheld.

| scenario | baseline ratio | current ratio | status |
|---|---:|---:|---|
| K2.5-tp8ep8-8k2k | 1.250x | 1.250x | unchanged |
| K2.5-tp8ep8-32k3k | 1.069x | 1.070x | unchanged |
| K2.5-tp4ep8dp2-8k2k | 2.118x | 2.120x | still failed |
| K2.5-tp4ep8dp2-32k3k | 1.255x | 1.090x | improved |
| K2.5-tp8ep8-8k2k-bt65536 | 1.120x | 1.120x | unchanged |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 1.252x | 1.230x | improved |

## Root Cause Of Non-Closure

| item | observation | implication |
|---|---|---|
| high-batch mixed miss | `mixed_prefill/forward_total` still misses decode_batch 56-64; 15 misses in the direct 8k run audit | 8k2k steady working point is not fully covered by B2b rows |
| symmetric DP schedule | single-replica per-GPU throughput = 291.673; two-replica lockstep per-GPU throughput = 291.673 | lockstep needs real phase asymmetry, not just `max(replica0, replica1)` over identical generated traces |
| measured replay still valid | 94k event rows reconstruct measured wall | mechanism exists in real vLLM trace, but simulator lacks the arrival/router asymmetry that creates it |

## Decision

Do not ship runtime DP lockstep in this phase. It would add code without changing the failing scenario and would hide the missing phase/asymmetry model.

Keep the `forward_total` path and the B2b rows because they improve 32k3k and do not regress the six-point table. Default AIC remains `No-Go`.

## Next Target

Phase447 should model or measure the DP phase/asymmetry source before another runtime lockstep attempt. The minimal next gate is: generated DP step sequences must reproduce the real 8k2k high-batch mixed miss/peer-prefill pattern before applying max coupling.
