# Phase413 Steady Decompose

Phase413 复用 Phase412 N=512 trace，只做离线 elapsed 分解；不改 runtime、PerfDatabase 或 gate。

## Verdict

- verdict: `prefill_occupancy_dominates`
- Phase414 target: `audit_chunked_prefill_wall_occupancy`
- steady target penalty: `1.357625`
- steady metric penalty: `1.357625`
- consistency gate: `passed`

## Decomposition

| block | ms | share |
|---|---:|---:|
| prefill wall | 805248.933241 | 0.647177 |
| DP peer-stall extra | 372.060000 | 0.000299 |
| clean decode latency gap | 438628.227887 | 0.352524 |

## Decode Audit

| clean decode steps | intrinsic ms | real mean ms | sim mean ms | real/sim | batch mean | batch p90 |
|---:|---:|---:|---:|---:|---:|---:|
| 106804 | 18.845000 | 27.634954 | 19.413327 | 1.422819 | 9.007793 | 9.000000 |

## Interpretation

- steady penalty 先从 raw metrics 重新计算，再和 Phase412 CSV 交叉核对。
- prefill wall 是同一墙钟窗口内任一 DP engine 正在 prefill 的时间。
- peer-stall extra 是 decode-only step 和 peer prefill 重叠时，相对同 batch clean decode median 多出来的 elapsed。
- clean decode latency gap 用 clean decode-only step 对 cb_sim pure-decode 模型逐 batch 对账。

## Boundary

- GPU/SSH: not used in Phase413.
- Runtime/PerfDatabase/gate: not modified.
- Phase405 penalty: not read.
- Default AIC: No-Go.
