# Phase447 sim fingerprint gate

结论: 指纹门未过。当前 sim 的 8k2k mixed decode batch p50 仍在真实 p10-p90 区间之外,因此本 phase 不允许合入 runtime DP lockstep。

| metric | sim | real low | real high | passed | reason |
|---|---:|---:|---:|---|---|
| mixed_decode_batch_p50 | 62.0 | 34.0 | 52.0 | False | sim_mixed_decode_batch_outside_real_band |
| mixed_step_count | 206 |  |  | False | sim_mixed_decode_batch_outside_real_band |

## Decision

- `runtime_lockstep_allowed=false`
- 下一步必须先建模 DP global admission / stale score routing / closed-loop wave,让 sim 自然生成真实相位与构成;不能手工注入 offset。
