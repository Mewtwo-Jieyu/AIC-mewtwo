# Phase410 Elapsed Penalty

Phase410 复用 Phase409 的同一份 GPU trace，只做离线 elapsed 归因；不改 runtime、PerfDatabase 或 gate。

## Verdict

- verdict: `elapsed_lockstep_still_insufficient`
- elapsed penalty: `1.005902`
- needed penalty: `1.887133`
- penalty gate: `failed`
- Phase411 target: `recheck_tail_imbalance_or_missing_dp_cost`

## Summary

| scenario | real wall ms | counterfactual wall ms | intrinsic decode ms | stalled decode steps | dp extra ms | tail single-engine ms |
|---|---:|---:|---:|---:|---:|---:|
| K2.5-tp4ep8dp2-32k3k | 1040198.320000 | 1034095.300000 | 25.840000 | 8 | 22964.780000 | 270931.080000 |

## Interpretation

- elapsed 里确实能看到 peer-prefill-stalled decode：这些 decode-only step 的耗时远高于 intrinsic decode。
- 但把这些 stalled decode 换回 intrinsic 后，elapsed penalty 仍远低于 needed penalty；这条证据不足以证明 DP lockstep elapsed alone 解释 1.887。
- 当前最大结构信号是 engine wall 不对称和长 tail：后续应先查 tail imbalance 或其他缺失 DP cost，而不是直接落 runtime lockstep 模型。

## Boundary

- GPU/SSH: not used in Phase410.
- Runtime/PerfDatabase/gate: not modified.
- Phase405 penalty: not read.
- Default AIC: No-Go.
