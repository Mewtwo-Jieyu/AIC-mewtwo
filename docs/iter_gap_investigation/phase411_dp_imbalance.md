# Phase411 DP Imbalance

Phase411 复用 Phase409 trace 和 metrics，只做离线 DP 副本不对称归因；不改 runtime、PerfDatabase 或 gate。

## Verdict

- verdict: `steady_state_dp_underscaling_with_closed_set_tail`
- Phase412 target: `separate_request_assignment_tail_from_balanced_dp_cost`
- decomposition: `1.352194 x 1.395608 = 1.887133`
- needed penalty: `1.887133`
- decomposition gate: `passed`

## Summary

| scenario | tail engine | tail wall ms | tail penalty | balanced penalty | tail excess share | request ratio | decode length ratio | balanced batch ratio | dominant imbalance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| K2.5-tp4ep8dp2-32k3k | 0 | 270931.080000 | 1.352194 | 1.395608 | 0.397002 | 1.415094 | 1.000227 | 1.020880 | request_count |

## Interpretation

- tail 存在且很大，但 tail_excess_share 没到 80%，不能把 1.887 主要归成闭集 tail 伪影。
- prompt-token 反推请求分配约为 engine0/engine1 = request ratio；decode length ratio 接近 1，说明不是输出长度不均。
- balanced 阶段 generation_requests 均值接近，batch 分布不均不是主导；剩余 balanced penalty 仍需要单独查 DP steady-state cost。
- Phase412 应把 request-assignment tail 和 balanced DP cost 分开，不应直接给 simulator 加一个闭集 penalty。

## Boundary

- GPU/SSH: not used in Phase411.
- Runtime/PerfDatabase/gate: not modified.
- Phase405 penalty: not read.
- Default AIC: No-Go.
