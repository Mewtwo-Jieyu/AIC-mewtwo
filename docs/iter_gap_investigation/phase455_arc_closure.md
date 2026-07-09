# Phase455 TP8 scope closure

结论: `max_num_batched_tokens` scope 要落库,但 Phase454 TP8-8k B2b 行不能入库。

| 项 | 结果 | 处理 |
|---|---:|---|
| 无 scope 临时入库 | TP8-32k 从 1.163x 打到 1.703x | 必须加 scope |
| scope 维度 | `max_num_batched_tokens` 可由配置直接推导 | TP8 exact scope; DP2 保持 legacy scope |
| TP8-8k scoped 试入库 | 1.439x → 1.455x | 退回 193 行 |
| TP8-32k 污染审计 | 0 hit, 1.163x 不变 | 通过 |
| TP8-bt65536 污染审计 | 0 hit, 1.120x 不变 | 通过 |
| floor A/B | 1 improved, 5 unchanged, 0 regressed | 通过 |

## Decision

保留 schema 与查询 scope,不保留 TP8 B2b 行。原因很简单:scope 解决的是污染,但这批 TP8 行命中目标点后让目标点更差。

scope 的生效边界是:TP8 使用 `max_num_batched_tokens` 精确匹配;DP2 继续走旧 serving-state key。这个边界来自本轮 A/B:全局 exact scope 会让 dp2-32k3k 从 1.056x 回归到 1.173x,说明 DP2 旧行不能在本 phase 被重新分 regime。

因此 Phase455 没有达到 6/6 <= 15%,不提议把 MULTI_CONFIG 门限从 1.50 收紧到 1.15。当前唯一未收口点仍是 TP8-8k2k serving 态覆盖,下一步要先查 TP8 B2b 行的测量口径或缺失 regime,不能靠扩大命中硬吃。

## Handoff

fork 同步专门任务暂不启动为主线收官;可以先把本轮已验证的 scope 机制带走,但 TP8 行保持未入库状态。
