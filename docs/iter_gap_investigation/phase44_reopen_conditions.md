# Phase 44: Reopen Conditions

## 结论

Phase 44 后如果要重开 vLLM compiled body 建模，必须先满足下面的硬条件。没有 Go 条件就不能开 timing smoke，不能写 perf table，不能接默认模型。

## MoE WNA16 重开条件

| 条件 | 标准 |
|---|---|
| tuning config | `tuning_config_loaded=true` |
| fallback | `fallback=false` |
| 权重口径 | 明确 `loaded_weight=true`，或证明 random-weight 与 loaded-weight 对目标 kernel 等价 |
| 边界 | 只包 MoE WNA16 aggregate，不用 full `_model_forward` 差分 |
| key | 包含 hidden、intermediate、experts、topk、tokens、dtype、TP/DP/EP |
| 复现 | 单 key 多次稳定 |

## compiled comm 重开条件

| 条件 | 标准 |
|---|---|
| comm-only event boundary | 找到只包通信调用的边界 |
| 不改变语义 | 不用 profiler、NCCL debug timing、全局 sync |
| 来源归因 | 能区分 TP、EP/global、unknown |
| shape key | 包含 op、tensor shape、dtype、rank、group size、topology |
| 复现 | 单 key 多次稳定 |

## experimental latency 重开条件

| 条件 | 标准 |
|---|---|
| clean timing | 来自合格边界，不来自 profiler/debug/sync/residual |
| 单 key 复现 | 同 key 多次稳定 |
| holdout | 至少一个 shape 或 topology holdout |
| output | 仍固定 `valid_for_default=false` |
| 接口 | 只走 experimental 路径，不写默认 `PerfDatabase` |

## default AIC 重开条件

| 条件 | 标准 |
|---|---|
| 机制闭环 | 源码语义、runtime key、clean timing 三者一致 |
| 泛化验证 | 跨 shape、topology、模型或版本验证通过 |
| 默认安全 | 默认 validate 不退化 |
| 审核门槛 | 明确不是 residual / profiler gap / fallback correction |

## 当前状态

| 路线 | 当前能否重开 | 原因 |
|---|---|---|
| MoE WNA16 | 不能 | tuning config fallback |
| compiled comm | 不能 | 没有 comm-only event boundary |
| experimental latency | 不能 | 没有 clean timing |
| default AIC | 不能 | 没有泛化模型和默认验收 |
