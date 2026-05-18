# Phase 42: MoE WNA16 Perf Spec

## 结论

MoE WNA16 aggregate 现在只能保留 diagnostic shape key。任何后续 perf 数据想进入 experimental latency，必须解决真实 shape、真实 tuning config、权重口径和 fallback 状态。Phase 25 random-weight fallback timing 没有入表资格。

| 项 | 决策 |
|---|---|
| 当前是否能写 MoE WNA16 perf table | 不能 |
| Phase 38 WNA16 key 是否保留 | 保留，diagnostic-only |
| Phase 25 timing 是否可用 | 不可用，random weight + fallback |
| fallback=true 是否允许进表 | 不允许 |
| 下一步最低门槛 | real-shape + non-fallback tuning + 明确权重口径 |

## 当前 diagnostic key

| 字段 | 当前值 |
|---|---|
| backend | `vllm` |
| module | `DeepseekV2MoE` |
| kernel | `wna16` |
| hidden | `7168` |
| intermediate | `2048` |
| experts | `384` |
| topk | `8` |
| tokens_padded | `248` |
| tokens_actual | `241` |
| dtype | `bfloat16` |
| topology | `tp=4, dp=2, ep=8` |
| tuning_config_loaded | `false` |
| fallback | `true` |

## 数据资格

| 要求 | 标准 |
|---|---|
| model shape | 必须来自模型 config，包含 hidden、intermediate、experts、topk |
| runtime shape | 必须包含 tokens_padded、tokens_actual、phase、forward_regime |
| topology | 必须包含 TP/DP/EP/world size |
| precision | 必须包含 dtype、quantization、WNA16 kernel 标识 |
| tuning config | 必须明确 `tuning_config_loaded` 和 `fallback` |
| 权重口径 | 必须明确 `loaded_weight` / `random_weight` |
| 计时边界 | 必须只包 MoE aggregate 或明确 compiled aggregate 边界 |
| 默认安全 | 必须固定 `valid_for_default=false`、`perf_database=false` |

## 可进入 future experimental latency 的候选

| 数据 | 条件 |
|---|---|
| real-shape WNA16 timing | 真实 Kimi shape，non-fallback tuning config，权重口径明确 |
| loaded-weight module timing | 可独立调用 MoE aggregate，不走 `_model_forward` 残差 |
| compiled MoE aggregate timing | 能证明边界只覆盖 MoE aggregate，不混入 NCCL/global envelope |

## 无资格数据

| 数据 | 拒绝原因 |
|---|---|
| Phase 25 single-layer timing | random weight + fallback，不代表线上 compiled aggregate |
| Phase 36 profiler MoE family ms | profiler coverage 不完整，且改变时序 |
| MoE marlin kernel 名 | 只能证明 kernel family，不能证明模块耗时 |
| fallback timing | tuning config 未命中，不能作为真实性能数据 |
| full `_model_forward` 减法 | residual 相减，不是干净模块数据 |

## Future Key 草案

| 字段组 | 字段 |
|---|---|
| backend | `backend`, `vllm_version`, `module_class`, `experts_class` |
| model shape | `hidden`, `intermediate`, `experts`, `topk`, `shared_experts` |
| runtime shape | `tokens_padded`, `tokens_actual`, `phase`, `forward_regime` |
| topology | `tp`, `dp`, `ep`, `world_size` |
| precision | `dtype`, `quantization`, `moe_kernel=wna16` |
| tuning | `moe_tuning_config_file`, `tuning_config_loaded`, `fallback` |
| weights | `loaded_weight`, `random_weight` |
| boundary | `timing_source`, `valid_for_default=false`, `perf_database=false` |

## 停止条件

| 情况 | 处理 |
|---|---|
| tuning config 仍 fallback | 只能 diagnostic，不进表 |
| 只能用 random weight | 只能验证边界，不下性能结论 |
| 必须调用 `_model_forward` | 停止 MoE module benchmark，转 forward envelope |
| 需要 residual 相减 | 停止 |
| 想用 TRT-LLM MoE 表替代 | 拒绝，后端和 kernel 路径不同 |

## Phase 43 入口

| 文档 | 作用 |
|---|---|
| `phase43_moe_wna16_tuning_config_check.md` | 判断 WNA16 tuning config 是否 non-fallback |
| `phase43_perf_feasibility_go_nogo.md` | 汇总是否进入 Phase 44 timing smoke |
