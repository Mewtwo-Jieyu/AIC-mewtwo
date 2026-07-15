# Phase462 backend semantic profile v1

## 结论

版本语义已从 tokenizer 测量行和 runtime 散落常数收敛到精确分发层。默认路径仍为 null-only，引擎环默认关闭；未知 backend/version 直接失败，不猜测、不回退。

| 扩展维度 | 归属 | 本步行为 |
|---|---|---|
| 硬件 | `systems/data/<hardware>/<backend>/<version>/` | 继续由版本化 PerfDB 目录承载 |
| 模型 | 模型配置 | 不变 |
| backend 版本语义 | `backend_semantic_profiles.json` | 按 `(backend, version)` 精确分发 |

## vLLM 0.19.0 profile

| 字段 | 值 | runtime 使用点 |
|---|---:|---|
| `null_blocks_per_pool` | 1 | 物理块转 scheduler 可分配块 |
| `batch_queue_depth` | 2 | 诊断引擎环在途批深度 |
| `engine_loop_default_enabled` | false | 默认路径保持 legacy/null-only |
| `engine_loop_diagnostic_available` | true | 显式诊断场景才可开启 |
| `request_progress_semantics` | `sampled_computed_placeholder` | Async 状态语义校验 |
| `preemption_victim_policy` | `running_tail` | 容量不足时 victim 策略校验 |
| `admission_queue_order` | `running_then_waiting` | scheduler 顺序校验 |
| `stop_admission_after_preemption` | true | Async 抢占后的 admit 规则 |
| `stop_after_partial_prefill` | true | partial prefill 后停止 admit |
| `completion_release_order` | `future_then_release_then_submit` | 完成/释放顺序校验 |

`pipeline_parallel_sizes=[1]` 与 `async_scheduling=true` 登记 0.19 引擎环适用范围。未来 profile 若引入框架尚未实现的语义值，加载时直接失败；新增常数值只需新增 profile 和对应版本数据目录。

## 红绿门

| 检查 | 结果 |
|---|---|
| 未知 backend/version | fail closed |
| 有限 KV 容量但无 profile | fail closed |
| 显式 profile 与 PerfDB 版本不符 | fail closed |
| tokenizer 测量行不再拥有 queue depth | 通过 |
| 默认 null-only 六点 CSV 对 `bef7d8d0` | 逐字节相同 |
| 默认 CSV SHA256 | `6f48dacef975a6209474e9629938e3c6c20fe50a777ccd5b22a2735fdb72f3e2` |

边界：本步不改 PerfDB、不启用引擎环、不改变 Default AIC No-Go。
