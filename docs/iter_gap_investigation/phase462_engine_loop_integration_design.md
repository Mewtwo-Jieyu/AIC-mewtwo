# Phase462 Step 2c-1 引擎环集成设计评审

结论：设计面已封闭，可交用户评审；本步仍是 report-only，不授权 runtime 集成。

## 改动范围

| 运行路径 | 现状 | 2c-2 方案 | 交付边界 |
|---|---|---|---|
| `CBSimulator.run` | 串行调度→执行→调度 | 共享引擎环状态机 | 全门通过后进入默认路径 |
| `CBSimulator.run_multi_replica` | 按最早本地时钟串行推进 | 共享引擎环状态机 | 先提供能力，Step 3 再判默认启用面 |
| `CBSimulator._run_multi_replica_lockstep` | 同轮调度后按最慢 rank 推进 | 共享引擎环状态机 | 保留 lockstep max，只替换 rank 内引擎环 |

实现应只新增一个共享的内部引擎环，不复制三套状态机。状态至少显式区分 `sampled_output_tokens`、`computed_output_tokens` 与 `output_placeholders`；现有 `generated_tokens` 不能继续同时代表三种时态。纯 decode 批量跳步会绕过 future、input drain 和 KV 块边界，因此新路径禁用该优化，不补等价启发式。

到达层不是延迟常数：闭环 bench 在 `t=0` 暴露 `min(N,C)` 个请求，请求完成才释放下一条；tokenizer 按运行时 `32 requests / 2ms` 微批，批完成后向 EngineCore input queue 投递事件。实测 tokenizer 原语 WMAPE 为 5.92%，只在 ISL 8k/32k、batch 1-32 内有效；超出支持域直接 fail，不做降级或外推。
原语数值应放入 cb_sim 专用的版本化 measured-resource，记录公式、支持域和 Phase462 provenance；loader 做部署/模型/ISL/batch 精确匹配。它不进 PerfDB，也不能作为 scheduler 内的隐藏常数。

DP 边界：三条 CBSimulator API 都实现同一状态机能力，但本相不删除 `dp > 1 and ctx_tokens == isl` 的既有默认路由门。dp2-bt65536 先在 Step 3 用 per-rank 诊断路径重基线，不能因 2c 架构改造而静默切换默认消费路径。

## 明确不改

| 锁定项 | 原因 |
|---|---|
| scheduler admission / preemption trigger / victim selection | 已由 first-divergence 与源码逐拍证实对齐 |
| `max_bt` keying | Phase461 已完成精确匹配，本相无新反证 |
| PerfDB 与已归档 serving-state 行 | 先修 dynamics，再按旧门重审，不提前入库 |
| validate N=512 协议 | 参考口径不变 |
| MULTI_CONFIG gate | 6/6 前保持 1.50，Default AIC 继续 No-Go |
| workload 首调度爬坡 | 已定为 `first_schedule_trajectory_not_modeled` 边界，不把 oracle 时间戳回灌 runtime |

## 事件顺序

| 顺序 | 事件 | 必须保持的语义 |
|---:|---|---|
| 1 | workload 暴露 | 初始 `min(N,C)`；完成一条才释放下一条 |
| 2 | tokenizer 微批完成 | 由已过门 measured primitive 产生 input event |
| 3 | input drain | 在源码规定的调度采样点 drain，不拟合 tokenizer→EngineCore 常数 |
| 4 | 非阻塞预调度 | queue depth 与触发条件照 vLLM 0.19；调度后登记 placeholder |
| 5 | 执行完成 | placeholder 转 computed/sampled，更新 KV 与请求完成事件 |
| 6 | 下一轮 | future、输入和可预调度槽共同决定下一事件；不退回串行 while 近似 |

## 分层红绿

| 层 | 输入 | 通过门 | 失败动作 |
|---|---|---|---|
| oracle 回归锚 | 2a-3d judge-only 时间戳 | drain 100%、首调度步 ≥93.75%、前 16 步 16/16 | 停线，修状态机实现 |
| 短跑稳态 | t=0 暴露 + tokenizer 原语 | 32k 诊断抢占 10±2、稳态自抢占=0、重复 victim=0 | 任一不满足则停在 `--ab` 前 |
| 六点 `--ab` | N=512 统一参考 + 动态 trace | 逐点对预注册符号判卷；bt65536 巨 bucket 墙钟权重向 reference 包络塌缩 | 符号异常先归因，不进入级联 |

Oracle 只检查实现是否复现已证结构，不把历史 `10/2/0` 当成可接受的稳态结果。2c-2 的硬门是稳态自抢占归零；没有 fallback，也不靠阈值、抖动或随机去同步补救。
短跑原型的 giant-bucket `0/0` 只裁决暴露敏感度，不能充当覆盖证据；该信号必须在 2c-3 的 N=512 动态 trace 中重新计算。

## post-2c 预注册

| 场景 | 预测 | 当前判定 |
|---|---|---|
| tp8-8k2k | `worsen` | `pending_dynamic_ab` |
| dp2-8k2k | `improve_until_crossing` | `pending_dynamic_ab` |
| tp8-bt65536 | `worsen` | `pending_dynamic_ab` |
| dp2-bt65536 | `worsen` | `pending_dynamic_ab` |
| dp2-32k3k | `worsen` | `pending_dynamic_ab` |
| tp8-32k3k | `improve_until_crossing` | `pending_dynamic_ab` |

四点预注册 `worsen`、两点 `improve_until_crossing`。短期计分板变差不是事故；tp8-8k2k 可能跌出 15%。只要源码语义与红绿门成立，就不回滚正确语义，移动必须逐点归因。反过来，任何未预注册的符号都先停线调查。

`--ab` 判卷后按固定顺序级联：

1. 用新动态查询包络重审 Phase461 归档的 tp8-bt65536 138 条 mixed 实测行；仍用原 ≥95% 墙钟加权覆盖门、session 可比性门和异常行门，不能自动入库。
2. 重基线 DP Step 3，检查 per-rank 引擎环涌现去同步对 8.54x 跨 rank spread 的解释力；既有 65k 默认路由门在证据出来前保持不变。
3. 重基线 dp2-32k3k；再决定剩余残差，不把前两项的变化提前算进结论。
4. 只有 6/6 ≤15% 后才进入 gate 收紧、收官报告和 fork 交接。

## 文件与测试范围

2c-2 允许修改的生产面仅限 cb_sim 引擎环、显式到达层及其定向测试。scheduler 策略、PerfDB、keying、validate、gate 任一出现 diff 都视为越界并停线。每层测试使用独立 pathspec；先 oracle 单测，再短跑稳态，再全表 `--ab`。

## 当前判决

`status=ready_for_user_review`、`runtime_change_allowed=false`。

边界：`diagnostic_only=true`、`valid_for_default=false`、`perf_database=false`、`Default AIC=No-Go`。
