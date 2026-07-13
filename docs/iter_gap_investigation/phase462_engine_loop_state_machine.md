# Phase462 Step 2a-3d EngineCore 状态机原型

## 结论

本轮只有 target-run replay，没有 prediction-eligible 输入；无论结构签名是否通过，都不能解锁 2c。runtime、PerfDB、validate 与 gate 均不改。

## 部署状态机

| 项 | 运行时值 | 源码规则 |
|---|---:|---|
| pipeline parallel | 1 | serve.log 实际配置 |
| async scheduling | true | serve.log 明确启用 |
| executor | multiproc | 启动路径 |
| batch queue 深度 | 2 | `multiproc_executor.py:469-472` |
| EngineCore 路径 | `step_with_batch_queue` | `core.py:185-211` |

`core.py:421-494` 定义非阻塞预调度：队列未满且最老 future 未完成时立即返回；队列满或没有可调度 token 时才等待最老 future。`core.py:1136-1175` 定义每次 busy-loop 先 drain 全部 input，再进入 engine step。调度后的请求状态立即推进，因此第二个 batch 可在第一个 batch 完成前采样到新状态。

部署态实际使用 `AsyncScheduler`：`async_scheduler.py:15-59` 在完整 prefill 与 decode 调度后都登记 output placeholder；future 返回后才增加已采样 output。原型因此分开记录 sampled output、已进入 KV 的 computed output 和在途 placeholder。另一个此前 16 步审计覆盖不到的规则是 `scheduler.py:563-564`：本步只要发生过抢占，就整步跳过 WAITING admission。

## 判卷口径

| 读数 | 数据 |
|---|---|
| drain、首调度、future 完成边界 | Phase462 N=512 到达观测短跑，取前 128 请求 |
| 前 16 步、抢占目标 `10/0/0` | Phase462 N=128/C=128 抢占观测短跑，脚本从 JSONL 重算 |
| DP 汇合 | Phase461 dp2-bt65536 双 rank 诊断流 |

两个 TP8 短跑只共享部署配置和工作负载形状；arrival run 提供输入/队列时间轴，preemption run 提供动态签名，未把两者当成同一条逐步时间线。当前 tokenizer fit 也使用了目标 32k run 的 batch 起点、成员与样本，因此以下模式全部是 replay，不具备预测资格。

## 四签名判卷

| 模式 | drain 步 | 首调度步 | 前 16 步 | 抢占 / 自抢占 / 重复 victim | 签名门 | 2c 门 |
|---|---:|---:|---:|---:|---|---|
| target_run_fit_replay_plus_cb_cost | 27.34% | 93.75% | 100.00% | 10 / 2 / 0 | FAIL | FAIL |
| target_run_fit_replay_plus_future_oracle | 100.00% | 93.75% | 100.00% | 10 / 2 / 0 | FAIL | FAIL |
| target_run_tokenizer_oracle_plus_future_oracle | 100.00% | 93.75% | 100.00% | 10 / 2 / 0 | FAIL | FAIL |
| target_run_drain_oracle_plus_future_oracle | 100.00% | 93.75% | 100.00% | 10 / 2 / 0 | FAIL | FAIL |

模式说明：

- `target_run_fit_replay_plus_cb_cost`：复用目标 run 的 tokenizer 批边界与成员，单步时长为当前 cb_sim 成本。
- `target_run_fit_replay_plus_future_oracle`：再用同 run 的 queue-depth-shifted scheduler 时间戳恢复 future 完成边界。
- `target_run_tokenizer_oracle_plus_future_oracle`：进一步使用实测 tokenizer 完成时刻。
- `target_run_drain_oracle_plus_future_oracle`：把请求直接放到实测 drain 边界，是结构定位上界。

四种模式的 `prediction_eligible=false`，所以签名门即使通过，2c 门也必须保持 FAIL。

## 剩余首次分歧

| 项 | real | 原型 | 判定 |
|---|---:|---:|---|
| request 27 首调度步 | 1476 | 1475 | 未闭合 |
| 自抢占 | 0 | 2 | 未闭合 |

原型仍有自抢占：step 945 的 trigger/victim 都是 request 13，sampled output=928、computed output=928、在途 placeholder=1。 真实第一次抢占的 trigger/victim 是否相同、computed token 分别为多少，均从 observation 重算：`False`，32928/32927。不能用硬编码消除剩余分歧。

同时，cb-cost replay 的 drain 只有 27.3%；换成 future completion oracle 后仍须按表判卷。当前 cb_sim 首个 32k prefill 为 756.474ms，同 run future 边界为 2539.342ms，成本差会直接改变 busy-loop 的 input drain 采样点。这里的 future 边界是 queue-depth-shifted scheduler 时间戳给出的可见完成上界，不冒充精确 CUDA 执行时长。这是独立阻断，不能由状态机补丁掩盖。

## DP 汇合

结论：`single_rank_state_machine_not_validated`。Phase461 双 rank 数据有每 rank 构成与 busy time，但没有每 rank 的 EngineCore receive、预调度和 drain 时间戳，不能从 8.54x spread 反推出状态机自然生成了相位差。Step 3 暂不继承。

## 边界

| 字段 | 值 |
|---|---|
| diagnostic_only | true |
| valid_for_default | false |
| perf_database | false |
| runtime / validate / gate 改动 | 无 |
| Default AIC | No-Go |

源码快照：`core.py=896730e749cbcabb487ce50c703974594d197fc31a1d3b26fe096197d142d2d5`，`multiproc_executor.py=2011e7d3024f1f230db98b88acc63856ca4d1e74b77f676d8a8a66c4e0dc01ad`。本报告只做离线原型判卷。
