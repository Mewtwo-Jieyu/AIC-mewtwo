# Phase462 Step2a-3c: 到达原语与 admission 原型

结论：**Step2c 继续锁住。** bt65536 的 real 与 sim 都在单步调度 9 个 context request，所谓“每步最多 1-2 个新 context”只适用于 `max_bt == ISL`，不能被扩成 admission 上限。tokenizer 微批服务函数能从 780 个 batch / 1024 个 request pair 测得，但 `tokenizer complete -> EngineCore receive` 受在途模型执行和 input-queue drain 支配，不能由 `(total_prompt_tokens, batch_size)` 单独决定。更严格的反事实直接重放实测 tokenizer-complete 时刻仍未复现三签名，因此到达原语单独入 runtime 没有因果闭环。

| 决策叉 | real | sim | 判定 |
|---|---:|---:|---|
| bt65536 单步 context/fresh admission | 9 | 9 | 对齐；禁止修改 admission cap |
| 单步 context tokens | 65535 | 65535 | 对齐 |

real 取 Phase458 `Iteration(2)`：此前只有 request 0 完成 prefill，尚无可抢占 cohort，因此该步 9 个 context request 都属于首次 admission。sim 用同状态重放，并只统计 `WAITING && num_preemptions == 0`，排除了 recompute。

## 实测原语

`tokenizer_ms = 0.029270374 + 0.001003614837 * total_prompt_tokens + (-0.604326979) * batch_size`

| 指标 | 结果 | 门 |
|---|---:|---:|
| batch / request pair | 780 / 1024 | 完整 |
| tokenizer weighted MAPE | 5.915% | <=10%，通过 |
| tokenizer complete -> EngineCore receive 拟合 MAPE | 466.667% | <=10%，失败 |

负的 batch 系数是观测支持域内的批处理摊销结果；该式只允许用于 ISL 8k/32k、batch 1-32，不允许外推。后半段延迟不是 add/IPC 常数：EngineCore 只在 step 边界 drain input queue，时间戳因此混入当前 model execution 的剩余时间。

## 三签名原型

原型没有用拟合值硬凑，而是直接喂实测 tokenizer batch complete 时刻，再由 sim 自己在迭代边界接收。这个 oracle 比预测原语更强；oracle 不过，拟合版本不可能解锁 runtime。逐请求可见步只比较首个 14-request cohort，避开 arrival run OSL=3k 与抢占短跑 OSL=1.2k 的协议差。

| 签名 | 结果 | 目标 |
|---|---:|---:|
| 逐请求首次调度步吻合 | 7.143% (14 pairs) | 100% |
| 前 16 步形状吻合 | 6.250% | 100% |
| preemption | 26 | 10±2 |
| self-preemption | 5 | 0 |
| repeat victim event | 4 | 0 |

首次分歧进一步落到**事件循环时钟边界**：部署态 `EngineCore` 在 `batch_queue_size > 1` 时走 `step_with_batch_queue`。它先 `schedule -> execute_model(non_block=True)`，future 未完成且队列未满就立即返回，再预调度下一批；队列满后才等待最老 future。真实 Iteration(0) 执行期间，Iteration(1) 已被提前排成空拍，之后 tokenizer 完成的请求到下一次 input drain 才可见。当前 cb_sim 只有“schedule -> 等完整 iteration latency -> 接收 -> 再 schedule”的单时钟循环，因此一到第二步就看见新请求，抢占仍为原来的同步相位签名。

| 部署源码 | 已确认语义 |
|---|---|
| vLLM 0.19.0 `v1/engine/core.py:185-211` | `max_concurrent_batches > 1` 时启用 batch queue，并选择 `step_with_batch_queue` |
| `core.py:421-494` | 队列未满时非阻塞 schedule/execute 下一批；队列满后才等待最老 future |
| `core.py:1136-1175` | 每次 engine step 前 drain input queue |
| cb_sim `simulator.py:163-198` | schedule 后直接累计完整 iteration latency，再进入下一轮 |

部署 `core.py` SHA256：`896730e749cbcabb487ce50c703974594d197fc31a1d3b26fe096197d142d2d5`。

下一步不应进入 Step2c。先审计/原型化 vLLM `execute_model(..., non_block=True)` 下的 in-flight schedule/input-drain 状态机，验证它能否在不改 scheduler 语义的前提下复现三签名；若不能，正式把 ramp 输入流水线划出稳态模型边界。

本步 `report-only / diagnostic_only=true / valid_for_default=false / perf_database=false`；未改 runtime、PerfDB 或 validation gate，Default AIC 维持 No-Go。
