# Phase462 Step2a-2: 状态演化源码审计

结论：计划中的“多 prefill 同步完成 -> 同相跨块 -> 自抢占”不是首次分歧的起点。部署态 vLLM 与 cb_sim 的 prefill 上限、running/waiting 顺序和 16-token block 申请语义一致。唯一能完整解释 Step2b 签名的差异是：cb_sim 在 t=0 把全部并发请求放入 waiting；真实 EngineCore 的下一请求在 Iteration(1) 尚不可见。删除这一个真实空拍后，前 16 步调度形状逐步 100% 对齐。

| 审计项 | real | sim | 判定 |
|---|---|---|---|
| prefill 打包 | threshold=0，首窗 max context req=2 | threshold=0，max=2 | 对齐，不是根因 |
| 完成节奏 | max decode 增量=1，Iteration(1) 有空拍 | max decode 增量=1，无空拍 | 空拍后 100% 对齐 |
| block 边界 | trigger-tail=1 token | trigger-tail=0 token | 申请算法对齐，相位不同 |

## 源码落点

| 位置 | 语义 |
|---|---|
| vLLM `config/scheduler.py:69-79,244-247` | 默认 `max_num_partial_prefills=1`，因此 `long_prefill_token_threshold` 保持 0；原计划假设的 4% cap 未启用 |
| vLLM `v1/core/sched/scheduler.py:384-479,574-757` | running 先消费预算，waiting 再消费剩余预算，与 cb_sim 顺序一致 |
| vLLM `v1/core/kv_cache_manager.py:361-397` | 按本步 `num_new_tokens` 计算新 block；Step2b 已证明两侧均在缺 1 block 时触发 |
| vLLM `v1/engine/async_llm.py:354-373,407-418` | 每个请求先 `process_inputs`，随后才 `add_request_async` 到 EngineCore |
| cb_sim `simulator.py:152-157` | 一次性将全部 concurrency 以 `arrival_time_ms=0.0` 填入 waiting |

部署源码 SHA256：`config/scheduler.py`=011cb6cda6625e9f05fc0a6479061e94ec5a449d984685f19985243e91c45a2d; `v1/engine/async_llm.py`=b101a3cd5ea4b7d0cf1852824d8fc96ec8c6a83c0396aad54ad9c1c8d69442c3; `v1/core/sched/scheduler.py`=9f3da2dfce94963e1e1cefc156e4239120f79c6eaf824b2829cac0ee7736b58a; `v1/core/kv_cache_manager.py`=0df69b7626195cbaabb33013d1e05ba6660243369171367f87313e4221bfffc8.

## 因果判定

真实第 0 步和 sim 相同；真实第 1 步只有 decode、没有 context。此时 KV 仍可容纳下一条完整 32k prompt，且 running=1 远低于 max_num_seqs，所以该空拍只能由下一请求尚未进入 EngineCore waiting 解释。把这一行从真实轨迹移除后，其后 16 步与 sim 完全一致；第一次抢占处因此表现为 real trigger 比 tail 领先 1 token，而 sim trigger 与 tail 同相并自抢占。

因此，recompute 巨 bucket 是抢占环路的放大器，不是首次分歧的来源。Step2c 不能修改 prefill cap、block 阈值或加入随机去同步。可执行修复仍被锁住：需要先把 EngineCore 可见到达建成无自由参数的输入原语，或正式声明 ramp/到达动力学不在模型范围。

## 六点符号预测

| 场景 | 修掉额外抢占后的单变量方向 | 风险 |
|---|---|---|
| K2.5-tp8ep8-8k2k | worsen | current_pass_at_risk |
| K2.5-tp4ep8dp2-8k2k | improve_until_crossing | current_pass_not_immediately_at_risk |
| K2.5-tp8ep8-8k2k-bt65536 | worsen | current_fail_worsens |
| K2.5-tp4ep8dp2-8k2k-bt65536 | worsen | current_fail_worsens |
| K2.5-tp4ep8dp2-32k3k | worsen | current_fail_worsens |
| K2.5-tp8ep8-32k3k | improve_until_crossing | current_pass_not_immediately_at_risk |

静态符号只用于证伪，不用于预测终值。尤其 tp8-8k2k 只有 1.543% sim 吞吐上升空间，语义修复可能把它推出 15%。

本步骤 `report-only / diagnostic_only=true / valid_for_default=false / perf_database=false`。未改 runtime、PerfDB 或 gate；Default AIC 维持 No-Go。
