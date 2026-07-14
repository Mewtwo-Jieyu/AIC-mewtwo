# Phase462 Step 2c-2c queue-order first divergence

结论：剩余自抢占不是 running 顺序语义错误，也不是 sampled/computed off-by-one。首个语义差是 KV 容量多算 1 block：real 可分配 `28,824`，sim 使用 `28,825`。这使 real 在 `schedule_seq=944` 先抢占队尾 13，sim 到下一步才由队尾 13 触发自抢占。Step 2c-2 继续停线，容量修复须单独确认。

| 门 | 结果 | 判定 |
|---|---:|---|
| 协议 | N128/C128/ISL32k/OSL1200/max_bt32000 | 对齐 |
| logging 开销 | 0.2114% | PASS，门值 <=2% |
| request ID | 128/128 | PASS |
| schedule_seq | 1..12032，无空洞 | PASS |
| 抢占关联 | 10/10 | PASS |
| 源码恢复 | 5/5 SHA256 一致 | PASS |
| 残留进程/GPU | 0/0 | PASS |

## First divergence

| 边界 | real | sim | 判定 |
|---|---|---|---|
| `schedule_seq=1` | 空缓存 free blocks=`28824` | free blocks=`28825` | 首个语义差：`kv_capacity_divergence` |
| `schedule_seq=944` 入参 | running=`[0..13]`，waiting=`[14..127]`，14 个请求 phase 全相同，free=`0` | 队列与 phase 完全相同，free=`1` | 只剩容量不同 |
| `schedule_seq=944` 输出 | scheduled running=`[0..12]`，preempted=`[13]` | scheduled running=`[0..13]`，preempted=`[]` | 首个行为差 |
| `schedule_seq=945` | 已把 13 放回 waiting | 13 仍在 running 并自抢占 | queue/phase 分叉是下游结果 |

全程 oracle 诊断对照为 real tokenizer 暴露 + real future 完成边界；只用于消掉 workload ramp 与执行时钟差，不进入交付路径。对齐了 12031 个非空 schedule；real 末尾多一个全空 quiescence schedule。sim 总签名为 `10 preemptions / 10 unique / 0 repeat / 2 self`，real 为 `10 / 10 / 0 / 0`。

## Causal chain

| 层 | 源码/配置 | 事实 |
|---|---|---|
| real 日志 | Phase458 serve log line 195；本次 capture line 200 | GPU KV cache size=`461,200` tokens，即物理块总数 `461200/16=28825` |
| vLLM 0.19 | `vllm/v1/core/block_pool.py:162-176,479-498`，SHA256 `2ea932628ac6a06607dd48ded7a908f705b55c3eb6801232264ee2db142c6cd9` | 初始化时从 free queue 永久取走 1 个 null block；可分配块=`num_gpu_blocks-1` |
| sim 配置 | `scripts/validate_cb_simulator.py:299-304,377-383` | 将物理块总数 `28825` 直接作为 scheduler 可分配容量 |
| sim scheduler | `src/aiconfigurator/sdk/backends/cb_simulator/scheduler.py:105-113` | 只在使用量 `> num_gpu_blocks` 时抢占，因此比 real 晚 1 block |

根因是“物理总块数”和“scheduler 可分配块数”混成同一字段。不能只把单场景常数改成 `28824`；正确修复应把 vLLM null-block 保留语义结构化进容量组装，并重审六场景。

## Preregistered suspects

| 嫌疑 | 核销 | 依据 |
|---|---|---|
| 被抢占请求回 waiting 的位置 | 排除 | vLLM `waiting.prepend_request` 与 sim `waiting.insert(0, victim)` 同为队首；首行为差发生前尚无抢占 |
| resume 与 new admit 的 running 相对顺序 | 排除 | 两侧按 waiting FCFS 取出后 append running；running/waiting 逐 ID 对齐至 seq944 |
| tokenizer 微批内 admit 顺序 | 排除 | `arrival_ordinal` 按原批内位置分配；128/128 映射完整，seq944 前可见集合与顺序一致 |
| sampled/computed 完成边界 off-by-one | 排除为首因 | seq944 的 14 个 per-request computed/placeholder/block_count 全相同；phase 首次分叉在 seq945，晚于容量与输出分叉 |

## Boundary

本步只增加临时 logging 挂钩、离线 analyzer、定向测试和诊断报告。未改 sim 容量/runtime、PerfDB、gate；未运行六点 `--ab`。80 MiB 完整 engine trace 与 3.2 MiB serve log 不入 git，原始文件保留在本地和 GPU 端，`raw_capture.sha256` 固化校验值；提交保留完整性门与 `first_divergence.json`。`tools.sha256` 记录 GPU 采集时工具，`analysis_tools.sha256` 记录最终离线判卷工具。`diagnostic_only=true`、`valid_for_default=false`、`perf_database=false`、`Default AIC=No-Go`。
