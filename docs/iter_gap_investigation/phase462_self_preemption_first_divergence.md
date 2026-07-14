# Phase462 Step 2c-2b self-preemption first divergence

结论：`step 8176` 的自抢占在 vLLM FCFS 源码中是合法分支，但同协议 real 短跑没有出现等价的 `trigger=running tail` 状态。证据不允许放宽硬门；Step2c-2 继续停线，下一分歧是 sim/real 的 running 队列顺序演化。

| 检查 | 结果 | 判定 |
|---|---:|---|
| sim 分配缺口 | 1 block | 与 real 的缺 1 block 对齐 |
| sim trigger 位置 | 13/13 | trigger 就是 running 队尾 |
| sim trigger 状态 | sampled=736, computed=736, placeholder=1 | 完整状态已冻结 |
| vLLM 源码 | `source_legal` | FCFS `running.pop()` 后显式处理 `preempted_req == request` |
| real 2b 决策 | 10 次 / 10 个唯一 victim | 全部 trigger != tail |
| real 同宏观状态 | waiting=31, running=14, 缺 1 block | trigger != tail，victim 是 peer |
| real waiting 覆盖 | 114 -> 3 | 已贯穿该 N=128 短跑的抢占发生区间 |
| Phase458 N=512 | trigger/victim relation observable=False | 只有聚合计数和 iteration 构成，不能补关系证据 |

源码依据：部署态 vLLM `scheduler.py:460-510` 在 block 分配失败后，FCFS 路径执行 `running.pop()`；若 pop 出的队尾就是当前 request，`scheduler.py:504-506` 明确停止继续抢占。`kv_cache_manager.py:360-389` 定义分配失败条件。本次核对源码 SHA256 为 `9f3da2dfce94963e1e1cefc156e4239120f79c6eaf824b2829cac0ee7736b58a` / `0df69b7626195cbaabb33013d1e05ba6660243369171367f87313e4221bfffc8`。

决策叉结果：`keep_gate_and_investigate_state_evolution`。源码只证明“给定该状态时自抢占合法”，不证明 sim 产生该状态的演化正确。更关键的是，real 在同样的 waiting=31、running=14、缺 1 block 状态下选择了 peer tail victim；首次分歧因此钉在 running 队列顺序或其上游 per-request phase 演化，而不是容量阈值。Phase458 又缺 request-id 关系，因此不能按 sim 的 1 次事件裁剪验收门。

下一步只提议、不执行 GPU：复用 2b logging-only 挂钩，在同一 N=128/C=128/ISL=32k/OSL=1200 协议下增加每步 running 顺序、trigger 位置、tail id、computed/placeholder/block 状态。现有日志时长已覆盖 waiting `114 -> 3`，缺的是队列维度，不是运行时长；无需盲目延长。

未改 runtime、PerfDB 或 gate；未运行六点 `--ab`。边界：`diagnostic_only=true`、`valid_for_default=false`、`perf_database=false`、`Default AIC=No-Go`。
