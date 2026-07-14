# Phase462 Step 2c-3 null-block capacity fix

结论：vLLM 日志给出物理 KV block 容量，每个 EngineCore 的 BlockPool 再永久保留 1 个 null block。cb_sim 已把物理块数与 scheduler 可分配块数分开，N128 的 seq=944/945 边界转绿；但短跑稳态仍有 2 次自抢占，硬门失败，六点 `--ab` 未放行。

| 检查 | 结果 | 判定 |
|---|---:|---|
| vLLM `GPU KV cache size` | 由 `kv_cache_config.num_blocks` 直接换算 token | 物理口径 |
| vLLM `BlockPool` | 初始化后从 free queue 取走 1 个 null block | 可分配=`physical-1` |
| DP2 | 每个 DP rank 各自创建 EngineCore/Scheduler/KVCacheManager/BlockPool | 每 replica 各扣 1 |
| N128 seq=944 | trigger=12，peer victim=13 | PASS |
| N128 seq=945 | 无自抢占 | PASS |
| oracle drain/首调度/前16步 | 100%/100%/16:16 | PASS |
| 短跑抢占/重复 victim | 10/0 | PASS |
| 短跑稳态自抢占 | 2（step 2167、10616） | FAIL，目标 0 |
| Phase462 定向测试 | 42 passed | PASS |
| 全量测试 | 1825 passed / 457 failed / 25 skipped | FAIL，当前 worktree 既有 CLI、数据与历史 phase 基线未收口 |

## Source audit

| 层 | 源码位置 | 口径 |
|---|---|---|
| 容量日志 | `vllm/v1/core/kv_cache_utils.py:1298-1319` | 直接读取 `kv_cache_config.num_blocks`，尚未扣 null block |
| null block | `vllm/v1/core/block_pool.py:150-176` | `self.null_block = self.free_block_queue.popleft()` |
| EngineCore 所有权 | `vllm/v1/engine/core.py:126-152,1036-1082` | 每个 DP rank 创建自己的 EngineCore 和 scheduler |
| 块池组装 | `vllm/v1/core/sched/scheduler.py:225`、`kv_cache_coordinator.py:44-55` | 每个 scheduler 创建自己的 BlockPool |

六场景配置继续保存日志对应的物理容量。runtime 的容量检查、full-sequence headroom、admission 与抢占统一消费 `num_allocatable_gpu_blocks`；没有逐场景减常数，也没有新增可调参数。

## Stop line

容量减 1 block 只应修正边界动态，不足以解释稳态状态机的全部差异。当前 t=0 短跑出现 2 次新的稳态自抢占，未满足既定硬门；因此不运行六点 `--ab`，不进入归档行、DP Step 3 或 dp2-32k3k 级联。

边界：`diagnostic_only=true`、`valid_for_default=false`、`perf_database=false`、`Default AIC=No-Go`。
