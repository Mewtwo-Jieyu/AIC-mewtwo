# Phase462 Step2a-3b: EngineCore 可见到达观测

结论：**观测门通过，Step2a-3c 可开始。** 本步骤只采时间边界与调度可见性，不拟合原语，不改 runtime、PerfDB 或 gate。

| 门 | 8k2k | 32k3k |
|---|---:|---:|
| benchmark 完整性 | 512/512 | 512/512 |
| logging 开销 | 0.138% | 0.052% |
| 四点同 ID 配对 | 512/512 | 512/512 |
| 运行时 tokenizer 配置 | 32 requests / 2ms | 32 requests / 2ms |
| 源码恢复 / GPU / 进程残留 | pass | pass |

## 四段时间

| 场景 | tokenizer 批数 | queue 等待 median/p90 | tokenizer 服务 median/p90 | batch complete -> EngineCore median/p90 | EngineCore -> first schedule median/p90 |
|---|---:|---:|---:|---:|---:|
| 32k3k | 390 | 2.104/2028.779ms | 30.247/33.019ms | 2406.053/2423.602ms | 503644.803/531589.165ms |
| 8k2k | 390 | 2.097/481.529ms | 8.182/8.761ms | 595.800/600.749ms | 78224.151/84799.196ms |

`EngineCore -> first schedule` 包含闭集并发下的 waiting 排队，只是观测边界，不是 tokenizer 原语。

## 微批与调度

| 场景 | batch size | 批数 | total prompt tokens | tokenizer 服务 median/p90 | 首次调度步数 median/max |
|---|---:|---:|---:|---:|---:|
| 32k3k | 1 | 385 | 32000 | 30.238/32.873ms | 1/1 |
| 32k3k | 2 | 1 | 64000 | 58.026/58.026ms | 2/2 |
| 32k3k | 29 | 1 | 928000 | 943.132/943.132ms | 29/29 |
| 32k3k | 32 | 3 | 1024000 | 1027.194/1031.050ms | 32/32 |
| 8k2k | 1 | 386 | 8000 | 8.178/8.754ms | 1.000/1 |
| 8k2k | 30 | 1 | 240000 | 225.902/225.902ms | 30/30 |
| 8k2k | 32 | 3 | 256000 | 236.671/238.885ms | 32/32 |

两个场景中 `new_context_count_max=1`。初始 29-32 请求 tokenizer 大微批在 EngineCore 可见后，被摊到同数量级的首次调度步；因此微批形成与 admission 错开是两个独立阶段。该读数只解锁 Step2a-3c 的离线原型，不能直接写成“一拍”或其他常数。

本步骤 `diagnostic_only=true / valid_for_default=false / perf_database=false`。Default AIC 维持 No-Go。
