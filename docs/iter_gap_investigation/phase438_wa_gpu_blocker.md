# Phase438 W-A GPU Blocker

结论：W-A 没有产出可入库数据。稳态 running 门过了，但这台 `worker-9dr4q` 上 vLLM torch profiler 一启动就触发 NCCL watchdog `CUDA error: unspecified launch failure`，EngineCore 死亡，最终 `trace_files=0`。因此没有执行 PerfDB 写入，也没有跑 A/B validate。

| 检查项 | 结果 |
|---|---|
| 场景 | `K2.5-tp4ep8dp2-8k2k-wa-steady` |
| 协议 | ISL 8000 / OSL 2000 / C 128 / prefix off / `max_num_batched_tokens=8000` |
| 环境变量 | 已按要求设置 `VLLM_ENABLE_CUDA_COMPATIBILITY=1`、CUDA/NVIDIA/NCCL/PyTorch `PATH` 与 `LD_LIBRARY_PATH` |
| 稳态门 | 通过，两个 engine 连续 3 次 running 都为 56 |
| profiler 路线 1 | `/stop_profile` 路径失败：`Requested callback is not found`，未落 trace |
| profiler 路线 2 | `start_only` 路径失败：`start_profile` 后 NCCL watchdog 报 `unspecified launch failure`，EngineCore 死亡 |
| nsys 回退 | 不可用，`nsys` 不在 PATH 或常见 CUDA/nsight 路径 |
| 自检 | `trace_files=0`，`mixed_prefill_high_batch_missing` |
| GPU 残留 | 清理后 `nvidia-smi --query-compute-apps` 与目标进程检查均为空 |
| 入库 | 未执行 |
| A/B | 未执行 |

本轮保留的有效证据：

| 文件 | 用途 |
|---|---|
| `phase438_wa_8k_steady/remote_run.log` | runner 控制流、稳态门、trace 文件数 |
| `phase438_wa_8k_steady/K2.5-tp4ep8dp2-8k2k-wa-steady/steady_running_checks.jsonl` | running >=56 的触发证据 |
| `phase438_wa_8k_steady/K2.5-tp4ep8dp2-8k2k-wa-steady/serve_wa_8k2k_steady_mixed_attempt1.log` | profiler 启动后 NCCL watchdog / EngineDeadError |
| `phase438_wa_8k_steady/K2.5-tp4ep8dp2-8k2k-wa-steady/window_checks.jsonl` | trace 自检失败证据 |

下一步不应继续在同一 profiler 路径上重试。可选路径只有两类：

| 路径 | 说明 |
|---|---|
| 换 GPU/镜像重跑 W-A | 需要 torch profiler 能稳定产出 `*.pt.trace.json.gz`，或节点有可用 `nsys` |
| 改采集协议 | 例如使用外部 nsys/CPU+CUDA timeline 工具；这会改变 Phase438 runner 和 extractor 输入格式，需要单独立项 |

Default AIC 维持 No-Go。
