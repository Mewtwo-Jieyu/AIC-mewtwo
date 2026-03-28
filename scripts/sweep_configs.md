# sweep_configs.py

批量评估不同并行配置的延迟和吞吐指标。

与 `aiconfigurator cli default` 的区别：`cli default` 只输出经过 SLA 过滤 + Pareto 筛选的 top-N 最优配置；本脚本输出**全量**评估结果，不做任何过滤。

## 用法

```bash
python scripts/sweep_configs.py \
  --model-path moonshotai/Kimi-K2.5 \
  --total-gpus 16 \
  --system h200_sxm \
  --backend vllm \
  --database-mode HYBRID \
  --isl 8192 --osl 2048 \
  --output sweep_results.csv
```

自定义搜索空间（默认 tp_list 不含 16，需手动指定）：

```bash
python scripts/sweep_configs.py \
  --model-path moonshotai/Kimi-K2.5 \
  --total-gpus 16 \
  --system h200_sxm \
  --backend vllm \
  --database-mode HYBRID \
  --isl 8192 --osl 2048 \
  --tp 1 2 4 8 16 \
  --output sweep_results.csv
```

## 参数

### 必填

| 参数 | 说明 |
|------|------|
| `--model-path` | HuggingFace 模型路径或本地路径 |
| `--total-gpus` | 可用 GPU 总数 |
| `--system` | 硬件系统名称（如 `h200_sxm`、`gb200`） |

### 可选

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--backend` | `trtllm` | 推理后端（`trtllm`、`vllm`、`sglang`） |
| `--database-mode` | `SILICON` | 性能数据库模式（`SILICON`、`HYBRID`、`EMPIRICAL`、`SOL`） |
| `--isl` | 4000 | 输入序列长度 |
| `--osl` | 1000 | 输出序列长度 |
| `--output`, `-o` | 无 | 输出 CSV 文件路径 |
| `--sort-by` | `tokens/s/gpu` | 排序列名 |

### 搜索空间覆盖

不指定时使用 SDK 内置的默认搜索空间。指定后会覆盖对应维度的搜索列表。

| 参数 | 示例 | 说明 |
|------|------|------|
| `--tp` | `--tp 1 2 4 8 16` | Tensor Parallelism 候选值 |
| `--pp` | `--pp 1 2` | Pipeline Parallelism 候选值 |
| `--dp` | `--dp 1 2 4 8` | Data Parallelism 候选值 |
| `--moe-tp` | `--moe-tp 1` | MoE Tensor Parallelism 候选值 |
| `--moe-ep` | `--moe-ep 1 2 4 8 16 32` | MoE Expert Parallelism 候选值 |

## 输出列

| 列名 | 含义 |
|------|------|
| `tp` / `pp` / `dp` / `moe_tp` / `moe_ep` | 并行配置参数 |
| `GPUs` | 该配置实际使用的 GPU 数量 |
| `bs` | Batch size |
| `concurrency` | 并发请求数 |
| `request_rate` | 请求速率 (req/s) |
| `memory` | 显存占用 (GiB) |
| `TTFT_ms` | Time To First Token，首 token 延迟 (ms) |
| `TPOT_ms` | Time Per Output Token，每 token 延迟 (ms) |
| `req_latency_ms` | 端到端请求延迟 (ms) |
| `tokens/s` | 集群总吞吐 |
| `tokens/s/gpu` | 单卡吞吐 |
| `tokens/s/user` | 单用户视角吞吐（请求级） |

## 工作原理

1. 调用 `build_default_task_configs()` 构建 TaskConfig（与 `aiconfigurator cli default` 相同的逻辑）
2. 如果用户指定了 `--tp` 等参数，覆盖 TaskConfig 中的搜索空间列表
3. 调用 `TaskRunner.run_agg()` 执行全量评估（内部通过 `enumerate_parallel_config()` 枚举所有满足约束的合法配置）
4. 直接输出原始 `pareto_df`，不做 Pareto 过滤或 SLA 筛选

OOM 的配置不会出现在结果中（被 SDK 内部静默跳过）。
