# Kimi-K2.5 vllm + h200_sxm 支持记录

**日期**: 2026-03-18
**分支**: pr-403-kimi-k2.5
**目标**: 让 `moonshotai/Kimi-K2.5` 能在 vllm + h200_sxm 上运行 aiconfigurator

---

## 问题 1: `--enable-wideep` CLI 参数不识别

**现象**:

```
aiconfigurator: error: unrecognized arguments: --enable-wideep
```

**原因**: 源码中已添加 `--enable-wideep` 参数（commit `5cca3e7`），但安装的包版本过旧，未包含该参数。

**解决**: `pip install -e .` 重新安装即可。

---

## 问题 2: h200_sxm + trtllm 缺少 wideep 性能数据

**现象**:

```
PerfDataNotAvailableError: WideEP All2All perf table is missing for system='h200_sxm', backend='trtllm', version='1.2.0rc5'
```

**原因**: wideep 数据文件只存在于以下系统/后端组合：


| 系统       | trtllm                 | sglang                           | vllm |
| -------- | ---------------------- | -------------------------------- | ---- |
| gb200    | 有 (1.2.0rc5, 1.2.0rc6) | 有 (0.5.8.post1)                  | 无    |
| h200_sxm | **无**                  | **有** (0.5.6.post2)              | 无    |
| h100_sxm | **无**                  | **有** (0.5.6.post2, 0.5.8.post1) | 无    |


**附带发现**: `perf_database.py:5368` 的 `_wideep_alltoall_data is None` 检查在 HYBRID fallback 的 try/except 之前触发，导致 `--database-mode HYBRID` 的经验值 fallback 没有生效。这是一个 bug，但本次未修复。

**解决**: 用户决定放弃 wideep，改用 vllm 后端。

---

## 问题 3: vllm 后端不支持 DEEPSEEK 模型族

**现象**:

```
NotImplementedError: AIConfigurator does not yet support DEEPSEEK models for VLLM backend.
```

**原因**: `task.py:717-718` 有一个显式的验证检查，阻止所有 DEEPSEEK 模型在 vllm 上运行。

**分析**: 经调查，vllm 后端实际上已有 DeepSeek 所需的：

- `MoEDispatch` vllm 代码路径 (`operations.py:690-707`)
- MLA 性能数据 (`h200_sxm/vllm/0.12.0/context_mla_perf.txt`, `generation_mla_perf.txt`)
- MoE 性能数据 (`h200_sxm/vllm/0.12.0/moe_perf.txt`)
- AllReduce 数据 (`custom_allreduce_perf.txt`)

**修改**: 注释掉验证检查。

```python
# task.py:717-718
# if self.backend_name == "vllm" and get_model_family(self.model_path) == "DEEPSEEK":
#     raise NotImplementedError(...)
```

---

## 问题 4: vllm 生成的并行配置无效 (moe_tp > 1 AND moe_ep > 1)

**现象**:

```
AssertionError: vllm does not support MoE TP and MoE EP at the same time
```

**原因**: 三个问题叠加：

1. `task.py` 中 `build_disagg_parallel_lists()` 和 `_agg_defaults_layer()` 为 vllm 设置了 `moe_tp_list=[1,2,4,8]` 和 `moe_ep_list=[1,2,4,8]`，允许两者同时 > 1
2. `utils.py:156-157` 的 vllm 枚举过滤器是 `pass # TODO`，没有过滤无效配置
3. vllm 的 `MoEDispatch.query()` 断言 `moe_tp == 1 OR moe_ep == 1`

**参考**: sglang 已正确处理 — wideep 时 `moe_tp=[1]`，非 wideep 时 `moe_ep=[1]`。

**修改**:

### 文件 1: `src/aiconfigurator/sdk/task.py` — `build_disagg_parallel_lists()` (约 line 214)

```python
# 修改前:
prefill_worker_config["moe_tp_list"] = parallel_config_list  # [1,2,4,8]
prefill_worker_config["moe_ep_list"] = parallel_config_list
# 同样 decode_worker_config

# 修改后:
prefill_worker_config["moe_tp_list"] = [1]
prefill_worker_config["moe_ep_list"] = [1, 2, 4, 8, 16, 32]  # 扩大搜索空间
# decode 同理，moe_ep 最大到 64
```

### 文件 2: `src/aiconfigurator/sdk/task.py` — `_agg_defaults_layer()` (约 line 379)

```python
# 修改前:
worker_config["moe_tp_list"] = [1, 2, 4, 8]
worker_config["moe_ep_list"] = [1, 2, 4, 8]

# 修改后:
worker_config["moe_tp_list"] = [1]
worker_config["moe_ep_list"] = [1, 2, 4, 8, 16, 32, 64]
```

### 文件 3: `src/aiconfigurator/sdk/utils.py` — `enumerate_parallel_config()` (约 line 156)

```python
# 修改前:
elif backend == common.BackendName.vllm:
    pass  # TODO

# 修改后:
elif backend == common.BackendName.vllm:
    if moe_tp > 1 and moe_ep > 1:
        continue
```

---

## 问题 5: 搜索空间太小导致内存不足

**现象**:

```
RuntimeError: No results found: the model does not fit in GPU memory for any parallel configuration.
```

**原因**: 原始 `moe_ep_list = [1,2,4,8]` 对 384 experts 的模型太小。EP=8 时每 GPU 48 experts，H200 的 141GB HBM 放不下。

**修改**: 在问题 4 的修改中一并解决 — 将 vllm 的搜索空间扩展到：

- `num_gpu_per_worker`: 最大 64
- `moe_ep_list`: 最大 64
- `dp_list`: 最大 64
- 模式参照 sglang+wideep 的搜索空间

---

## 问题 6: TTFT/TPOT 延迟约束过紧

**现象**:

```
No configuration satisfied the TTFT/TPOT or request-latency constraints.
```

**原因**: 默认 TTFT=2000ms, TPOT=30ms。HYBRID 模式下 vllm + h200_sxm 的经验估算延迟超过了默认约束。这不是代码 bug，是缺少真实 benchmark 数据导致估算偏保守。

**临时解决**: 放宽约束参数：

```bash
aiconfigurator cli default \
  --model-path moonshotai/Kimi-K2.5 \
  --total-gpus 128 \
  --system h200_sxm \
  --backend vllm \
  --database-mode HYBRID \
  --ttft 10000 --tpot 100
```

---

## 重要风险: HYBRID 模式估算误差

当前 vllm + h200_sxm + DeepSeek/Kimi-K2.5 的组合**完全没有真实 benchmark 数据**（SILICON 数据），所有性能估算都依赖 HYBRID 模式的经验公式。这会带来以下误差风险：

### 受影响的操作及误差来源

| 操作 | 数据来源 | 误差风险 |
| --- | --- | --- |
| MoE compute (`moe_perf.txt`) | 有真实数据 | 低 — h200_sxm/vllm/0.12.0 下存在 |
| MLA context/generation (`context_mla_perf.txt`, `generation_mla_perf.txt`) | 有真实数据 | 低 — h200_sxm/vllm/0.12.0 下存在 |
| GEMM (`gemm_perf.txt`) | 有真实数据 | 低 |
| MoE AlltoAll / dispatch 通信 | HYBRID 经验值 | **高** — 通信延迟随拓扑、EP size 变化大，经验公式可能严重偏离 |
| Custom AllReduce | 有真实数据 (`custom_allreduce_perf.txt`) | 低 — 但仅覆盖部分 GPU 数量 |
| NCCL 集合通信 | 有真实数据 | 中 — 数据可能不覆盖大 EP (16/32/64) 场景 |
| 内存估算 | 委托给 TRTLLMBackend 计算 | **中** — vllm 的实际内存开销模式可能与 trtllm 不同 |

### 具体影响

1. **TTFT/TPOT 估算偏差**: 这是问题 6 的根因。HYBRID 经验值可能系统性地高估或低估延迟，导致：
   - 高估时：合理的配置被排除（当前遇到的情况），需要放宽约束才能出结果
   - 低估时：推荐的配置在真实环境下可能不满足 SLA

2. **Pareto 排序失真**: 即使放宽约束出了结果，top-N 配置的排序可能不反映真实性能，导致推荐的"最优"配置并非实际最优

3. **大 EP 场景缺乏验证**: 扩展搜索空间到 EP=32/64 后，NCCL alltoall/all_gather 在这些规模下的通信延迟缺少真实数据点，插值/外推误差更大

### 建议

- 当前的 HYBRID 结果仅供**参考和初筛**，不应直接用于生产部署决策
- 获取 vllm + h200_sxm 上 DeepSeek/Kimi-K2.5 的真实 benchmark 数据后，使用 `--database-mode SILICON` 重新评估
- 特别需要补充：大 EP (16/32/64) 场景下的 NCCL alltoall 通信数据

---

## 修改文件汇总


| 文件                                                               | 修改内容                                                    |
| ---------------------------------------------------------------- | ------------------------------------------------------- |
| `src/aiconfigurator/sdk/task.py:717-718`                         | 注释掉 DEEPSEEK + vllm 的 NotImplementedError 验证            |
| `src/aiconfigurator/sdk/task.py` `build_disagg_parallel_lists()` | vllm: `moe_tp_list=[1]`, 扩展 `moe_ep/dp/num_gpu` 到 32/64 |
| `src/aiconfigurator/sdk/task.py` `_agg_defaults_layer()`         | vllm: 同上，扩展搜索空间                                         |
| `src/aiconfigurator/sdk/utils.py` `enumerate_parallel_config()`  | 实现 vllm 过滤器：`moe_tp > 1 and moe_ep > 1` 时 continue      |


## 未解决的已知问题

1. **HYBRID fallback bug**: `perf_database.py:5368` 的 None 检查在 HYBRID try/except 之前触发，导致 fallback 无效
2. **vllm 无 wideep 支持**: 代码和数据都不存在，如需支持需要大量开发
3. **HYBRID 估算精度**: 缺少 vllm + h200_sxm 的真实 DeepSeek benchmark 数据，HYBRID 经验值可能不准确

