# CB Simulator 后续工作 TODO

> 截止 2026-04-24，Phase 4 完成后的遗留项

## 高优先级

### TODO-1: EP All2All 通信物理建模

**背景**: 当前 EP8 场景使用 90ms 常数 overhead，本质上是从在线 profiling 逆推的经验值，不是真正的物理建模。

在线 profiling 显示 prefill/mixed 迭代缺口达 500–700ms（cb_sim 偏快 2–3x），主要来源推断为 MoE Expert Parallel All2All 通信——每层 forward pass 需要 2 次 All2All（dispatch + combine），在 H200 高带宽互连上仍有显著延迟。

TRT-LLM 侧已有完整的 All2All 建模（`TrtLLMWideEPMoEDispatch`）：
- 支持 inter-node（RDMA）和 intra-node（NVLink/PCIe）
- 以 `wideep_alltoall_perf.txt` benchmark 数据作为 perf database 输入
- 通过 `bytes_per_token * num_tokens + overhead` 的线性模型预测延迟

**行动项**:
1. 在 H200 集群上运行 `collector/slurm_comm_collector/collect_trtllm_alltoall.py`，采集 H200 NVLink All2All benchmark 数据（不同 message size × expert count × TP 组合）
2. 将数据导入 vLLM perf database（新增 `alltoall_perf.txt` 或扩展现有格式）
3. 在 `iteration_latency.py` 中增加 EP All2All 通信项（按 tokens/experts/layers 计算）
4. 删除 `_CB_SIM_8GPU_EP_DECODE_OVERHEAD_MS = 90.0` 常数

**验证方法**: 用 `diagnose_cb_iter_latency.py` 对比 ep8 场景 per-iteration latency，prefill/mixed 缺口应从 500ms 降到 ≤100ms。

---

### TODO-2: 前缀缓存（Prefix Caching）建模

**背景**: vLLM v1 默认启用 prefix caching（`enable_prefix_caching=True`）。共享前缀的请求批次中，KV cache 命中的 token 不需要重新计算 attention，节省大量计算。

当前 cb_sim 完全忽略前缀缓存效果，对高命中率场景（如 system prompt 复用）存在系统性低估吞吐的风险。

**行动项**:
1. 在 `CBSimConfig` 增加 `prefix_cache_hit_rate: float = 0.0`
2. 在 `IterationLatencyCalculator.compute()` 中，按命中率折减 context attention ops（命中部分直接从 KV cache 读，延迟按 memory bandwidth 计算而不是 compute）
3. 更新 `validate_cb_simulator.py`，增加有/无前缀缓存的场景对比

---

## 中优先级

### TODO-3: Dense 模型验证（Llama-3）

**背景**: 所有验证数据均来自 Kimi-K2.5（MoE, 36B activated/236B total）。cb_sim 的 `run_static` 对 dense 模型和 MoE 模型使用不同的 kernel latency 数据库，但 overlap_factor/overhead 参数是否在 dense 模型上同样适用尚未验证。

**行动项**:
1. 收集 Llama-3-70B / Llama-3.1-405B 在 H100/H200 上的实测 throughput + TTFT 数据
2. 运行 `validate_cb_simulator.py --model llama3`
3. 如精度不达标，检查是否需要 dense-specific 参数调整

---

### TODO-4: `long_prefill_token_threshold` 和 `preemption` 参数准确性

**背景**: Phase 2 实现了这两个调度机制，但参数来自 vLLM 源码的静态分析，未对照在线服务实际配置验证：

- `long_prefill_token_threshold`：在线服务是否使用了与 `max_model_len * 0.04` 不同的值？
- `preemption`：在线服务实际驱逐频率有多高？cb_sim 是否准确再现了驱逐模式？

**行动项**:
1. 对照在线服务的 vLLM 启动参数确认 `max_num_partial_prefills` 设置
2. 从 `diagnose_cb_iter_latency.py` 输出中统计实际 preemption 事件频率，与 cb_sim 模拟对比

---

## 低优先级

### TODO-5: `overlap_factor` 跨配置/模型泛化

当前 overlap_factor=0.0 基于 Kimi-K2.5 + H200 × 16 的 profiling 数据确定。理论上这个参数和：
- GPU 互连拓扑（NVLink vs PCIe）
- TP 规模（all-reduce 延迟占比）
- MoE vs dense（dispatch overhead 比例）

都有关系。未来若要支持新硬件（如 GB200、AMD MI300X），需要重新 profile 确认最优 α 值。

### TODO-6: CB Simulator 压力测试 + 边界条件

当前单元测试覆盖了主要路径，但以下边界条件覆盖不足：
- 极大 batch（bs=512+）
- 混合 ISL 分布（部分 long context + 部分 short context 在同一批次）
- KV cache 耗尽频繁驱逐场景
- 多轮对话（每轮 prefix 快速增长）
