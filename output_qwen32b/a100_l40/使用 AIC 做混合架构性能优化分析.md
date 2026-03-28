# 实验设置

| 参数       | 说明                                                                                     |
| -------- | -------------------------------------------------------------------------------------- |
| **模型**   | Qwen/Qwen3-32B                                                                         |
| **系统**   | A100: `a100_sxm`, L40: `l40s`                                                          |
| **场景定义** | Short: `isl=512, osl=128`<br>Medium: `isl=2048, osl=512`<br>Long: `isl=4000, osl=1000` |
| **架构类型** | Pure Agg: 纯聚合模式<br>Pure Disagg: 纯解耦模式<br>Mix Disagg: 混合解耦模式（L40+A100 或 A100+L40）       |
## 1.1 关键指标说明

- **tokens/s/user**：单用户处理速度，反映用户体验 
	- `tokens_s_user = 1000 / tpot` 
	- `tpot = generation_latency / (osl - 1)`（osl > 1, ms）
- **tokens/s/gpu**：GPU利用率效率，反映硬件资源使用效率
	- **静态模式**：`tokens_s_gpu = tokens_s / (tp * pp * dp)`，其中 `tokens_s = seq_s * osl`（非 static_gen） 
	- **聚合模式**：`tokens_s_gpu = output_throughput / (pp * tp * dp)`
	- **disagg 模式**：`tokens_s_gpu = tokens_s / num_total_gpus`，其中 `num_total_gpus` 为 prefill 与 decode GPU 总和
- **TTFT (Time To First Token)**：首token延迟，反映响应速度
- **TPOT (Time Per Output Token)**：每token生成延迟，反映输出流畅度
- **收益分析维度**：以 Pure A100 Agg 为基线，比较其他架构
	- 在用户性能上的提升/下降
	- GPU效率上的提升/下降
	- 成本上的提升/下降
## 1.2 坐标轴计算逻辑

- **tokens/s/user**：仅反映生成阶段延迟，与TPOT强相关，用于SLA约束。
- **tokens/s/gpu**：反映整体硬件利用率，包含并行度（TP/PP/DP）归一化，用于吞吐优化。

两者在Pareto前沿上构成权衡关系：X轴为tokens/s/user（用户体验），Y轴为tokens/s/gpu（硬件效率）

- **静态模式**：`tokens_s_gpu = tokens_s / (tp * pp * dp)`，其中 `tokens_s = seq_s * osl`（非 static_gen） 
	```python base_backend.py
	# 计算序列吞吐和token吞吐 
	seq_s = global_bs / request_latency * 1000 * model.config.pp_size 
	tokens_s = seq_s * osl 
	# 非static_gen模式 
	
	# 归一化到每GPU 
	seq_s_gpu = seq_s / (tp * pp * dp) 
	tokens_s_gpu = tokens_s / (tp * pp * dp) 
	
	# 用户侧速度 
	tokens_s_user = 1000 / tpot
	```
- **聚合模式**：`tokens_s_gpu = output_throughput / (pp * tp * dp)`
```python 
# 输出吞吐已包含pp缩放 
output_throughput = 1000 / (mix_steps * mix_latency + genonly_steps * genonly_latency) * b * (osl - 1) output_throughput *= scale_factor # pp * attention_dp_size 

# 每GPU吞吐 
tokens_s_gpu = output_throughput / (pp * tp * dp) 
tokens_s_user = 1000 / tpot
```
- **disagg 模式**：`tokens_s_gpu = tokens_s / num_total_gpus`，其中 `num_total_gpus` 为 prefill 与 decode GPU 总和
```python
# 系统序列吞吐取prefill/decode最小值 
seq_s = min(prefill_seq_s * prefill_workers * 0.9, decode_seq_s * decode_workers * 0.92) 
# 总token数和GPU数 
tokens_s = seq_s * osl num_total_gpus = prefill_gpus * prefill_workers + decode_gpus * decode_workers 
# 每GPU吞吐 
tokens_s_gpu = tokens_s / num_total_gpus 
tokens_s_user = decode_summary_dict["tokens/s/user"] # 来自decode侧
```
# 综合Pareto前沿分析
>帕累托曲线上的点是所有“非支配”配置（即无法在不降低另一指标的情况下提升一个指标），并不全部满足你设置的 SLA（如 TPOT 或 request_latency）
>
>CLI 在绘制曲线时使用完整帕累托前沿，但在标记 `x` 时会先按 SLA 过滤

- **帕累托前沿生成**：`get_pareto_front` 仅基于“非支配”关系筛选，不考虑 SLA
- **x 标记逻辑**：在 picking 阶段，代码先用 SLA 约束（如 `tpot <= target`）过滤帕累托前沿，得到候选集；若候选集非空，再按 tokens/s/gpu 降序取 top N
- **无满足 SLA 时的回退**：若无点满足 SLA，会回退到从整个帕累托前沿选最接近的配置，用 `x` 强调“在满足 SLA 前提下最优”的点

<div style="display: flex; justify-content: space-between; align-items: center;">
  <div style="flex: 0 0 45%; text-align: center;">
    <a href="custom_pareto_tokens_gpu_vs_user_merged.png" target="_blank">
      <img src="custom_pareto_tokens_gpu_vs_user_merged.png" style="width: 100%; max-width: 1000px; height: auto;" />
    </a>
    <p style="font-size: 12px; color: #666; margin-top: 8px;">点击查看大图</p>
  </div>
  <div style="flex: 0 0 45%; text-align: center;">
    <a href="custom_pareto_tokens_gpu_vs_user_cost_merged 5.png" target="_blank">
      <img src="custom_pareto_tokens_gpu_vs_user_cost_merged 5.png" style="width: 100%; max-width: 1000px; height: auto;" />
    </a>
    <p style="font-size: 12px; color: #666; margin-top: 8px;">点击查看大图</p>
  </div>
</div>

## Short 场景 (isl=512, osl=128)

<div style="display: flex; justify-content: space-between; align-items: center;">
  <div style="flex: 0 0 45%; text-align: center;">
    <a href="custom_pareto_tokens_gpu_vs_user_merged 3.png" target="_blank">
      <img src="custom_pareto_tokens_gpu_vs_user_merged 3.png" style="width: 100%; max-width: 1000px; height: auto;" />
    </a>
    <p style="font-size: 12px; color: #666; margin-top: 8px;">点击查看大图</p>
  </div>
  <div style="flex: 0 0 45%; text-align: center;">
    <a href="custom_pareto_tokens_gpu_vs_user_cost_merged 6.png" target="_blank">
      <img src="custom_pareto_tokens_gpu_vs_user_cost_merged 6.png" style="width: 100%; max-width: 1000px; height: auto;" />
    </a>
    <p style="font-size: 12px; color: #666; margin-top: 8px;">点击查看大图</p>
  </div>
</div>

### 数据记录
```dataview
TABLE WITHOUT ID
experiment AS "实验",
round(tokens_per_sec_per_gpu, 2) AS "tokens/s/gpu",
round(tokens_per_sec_per_user, 2) AS "tokens/s/user",
round(tokens_per_10k_rmb, 2) AS "tokens/s/10k_rmb",
round(num_total_gpus, 2) AS "GPUs",
round(ttft_ms, 2) AS "TTFT(ms)",
round(tpot_ms, 2) AS "TPOT(ms)"
FROM "AIConfigurator/short_results.csv"
```
### 收益分析 (基线: Pure A100 Agg)

| 架构                        | tokens/s/user      | tokens/s/gpu        | tokens/s/10k_rmb | 综合评价             |
| ------------------------- | ------------------ | ------------------- | ---------------- | ---------------- |
| **基线**: a100_short_agg    | 51.02              | 289.44              | 661.59           | -                |
| l40_p_a100_d_short_disagg | **72.99 (+43%)** ⭐ | 210.10 (-27%)       | 672.31 (+1.6%)   | **最佳用户体验**，成本持平  |
| a100_short_disagg         | 50.04 (-2%)        | **373.42 (+29%)** ⭐ | 426.77 (-35%)    | **最佳GPU效率**，成本较高 |
| l40_short_disagg          | 57.24 (+12%)       | 3.32 (-99%)         | 17.70 (-97%)     | 用户性能尚可，但成本极高     |
| a100_p_l40_d_short_disagg | 57.24 (+12%)       | 3.32 (-99%)         | 10.62 (-98%)     | 用户性能尚可，但成本极高     |
### 核心结论

1. **用户体验优先**：`l40_p_a100_d_short_disagg` 表现最优，用户性能提升 43%，成本几乎持平
   - 首token延迟(TTFT): 246.73ms (优于基线的 455.75ms)
   - 适合对延迟敏感的短对话场景

2. **GPU效率优先**：`a100_short_disagg` GPU效率提升 29%，但需要权衡成本
   - 适合高吞吐需求场景，但需承担更高成本

3. **成本优化**：混合架构(l40+a100)在保持用户体验的同时成本最优
   - 纯L40架构成本效益极差，不建议使用

## Medium 场景 (isl=2048, osl=512)

<div style="display: flex; justify-content: space-between; align-items: center;">
  <div style="flex: 0 0 45%; text-align: center;">
    <a href="custom_pareto_tokens_gpu_vs_user_merged 2.png" target="_blank">
      <img src="custom_pareto_tokens_gpu_vs_user_merged 2.png" style="width: 100%; max-width: 1000px; height: auto;" />
    </a>
    <p style="font-size: 12px; color: #666; margin-top: 8px;">点击查看大图</p>
  </div>
  <div style="flex: 0 0 45%; text-align: center;">
    <a href="custom_pareto_tokens_gpu_vs_user_cost_merged 7.png" target="_blank">
      <img src="custom_pareto_tokens_gpu_vs_user_cost_merged 7.png" style="width: 100%; max-width: 1000px; height: auto;" />
    </a>
    <p style="font-size: 12px; color: #666; margin-top: 8px;">点击查看大图</p>
  </div>
</div>

### 数据记录

```dataview
TABLE WITHOUT ID
experiment AS "实验",
round(tokens_per_sec_per_gpu, 2) AS "tokens/s/gpu",
round(tokens_per_sec_per_user, 2) AS "tokens/s/user",
round(tokens_per_10k_rmb, 2) AS "tokens/s/10k_rmb",
round(num_total_gpus, 2) AS "GPUs",
round(ttft_ms, 2) AS "TTFT(ms)",
round(tpot_ms, 2) AS "TPOT(ms)"
FROM "AIConfigurator/medium_results.csv"
```
### 收益分析 (基线: Pure A100 Agg)

| 架构 | tokens/s/user | tokens/s/gpu | tokens/s/10k_rmb | 综合评价 |
|------|--------------|--------------|-------------------|---------|
| **基线**: a100_medium_agg | 50.16 | 230.78 | 527.49 | - |
| a100_medium_disagg | 52.63 (+5%) | 194.04 (-16%) | 443.52 (-16%) | 用户性能略优，成本较高 |
| **l40_p_a100_d_medium_disagg** | 25.78 (-49%) | **280.08 (+21%)** ⭐ | **689.43 (+31%)** ⭐⭐ | **最佳成本效益**，但用户体验受损 |
| a100_p_l40_d_medium_disagg | 25.29 (-50%) | 69.95 (-70%) | 172.18 (-67%) | 性能和成本均不理想 |
| l40_medium_disagg | 25.29 (-50%) | 69.95 (-70%) | 248.71 (-53%) | 性能和成本均不理想 |
### 核心结论

1. **成本优化推荐**：`l40_p_a100_d_medium_disagg` 在中长序列场景下表现出色
   - GPU效率提升 21%，成本效率提升 31%
   - 用户性能下降约 50%，需评估SLA要求

2. **用户体验保留**：`a100_medium_disagg` 保持用户体验的同时略有提升
   - 适合对延迟要求不严格但追求整体性能的场景

3. **架构选择建议**：
   - 对用户体验有严格要求：使用 Pure A100 Agg 或 A100 Disagg
   - 追求成本效益：使用 L40 Prefill + A100 Decode 混合架构
   - 纯L40架构在medium场景下表现不佳
## Long 场景 (isl=4000, osl=1000)

<div style="display: flex; justify-content: space-between; align-items: center;">
  <div style="flex: 0 0 45%; text-align: center;">
    <a href="custom_pareto_tokens_gpu_vs_user_merged 1.png" target="_blank">
      <img src="custom_pareto_tokens_gpu_vs_user_merged 1.png" style="width: 100%; max-width: 1000px; height: auto;" />
    </a>
    <p style="font-size: 12px; color: #666; margin-top: 8px;">点击查看大图</p>
  </div>
  <div style="flex: 0 0 45%; text-align: center;">
    <a href="custom_pareto_tokens_gpu_vs_user_cost_merged 8.png" target="_blank">
      <img src="custom_pareto_tokens_gpu_vs_user_cost_merged 8.png" style="width: 100%; max-width: 1000px; height: auto;" />
    </a>
    <p style="font-size: 12px; color: #666; margin-top: 8px;">点击查看大图</p>
  </div>
</div>

### 数据记录
```dataview
TABLE WITHOUT ID
experiment AS "实验",
round(tokens_per_sec_per_gpu, 2) AS "tokens/s/gpu",
round(tokens_per_sec_per_user, 2) AS "tokens/s/user",
round(tokens_per_10k_rmb, 2) AS "tokens/s/10k_rmb",
round(num_total_gpus, 2) AS "GPUs",
round(ttft_ms, 2) AS "TTFT(ms)",
round(tpot_ms, 2) AS "TPOT(ms)"
FROM "AIConfigurator/long_results.csv"
```
### 收益分析 (基线: Pure A100 Agg)

| 架构                           | tokens/s/user | tokens/s/gpu        | tokens/s/10k_rmb | 综合评价               |
| ---------------------------- | ------------- | ------------------- | ---------------- | ------------------ |
| **基线**: a100_long_agg        | 53.32         | 180.54              | 412.65           | -                  |
| **a100_long_disagg**         | 51.94 (-3%)   | 167.44 (-7%)        | 382.72 (-7%)     | 性能略降，成本略优          |
| **l40_p_a100_d_long_disagg** | 19.58 (-63%)  | **243.40 (+35%)** ⭐ | 599.13 (+45%) ⭐⭐ | **最佳成本效益**，但用户体验下降 |
| l40_long_disagg              | 16.98 (-68%)  | 105.57 (-42%)       | 375.36 (-9%)     | 纯L40架构性能不佳         |
| a100_p_l40_d_long_disagg     | 16.98 (-68%)  | 105.57 (-42%)       | 259.86 (-37%)    | 性能和成本均不理想          |
| l40_long_agg                 | 16.84 (-68%)  | 64.33 (-64%)        | 36.76 (-91%)     | 成本效率极低             |
### 核心结论

1. **最佳成本效益**：`l40_p_a100_d_long_disagg` 在长序列场景下表现突出
   - GPU效率提升 35%，成本效率提升 45%
   - 用户性能下降约 63%，需要权衡

2. **用户体验保留**：`a100_long_disagg` 在保持用户体验的同时略微降低成本
   - 性能下降仅 3%，适合对用户体验有严格要求的场景

3. **长序列场景特点**：
   - 混合架构(L40 Prefill + A100 Decode)在成本优化方面优势明显
   - 纯L40架构在长序列场景下性价比极低，不建议使用

# 综合分析与决策建议

## 跨场景性能对比

| 场景 | 用户性能最佳 | GPU效率最佳 | 成本效益最佳 |
|------|-------------|-------------|-------------|
| **Short** (512/128) | l40_p_a100_d (+43%) | a100_disagg (+29%) | l40_p_a100_d (+1.6%) |
| **Medium** (2048/512) | a100_disagg (+5%) | l40_p_a100_d (+21%) | l40_p_a100_d (+31%) |
| **Long** (4000/1000) | a100_agg (基线) | l40_p_a100_d (+35%) | l40_p_a100_d (+45%) |
## 关键发现

### 1. 混合架构的优势趋势

**L40 Prefill + A100 Decode** 架构随着序列长度增加，优势愈发明显：

| 场景 | 用户性能变化 | GPU效率变化 | 成本效率变化 |
|------|-------------|-------------|-------------|
| Short | +43% ⭐ | -27% | +1.6% |
| Medium | -49% | +21% | +31% ⭐ |
| Long | -63% | +35% | +45% ⭐⭐ |

**趋势分析**：
- 短序列：用户体验优先，首token延迟优势显著
- 中长序列：成本效益优势显现，但用户体验下降
### 2. Pure A100架构的稳定性

Pure A100 Disagg 在各场景下表现稳定：

| 场景 | 用户性能 | GPU效率 | 成本效率 |
|------|---------|---------|---------|
| Short | -2% | +29% | -35% |
| Medium | +5% | -16% | -16% |
| Long | -3% | -7% | -7% |

**结论**：适合追求稳定性能、对成本不敏感的场景
### 3. 不推荐架构
| 架构 | 主要问题 | 适用场景 |
|------|---------|---------|
| 纯L40 Disagg | 用户性能-68%，成本效率-97% | 几乎无适用场景 |
| A100 Prefill + L40 Decode | 用户性能-50~68%，成本效率-37~98% | 不推荐 |
## 场景化决策建议

### 对话场景分类与推荐架构

| 场景类型 | 特征 | 推荐架构 | 理由 |
|---------|------|---------|------|
| **实时短对话** | isl<1000, osl<256<br>延迟敏感 | **l40_p_a100_d** | 用户性能+43%，成本持平，TTFT优化明显 |
| **标准对话** | 1000<isl<3000<br>均衡需求 | **a100_disagg** | 性能稳定，用户体验略优(+5%) |
| **长文档分析** | isl>3000, osl>1000<br>成本敏感 | **l40_p_a100_d** | 成本效益+45%，适合批处理场景 |
| **多用户高并发** | 高吞吐需求<br>SLA严格 | **a100_agg** 或 **a100_disagg** | 稳定性能，可预测的SLA |
### 成本优化策略

1. **梯度部署策略**
   - 低峰期：使用混合架构降低成本
   - 高峰期：使用Pure A100确保SLA
   - 批处理：混合架构最大化成本效益

2. **资源配比建议**
   - Prefill:Decode GPU比例：2:8 至 3:7 (基于l40_p_a100_d配置)
   - 副本数(replicas)：根据SLA要求动态调整
## 技术启示

1. **分离架构的价值**：随着序列长度增加，prefill/decode分离的成本优势越明显
2. **GPU异构性利用**：L40适合计算密集的prefill，A100适合内存密集的decode
3. **帕累托权衡**：不存在"一方案适配所有场景"的解决方案，需根据具体SLA和成本约束选择

---
# 补充更新
## 成本计算逻辑更新
| 版本     | 成本计算模型                                | 适用场景                              |
| ------ | ------------------------------------- | --------------------------------- |
| **旧版** | "总 GPU 数 → 向上取整 → 统一单价"               | 聚合架构 (agg) / 单 GPU 类型实验           |
| **新版** | "Prefill/Decode 分离计费 + replicas 保守放大" | 分离架构 (disagg) / 混合 GPU 类型 / 多副本部署 |

| **资源参数利用**                | **旧版** | **新版**                                                                                      | **影响说明**                         |
| ------------------------- | ------ | ------------------------------------------------------------------------------------------- | -------------------------------- |
| replicas                  | 完全忽略   | 显式使用：`total_nodes = replicas × nodes_per_replica`                                           | replica>1 时成本按"每副本独占"保守估算，避免低估   |
| workers / gpus_per_worker | 完全忽略   | 显式使用：  <br>`p_gpus = (p)workers × p_gpus_worker`  <br>`d_gpus = (d)workers × d_gpus_worker` | 能识别 p/d 资源不对称场景（如 p=10 卡, d=2 卡） |
| 缺失参数回退                    | 无      | 若 `p_gpus_worker` 缺失，尝试用 `(p)tp×(p)pp×(p)dp`反推                                              | 增强鲁棒性，减少因字段缺失导致的计算失败             |

<div style="display: flex; justify-content: space-between; align-items: center;">
  <div style="flex: 0 0 45%; text-align: center;">
    <a href="custom_pareto_tokens_gpu_vs_user_cost_merged 4.png" target="_blank">
      <img src="custom_pareto_tokens_gpu_vs_user_cost_merged 4.png" style="width: 100%; max-width: 1000px; height: auto;" />
    </a>
    <p style="font-size: 12px; color: #666; margin-top: 8px;">点击查看大图</p>
  </div>
  <div style="flex: 0 0 45%; text-align: center;">
    <a href="custom_pareto_tokens_gpu_vs_user_cost_merged 5.png" target="_blank">
      <img src="custom_pareto_tokens_gpu_vs_user_cost_merged 5.png" style="width: 100%; max-width: 1000px; height: auto;" />
    </a>
    <p style="font-size: 12px; color: #666; margin-top: 8px;">点击查看大图</p>
  </div>
</div>

---




