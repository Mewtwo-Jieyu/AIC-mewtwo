# h200_sxm_halfbw 新硬件适配记录（带宽减半实验）

## 1. 背景与目标

- **目标**：在不采集新 perf 数据的前提下，基于现有 `h200_sxm` 构造一块“HBM/互联带宽减半”的**虚构硬件**，验证 AIC 的新硬件适配流程是否跑通，并用于分析带宽对配置/性能的影响。
- **约束**：不改动 SDK 代码逻辑，仅通过新增系统 YAML + 更新 `support_matrix.csv` + 自定义实验 YAML 的方式完成。

## 2. 系统 YAML 构造

- **文件位置**：`src/aiconfigurator/systems/h200_sxm_halfbw.yaml`
- **data_dir 复用**：直接复用 `h200_sxm` 的 perf 数据目录：

  ```yaml
  data_dir: data/h200_sxm  # reuse h200_sxm perf data for synthetic half-bandwidth system
  ```

- **HBM 与互联带宽缩放**：在保持显存容量、FLOPS 等一致的前提下，将关键带宽字段缩小为原来的 1/2：

  - `gpu.mem_bw: 4800000000000 -> 2400000000000`
  - `node.inter_node_bw: 25000000000 -> 12500000000`
  - `node.intra_node_bw: 450000000000 -> 225000000000`

- 其他字段保持与 `h200_sxm` 一致：

  - 显存容量：`mem_capacity: 151397597184`
  - Tensor Core FLOPS：`float16_tc_flops / int8_tc_flops / fp8_tc_flops`
  - 能耗与架构：`power: 700`, `sm_version: 90`
  - NCCL / 预留显存：复用原有 `nccl_mem` 与 `other_mem` 配置。

这样得到一个“只在带宽维度变差”的合成系统，用于做对比实验。

## 3. 支持矩阵补充

- **文件位置**：`src/aiconfigurator/systems/support_matrix.csv`
- 初始运行 `aiconfigurator cli support`：

  ```bash
  aiconfigurator cli support \
    --model-path Qwen/Qwen3-32B \
    --system h200_sxm_halfbw \
    --backend trtllm
  ```

  输出中提示：

  - `Model 'Qwen/Qwen3-32B' not found in support matrix.`
  - `Version: None`，`Aggregated Support: NO`，`Disaggregated Support: NO`

  说明当前 `support_matrix.csv` 里**没有任何关于 `h200_sxm_halfbw` 的记录**，只做了架构级别的 fallback 推断。

- 为了“先跑通链路”，直接**捏造**一组 `PASS` 记录，复用 `h200_sxm` 在 TRTLLM 上的版本组合，在 `support_matrix.csv` 中新增：

  ```text
  Qwen/Qwen3-32B,Qwen3ForCausalLM,h200_sxm_halfbw,trtllm,1.0.0rc3,agg,PASS,
  Qwen/Qwen3-32B,Qwen3ForCausalLM,h200_sxm_halfbw,trtllm,1.0.0rc3,disagg,PASS,
  Qwen/Qwen3-32B,Qwen3ForCausalLM,h200_sxm_halfbw,trtllm,1.2.0rc5,agg,PASS,
  Qwen/Qwen3-32B,Qwen3ForCausalLM,h200_sxm_halfbw,trtllm,1.2.0rc5,disagg,PASS,
  ```

- 更新后，再次运行：

  ```bash
  aiconfigurator cli support \
    --model-path Qwen/Qwen3-32B \
    --system h200_sxm_halfbw \
    --backend trtllm \
    --backend-version 1.2.0rc5
  ```

  即可看到：

  - `Version: 1.2.0rc5`
  - `Aggregated Support: YES`
  - `Disaggregated Support: YES`

这一步完成了“新 system 名称在 CLI support 路径上的接入”。

## 4. 实验 YAML 与对比设置

- **实验配置文件**：`exp_yaml/fake_h200_halfbw.yaml`
- 该 YAML 同时定义了 **半带宽** 与 **原带宽** 的 agg/disagg 四个实验，用于对比：

  - `exp_h200_halfbw_agg`
  - `exp_h200_halfbw_disagg`
  - `exp_h200_fullbw_agg`
  - `exp_h200_fullbw_disagg`

- 以半带宽聚合模式为例：

  ```yaml
  exp_h200_halfbw_agg:
    mode: "patch"
    serving_mode: "agg"
    model_path: "Qwen/Qwen3-32B"
    total_gpus: 16
    system_name: "h200_sxm_halfbw"
    backend_name: "trtllm"
    profiles: []
    config:
      worker_config:
        gemm_quant_mode: "float16"
        moe_quant_mode: "float16"
        fmha_quant_mode: "float16"
        kvcache_quant_mode: "float16"
    isl: 2048
    osl: 512
    ttft: 1000.0
    tpot: 50.0
  ```

- 对应的 full 带宽实验仅将 `system_name` / `decode_system_name` 切换为 `h200_sxm`，其余参数保持一致，用于做“仅带宽不同”的横向对比。

- 这些实验通过 `aiconfigurator cli exp` 运行，并将结果用于后续自定义帕累托图（见 `scripts/plot_custom_pareto_cost_v3.py` 与 `output_scale/scale_mem_bw/` 目录下的图像）。

## 5. 运行命令示例（本地 aiconf 环境）

在本地 conda 环境 `aiconf` 中，典型验证流程：

```bash
conda activate aiconf
cd /Users/zhao_penelope/shailab/aiconfigurator

# 1）快速支持性检查
PYTHONPATH=src aiconfigurator cli support \
  --model-path Qwen/Qwen3-32B \
  --system h200_sxm_halfbw \
  --backend trtllm \
  --backend-version 1.2.0rc5

# 2）exp 模式跑带宽缩放实验
PYTHONPATH=src aiconfigurator cli exp \
  --config exp_yaml/fake_h200_halfbw.yaml \
  --save-dir output_scale/scale_mem_bw
```

其中 `PYTHONPATH=src` 是为了让 Python 在本地源码树下正确找到 `aiconfigurator` 包（相当于把 `src` 加到模块搜索路径），方便在“源码开发模式”下直接运行 CLI。

## 6. 总结

- **h200_sxm_halfbw 的本质**：复用 `h200_sxm` 的 perf 数据，仅通过 YAML 中的 `mem_bw` / 互联带宽参数来构造一个更受带宽瓶颈限制的“虚拟系统”。
- **适配步骤**：
  1. 在 `systems/` 下新增 `h200_sxm_halfbw.yaml`，数据目录指向 `data/h200_sxm`。
  2. 在 `support_matrix.csv` 中为目标模型/后端版本添加 `PASS` 记录，确保 `cli support` 能识别新 system。
  3. 编写实验 YAML（`fake_h200_halfbw.yaml`）同时覆盖半带宽与原带宽的 agg/disagg，对比带宽缩放下的最优配置与帕累托前沿变化。
- 这套流程可作为后续“新硬件首轮快速模拟/预研”的模板，在真实数据尚未完全采集前先跑通端到端链路。

