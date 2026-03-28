[DeepWiki - Analysis](https://deepwiki.com/ai-dynamo/aiconfigurator/1-overview)

# 新硬件适配指南（面向开发者）

本文件说明在 AIConfigurator 中适配一套**全新 GPU 系统**（例如新一代 SXM/PCIe 服务器）时，需要：

- **修改或新增哪些文件 / 目录**
- **理解和收集哪些关于新硬件的关键信息**
- **在代码层面需要注意哪些接口和验证路径**

整体目标：让新硬件在 `default/exp/generate/support` 四种 CLI 模式下，都能像现有的 `h100_sxm/h200_sxm/b200_sxm/gb200_sxm/a100_sxm/l40s` 一样，被完整地建模、搜索和验证。

---

## 总体流程一览

1. **准备硬件信息**：根据芯片 datasheet、系统 BOM、网络拓扑文档，整理 GPU 和节点级别的参数。
2. **新增系统 YAML**：在 `src/aiconfigurator/systems/` 下创建 `<system_name>.yaml`，描述 GPU 能力和节点拓扑。
3. **采集性能数据**：使用 `collector/` 下各后端的采集脚本，在新硬件上跑 GEMM/Attention/MoE/NCCL 等基准，输出到对应 `data/<system_name>/...` 目录。
4. **更新支持矩阵**：在 `src/aiconfigurator/systems/support_matrix.csv` 中补充新硬件 + 后端 + 版本 + 模式（agg/disagg）的支持记录，或使用 `tools/support_matrix/generate_support_matrix.py` 自动生成。
5. **CLI 验证**：通过 `aiconfigurator cli support` / `cli default` 对典型模型做一轮回归，确认新硬件在 agg/disagg 下都能得到合理配置。
6. **（可选）模板与后端参数映射增强**：如果新硬件需要特殊的后端参数或部署模板，在 `generator/config` 与 `backend_config_mapping.yaml` 里做有针对性的扩展。

---

## 必需的文件 / 目录修改

### 1. 新增系统定义文件：`src/aiconfigurator/systems/<system_name>.yaml`

在 `src/aiconfigurator/systems/` 目录下为新硬件创建一个 YAML 文件，例如 `gb200_sxm.yaml`、`h200_sxm.yaml` 的同级兄弟：

- **文件名即系统名**：`<system_name>.yaml`，例如 `b200_sxm.yaml`。
- 顶层关键字段：
  - **`data_dir`**：性能数据库相对路径，例如 `data/gb300`。
  - **`gpu`**：单卡能力相关参数。
  - **`node`**：节点 / 机架拓扑与带宽、延迟。
  - **`misc`**：NCCL 内存开销、其它预留内存、NCCL 版本等经验值。

可以参考已有的系统文件（如 `b200_sxm.yaml`、`h200_sxm.yaml`、`a100_sxm.yaml` 等），典型结构如下（节选）：

```yaml
data_dir: data/gb300  # relative to systems_dir
gpu:
  mem_bw: 8000000000000                  # HBM 实测或 datasheet 带宽 (Byte/s)
  mem_bw_empirical_scaling_factor: 0.8   # 经验修正系数（< 1 表示实际可用带宽）
  mem_empirical_constant_latency: 0.000003  # 内存访问常数延迟 (s)
  mem_capacity: 298013687808             # 单卡显存容量 (Byte)
  float16_tc_flops: 2500000000000000     # FP16 Tensor Core 理论/有效 FLOPS
  int8_tc_flops: 165000000000000         # INT8 Tensor Core FLOPS
  fp8_tc_flops: 5000000000000000         # FP8 Tensor Core FLOPS
  fp4_tc_flops: 15000000000000000        # FP4 Tensor Core FLOPS
  power: 1400                            # TDP (Watt)
  sm_version: 103                        # SM 架构版本（决定 kernel 选择）

node:
  num_gpus_per_node: 4
  num_gpus_per_rack: 72
  inter_rack_bw: 25000000000             # 机架间 IB 带宽 (Byte/s per GPU, single dir)
  inter_node_bw: 900000000000            # 机架内 NVSwitch 带宽
  intra_node_bw: 900000000000            # 节点内 NVLink 带宽
  pcie_bw: 64000000000                   # PCIe 单向带宽 (Byte/s)
  p2p_latency: 0.00001                   # GPU–GPU P2P 延迟 (s)
  inter_rack_latency: 0.000005           # 机架间网络延迟 (s)

misc:
  nccl_mem:
    1: 0
    2: 358612992
    4: 411041792
    8: 411041792
  other_mem: 3758096384
  nccl_version: "2.27"
```

**注意：**

- 所有带宽统一用 **Byte/s**，延迟用 **秒**，容量用 **Byte**，确保与现有系统保持一致。
- `mem_bw_empirical_scaling_factor`、`mem_empirical_constant_latency` 等参数来自经验和微基准测试，可以先用**保守估计**，再通过回归慢慢调优。

### 2. 组织性能数据库：`src/aiconfigurator/systems/data/<system_name>/...`

性能数据库由 `PerfDatabase` 统一管理，目录结构约定如下（以 `gb300` 为例）：

```text
src/aiconfigurator/systems/
├── gb300.yaml
└── data/
    └── gb300/
        ├── trtllm/
        │   └── 1.2.0rc5/
        │       ├── gemm_perf.txt
        │       ├── attention_perf.txt
        │       ├── moe_perf.txt
        │       └── nccl_perf.txt
        ├── vllm/
        │   └── 0.12.0/
        │       └── ...
        └── sglang/
            └── 0.5.6.post2/
                └── ...
```

要求：

- **目录名称**：
  - 顶层目录名必须与 YAML 中的 `data_dir` 一致。
  - 二级目录是后端名：`trtllm` / `vllm` / `sglang`。
  - 三级目录是**后端版本**（与 CLI 中 `--backend-version` / `--generated-config-version` 对齐）。
- **文件格式**：
  - `gemm_perf.txt` / `attention_perf.txt` / `moe_perf.txt` / `nccl_perf.txt` 等完全复用现有格式；可以参考 `a100_sxm` 等系统下的同名文件。
  - 不同后端的文件字段略有差异，但 `PerfDatabase` 已为每种后端定义了解析逻辑，只要格式对齐就无需改代码。

性能数据可以通过 `collector/` 下的脚本批量采集，见后文“数据采集与 collector 使用”一节。

### 3. 更新支持矩阵：`src/aiconfigurator/systems/support_matrix.csv`

`support_matrix.csv` 记录了**模型 × 架构 × 系统 × 后端 × 版本 × 模式(agg/disagg)** 的测试结果，供：

- CLI `support` 模式快速查询是否“支持”
- 文档和 README 中展示支持矩阵

典型行示例（节选）：

```text
HuggingFaceID,Architecture,System,Backend,Version,Mode,Status,ErrMsg
Qwen/Qwen2.5-72B,Qwen2ForCausalLM,gb200_sxm,trtllm,1.0.0rc6,disagg,PASS,
Qwen/Qwen2.5-72B,Qwen2ForCausalLM,gb200_sxm,trtllm,1.2.0rc5,agg,PASS,
...
```

为新硬件添加条目时：

- **System**：使用新系统名（如 `gb300` 或 `gb300_sxm`，与 YAML 文件名一致）。
- **Backend / Version**：来自你在 `data/<system_name>/<backend>/<version>/` 下准备的性能数据库。
- **Mode**：`agg` 或 `disagg`。
- **Status**：
  - `PASS`：说明 AIConfigurator 能在该组合下找到可行配置。
  - `FAIL`：会带上 `ErrMsg`，记录失败原因（模型配置不兼容、缺少 perf 表等）。

支持矩阵行可以通过自动化脚本生成：

- 参考 `tools/support_matrix/generate_support_matrix.py` 和 `tools/support_matrix/support_matrix.py`。
- 推荐在 CI 或专门机器上跑一轮 `support_matrix` 生成，确认新硬件在主流模型上的“覆盖面”。

### 4. CLI 与帮助文档中的 system 名称（可选）

在 `src/aiconfigurator/cli/main.py` 中，`--system` 参数的帮助文本给出了若干示例：

```python
parser.add_argument(
    "--system",
    type=str,
    required=True,
    help="System name (GPU type). Example: h200_sxm,h100_sxm,b200_sxm,gb200_sxm,a100_sxm,l40s.",
)
```

对新硬件来说：

- **功能上**：只要你提供了 `<system_name>.yaml` 和对应 `data/` 目录，CLI 会自动识别，无需额外硬编码。
- **可读性上**：可以在 help 文本里补充新系统名，方便用户发现，例如加入 `gb300_sxm`。

---

## 需要掌握的新硬件信息（what & where）

为了把新硬件建模到 AIConfigurator 中，需要提前弄清楚以下几个层面的信息。

### 1. GPU 芯片级信息（单卡）

- **显存容量**：`mem_capacity`
  - 来源：`nvidia-smi` / datasheet。
  - 影响：能否装下给定模型权重和 KV cache；影响并行度上限。
- **显存带宽**：`mem_bw`
  - 来源：datasheet / `bandwidthTest` 一类基准。
  - 与 `mem_bw_empirical_scaling_factor` 一起决定 LLM 中 memory-bound kernel 的吞吐。
- **内存访问常数延迟**：`mem_empirical_constant_latency`
  - 来源：对 attention/GEMM kernel 做小 batch、短序列测试拟合得到。
- **Tensor Core FLOPS（按精度）**：
  - `float16_tc_flops` / `fp8_tc_flops` / `fp4_tc_flops` / `int8_tc_flops`。
  - 来源：datasheet 理论峰值，或根据实际 GEMM 基准做略微下调。
- **功耗上限**：`power`
  - 用于能耗建模与帕累托分析中“性能 vs 能耗”的维度。
- **SM 架构版本**：`sm_version`
  - 决定 kernel 选择、支持的算子/精度（如 FP8/FP4 是否原生支持）。

### 2. 节点 / 集群拓扑信息

在 YAML 的 `node` 字段中，需要描述：

- **GPU 数量**：
  - `num_gpus_per_node`：单节点 GPU 数。
  - `num_gpus_per_rack`：整机架 GPU 数（影响跨节点通信拓扑）。
- **带宽（Byte/s, 单向）**：
  - `intra_node_bw`：同一节点内 NVLink 带宽。
  - `inter_node_bw`：同一机架内节点之间通过 NVSwitch 等互联的带宽。
  - `inter_rack_bw`：不同机架之间的 IB / RoCE 带宽。
  - `pcie_bw`：GPU 与 CPU / NIC 间的 PCIe 带宽。
- **延迟（秒）**：
  - `p2p_latency`：节点内 GPU–GPU P2P 延迟。
  - `inter_rack_latency`：跨机架通信 RTT。

这些参数直接影响：

- TP / PP / DP / MoE-TP / MoE-EP 等策略下的**通信开销估计**。
- 聚合 vs 分布式（agg vs disagg）模式下**prefill/decode 分工**的最优点。

### 3. NCCL 与其它内存开销

在 YAML 的 `misc` 部分：

- **`nccl_mem` 映射**：
  - key 为 GPU 数量（world size），value 为 NCCL 额外占用的显存（Byte）。
  - 这些值可以通过在不同 world size 下跑 NCCL allreduce/alltoall，记录显存占用变化获得。
- **`other_mem`**：
  - 框架本身 / runtime / driver / 其他进程占用的保底显存。
  - 建议宽松一些，以避免边缘 case 上 OOM。
- **`nccl_version`**：
  - 对 debug 很重要，尤其是遇到通信挂起 / 性能不达标问题时。

### 4. 后端与版本支持情况

新硬件通常需要明确：

- 支持哪些后端：`TensorRT-LLM` / `vLLM` / `SGLang`。
- 每个后端在哪些版本上经过验证，例如：
  - `trtllm`: `1.0.0`, `1.2.0rc5`, `1.0.0rc6` 等。
  - `vllm`: `0.12.0`, `0.14.0` 等。
  - `sglang`: `0.5.6.post2`, `0.5.8` 等。
- 各版本在不同模型上的兼容性（MoE 支持、FP8/FP4 支持、动态 KV cache 策略等），最终会反映在：
  - `support_matrix.csv` 的 PASS/FAIL
  - `PerfDatabase` 中**是否存在对应版本的 perf 文件**

---

## 数据采集与 `collector/` 使用

`collector/` 目录包含了针对不同后端和算子类型的性能采集脚本，包括：

- `collector/trtllm/*.py`：TRT-LLM 上的 GEMM / Attention / MoE / MLA / Mamba2 等采集。
- `collector/vllm/*.py`：vLLM 上的 GEMM / Attention / MoE / MLA 等采集。
- `collector/sglang/*.py`：SGLang 上的 attention / MLA / wide&deep MoE 等采集。
- `collector/collect_nccl.py`、`collector/collect_all_reduce.py`：通信（NCCL allreduce 等）性能采集。
- `collector/deep_collector/*`：更细粒度的 intra-node / inter-node 通信测试工具。

典型使用路径（概念层面）：

1. 在目标新硬件集群上安装好对应版本的 `trtllm` / `vllm` / `sglang`。
2. 参考现有系统的采集命令，在新硬件上运行同一套 `collector/*` 脚本。
3. 将生成的 `*_perf.txt` 文件放入：
   - `src/aiconfigurator/systems/data/<system_name>/<backend>/<version>/`。
4. 使用 `tools/sanity_check/validate_database.ipynb` 或 SDK 中的简单查询调用，确认 perf 文件能被 `PerfDatabase` 正确解析。

**建议**：

- 新硬件首轮适配时，可以优先支持**一个主流后端 + 一个版本**（例如 `trtllm 1.2.0rc5`），先跑通整条链路；再扩展到更多版本和后端。

---

## 性能数据库与 SDK 中的关键代码

新硬件适配以“**数据驱动**”为主，核心逻辑已经在 SDK 中抽象好，一般**不需要**新增代码。理解以下几个入口，有助于 debug：

- **系统路径与 YAML 加载**
  - `src/aiconfigurator/sdk/perf_database.py` 中：
    - `get_systems_paths()` / `set_systems_paths()`：控制系统 YAML 和数据查找路径。
    - `get_supported_databases()`：扫描所有 `<system>.yaml` 和对应 `data_dir`，枚举已有的 (system, backend, version) 组合。
- **数据库加载**
  - `get_database(system, backend, version, ...)`：
    - 根据 `<system>.yaml` 里的 `data_dir` 拼出路径。
    - 校验目录是否存在，构造 `PerfDatabase` 实例并缓存。
- **量化 / 模式支持推断**
  - `_update_support_matrix()`（`PerfDatabase` 内部方法）：
    - 根据实际加载到的 GEMM / attention / MoE / NCCL 表，自动推断当前 (system, backend, version) 支持哪些 quant 模式。
    - 不同后端（`trtllm`/`vllm`/`sglang`）有各自分支处理方式。

因此，对新硬件来说：

- **只要目录和文件布局与现有系统一致**，并提供了格式正确的 `*_perf.txt` 文件，`PerfDatabase` 就能自动接入，无需改动 Python 代码。
- 只有在以下情况时，才需要修改 SDK：
  - 引入**全新的算子类型**或性能模型（例如新的 MLA 变体）。
  - 引入**全新后端**（不是 TRTLLM / vLLM / SGLang）。
  - 需要改变数据库模式（例如新增 HYBRID/EMPIRICAL 之外的新模式）。

---

## 生成器与后端配置映射（可选增强）

如果新硬件在某个后端上需要特殊参数（例如额外的 CUDA graph 配置、特殊 MoE all2all backend 等），可以通过 **生成器配置映射** 支持。

相关文件：

- `src/aiconfigurator/generator/config/backend_config_mapping.yaml`
  - 负责在生成器内部的统一参数（如 `max_batch_size`、`max_seq_len`、`kv_transfer_backend` 等）与具体后端 CLI / config 字段之间做映射。
- `src/aiconfigurator/generator/config/backend_templates/...`
  - 不同后端 / 部署形态（agg/disagg, bare-metal/K8s）的模板。

对于新硬件来说，一般**不需要**在这里做改动，除非：

- 为了适配新硬件上的某些优化功能（如专用的 KV 传输后端、特殊 MoE all2all 后端），需要在生成的脚本或配置中加参数。
- 后端在新硬件上有不同的默认行为，需要调整配置以获得稳定性能。

在这种情况下，可以：

- 在 `backend_config_mapping.yaml` 中新增或修改对应 `param_key` 的后端字段。
- 在 backend 模板中使用这些参数，生成符合新硬件推荐实践的配置。

---

## 验证步骤与推荐 Checklist

适配完成后，建议按下面顺序验证新硬件：

1. **快速支持性检查**

```bash
aiconfigurator cli support \
  --model-path Qwen/Qwen3-32B \
  --system <new_system_name> \
  --backend trtllm
```

预期：

- 如果 `support_matrix.csv` 里已经有对应行，CLI 会直接报告 PASS/FAIL，并给出原因。
- 如果尚未写入 `support_matrix.csv`，则根据实际运行结果更新该文件。

2. **默认模式端到端验证**

```bash
aiconfigurator cli default \
  --model-path Qwen/Qwen3-32B \
  --total-gpus 8 \
  --system <new_system_name> \
  --backend trtllm \
  --ttft 600 --tpot 50 \
  --isl 4000 --osl 500 \
  --save-dir results_new_system
```

检查：

- 是否成功搜索出 agg 和 disagg 的 Pareto 前沿。
- `results_new_system` 目录下是否生成 `agg/` 和 `disagg/` 对应的 top1 / topn 配置与脚本。

3. **配置生成模式（generate）健壮性**

```bash
aiconfigurator cli generate \
  --model-path Qwen/Qwen3-32B \
  --total-gpus 8 \
  --system <new_system_name> \
  --backend trtllm
```

确认：

- 不依赖完整 sweep，也能生成一个在新硬件上可跑通的 agg 配置。

4. **support_matrix 自动化回归（可选）**

- 使用 `tools/support_matrix/generate_support_matrix.py` 对常见模型家族（GPT, LLaMA, Qwen, DeepSeek 等）跑一轮，生成或更新 `support_matrix.csv`。
- 对所有 `FAIL` 项分析原因：是模型结构不兼容，还是 perf 表缺失 / 不足。

---

## 常见问题与调试建议

- **缺少 perf 表 / 数据库报错**
  - 症状：类似 `PerfDataNotAvailableError: Context attention perf table is missing for system='...'`。
  - 检查：
    - 目录 `data/<system>/<backend>/<version>/attent*` 是否存在且格式正确。
    - `support_matrix.csv` 中是否错误地标记了某个 (system, backend, version) 为 `PASS`。

- **TP/PP/MoE 拓扑不可行**
  - 症状：`No results found for any parallel configuration`。
  - 排查方向：
    - 检查 `mem_capacity`、`other_mem` 和 `nccl_mem` 是否太保守，导致可用显存过小。
    - 检查模型本身的 head 数 / hidden size 是否与默认 TP 倍数不匹配。

- **通信开销异常大或分布式模式效果不好**
  - 排查：
    - 核实 `intra_node_bw` / `inter_node_bw` / `inter_rack_bw` 是否明显偏低。
    - 核实 `p2p_latency` / `inter_rack_latency` 是否与实际网络 RTT 对齐。

---

## 总结

- **最核心的工作**：为新硬件准备**合理的系统 YAML** 和 **完整的 perf 数据库目录**，并在 `support_matrix.csv` 中记录通过验证的组合。
- **大部分逻辑已经抽象在 SDK 中**，新硬件适配以“填充数据”为主，而不是大规模改代码。
- 只有在引入**全新后端 / 算子类型 / 模式**时，才需要修改 `perf_database.py`、`operations.py` 或 backend 实现。

