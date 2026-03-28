# Kimi-K2.5 模型适配操作记录（AIConfigurator）

> 面向仓库：`AIC_v0`（AIConfigurator）。本文综合 `Project.md` 的项目整体流程与 `documentations/kimi-k2.5模型适配.md` 的具体改动/排障过程，整理为一份**可复现的操作记录**，并给出**局限性分析**与 **future work**。

---

## 背景与目标

- **目标**：让 AIConfigurator 能正确识别并建模 `moonshotai/Kimi-K2.5`，并在指定硬件系统与后端（本次以 **vLLM** 为主）上完成 `cli default` 搜索与产物生成；对常见失败点提供可操作的排障路径。
- **范围**：
  - **模型适配**：从 HuggingFace `config.json` 解析结构参数 → architecture 映射 → 建模与搜索空间合法性处理。
  - **性能库**：当 perf 表维度覆盖不足（或缺 key）导致插值崩溃时，提供短期 fallback 以“先跑通搜索”。
  - **不在范围**：对 Kimi-K2.5 的真实端到端性能做全面校准（需要额外 benchmark 与数据补采）。

---

## 关键概念（与 AIC 流程对齐）

- **AIC 输入**：模型（`--model-path` 或 HF id）、系统（`--system`）、后端（`--backend`）、SLA（TTFT/TPOT）、ISL/OSL、总卡数（`--total-gpus`）。
- **AIC 的模型适配核心链路**：
  - 读取 HuggingFace `config.json`
  - `architectures[0]` → 映射到内部 `ModelFamily`
  - 从结构参数拼装 ops（GEMM / Attention / MoE / 通信…）→ 通过性能库插值估算 TTFT/TPOT
- **产物目录**：`--save-dir` 下生成可部署的配置、脚本与 k8s 清单（详见 `Project.md` 与项目 docs）。

---

## 前置条件与环境准备

### 代码与依赖

- **Python**：>= 3.9
- **建议**：使用项目推荐的依赖版本（参考 `Project.md` / `pyproject.toml`）

### 重要：性能数据（Git LFS）

如果运行 `default/exp` 时出现如下异常/现象，优先怀疑 perf 数据仍是 LFS pointer：

- 报错类似 `KeyError: 'gemm_dtype'`、CSV 列缺失
- perf 文件内容出现 `version https://git-lfs.github.com/spec/v1`

处理方式（在仓库根目录）：

```bash
git lfs install
git lfs pull
```

---

## 模型信息确认（Kimi-K2.5）

### HuggingFace 标识与 architecture

- **HF id**：`moonshotai/Kimi-K2.5`
- **architecture（AIC 用于映射 ModelFamily）**：`KimiK25ForConditionalGeneration`
- **建议 ModelFamily**：`MOE`

### AIC 建模“必须有”的结构参数（来自 HF config）

Kimi-K2.5 的顶层是多模态 `ForConditionalGeneration`，文本结构位于 `text_config`。本次适配要求 AIC 在缺失顶层关键字段时从 `text_config` 读取：

- **Transformer**
  - `num_hidden_layers=61`
  - `hidden_size=7168`
  - `num_attention_heads=64`
  - `num_key_value_heads=64`
  - `head_dim=112`（推导：7168 / 64）
  - `intermediate_size=18432`
  - `vocab_size=163840`（技术报告口径：160K；HF config 往往给精确值 163840）
  - `max_position_embeddings=262144`（技术报告口径：256K；HF config 往往给精确值 262144）
- **MoE**
  - `num_experts=384`（HF 字段：`n_routed_experts`）
  - `num_experts_per_tok(topk)=8`
  - `moe_intermediate_size=2048`
  - `n_shared_experts=1`
  - `first_k_dense_replace=1`（“前若干层 dense，其余为 MoE”）

> 以上数值以 `documentations/kimi-k2.5模型适配.md` 的整理为准，最终仍以 HF `config.json` 为权威来源。

---

## AIC 侧硬约束与搜索空间注意事项

### 并行合法性（必须满足，否则会 assert / 直接失败）

- **TP 约束（硬约束）**：`num_attention_heads % tp_size == 0`
  - Kimi-K2.5：64 heads → `tp ∈ {1,2,4,8,16,32,64}`
- **PP 约束（强烈建议）**：`num_hidden_layers % pp_size == 0`
  - Kimi-K2.5：61 layers（质数）→ **建议默认仅 `pp=1`**（否则会 warning 且引入估算误差）

### vLLM + MoE 约束

vLLM 场景下需要避免枚举到不支持的 MoE 并行组合（否则会导致所有候选被跳过，最终“没有结果”）：

- **约束**：不同时启用 `moe_tp>1` 且 `moe_ep>1`

---

## 代码与数据改动摘要（本次适配覆盖点）

> 这里给的是“变更意图与作用点”。逐条细节、行号与原始排障日志请以 `documentations/kimi-k2.5模型适配.md` 为准。

### 1) architecture → ModelFamily 映射

- **目的**：让 AIC 将 `KimiK25ForConditionalGeneration` 识别为 `MOE` 家族路径。
- **涉及文件**：`src/aiconfigurator/sdk/common.py`

### 2) 兼容多模态 `text_config`（解析文本结构参数）

- **目的**：当顶层 `config.json` 缺少 `num_hidden_layers/hidden_size/...` 时，从 `text_config` 读取结构参数；同时保留顶层 `architectures` 用于家族映射一致性。
- **涉及文件**：`src/aiconfigurator/sdk/utils.py`

### 3) vLLM（不量化）时忽略 HF “checkpoint 存储格式”量化标记

- **目的**：避免将 `compressed-tensors` 这类 HF 标记误判为 AIC 的可建模量化模式，从而触发 `Unsupported quant algorithm`。
- **涉及文件**：`src/aiconfigurator/sdk/utils.py`

### 4) perf 数据库加载失败的报错增强

- **目的**：当 `system/backend/version` 不匹配或数据缺失时，直接给出可用版本列表与路径信息，避免 `NoneType` 难定位。
- **涉及文件**：`src/aiconfigurator/sdk/task.py`

### 5) Attention/MLA perf 插值维度只有单点时的短期退化

- **目的**：perf 表在某维度（例如只采了一个点）导致插值断言失败时，允许退化为 nearest 以保证搜索可继续运行。
- **涉及文件**：`src/aiconfigurator/sdk/perf_database.py`

### 6) Attention/MLA perf 缺失 key 的安全查询 + nearest fallback（含 head_dim 缩放）

- **目的**：避免 `defaultdict` 通过 `[...]` 索引隐式造空表，最终在插值时对空 keys 崩溃；对 `head_size` 缺失使用 nearest（如 128）并按 \(112/128\) 做线性缩放，保证搜索不断。
- **涉及文件**：`src/aiconfigurator/sdk/perf_database.py`

### 7) vLLM + MoE 并行组合过滤

- **目的**：过滤掉 vLLM 不支持的 `moe_tp>1 && moe_ep>1` 组合，避免 “No results found for any parallel configuration”。
- **涉及文件**：`src/aiconfigurator/sdk/utils.py`

### 8) 扩展 vLLM 默认并行列表以支持 16/32 卡搜索

- **目的**：解决 `--total-gpus 32` 传入后仍只枚举到 8（默认列表硬编码为 `[1,2,4,8]`）的问题；扩展到更大并行候选后仍由 `--total-gpus` 裁剪上限。
- **涉及文件**：`src/aiconfigurator/sdk/task.py`

### 9) 按技术报告启用 Kimi 的 MLA 查表与建模路径

- **目的**：tech report 明确 Kimi-K2.5 的 Attention Mechanism 为 **MLA**；因此在仿真中应优先使用 `*_mla_perf.txt` 对应的 perf 表与 MLA ops，而不是通用 attention 表（`*_attention_perf.txt`）。
- **涉及文件**：
  - `src/aiconfigurator/sdk/task.py`
    - 逻辑从“仅 DeepSeek 走 MLA 表”扩展到：当 `architecture=="KimiK25ForConditionalGeneration"` 时同样选择 `context_mla/generation_mla`（SGLang WideEP 场景选择 `wideep_context_mla/wideep_generation_mla`）。
  - `src/aiconfigurator/sdk/models.py`
    - 在 `MOEModel` 中针对 `KimiK25ForConditionalGeneration` 将注意力 op 从 `ContextAttention/GenerationAttention` 切换为 `ContextMLA/GenerationMLA`（op 名仍保持 `context_attention/generation_attention` 以兼容下游匹配逻辑）。

### 10) 源码运行时版本号兼容

- **目的**：未安装包的开发态运行不因 `PackageNotFoundError` 失败。
- **涉及文件**：`src/aiconfigurator/__init__.py`

### 11) 预下载模型配置缓存（model_configs）

- **目的**：为 `moonshotai/Kimi-K2.5` 提供本地 HF config 缓存，便于离线/稳定复现，避免每次都从 HuggingFace 拉取 `config.json`。
- **涉及文件**：
  - `src/aiconfigurator/model_configs/moonshotai--Kimi-K2.5_config.json`
    - 内容包含 `architectures=["KimiK25ForConditionalGeneration"]` 与 `text_config` 下的关键结构参数（61 层、7168 hidden、64 heads、384 experts、topk=8、vocab≈160K、context≈256K 等），与技术报告/实际适配一致。
  - `src/aiconfigurator/sdk/common.py` 中的 `DefaultHFModels`
    - 已将 `moonshotai/Kimi-K2.5` 加入集合，使 `get_model_config_from_model_path("moonshotai/Kimi-K2.5")` 会优先走本地 `model_configs` 缓存。

### 12) 系统性能数据更新

- **目的**：配合上述逻辑，完善/更新系统 perf 数据文件（`systems/data/...`）。
- **涉及路径**：`src/aiconfigurator/systems/data/<system>/<backend>/<version>/...`

---

## 操作步骤（可复现）

### 步骤 0：拉取 perf 数据（如使用 Git LFS）

```bash
git lfs install
git lfs pull
```

### 步骤 1：基础可用性验证（建议先做）

优先跑 `support` 模式，快速验证模型/系统/后端组合是否可被 AIC 正确识别（能更早暴露 architecture 映射、TP/PP 等硬约束问题）。

> 命令形态请参考项目 `cli_user_guide`；若你们内部已约定了 `support_matrix.csv` 的回归流程，也建议在此阶段同步更新。

### 步骤 2：跑 default 搜索（本次复现命令）

```bash
aiconfigurator cli default \
  --model-path moonshotai/Kimi-K2.5 \
  --system h200_sxm \
  --backend vllm \
  --total-gpus 16 \
  --isl 4000 --osl 500 \
  --ttft 2000 --tpot 50 \
  --save-dir results_kimi_vllm_default_16g
```

### 步骤 3：检查结果与产物

`--save-dir` 会生成可部署产物，通常结构形如：

- `results_xxx/agg/top1/`：聚合模式 top1（含 `run_0.sh`、`k8s_deploy.yaml`、`bench_run.sh`、`generator_config.yaml` 等）
- `results_xxx/disagg/top1/`：解耦模式 top1（含 `prefill_config.yaml/decode_config.yaml`、多节点 `run_*.sh` 等）
- `best_config_topn.csv`、`pareto.csv`、`pareto_frontier.png`：搜索结果与可视化

> 具体文件名/层级以项目当前 generator 输出为准；如目录结构与预期不一致，优先对照仓库的部署指南与 generator 文档。

---

## 本次运行记录结论（来自既有排障文档）

- **阶段 1（perf 插值崩溃）**：`values is None or empty` → 通过 Attention/MLA 查询的安全 fallback/退化策略后解决
- **阶段 2（vLLM 并行约束）**：`vllm does not support MoE TP and MoE EP at the same time` → 通过过滤非法并行组合后解决
- **阶段 3（total-gpus 未生效）**：传 `--total-gpus 32` 仍只枚举 `<=8` → 扩展 vLLM 默认并行列表后解决
- **结果**：
  - **agg**：完成并产出结果；最佳解示例为 `tp=16, pp=1, dp=1`，因此实际使用卡数为 16（在 32 卡预算下选择 16 即可满足约束并达到更优指标）
  - **disagg**：本次运行无结果（需要后续进一步分析/补齐数据与策略）

---

## 局限性分析（当前实现/流程的边界）

### 1) vLLM perf 表覆盖不足导致的 “退化估算”

- **现状**：对单点维度插值、缺 key 时做 nearest fallback（含 head_dim 线性缩放）可以让搜索继续跑通。
- **局限**：退化策略会引入不可忽视的误差，尤其在 Kimi-K2.5 这类 head_dim（112）与 perf 表常见采样（如 128）不匹配时，线性缩放只是一阶近似。

### 2) Kimi-K2.5 的 MLA 机制与 perf 覆盖/口径差异带来的误差风险

- **现状**：已按 tech report 将 Kimi-K2.5 切换到 MLA 查表与 `ContextMLA/GenerationMLA` ops 建模路径。
- **局限**：
  - 现有 `*_mla_perf.txt` 的维度覆盖未必完整覆盖 Kimi 的关键形态（例如 `head_dim=112`、kv_heads/seq_len/batch 等），仍可能触发 fallback/退化估算；
  - 不同后端（vLLM/TRTLLM/SGLang）对 MLA 的实现细节不同，单纯“切到 MLA 表”并不必然保证绝对 TTFT/TPOT 精度。
  - tech report 提到 “1 层 dense + 其余 MoE” 的层内结构差异（如 `first_k_dense_replace=1`）目前尚未在 ops 构建中逐层显式建模，仍属于结构近似。

### 3) PP 搜索空间在 61 层模型上天然受限

- **现状**：PP 需要层数可整除才能可靠建模/划分；61 层使得 `pp>1` 基本不可用。
- **局限**：会限制搜索空间多样性，尤其在大卡数预算下可能影响找到最优 Pareto 前沿的能力（只能更多依赖 TP/DP/MoE 并行）。

### 4) disagg（prefill/decode 拆分）无结果的原因尚未闭环

- **现状**：现阶段结论是“搜索返回空结果”，尚未给出可复现的最小失败原因分类（SLA 过紧、并行空间过窄、OOM、perf 缺失、约束冲突等）。
- **局限**：缺少系统化的失败归因会阻碍后续优化路径选择（该补 perf、该放约束，还是该改建模/搜索策略）。

### 5) support_matrix 与回归机制仍需制度化

- **现状**：`support_matrix.csv` 能承载 PASS/FAIL 与错误原因，但新模型新增条目与回归执行尚需明确流程化入口（脚本/CI）。
- **局限**：没有可重复的准入/回归，适配容易“只在开发机上可用”，难以长期维护。

---

## Future Work（建议按优先级推进）

### P0：把 “跑通” 变成 “可信”

- **补采/扩充 vLLM MLA perf 表**：至少补齐对 Kimi-K2.5 关键维度（如 `head_dim=112`、kv_heads、seq_len、batch 等）的多点采样，尽可能消除插值退化路径。
- **基准对齐与校准**：选取少量代表性配置（不同 TP/DP）做真实 benchmark，将 AIC 估算与实测的 TTFT/TPOT/吞吐对齐，必要时引入 backend-specific 校准因子。

### P1：disagg 无结果的系统化归因与修复

- **失败归因分桶**：把 “无结果” 拆成 OOM、硬约束冲突、perf 缺失、SLA 不可达、搜索空间过窄等类别，并在日志/报错层面给出显式原因。
- **扩大 disagg 搜索空间策略**：在不违反硬约束的前提下，引入更合理的 prefill/decode worker 组合、并行候选与 batch 策略；必要时针对 61-layer 的 PP 不可用情况调整 disagg 的默认策略。

### P1：模型解析与家族映射的鲁棒性

- **统一多模态/子 config 解析策略**：将 “顶层缺字段 → 下钻 `text_config`” 的逻辑推广为通用机制（并补充测试用例），减少后续类似模型适配成本。
- **架构名变化的监控**：对 `architectures[0]` 新增/变更提供更明确的提示（例如建议用户添加映射项），避免 silent fallback。

### P2：约束与可解释性增强

- **约束前置检查**：在枚举并行前对 TP/PP、MoE 并行限制做显式过滤与统计，让用户一眼看到“合法候选还剩多少”。
- **perf 覆盖诊断工具**：给定模型结构与后端版本，输出 perf 表缺失的 key/维度覆盖情况，指导 collector 补采。

### P2：支持矩阵与回归体系

- **制度化 support_matrix 流程**：提供一键脚本/CI 作业，自动运行关键组合并把结果落到 `support_matrix.csv`（含失败原因摘要）。
- **最小回归集**：为 MoE 大模型与多模态模型建立最小回归用例集合，防止后续 perf 表/解析逻辑修改引入回归。

---

## 附：常见问题快速定位（Checklist）

- **报 architecture 不支持**：补 `ARCHITECTURE_TO_MODEL_FAMILY` 映射
- **报 `num_heads ... divisible by tp_size`**：调整 tp 候选集合（确保 `heads%tp==0`）
- **perf 插值断言失败 / values empty**：优先确认 Git LFS；其次检查 perf 表维度覆盖与缺 key
- **vLLM MoE 并行 assert**：避免 `moe_tp>1 && moe_ep>1`
- **`--total-gpus` 看起来没生效**：检查后端默认并行列表是否覆盖到 16/32，并确认是否被上限裁剪

