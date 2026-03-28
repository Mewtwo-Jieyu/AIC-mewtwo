

AIC 的模型适配核心是：读取 HuggingFace config.json → 用 architectures[0] 映射到内部 ModelFamily → 在 sdk/models.py 里用一组 ops（GEMM/FMHA/MoE/Comm…）拼出模型 → 用性能库插值估算端到端 TTFT/TPOT（见 docs/add_a_new_model.md）。

---

必需信息（AIC 直接读取 / 建模用）
来自模型的 config.json（本项目实际会读这些字段，见 src/aiconfigurator/sdk/utils.py 的 _parse_hf_config_json）：
architectures[0]：决定属于哪个内部家族（LLAMA/MOE/DEEPSEEK/…）
    num_hidden_layers
    hidden_size
    num_attention_heads
    vocab_size
    max_position_embeddings
    可选但强烈建议（不填会变成 0，可能导致估算/校验异常）
        num_key_value_heads（GQA/MQA）
        intermediate_size
        head_dim 或 attention_head_dim（否则用 hidden_size/num_attention_heads 推）

MoE 模型还需要（AIC 识别并读取，见同函数）：
    num_experts_per_tok（topk）
    num_local_experts 或 n_routed_experts 或 num_experts
    moe_intermediate_size（否则退化用 intermediate_size）

量化/精度默认值（可选，但会影响估算与后端参数默认值；见 src/aiconfigurator/sdk/models.py 的 _infer_quant_modes_from_raw_config）：
    quant_algo（fp8/fp8_block/nvfp4/mxfp4/float16…）
    quant_dynamic
    kv_cache_quant_algo

配时“必须人工确认”的约束（否则 support/搜索会出错）
    TP 可行性：num_attention_heads % tp_size == 0，否则会直接 assert（见 BaseModel 初始化）。
    这也是你们 support_matrix.csv 里 Qwen2.5 1.5B/7B 在 TP=8 报错的根因：head 数 12/28 不能被 8 整除。
    PP 可行性：num_hidden_layers % pp_size == 0 不满足会给 warning（会带来估算误差）。
    MoE block-quant 形状约束：某些 MoE + fp8_block 会对 moe_intermediate_size/moe_tp 与 block size 的整除关系做校验（你们 matrix 里也有相关 FAIL）。

---

AIC 代码里需要补充/添加哪些地方（按“新模型的新程度”分三类）
docs/add_a_new_model.md 已把流程分成 3 种情况 - “你要改哪些文件”。

情况 A：只是现有架构的“普通变体”（推荐优先走这条）
特征：还是 Dense Transformer（或你们已支持的 MoE/DeepSeek 形态），只是层数/hidden/heads/GQA/上下文长度不同。
你要做的事：
    确认模型的 architectures[0] 是否已被识别
        映射表在 src/aiconfigurator/sdk/common.py 的 ARCHITECTURE_TO_MODEL_FAMILY
        例如：Qwen2ForCausalLM、Qwen3ForCausalLM 被映射到 "LLAMA"；Qwen3MoeForCausalLM 映射到 "MOE"。
    如果 architecture 新名字没在映射表里：只加一行映射即可
        改：src/aiconfigurator/sdk/common.py
        例："YourModelForCausalLM": "LLAMA"（或 "MOE"/"DEEPSEEK"）
    不需要新增模型类：直接用 --model-path org/your-model 跑即可（AIC 会拉取/读取 config）。
你要准备的模型信息：上面“必需信息”那一套 config.json 字段必须齐全且合理。

情况 B：架构属于已支持家族，但需要新增“性能数据点”
典型：MoE 的 num_experts/topk/moe_intermediate_size 是新组合；或 Attention 变体导致现有 perf table 覆盖不足。
你要做的事：
    在 collector/ 对应算子收集脚本里扩展测试用例（例如 MoE：collector/trtllm/collect_moe.py 里扩展 cases）
    跑 collector 生成新的 *_perf.txt（或同类数据文件）
    用新数据更新系统性能库
        位置类似：src/aiconfigurator/systems/data/<system>/<backend>/<version>/...
    仍然需要确保 ARCHITECTURE_TO_MODEL_FAMILY 有映射（同情况 A）

情况 C：模型引入了 AIC 还不支持的新“算子/结构”
典型：Mamba/Conv/其他非标准 Transformer 组件。
你要做的事（这是最重的路径）：
    src/aiconfigurator/sdk/operations.py：新增 Operation（例如 Conv）
    src/aiconfigurator/sdk/perf_database.py：新增 query_xxx 与数据加载逻辑
    collector/：新增/扩展采集脚本，采集该算子数据
    src/aiconfigurator/sdk/models.py：新增一个新的模型类（通常也意味着新增一个 ModelFamily）
    src/aiconfigurator/sdk/common.py：把新的 ModelFamily 加入集合，并补映射

---

“操作手册”：从 0 到 1 适配一个新模型
1) 判断你属于哪种适配路径
    先看模型 config.json 的 architectures[0]
        若只是名字新、结构仍是现有家族 → 情况 A
        若是 MoE 且 topk/experts/中间层很特殊 → 大概率情况 B
        若结构/算子不在现有 ops（非 GEMM/FMHA/MoE/Comm/…）→ 情况 C

2) 准备“模型信息包”（强烈建议你们内部建一个模板）
    HuggingFaceID（或本地路径）
    关键结构参数：layers/hidden/heads/kv_heads/intermediate/head_dim/context/vocab
    MoE：topk/num_experts/moe_intermediate_size
    量化：quant_algo/kv_cache_quant_algo/quant_dynamic（如有）
    建议同时给出可行的 TP/PP 候选集合（避免踩 num_heads % tp != 0）
    你们已经有业务侧信息模板（documentations/需求文档.md），但那是“容量评估输入”。模型适配这里更关注“架构参数 + 算子数据覆盖”。

3) 代码/配置改动（最常见：情况 A）
    修改 src/aiconfigurator/sdk/common.py：补 ARCHITECTURE_TO_MODEL_FAMILY
    （通常不需要动）src/aiconfigurator/sdk/models.py、generator templates

4) 运行支持性验证（推荐作为准入门槛）
    用 CLI 的 support 模式验证：src/aiconfigurator/cli/main.py 的 _run_support_mode 会读 support_matrix 并尝试解析 architecture 后做判断。
    如果 support matrix 没有该模型的精确行，AIC 会用“同 architecture 的多数投票”推断支持（见 common.check_support）。

5) 更新 support_matrix.csv（让适配“可追踪、可回归”）
    文件：src/aiconfigurator/systems/support_matrix.csv
    建议流程：
        先加一行（或多行）新模型的组合（System/Backend/Version + agg/disagg）
        跑你们的 support_matrix 生成/测试工具（仓库里有 tools/support_matrix/support_matrix.py 的痕迹）
        让 PASS/FAIL 结果与错误栈进入 matrix，后续一眼能看出“失败原因是 tp 不可整除 / MoE block 约束 / 缺数据点”等

6) 如果出现 FAIL，按错误类型快速定位
    num_heads ... divisible by tp_size：调整搜索空间的 tp 列表（或你的默认 tp 候选），不要让它枚举到不合法 tp
    MoE quant/block 校验失败：要么换量化策略，要么补齐更合适的 perf 覆盖与约束处理
    “不支持 architecture”：补 ARCHITECTURE_TO_MODEL_FAMILY
    性能表缺文件/缺数据点：走情况 B 的采集与更新系统数据

---

从技术报告中可提取并映射到 AIC 的参数

HuggingFaceID
    moonshotai/Kimi-K2.5（报告首页脚注）

MoE（AIC 会用到 topk / num_experts / moe_intermediate_size）
    num_experts：384 experts（报告 4.1/4.2 附近描述 “utilizing 384 experts”）
    topk（= num_experts_per_tok）：8 activated per token（同段落）
    稀疏度：报告写 “sparsity of 48”（这不是 AIC 的字段，但可用来 sanity check）
    moe_intermediate_size：报告未给出（AIC 需要从 HF config.json 读 moe_intermediate_size 或退化到 intermediate_size）

上下文长度 / 最大位置（AIC 用 context_length）
    训练/激活到的长上下文：32768 → 262144（Table 3 行 “Sequence length 32768→262144”）
    这通常对应 AIC 里的 context（= max_position_embeddings）应为 262144（最终以 HF config.json 为准，因为有的模型会用 RoPE scaling/YaRN 并不一定等价于 max_position_embeddings 字段）

参数规模（AIC 不直接用，但有助于核对）
    总参数量：1.04T total parameters
    激活参数量：32B activated parameters
    说明：这两个不会直接进入 AIC 的 config.json 解析字段，但能帮助你核对是不是你要的 checkpoint/版本。


报告里没给、但 AIC 适配“必须有”的字段（需要从 HF config.json 补齐）
AIC 在 src/aiconfigurator/sdk/utils.py::_parse_hf_config_json 里会读这些（缺失会报错或变 0）：
    architectures[0]（决定 ModelFamily 映射）
    num_hidden_layers
    hidden_size
    num_attention_heads
    num_key_value_heads（GQA/MQA，强烈建议有）
    intermediate_size
    vocab_size
    head_dim / attention_head_dim（否则会用 hidden_size/num_attention_heads 推）
    MoE 的 moe_intermediate_size（报告未给，HF 往往会给）

你可以直接用在 AIC 里的“参数汇总模板”（报告 + 待补齐项）
    模型ID：moonshotai/Kimi-K2.5
    模型类型：MoE Transformer（报告明确）
    MoE：
        num_experts: 384
        topk (num_experts_per_tok): 8
        moe_intermediate_size: (从 HF config.json 取)
    上下文：
        context（目标长上下文）: 262144（Table 3）
        max_position_embeddings（HF 字段）: (从 HF config.json 取；若不等于 262144，需要按模型实现确认)
    其余结构（全部从 HF config.json 取）：layers/hidden/heads/kv_heads/intermediate/vocab/head_dim


# 操作记录

## 代码改动记录（为支持 `moonshotai/Kimi-K2.5`）

### 1) 增加 HuggingFace architecture → AIC ModelFamily 映射

- **文件**：`src/aiconfigurator/sdk/common.py`
- **位置**：L284
- **改动**：在 `ARCHITECTURE_TO_MODEL_FAMILY` 增加：
  - `KimiK25ForConditionalGeneration` → `MOE`

### 2) 兼容多模态 config 的 `text_config`（解析文本模型结构参数）

- **文件**：`src/aiconfigurator/sdk/utils.py`
- **位置**：L362-L374
- **改动**：在 `_parse_hf_config_json()` 中新增逻辑：
  - 若顶层 `config.json` 缺少 `num_hidden_layers/hidden_size` 且存在 `text_config`，则从 `text_config` 读取模型结构参数；
  - 同时保留顶层 `architectures`，用于 AIC 的 `ModelFamily` 映射一致性。

### 3) vLLM（不量化）场景：忽略 HF `compressed-tensors` 等不支持的量化标记

- **文件**：`src/aiconfigurator/sdk/utils.py`
- **位置**：L652-L658
- **改动**：在 `_infer_quantization_fields()` 中增加白名单：
  - 仅保留 AIC 支持/可建模的 `quant_algo`（`fp8/fp8_block/nvfp4/mxfp4/float16`）
  - 对 HF `quant_method=compressed-tensors` 这类“checkpoint 存储格式”标记直接忽略，避免 downstream 报 `Unsupported quant algorithm`，从而按 FP16/BF16（不量化）路径继续。

### 4) perf 数据库加载失败时的报错增强（避免 NoneType & 给出可用版本）

- **文件**：`src/aiconfigurator/sdk/task.py`
- **位置**：L1042-L1062
- **改动**：`TaskRunner._get_database()` 若 `get_database(...)` 返回 `None`：
  - 直接抛 `PerfDataNotAvailableError`
  - 错误信息包含：`system/backend/version`、该 system/backend 下可用版本列表、以及当前 `systems_paths`

### 5) vLLM attention perf 覆盖不足时：插值单点退化（保证 default 能跑通）

- **文件**：`src/aiconfigurator/sdk/perf_database.py`
- **位置**：
  - L2967-L2986：`_nearest_1d_point_helper()` 支持 `len(values)==1` 时返回 `(v, v)`（nearest 退化）
  - L2956-L2962：当 `x_left == x_right` 时不做 `_interp_1d`，直接取单点值
- **作用**：避免 `values is None or len(values) < 2` 这类插值断言导致整个 pareto 搜索失败；后续可通过补采 perf 表提高精度。

### 6) vLLM attention perf 缺失 key 时：避免 `defaultdict` 隐式造空表导致插值崩溃（并对 head_dim 做 nearest+缩放）

- **现象**：运行 `aiconfigurator cli default` 时出现：
  - `ValueError: values is None or empty Failed to query context attention data ... head_size=112 ... Consider using HYBRID mode.`
- **根因**：`load_context_attention_data()` 生成的是多层 `defaultdict`。当 perf 表缺少某个维度 key（本例常见是 `head_dim=112`），使用 `[...]` 索引不会 `KeyError`，而是**隐式创建空 dict**，最终在 `_interp_*` 中对空 `data.keys()` 触发 `values is None or empty`。
- **文件**：`src/aiconfigurator/sdk/perf_database.py`
- **改动**：
  - `query_context_attention()` / `query_generation_attention()`：用 `.get()` 做安全 lookup，发现缺失时做 nearest fallback（kv_heads/head_size/window_size），并对 **head_size 缺失**采用最近点（如 128）并按 \(112/128\) 对 latency/energy 线性缩放，保证搜索可继续进行。

### 7) vLLM + MoE 并行约束：过滤掉 `moe_tp>1 && moe_ep>1`

- **现象**：`AssertionError: vllm does not support MoE TP and MoE EP at the same time`，导致所有候选并行组合被跳过，最终 `No results found for any parallel configuration`。
- **文件**：`src/aiconfigurator/sdk/utils.py`
- **改动**：`enumerate_parallel_config()` 在 `backend==vllm` 且 MoE 模型时，直接跳过 `moe_tp>1 && moe_ep>1` 的组合（与 `operations.py` 的约束一致）。

### 8) `--total-gpus` 传入 32 但枚举仍只到 8：扩展 vLLM 默认搜索空间到 16/32

- **现象**：命令行传 `--total-gpus 32`，但日志 `Listing parallelism configs` 仍只出现 `dp<=8`；于是仍然会被判定 OOM（“model does not fit in GPU memory”）。
- **根因**：vLLM + MoE 的默认并行列表在 `task.py` 被硬编码为 `[1,2,4,8]`；`total_gpus` 只会“裁剪上限”，不会自动扩展到 16/32。
- **文件**：`src/aiconfigurator/sdk/task.py`
- **改动**：
  - agg：`TaskConfigFactory._agg_defaults_layer()` 的 vLLM 分支把并行列表扩展为 `[1,2,4,8,16,32,64,128]`
  - disagg：`build_disagg_parallel_lists()` 的 vLLM 分支同样扩展；最终仍由 `--total-gpus` 做上限裁剪。

### 9) 开发态从源码运行：未安装包时 `__version__` 取值兼容

- **现象**：从源码 `PYTHONPATH=src python -m ...` 运行时，`importlib.metadata.version("aiconfigurator")` 可能抛 `PackageNotFoundError`。
- **文件**：`src/aiconfigurator/__init__.py`
- **改动**：捕获 `PackageNotFoundError` 并回退 `__version__="0.0.0"`，便于源码调试（不影响正常 pip/conda 安装版本）。

---

## 本次排障/验证记录（2026-03-17）

### 复现命令

```bash
aiconfigurator cli default \
  --model-path moonshotai/Kimi-K2.5 \
  --system h200_sxm \
  --backend vllm \
  --total-gpus 32 \
  --isl 4000 --osl 500 \
  --ttft 2000 --tpot 50 \
  --save-dir results_kimi_vllm_default_32g
```

### 关键过程与结论

- **阶段 1（perf 插值崩溃）**：`values is None or empty` → 通过「条目 6」修复后消失。
- **阶段 2（vLLM 并行约束）**：`vllm does not support MoE TP and MoE EP at the same time` → 通过「条目 7」过滤后消失。
- **阶段 3（total-gpus 未生效）**：虽然输入 `--total-gpus 32`，但枚举仍 `dp<=8` → 通过「条目 8」扩展搜索空间后，枚举出现 `dp=16` 等组合。
- **最终结果**：
  - agg：实验成功产出结果（日志示例：`Experiment agg completed with 43 results.`），最佳解使用 `tp=16, pp=1, dp=1`（即 `gpus/worker=16`），因此实际 **`total_gpus (used)=16`**（32 卡预算下 AIC 选择了 16 卡即可满足 SLA 并达到更优吞吐/每卡表现）。
  - disagg：本次运行返回空结果（需要进一步放宽约束、增加预算、或调整 disagg 搜索/配置策略）。

---

# 遇到的问题和修正debug

---

## 步骤 A：把“模型结构参数”准备成 AIC 输入（Kimi-K2.5）

### A.1 基本信息

- **HuggingFaceID**：`moonshotai/Kimi-K2.5`
- **AIC 识别到的 architecture**：`KimiK25ForConditionalGeneration`
- **建议 AIC ModelFamily**：`MOE`（用于走 `MOEModel` 的建模路径）

### A.2 AIC 必需结构参数（用于建模与合法性校验）

以下参数建议以 HuggingFace `config.json` 为准（Kimi-K2.5 的这些字段位于 `text_config`）：

- **Transformer**
  - **num_hidden_layers**：61
  - **hidden_size**：7168
  - **num_attention_heads**：64
  - **num_key_value_heads**：64
  - **head_dim（推导）**：112（= 7168 / 64）
  - **intermediate_size（FFN）**：18432
  - **vocab_size**：163840（≈160K）
  - **max_position_embeddings / context**：262144（=256K）

- **MoE**
  - **num_experts**：384（HF 字段：`n_routed_experts`）
  - **topk（num_experts_per_tok）**：8
  - **moe_intermediate_size（per expert）**：2048
  - **n_shared_experts**：1
  - **first_k_dense_replace / Dense layers**：1（用于“前若干层 dense，其余为 MoE”的配置）

### A.3 AIC 侧必须满足的并行约束（否则会直接报错或显著增大误差）

- **TP 约束（硬约束）**：`num_attention_heads % tp == 0`
  - Kimi-K2.5：64 heads → tp 可选 `{1,2,4,8,16,32,64}`

- **PP 约束（建议强约束）**：`num_hidden_layers % pp == 0`
  - Kimi-K2.5：61 layers（质数）→ **建议默认只允许 `pp=1`**（否则 AIC 会 warning 且引入估算误差）

---

## 步骤 B：决定推理后端与精度（本次：vLLM，暂不量化）

### B.1 决策

- **backend**：vLLM
- **量化**：暂不考虑（按 FP16/BF16 路径使用；AIC 侧建议保持 `quant_algo=None`，以免误触发 fp8/fp4 路径）

### B.2 关键注意事项（vLLM + Kimi-K2.5）

- **Kimi-K2.5 的 HF 顶层是多模态 `ForConditionalGeneration`**，文本模型结构在 `text_config`，AIC 必须走上述兼容解析逻辑。
- HF config 内包含 `quantization_config`（compressed-tensors/4bit 等），但 **AIC 当前的量化模式推断主要依赖 `quant_algo/kv_cache_quant_algo`**；
  - 本次“不考虑量化”时，应以 vLLM 的 FP16/BF16 运行方式为准，避免把 HF 的量化配置误当成 AIC 的 quant_mode。
- **MLA/DeepSeek 风格字段存在**（如 `qk_nope_head_dim/qk_rope_head_dim/kv_lora_rank`），这可能导致：
  - AIC 当前用“通用 MoE + 标准 attention ops”估算时，对 vLLM 的 attention 延迟可能偏差较大；
  - 需要后续用 vLLM 的基准测试结果做一次校验（至少对 TTFT/TPOT 的量级与趋势进行 sanity check）。

### B.3 验证建议（最小闭环）

- 先跑一组最小配置，验证 AIC 能完成 `get_model_config_from_model_path('moonshotai/Kimi-K2.5')` 并进入 vLLM 流程；
- 搜索空间建议：
  - `pp_list=[1]`
  - `tp_list` 从 `{1,2,4,8,16}` 起步（按你们卡数再扩展）

### B.4 常见阻塞项：Perf 数据为 Git LFS 指针（未拉取）

如果运行 `default`/`exp` 时出现类似报错：
- `KeyError: 'gemm_dtype'`（CSV 列缺失）
- 或日志显示 perf 文件内容是 `version https://git-lfs.github.com/spec/v1`

通常原因是：`src/aiconfigurator/systems/data/<system>/<backend>/<version>/*.txt` 仍是 **Git LFS pointer**，真实 perf 表未下载。

处理方式：

```bash
git lfs install
git lfs pull
```

然后重跑 `aiconfigurator cli default ...`。

### B.5 常见阻塞项：vLLM attention perf 维度覆盖不足（插值需要至少 2 个点）

如果出现错误类似：
- `AssertionError: values is None or len(values) < 2 Failed to query context attention data ...`

说明某个维度（例如 `num_heads`）在当前 perf 表里只有 1 个采样点，插值无法进行。

处理策略：
- **短期**：退化为 nearest（单点维度不插值）以保证 AIC 能跑通搜索；
- **长期**：补采 vLLM attention 表在该维度的更多采样点（至少 2 个），提高估算精度。

