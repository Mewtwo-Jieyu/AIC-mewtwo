## 背景：现有“最优点”定义回顾

- **best_config_topn.csv（当前 CLI 定义的最优点）**
  - 先在底层 `InferenceSession` 中用 TTFT / TPOT / `request_latency` 约束过滤掉不可行配置，得到 `pareto_df`。
  - CLI 在 `process_experiment_result` 里：
    - 计算集群级吞吐：`tokens/s/gpu_cluster`。
    - 若给了 `request_latency`：只保留 `request_latency <= target_request_latency` 的点；
    - 否则：只保留 `tpot <= target_tpot` 的点。
    - 按 `parallel` / `(d)parallel` 分组，每组只保留 `tokens/s/gpu_cluster` 最高的一行。
    - 在剩余集合上按 `tokens/s/gpu_cluster` 降序（必要时加一个次级排序列）取前 `top_n`，写成 `best_config_topn.csv`。
  - **本质**：这是一个“在满足 SLA 前提下，最大化集群每卡吞吐（tokens/s/gpu_cluster）”的筛选逻辑，偏吞吐优先，而不是成本/性价比优先。

- **pareto.csv**
  - 是在 \((X, \text{tokens/s/gpu\_cluster})\) 平面上的帕累托前沿点集，其中：
    - 若无 `request_latency` SLA：\(X = \text{tokens/s/user}\)，希望越大越好；
    - 若有 `request_latency` SLA：\(X = \text{request_latency}\)，希望越小越好（实现中通过取负转成“越大越好”）。
  - `get_pareto_front` 会在 \((X, Y)\) 二维空间里删除被其他点同时压制的点，得到一条前沿曲线，写成 `pareto.csv`。

- 更详细的现有逻辑，可参考已有文档 `documentations/pareto_best_config_analysis.md`。

---

## 目标：在两张图上自动找出“跨架构双优点”

在现有文档/分析中，跨架构对比通常同时看两张 2D 图：

- **图 1（性能视角）**：`tokens/s/gpu`（或 `tokens/s/gpu_cluster`） vs `tokens/s/user`
- **图 2（成本视角）**：`tokens/s/10k_rmb` vs `tokens/s/user`

在目前的分析里，两张图上标出来的“最优点”**本质上都是按照吞吐筛选得到的同一批配置点**：

- 先由 CLI / `best_config_topn` 按 `tokens/s/gpu_cluster`（在 SLA 下）选出若干吞吐最优配置；
- 然后：
  - 在 **性能图** 上，把它们画成 `tokens/s/gpu vs tokens/s/user`；
  - 在 **成本图** 上，对同一批点额外算出 `tokens/s/10k_rmb`，画成 `tokens/s/10k_rmb vs tokens/s/user`。

我们现在想做的是：**在这批 baseline 吞吐最优点的基础上，额外找出一类“相对更优”的点集合**：

- 以某个 baseline 最优配置点 \(p_\text{base}\) 为参照，在两张图之一或两张图中，找到那些在 \((X, Y)\) 上同时更优的点：
  - 在性能图上：\(X=\text{tokens/s/user}, Y=\text{tokens/s/gpu}\)，要求 \(X > X_\text{base}\) 且 \(Y > Y_\text{base}\)；
  - 或者在成本图上：\(X=\text{tokens/s/user}, Y=\text{tokens/s/10k\_rmb}\)，要求 \(X > X_\text{base}\) 且 \(Y > Y_\text{base}\)；
- 先收集这些“在某一张图上对 baseline 做到 x、y 双优”的候选点集合；
- 再按**另一张图的纵轴**（另一维度的 `Y`）做排序，选出每个架构/场景下最有代表性的那一个。

为了让这件事可复用、可自动化，希望在现有 CLI / 脚本基础上：

- **增加一套“跨架构双优/支配点”筛选逻辑**，输入为现有 CSV；
- 自动标记并输出：
  - 每个架构在“性能+成本”双视角下的**支配点**（若存在）；
  - 同时旁路输出现有 AIC 定义的**吞吐最优点**（`best_config_topn`）作为对照。

### Medium 场景的直观例子：为什么“最优”不能只按吞吐选？

以 Medium 场景为例（`isl=2048, osl=512`），现有文档里通常会选：

- 在 **tokens/s/gpu vs tokens/s/user** 这张“性能图”上：
  - `a100_medium_agg` 的某个点作为 A100 聚合基线；
  - `l40_p_a100_d_medium_disagg` 的某个点作为混合架构的代表；
  - 这两个“最优点”在这张图里**未必互相支配**：很可能一个在 `tokens/s/gpu` 更高，另一个在 `tokens/s/user` 更高，它们都在各自架构的前沿上。

但如果切换到 **tokens/s/10k_rmb vs tokens/s/user** 的“成本图”：

- 在 `tokens/s/user ≈ 60` 一带，可能存在一个 `disagg` 点：
  - 它相对 `a100_medium_agg` 当前的“吞吐最优点”来说，`tokens/s/user` 不低，甚至略高；
  - 同时 `tokens/s/10k_rmb` 明显更高（成本效益更好）；
  - 换句话说，在“成本指标 + 用户吞吐”这两个维度上**同时支配**当前的 A100 基线点。
- 这个 disagg 点在“性能图”上，未必是所有可行配置里 `tokens/s/gpu` 最高的那个，因此在现有 `best_config_topn`（吞吐优先）筛选流程中，很容易被**直接过滤掉**。

这个例子说明：

- 现有 `best_config_topn` 的“最优”本质上是**吞吐单目标最优**；
- 一旦我们希望把“成本/性价比”也纳入目标或约束，就会出现“在某张图（例如成本图）上能双优/支配”的点，而这些点**不一定是纯吞吐意义上的最优点**；
- 本文设计的“双优/支配点”算法，就是为了在这种情况下，系统性地找出这类“相对双优”的配置，并与原始吞吐最优点并列输出，而不是互相替代。

---

## 数据来源与坐标系统一

### 1. 单实验层面：`pareto.csv` + `best_config_topn.csv`

对于某个具体 experiment（例如 `a100_medium_disagg`）：

- `pareto.csv` 给出在 \((X, \text{tokens/s/gpu\_cluster})\) 平面上的帕累托前沿；
- 同目录下的 `best_config_topn.csv` 给出了在该 experiment 内部、在给定 SLA 下的吞吐最优点。

这部分更偏“单架构/单实验内部的最优解”，主要用于：

- 保留 AIC 原始定义的“吞吐优先最优点”；
- 作为 cross-arch 分析时的**参考基线**。

### 2. 跨实验/跨架构层面：`collect_results_cost_sweep.py` 的汇总 CSV

`scripts/collect_results_cost_sweep.py` 在同一批实验结果上、在不同成本假设下重复计算 `tokens_per_10k_rmb`，并输出：

- 按场景的汇总表，例如：
  - `results_summary_baseline_l40_1.5w_vs_a100_3.5w.csv`
- 一个合并所有成本场景的总表：
  - `results_summary_<prefix>_all_scenarios.csv`

单个汇总 CSV 的典型字段（示例）：

- `experiment`：如 `l40_p_a100_d_medium_disagg`，可视为“架构/部署模式”的标签；
- `tokens_per_sec_per_gpu`：对应性能图的纵轴 \(Y_\text{perf}\)；
- `tokens_per_sec_per_user`：对应两张图的横轴 \(X\)，用户体验视角；
- `tokens_per_10k_rmb`：对应成本图的纵轴 \(Y_\text{cost}\)；
- `ttft_ms, tpot_ms, request_latency_ms`：SLA 相关字段；
- `isl, osl`：序列长度场景；
- `gpu_type` / `p_gpu_type` / `d_gpu_type`：硬件架构标签。

**后续“跨架构双优点”筛选建议完全基于这一层的汇总 CSV 来做**，因为：

- 已经是“在各自 experiment 内，按照 AIC 逻辑挑出来的最优/代表性配置”；
- 自带成本信息（`tokens_per_10k_rmb`），能直接驱动成本视角的图；
- 便于在一张表里对比多种架构/模式。

> 直观理解：`results_summary_*.csv` 中的每一行，正是“两张图上的一个点”，只是分别投影在：
> - 图 1：\((X, Y_\text{perf}) = (\text{tokens/s/user}, \text{tokens/s/gpu})\)
> - 图 2：\((X, Y_\text{cost}) = (\text{tokens/s/user}, \text{tokens/s/10k\_rmb})\)

---

## 算法设计：跨架构“双优/支配点”筛选

下面的算法是“逻辑设计”，可以在后续用 pandas/numpy 实现；此处只给出步骤，不写具体代码实现。

### 1. 筛选输入行：保证“同场景 + 同 SLA”

目标是比较**同一业务场景**下的不同架构，因此需要先在汇总表里做一次过滤：

- **场景过滤**：
  - 以 `isl, osl` 列筛选（如 Short/Medium/Long）；
  - 或者直接由上游把不同场景拆成多个 `results_summary_xxx.csv`，每个文件只含一个 `(isl, osl)`。
- **SLA 过滤**：
  - 按 `ttft_ms, tpot_ms, request_latency_ms` 过滤出满足目标 SLA 的行；
  - 逻辑与 CLI 中 `get_best_configs_under_*_constraint` 一致：
    - 若有 `request_latency_ms > 0`：只保留 `request_latency_ms <= target_request_latency` 的行；
    - 否则：只保留 `tpot_ms <= target_tpot` 的行。
- **成本场景过滤（可选）**：
  - 若使用 `results_summary_all_scenarios.csv`，可以先按 `cost_scenario` 选中一个成本假设（如 `baseline_l40_1.5w_vs_a100_3.5w`）。

记经过上述过滤后的 DataFrame 为 \(D\)，其中每一行 \(i\) 对应一个点：

- \(X_i = \text{tokens\_per\_sec\_per\_user}_i\)
- \(Y^\text{perf}_i = \text{tokens\_per\_sec\_per\_gpu}_i\)
- \(Y^\text{cost}_i = \text{tokens\_per\_10k\_rmb}_i\)
- 架构标签：`arch_i`（可由 `experiment` 或 `gpu_type/p_gpu_type/d_gpu_type` 正规化得到）。

### 2. 定义“双优/支配”关系

我们希望找到这样的点 \(p\)：

- 在**性能图**中不被其他点压制：
  - 按 \((X, Y^\text{perf})\) 最大化的帕累托前沿；
- 同时在**成本图**中也不被其他点压制：
  - 按 \((X, Y^\text{cost})\) 最大化的帕累托前沿。

更形式化地：

- 在集合 \(D\) 上定义两套帕累托前沿：
  - 性能前沿 \(F_\text{perf}\)：在二维空间 \((X, Y^\text{perf})\) 下，保留所有**不存在**另一点 \(j\) 使得
    \[
      X_j \ge X_i,\quad Y^\text{perf}_j \ge Y^\text{perf}_i,\quad
      \text{且至少一项严格大于}
    \]
    的点 \(i\)。
  - 成本前沿 \(F_\text{cost}\)：在二维空间 \((X, Y^\text{cost})\) 下，以同样规则筛选。
- 定义**跨图双优点集合**：
  \[
    F_\text{dual} = F_\text{perf} \cap F_\text{cost}
  \]
  即：同时位于两条前沿上的点——在性能图和成本图两个视角下都不被其他点压制。

> 直觉：
> - 若某个架构只在性能图好看、但在成本图明显被别的架构“吊打”，它不会落在 \(F_\text{dual}\) 里；
> - 只有那些在“性能 vs 用户体验”和“成本 vs 用户体验”两个平面上都属于前沿的点，才是我们要强调的“相对双优/支配点”。

### 3. 按架构归一：每个架构只保留一个“代表性双优点”

对于同一个 `arch`，\(F_\text{dual}\) 中可能出现多行（例如同一架构在 8/16/32 GPU 规模下都有双优点）。为了便于对比和输出，需要按架构做一次归一：

- 对于每个架构 \(a\)：
  1. 取出 \(F_\text{dual}\) 中所有 `arch_i = a` 的点集合 \(S_a\)。
  2. 若 \(S_a\) 为空：
     - 表示该架构在当前场景 + 成本假设下，没有跨图双优点；
     - 可以选择不输出，或仅在结果表中打上 `dual_optimal = False` 标记。
  3. 若 \(S_a\) 含有多于 1 个点：
     - **主排序指标**：根据分析目标选择：
       - 如果更关心**成本敏感场景**：主排序用 `Y^\text{cost}`（`tokens_per_10k_rmb`）降序；
       - 如果更关心**性能敏感场景**：主排序用 `Y^\text{perf}`（`tokens_per_sec_per_gpu`）降序。
     - **副排序指标**：用“另一张图的纵轴”做 tie-break（对应你提出的“如果有多个就按照另一个图的 y 轴去排序取 top1”）：
       - 成本优先时：副排序用 `Y^\text{perf}`；
       - 性能优先时：副排序用 `Y^\text{cost}`。
     - 取排序后的第一行，作为该架构在当前场景下的**代表性双优点**。

最终得到一个集合：

- \(\text{DualOptimalArchPoints} = \{(a, i_a)\}\)，每个架构至多一个代表性双优点。

### 4. 对齐并输出“吞吐最优点”（AIC 原始定义）

在跨架构分析结果里，建议同时给出：

- **AIC 吞吐最优点（baseline）**：
  - 对每个 experiment / 架构，都有一个“在 SLA 下最大化 `tokens/s/gpu_cluster`”的点；
  - 在基于 `results_summary_*.csv` 的分析里，可以直接把这一点理解为“CSV 的原始那一行”（因为这些汇总通常就是从 `best_config_topn` 择优而来）；
  - 或者在需要完全精确复现 CLI 逻辑时：
    - 从对应子目录下重新读 `best_config_topn.csv`；
    - 通过 `(isl, osl, experiment)` 等键把它 join 回汇总结果。
- **双优点（新逻辑）**：
  - 在上一步得到的 `DualOptimalArchPoints` 上打上 `dual_optimal = True` 标记；
  - 其余行 `dual_optimal = False`。

在输出表中，可以为每个架构增加两类标记列，例如：

- `is_aic_throughput_best`：该行是否对应 AIC 的吞吐最优点（通常为 1 条/架构）；
- `is_dual_optimal`：该行是否是“双优/支配点”代表；
- `dominates_arches`（可选）：列出被该点同时压制的其他架构列表，便于生成解释性文档。

---

## 集成路径建议

下面给出一条尽量“侵入性小”、但和现有工具链对齐的实现路径，并附带一条未来可以接到 CLI 的可选路径。

### 路径 A：在 `scripts/collect_results_cost_sweep.py` 上叠加双优点分析（推荐起步方案）

1. **扩展脚本输出**
   - 现有 `run_cost_scenarios` 已经会在目录下生成：
     - 若干 `results_summary_<scenario>.csv`；
     - 一个 `results_summary_<prefix>_all_scenarios.csv` 合并表。
   - 新增一个分析入口，例如逻辑函数：
     - `analyze_cross_arch_dual_optimal(base_dir: str, summary_file: str) -> Path`
   - 该函数内部：
     1. 读取 `summary_file`（可以是单场景，也可以是 `_all_scenarios`）；
     2. 按本设计文档中的步骤完成：
        - 场景/SLA/成本场景过滤；
        - 性能前沿 \(F_\text{perf}\) + 成本前沿 \(F_\text{cost}\) 计算；
        - 求交集 \(F_\text{dual}\)，再按架构做代表点选取；
        - 打标 `is_dual_optimal` / `is_aic_throughput_best` 等；
     3. 输出一个新的 CSV，例如：
        - `cross_arch_dual_optimal_<scenario>.csv`
        - 或 `results_summary_<prefix>_dual_optimal.csv`。

2. **脚本 CLI 层面的调用方式**
   - 在 `collect_results_cost_sweep.py` 的 `main` 里，增加一个可选 flag，例如：
     - `--analyze-dual-optimal`：在跑完所有成本场景后，自动对 `_all_scenarios` 做一次双优分析；
     - 或 `--dual-optimal-only`：只读取现有 `_all_scenarios` 做分析，不重复扫描实验目录。

3. **可与现有图表脚本联动**
   - 现有 `scripts/plot_cost_debug_disagg.py` 已经定义了：
     - 性能相关曲线：`tokens/s/user` vs `tokens/s/10k`；
     - 基于 `best_config_topn` 的标记点。
   - 可以在后续迭代中：
     - 读取 `cross_arch_dual_optimal_*.csv`；
     - 对其中 `is_dual_optimal=True` 的点做特殊标记（颜色/形状），在两张图上同时高亮出“跨架构支配点”。

### 路径 B：未来接入主 CLI（可选的增强）

如果希望把“双优点”分析做成用户一键可用的 CLI 入口，可以考虑：

- 在 `src/aiconfigurator/cli/main.py` 中增加一个子模式或子命令，例如：
  - `aiconfigurator cli analyze-arch` 或在 `exp` 模式下增加 `--analyze-arch-dual-optimal` 选项；
- 在实现上仍然复用 Path A 中的同一套 DataFrame 操作逻辑：
  - CLI 只是负责解析参数、找到结果目录、调用分析函数、打印/保存结果；
  - 具体的帕累托计算和 CSV 读写逻辑保持在 SDK/脚本层。

这样可以保证：

- **短期**：不动核心 CLI 代码，只在 `scripts/` 和 `documentations/` 里迭代分析逻辑；
- **中期**：等分析方法稳定后，再逐步上升为面向最终用户的 CLI 功能。

---

## 输出结果示意（概念级）

假设在某个 Medium 场景 + 基线成本假设下，我们对 A100/L40 相关的几个 experiment 做了双优分析，可能得到这样一张汇总表（字段仅示意，非真实数据）：

| experiment                | arch_label      | tokens/s/gpu | tokens/s/user | tokens/10k_rmb | ttft_ms | tpot_ms | is_aic_throughput_best | is_dual_optimal | dominates_arches           |
|--------------------------|-----------------|--------------|---------------|----------------|---------|---------|------------------------|-----------------|----------------------------|
| a100_medium_agg          | A100_AGG        | 230.8        | 50.2          | 527.5          | 785.0   | 19.9    | ✅                      | ❌               | -                          |
| a100_medium_disagg       | A100_DISAGG     | 194.0        | 52.6          | 443.5          | 579.7   | 19.0    | ✅                      | ❌               | -                          |
| l40_medium_disagg        | L40_DISAGG      | 70.0         | 25.3          | 248.7          | 925.6   | 39.5    | ✅                      | ❌               | -                          |
| l40_p_a100_d_medium_disagg | L40P_A100D   | 280.1        | 25.8          | 689.4          | 925.6   | 38.8    | ✅                      | ✅               | A100_AGG, A100_DISAGG,... |

在这个示意中：

- `l40_p_a100_d_medium_disagg`：
  - 在性能图上：`tokens/s/gpu` 高于基线 A100 方案；
  - 在成本图上：`tokens/10k_rmb` 也高于基线；
  - 因此在两张图上都位于帕累托前沿，成为一个典型的“跨架构双优/支配点”，`is_dual_optimal=True`。
- `a100_medium_agg` 等方案：
  - 可能在性能或用户体验上有自身优势，但在成本维度上被 `L40_P_A100_D` 明显压制；
  - 因此不会同时落在两个前沿集合的交集里。

后续在文档/可视化中，就可以直接引用这张表，自动生成类似“表 3.4：跨场景性能对比”的结论，而不需要人工肉眼比对两张图。

---

## 小结

- **保持不变**：`best_config_topn.csv` 仍然是“在 SLA 下最大化集群吞吐”的吞吐优先最优点定义。
- **新增逻辑**：在 cost sweep 汇总表（`results_summary_*.csv`）上，构建性能图和成本图各自的帕累托前沿，并取其交集得到“跨图双优/支配点”。
- **输出形态**：以新的 CSV（如 `cross_arch_dual_optimal_*.csv`）的形式，给出每个架构在当前场景下的代表性双优点，并显式对照 AIC 原始的吞吐最优点。
- **集成路径**：优先在 `scripts/collect_results_cost_sweep.py` 叠加分析逻辑；稳定后再考虑接入 CLI 主入口。

