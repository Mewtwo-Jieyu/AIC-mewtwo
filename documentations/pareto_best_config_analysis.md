
- pareto.csv：是每个 experiment 在「集群吞吐 vs X 轴指标」(X = tokens/s/user 或 request_latency) 上的 帕累托前沿点集，是从 TaskRunner 返回的完整结果 pareto_df 上通过 get_pareto_front 算出来的。

- best_config_topn.csv：是在给定 SLA 约束（TPOT / request_latency）下，从 全部 pareto_df 点 中选出的、按「集群平均吞吐 tokens/s/gpu_cluster」排序的 Top‑N 最优配置，并按并行配置去重。

---
## Pareto曲线 + 最优配置点选择
### 1. SLA 约束是如何作用到 `best_config_topn.csv` 的？

#### 1.1 SLA 输入与内部搜索

CLI 的 TaskConfig 把用户的 SLA（ttft, tpot, request_latency）存到 runtime_config 里，传给 TaskRunner：

```python
# main.py Lines 600-616 
best_config_df, best_throughput, pareto_frontier_df, x_axis_col = process_experiment_result(task_config, task_result, top_n)
```

`TaskRunner.run_agg / run_disagg `在真正搜索时，会：
- 固定 ttft = 用户设定的 ttft
- 把 tpot 扩展成一个搜索列表（1–19ms 步长 1ms，20–295ms 步长 5ms）：

```python
# task.py Lines 952-961
runtime_config = config.RuntimeConfig(
    isl=task_config.runtime_config.isl,
    osl=task_config.runtime_config.osl,
    prefix=task_config.runtime_config.prefix,
    ttft=task_config.runtime_config.ttft,
    tpot=list(range(1, 20, 1)) + list(range(20, 300, 5)),
    request_latency=getattr(task_config.runtime_config, "request_latency", None),)
```

- 如果 用户给了 request_latency，agg_pareto / disagg_pareto 会先把 request_latency 拆成一系列 (ttft, tpot) 组合：

```python
# utils.py Lines 100-123
def enumerate_ttft_tpot_constraints(
    osl: int,
    request_latency: float,
    ttft: float | None = None,
) -> list[tuple[float, float]]:
    ...
    if ttft is None:
        ttft = request_latency * 0.95
    ...
    ttft_list = sorted([t for t in ttft_set if t < request_latency])
    return [(t, (request_latency - t) / (osl - 1)) for t in ttft_list]
```

- agg_pareto 里可以看到这一逻辑：

```python
# pareto_analysis.py Lines 75-94
if runtime_config.request_latency is not None and runtime_config.request_latency > 0:
    ttft_tpot_constraints = enumerate_ttft_tpot_constraints(
        runtime_config.osl, runtime_config.request_latency, runtime_config.ttft
    )
    ...
    for ttft_constraint, tpot_constraint in ttft_tpot_constraints:
        overwritten_runtime_config = copy.deepcopy(runtime_config)
        overwritten_runtime_config.ttft = ttft_constraint
        overwritten_runtime_config.tpot = tpot_constraint
        runtime_configs_to_evaluate.append(overwritten_runtime_config)
else:
    tpot_list = runtime_config.tpot if isinstance(runtime_config.tpot, list) else [runtime_config.tpot]
    ...
```

然后对每个 (ttft, tpot) 组合调用：

```python
# pareto_analysis.py Lines 105-113
summary = sess.find_best_agg_result_under_constraints(
    runtime_config=overwritten_runtime_config,
    top_k=10,
    max_batch_size=512,
    ctx_stride=512,
)
result_df = summary.get_summary_df()
```

> 这一步里，TTFT/TPOT/request_latency 的硬性约束已经在底层 InferenceSession 里被用来过滤不可行配置，因此写到 CSV 里的点本身就满足（或接近）这些约束。

#### 1.2 process_experiment_result：在 SLA 下选出 top‑N

CLI 真正决定 best_config_topn.csv 的地方在 process_experiment_result：

```python
# utils.pyLines 32-75
pareto_df = result["pareto_df"]
runtime_cfg = task_config.config.runtime_config
target_tpot = runtime_cfg.tpot
target_request_latency = runtime_cfg.request_latency
use_request_latency = target_request_latency is not None and target_request_latency > 0
total_gpus = getattr(task_config, "total_gpus", None) or 0

# 先算集群层面的 tokens/s/gpu_cluster
if pareto_df is not None and not pareto_df.empty:
    pareto_df["tokens/s/gpu_cluster"] = (
        pareto_df["tokens/s/gpu"]
        * (total_gpus // pareto_df["num_total_gpus"])
        * pareto_df["num_total_gpus"]
        / total_gpus
    )
    x_axis_col = "request_latency" if use_request_latency else "tokens/s/user"
    pareto_frontier_df = get_pareto_front(
        pareto_df,
        x_axis_col,
        "tokens/s/gpu_cluster",
        maximize_x=not use_request_latency,
        maximize_y=True,
    )
else:
    pareto_frontier_df = pd.DataFrame()
    x_axis_col = "request_latency" if use_request_latency else "tokens/s/user"
group_by_key = "(d)parallel" if task_config.serving_mode == "disagg" else "parallel"

if use_request_latency:
    best_config_df = get_best_configs_under_request_latency_constraint(
        total_gpus=total_gpus,
        pareto_df=pareto_df,
        target_request_latency=target_request_latency,
        top_n=top_n,
        group_by=group_by_key,
    )
else:
    best_config_df = get_best_configs_under_tpot_constraint(
        total_gpus=total_gpus,
        pareto_df=pareto_df,
        target_tpot=target_tpot,
        top_n=top_n,
        group_by=group_by_key,
    )
```

要点：
- 两个层级的 SLA：
	- 底层 InferenceSession 已经基于 (ttft, tpot, request_latency) 只返回“可行”的候选配置。
	- 这里在可行集合上 再加一层约束：
		- 若给了 request_latency 且 >0：只保留 request_latency <= target_request_latency 的点。
		- 否则：只保留 tpot <= target_tpot 的点。
- 目标函数：在剩余候选里，最大化 tokens/s/gpu_cluster（集群平均每 GPU 吞吐）。
- 并行配置去重：
	- group_by_key = "parallel"（agg）或 "(d)parallel"（disagg），保证每种并行配置只留一个最优点。

具体选优逻辑在 _get_best_configs_under_constraint：

```python
# pareto_analysis.pyLines 437-488
def _get_best_configs_under_constraint(
    total_gpus: int,
    pareto_df: pd.DataFrame,
    target_value: float,
    constraint_col: str,
    top_n: int = 1,
    group_by: str | None = None,
    *,
    secondary_sort_col: str | None = None,
    secondary_sort_ascending: bool = False,
) -> pd.DataFrame:
    ...
    candidate_configs = pareto_df[pareto_df[constraint_col] <= target_value].copy()
    ...
    if not candidate_configs.empty:
        # 计算集群级 tokens/s/gpu
        candidate_configs["tokens/s/gpu_cluster"] = (
            candidate_configs["tokens/s/gpu"]
            * (total_gpus // candidate_configs["num_total_gpus"])
            * candidate_configs["num_total_gpus"]
            / total_gpus
        )
        if group_by is not None:
            top_indexes = candidate_configs.groupby(group_by)["tokens/s/gpu_cluster"].idxmax()
            candidate_configs = candidate_configs.loc[top_indexes]
        sort_columns = ["tokens/s/gpu_cluster"]
        sort_ascending = [False]
       if secondary_sort_col and secondary_sort_col in candidate_configs.columns:
            sort_columns.append(secondary_sort_col)
            sort_ascending.append(secondary_sort_ascending)
        candidate_configs = (
            candidate_configs.sort_values(by=sort_columns, ascending=sort_ascending).head(top_n).reset_index(drop=True)
        )
        return candidate_configs
```

两个包装函数定义了具体的 SLA 语义：

```python
# pareto_analysis.pyLines 497-514
def get_best_configs_under_tpot_constraint(...):
    return _get_best_configs_under_constraint(
        ...,
        constraint_col="tpot",
        ...,
        secondary_sort_col="tokens/s/user",   # 同吞吐下更偏好 per‑user 吞吐大的
        secondary_sort_ascending=False,
    )
```

```python
# 517:534:src/aiconfigurator/sdk/pareto_analysis.py
def get_best_configs_under_request_latency_constraint(...):
    return _get_best_configs_under_constraint(
        ...,
        constraint_col="request_latency",
        ...,
        secondary_sort_col="request_latency",  # 同吞吐下偏好延迟更低
        secondary_sort_ascending=True,
    )
```

总结：
- 设定的 ttft / tpot / request_latency：
	- 先在底层决定“有哪些配置点会被评估并写进 pareto_df”；
	- 然后在 CLI 顶层再做一次「constraint_col <= 目标值」的过滤，并在每个并行拓扑下选出集群吞吐最高的配置，按吞吐排序取前 top_n 行，这就是 best_config_topn.csv。

---
### 2. pareto.csv 是如何得到的？从中如何筛选“最优配置点”？

#### 2.1 pareto.csv 的来源

TaskRunner.run_* 返回的 result["pareto_df"] 是 所有可行配置 的结果表。process_experiment_result 在上面先算出了 tokens/s/gpu_cluster，然后调用 get_pareto_front 得到帕累托前沿：

```python
# utils.pyLines 47-54
x_axis_col = "request_latency" if use_request_latency else "tokens/s/user"
pareto_frontier_df = get_pareto_front(
    pareto_df,
    x_axis_col,
    "tokens/s/gpu_cluster",
    maximize_x=not use_request_latency,
    maximize_y=True,
)
```

get_pareto_front 的核心逻辑：

```python
# pareto_analysis.pyLines 268-311
def get_pareto_front(df, x_col, y_col, *, maximize_x=True, maximize_y=True):
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.sort_values(by=x_col)
    def is_pareto(costs: np.ndarray) -> np.ndarray:
        is_better = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_better[i]:
                # 移除被当前点支配的点
                is_better[is_better] = np.any(costs[is_better] > c, axis=1)
                is_better[i] = True
        return is_better
    working = df[[x_col, y_col]].copy()
    if not maximize_x:
        working[x_col] = -working[x_col]
    if not maximize_y:
        working[y_col] = -working[y_col]
    costs = working[[x_col, y_col]].values
    is_pareto_front = is_pareto(costs)
    pareto_front = df[is_pareto_front]
    return pareto_front.sort_values(by=x_col).reset_index(drop=True)
```

- 当 X 轴是 tokens/s/user（越大越好）时：maximize_x=True，得到「在 tokens/s/user–tokens/s/gpu_cluster 平面上不被任何其他点同时压制」的前沿。
- 当 X 轴是 request_latency（越小越好）时：先对 X 取负，把「小更好」转换成「大更好」，再做同样的 Pareto 过滤。

随后，在 save_results 里，这个 pareto_fronts[exp_name] 被直接写成对应 experiment 目录下的 pareto.csv：

```python
# report_and_save.pyLines 529-541
for exp_name, pareto_df in pareto_fronts.items():
    exp_dir = os.path.join(safe_result_dir, exp_name)
    safe_mkdir(exp_dir, exist_ok=True)
    # 1. best_config_topn.csv
    best_config_df = best_configs.get(exp_name)
    if best_config_df is not None:
        best_config_df.to_csv(os.path.join(exp_dir, "best_config_topn.csv"), index=False)
    # 2. pareto.csv = 已经是 Pareto 前沿的 DataFrame
    if pareto_df is not None:
        pareto_df.to_csv(os.path.join(exp_dir, "pareto.csv"), index=False)
```

因此：
- pareto.csv 不是“所有点”，而是已经做过一次帕累托筛选后的结果（在 tokens/s/gpu_cluster vs X 轴空间）。

#### 2.2 从 pareto.csv 再筛选「最优配置」

CLI 内部选 best_config_topn 用的是==原始 pareto_df==，而不是已经压缩过的 pareto_frontier_df（参见上面 process_experiment_result，传给 get_best_configs_* 的是 pareto_df）。

这意味着：
- 从 理论上，best_config_topn.csv 中的点不一定都是 pareto.csv 上的点（虽然通常会高度重合），因为：
	- 帕累托前沿是针对 (X, tokens/s/gpu_cluster) 二维空间；
	- top‑N 选取是额外加上 SLA 约束 + 并行配置去重之后，再按 tokens/s/gpu_cluster 排序挑出来的。
- 但在实际使用中，已经通过 InferenceSession + SLA 预筛选，再在 pareto_df 上筛一次，最终 top‑N 点一般都在直观意义上的“好区域”。

如果只拿到 pareto.csv 而没有原始 pareto_df，想“复现” CLI 的筛选逻辑，可以参考：
1. 从 pareto.csv 读入为 df。
2. 计算（如果缺的话）tokens/s/gpu_cluster：

```python
       df["tokens/s/gpu_cluster"] = ( 
           df["tokens/s/gpu"] 
           * (total_gpus // df["num_total_gpus"])  
           * df["num_total_gpus"]   
           / total_gpus  
       ) 
```

3. 按需过滤：
	- 如果使用 TPOT SLA：df = df[df["tpot"] <= target_tpot]
	- 如果使用 request latency SLA：df = df[df["request_latency"] <= target_request_latency]
4. 按并行配置去重（如果你有 parallel / (d)parallel 列，可以模仿 CLI）：

```python
       group_col = "(d)parallel" if is_disagg else "parallel"
       idx = df.groupby(group_col)["tokens/s/gpu_cluster"].idxmax()
       df = df.loc[idx]
```

5. 最后排序 + 选前 N：
```python
       sort_cols = ["tokens/s/gpu_cluster"]
       asc = [False]
       if use_request_latency:
           sort_cols.append("request_latency")
           asc.append(True)
       else:
           sort_cols.append("tokens/s/user")
           asc.append(False)
       best_config_topn = df.sort_values(sort_cols, ascending=asc).head(top_n) 
```

这基本就复刻了 _get_best_configs_under_constraint + 包装函数的行为。

---
### 3. 可能遇到输出为空的情况

- 当用户给的 `ttft/ tpot/ request_latency` 太紧，找不到配置能同时满足这些约束。
- 在 `TaskRunner.run_agg/run_disagg `里，底层的`InferenceSession.find_best_*_under_constraints` 会把不符合的组合过滤掉。
- 如果所有 (ttft,tpot) 组合都被删光了，返回给上层的 result["pareto_df"] 要么是 None，要么是一个空的 DataFrame。
- `process_experiment_result `对这个 `pareto_df` 的第一道检查就是：

```python
if pareto_df is not None and not pareto_df.empty:
    pareto_df["tokens/s/gpu_cluster"] = …
    pareto_frontier_df = get_pareto_front(…)
else:
    pareto_frontier_df = pd.DataFrame()
    x_axis_col = …
```

- 在选 `best_config_topn` 的那一层，约束过滤写得很简单：如果 candidate_configs 是空的，函数根本不会 return 任何东西，隐式返回 None

```python
candidate_configs = pareto_df[pareto_df[constraint_col] <= target_value].copy()
…
if not candidate_configs.empty:
    …  # 计算 tokens/s/gpu_cluster、按 parallel 分组、排序、取 top_n
    return candidate_configs
```

- `get_best_configs_under_tpot_constraint`/ `…_request_latency… `也都是这么封装的。因此可能得到 None

```python
best_config_df = get_best_configs_under_…
```

- 最后写文件的时候，`report_and_save.py` 只在 `best_config_df is not None` 时才生成 best_config_topn.csv：

```python
best_config_df = best_configs.get(exp_name)
if best_config_df is not None:
    best_config_df.to_csv(...)
```

- 所以当约束过严时:
	- pareto.csv 可能是空表（或者根本没有这个文件），
	- best_config_topn.csv 根本不会被写出，或者写的是一个仅含表头的空文件；
	- 命令行输出/日志里通常会有 “no valid configuration under constraints” 之类的提示。

- 同样的逻辑也适用于 request_latency 的枚举分解：如果对某个 (ttft,tpot) 组合，没有任何模型设定能在给定 osl/isl 下满足 request_latency，该组合对应的 summary 就会空，最终整个 runtime_configs_to_evaluate 迭代结束后仍然可能没有有效行，pareto_df 依旧为空。

#### 3.1 解决办法

- 放宽 SLA：增大 ttft、放松 tpot、或者直接不要指定request_latency，让搜索空间更大一些
- 检查 total_gpus、isl、osl 等系统参数，确认它们和模型/后端组合在逻辑上是可行的。
- 在调试时可以在运行结果目录下手动查看 pareto_df/pareto.csv内容，确认是不是根本没有任何行。

---
## 示例
### 目录整体

- 模型：Qwen/Qwen3-32B
- 系统：a100_sxm, l40s
- backend：trtllm
- SLA：
- isl=2048
- osl=512
- ttft=800ms
- tpot=20ms
- 没有显式 request_latency（默认 0）

在这下面有 6 个子目录：a100_medium_agg / a100_medium_disagg / l40_medium_agg / l40_medium_disagg / a100_p_l40_d_medium_disagg / l40_p_a100_d_medium_disagg，每个都是一个 experiment，对应自己的 pareto.csv 和 best_config_topn.csv。

下面我用你最关心的 a100_medium_disagg 这个子目录做“从 pareto.csv 到 best_config_topn.csv”的具体示例。

---

### 1. 这个 experiment 的 pareto.csv 里是什么？

a100_medium_disagg/pareto.csv 的每一行是一个 disagg 配置点，比如第一行（字段很多，这里只看关键几列）：

```csv
model,isl,osl,...,concurrency,...,ttft,tpot,request_latency,...,tokens/s/gpu,tokens/s/user,...,num_total_gpus,...,(p)parallel,...,(d)parallel,...,power_w,tokens/s/gpu_cluster

Qwen/Qwen3-32B,2048,512,...,384,...,579.748,53.907,28126.225,...,0.802,18.551,...,16,...,(p)parallel=tp2pp1dp1etp1ep1,...,(d)parallel=tp2pp1dp1etp1ep1,...,power_w=0.0, tokens/s/gpu_cluster=410.423
...
```

要点：
- 每一行是某个 (p)parallel / (d)parallel + workers + bs + concurrency 的 完整配置；
- ttft / tpot / request_latency 是 预测出来的响应时延；
- tokens/s/gpu、tokens/s/user、tokens/s/gpu_cluster 是不同层级的吞吐：
- tokens/s/gpu：单 GPU 平均吞吐；
- tokens/s/user：单用户（一个 request）平均吞吐；
- tokens/s/gpu_cluster：把这个配置复制铺满你 CLI 里给的 总 GPU 数 total_gpus 后，折算成“集群平均每 GPU 吞吐”。它由 CLI 在 process_experiment_result 里根据公式算出来，然后一起写进 CSV。

pareto.csv 中这些行本身已经是 在 (X, tokens/s/gpu_cluster) 平面上的帕累托前沿点（X = tokens/s/user，因为这里没有 request_latency SLA），是从内部更大的候选集合继续压缩过一轮的结果。

---
### 2. CLI 怎么在这个目录里选出 best_config_topn.csv？

同一个子目录下的 best_config_topn.csv（disagg）现在只有两行：

```csv
model,isl,osl,...,concurrency,...,ttft,tpot,request_latency,...,tokens/s/gpu,tokens/s/user,...,num_total_gpus,...,(p)parallel,...,(d)parallel,...,power_w,tokens/s/gpu_cluster

Qwen/Qwen3-32B,2048,512,...,64,...,579.748,19.001,10289.259,...,0.379,52.628,...,16,...,(p)parallel=tp2pp1dp1etp1ep1,...,(d)parallel=tp8pp1dp1etp1ep1,...,0.0,194.039

Qwen/Qwen3-32B,2048,512,...,72,...,579.748,18.979,10278.017,...,0.349,52.689,...,16,...,(p)parallel=tp2pp1dp1etp1ep1,...,(d)parallel=tp4pp1dp1etp1ep1,...,0.0,178.848
```

结合我们前面读的 CLI 代码，你可以把从 pareto.csv 到这两行的过程理解为：
1. 取这个 experiment 的完整结果 DataFrame：pareto_df = result["pareto_df"]。
2. 计算集群指标 tokens/s/gpu_cluster（在你看到的 CSV 里这已经算好的，最后一列）：
	- 这里的 total_gpus 就是你在 CLI 里传的集群规模（通常是成百上千），决定“这个配置复制铺满集群后”的集群平均每卡吞吐。

```python
# utils.pyLines 40-46
   pareto_df["tokens/s/gpu_cluster"] = (
       pareto_df["tokens/s/gpu"]
       * (total_gpus // pareto_df["num_total_gpus"])
       * pareto_df["num_total_gpus"]
       / total_gpus
   )
```

3. 因为这个任务没有设 request_latency（只有 tpot=20）CLI 在 process_experiment_result 里会走 TPOT 分支：

   ```python
   # utils.pyLines 60-75
   use_request_latency = target_request_latency is not None and target_request_latency > 0
   ...
   if use_request_latency:
       ...  # 走 request_latency 约束
   else:
       best_config_df = get_best_configs_under_tpot_constraint(
           total_gpus=total_gpus,
           pareto_df=pareto_df,
           target_tpot=target_tpot,   # 这里就是目录名里的 tpot=20
           top_n=top_n,
           group_by=group_by_key,     # disagg: "(d)parallel"
       )
   ```

4. `get_best_configs_under_tpot_constraint `具体做了三件事（在这个子目录就对应你看到的那两行）：

```python
# pareto_analysis.pyLines 497-514
   def get_best_configs_under_tpot_constraint(...):
       return _get_best_configs_under_constraint(
           constraint_col="tpot",
           secondary_sort_col="tokens/s/user",
           secondary_sort_ascending=False,
       )
```

展开 `_get_best_configs_under_constraint` ：
- 先用 SLA 过滤：tpot <= 20
在 a100_medium_disagg/pareto.csv 里：
- 前几行的 tpot 是 53.9ms、49.3ms、29.7ms、24.97ms，都被 SLA 过滤掉；
- 从第 6 行开始（tpot=19.001ms、18.979ms、17ms … 9.9ms）才满足 tpot <= 20，成为候选。
- 对每种 (d)parallel 只留一个最佳点：

```python
# pareto_analysis.pyLines 477-479
     if group_by is not None:
         top_indexes = candidate_configs.groupby(group_by)["tokens/s/gpu_cluster"].idxmax()
         candidate_configs = candidate_configs.loc[top_indexes]
```

在这个子目录里，可以看到：
- 一类是 "(d)parallel" = tp8pp1dp1etp1ep1；
- 一类是 "(d)parallel" = tp4pp1dp1etp1ep1；
- 每一类中，tokens/s/gpu_cluster 最大的那一行，就分别成为 best_config_topn.csv 的第 1、2 行。
- 在所有候选并行配置之间排序取 Top‑N：
	- 主排序键：tokens/s/gpu_cluster（降序）——越高越好；
	- 次排序键：tokens/s/user（降序）——在同吞吐下，优先每用户吞吐更高的。

在 a100_medium_disagg/best_config_topn.csv 里：
- 第 1 行：
	- tpot=19.001ms（满足 SLA）
	- (d)parallel=tp8pp1dp1etp1ep1
	- concurrency=64，tokens/s/gpu_cluster=194.039 ——在所有 tp8 decode 并行中是吞吐最高的，也是所有并行配置中吞吐最高的；
- 第 2 行：
	- tpot=18.979ms（满足 SLA）
	- (d)parallel=tp4pp1dp1etp1ep1
	- concurrency=72，tokens/s/gpu_cluster=178.848 ——在 tp4 decode 并行里是吞吐最高的，在全局排序里排第二。

---
### 3. 其他子目录（a100_agg / l40 / hetero）的含义是一样的

在同一个 a100_l40_medium 结果根下的其他 5 个子目录：
- a100_medium_agg / l40_medium_agg
- a100_medium_disagg / l40_medium_disagg
- a100_p_l40_d_medium_disagg / l40_p_a100_d_medium_disagg

它们的 pareto.csv 和 best_config_topn.csv 都是用同一套逻辑生成的，只是：
- system / (p)/(d)system 不同（A100 / L40 或 hetero 混搭）；
- 某些目录是 agg（只看 parallel），有些是 disagg（按 (d)parallel 分组）；
- SLA 参数仍然来自这一整次 run 的根目录名（ttft=800,tpot=20），
- 若将来你通过 CLI 传 --request_latency，那么 x_axis 会切到 request_latency，并走 get_best_configs_under_request_latency_constraint 分支，其他逻辑完全类似。

