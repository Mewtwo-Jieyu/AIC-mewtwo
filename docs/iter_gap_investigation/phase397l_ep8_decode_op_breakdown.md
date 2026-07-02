# Phase397l — vLLM 0.19 ep8 decode 逐算子分解（仅测量）

> **范围**：在 H200 0.19 节点上用 vLLM 原生 torch profiler 对 ep8 稳态 decode 做一次
> 逐算子墙钟 + 重叠结构实测，为后续"用实测通信表替换 `ep8_per_iteration_overhead_ms=90`"
> 和"重建 overlap 模型"提供 ground truth。
> **本 phase 只测量 + 出报告，不改任何代码 / DB / gate 常量。**

## 1. 结论速览（TL;DR）

phase397k 已证 cb_sim 对 0.19 decode 有 ~5x 低估（吞吐），等价于 tpot 被高估 ~5x。本 phase
用实测把这 5x 拆清楚，结论与 phase397k 的"魔法数假说"**部分吻合、部分推翻**：

1. **`ep8_per_iteration_overhead_ms = 90.0` 是纯拟合数，没有物理对应**。实测 tp8ep8-8k2k
   稳态 decode 单迭代 **墙钟 36.85 ms**，其中**全部通信只有 6.18 ms/iter（15.7%）**，
   且是 **all-reduce**（`multimem_all_reduce`）——**根本没有 all2all dispatch/combine 算子**。
   90 ms 是真实通信的 **14.6x**、是整个迭代的 **2.4x**。它在 cb_sim 的 decode 预测里占了 **51%**。

2. **decode 基本是串行的，不是"真实系统靠重叠跑快了"**。实测 GPU-busy 39.39 ms ≈ 墙钟 36.85 ms
   （overlap 比 **1.07**，只重叠了 ~7%）。所以 phase397k 里"cb_sim 全串行求和、真实靠 overlap 藏通信"
   的假说**不成立**——真实系统本身也几乎全串行。

3. **即便把 90 ms 清零，cb_sim 串行和仍高估 2.36x，且几乎全在 `generation_moe`**：cb_sim 给
   `generation_moe = 48.9 ms`，实测 MoE 只有 **9.5 ms（5.13x 高估）**；`generation_attention = 31.9 ms`
   vs 实测 16.9 ms（1.89x）。**MoE 建模是最大单点误差（~39 ms 超算），比 90 ms 之外的所有误差都大。**

一句话：**5x = 90 ms 假通信（51%）＋ MoE 表 5x 超算（~39 ms）＋ attention 1.9x 超算（~15 ms）**，
与"重叠"无关。下一步应先修 `generation_moe`，再拿实测 6.2 ms all-reduce 替换 90 ms，而不是再拟合一个数。

## 2. 配置矩阵与方法

| 点 | 拓扑 | ISL/OSL | max_bt | 并发 | util | 实测 decode bs |
|---|---|---|---|---|---|---|
| 主 | `tp8ep8-8k2k` | 8000 / 2000 | 8000 | 128 | 0.80 | 128 |
| DP 对照 | `tp4dp2ep8-8k2k` | 8000 / 2000 | 8000 | 128 | 0.75 | 69（每 DP 副本 ~64） |

- 模型 `moonshotai/Kimi-K2.5`（vLLM 0.19.0，H200×8，driver 570.133.20）。
- **采集**：[collector/vllm/run_phase397l_decode_profile.sh](../../collector/vllm/run_phase397l_decode_profile.sh)。
  起 serve 时挂 `--profiler-config`（`profiler=torch`、`torch_profiler_record_shapes=true`、
  `warmup_iterations=3`、`active_iterations=25`），benchmark 起跑 45 s 进入稳态 decode 后
  `POST /start_profile` → 抓 25 个 decode step → `POST /stop_profile`；同时开 `NCCL_DEBUG=INFO`
  `NCCL_DEBUG_SUBSYS=COLL,INIT`。产物写节点本地 `/tmp` 再拷回。
- **逐算子分解**：[scripts/analyze_phase397l_op_breakdown.py](../../scripts/analyze_phase397l_op_breakdown.py)
  解析每 rank 的 `profiler_out_<rank>.txt`（torch `key_averages()` 表），按 self-CUDA 归类；
  用 `ProfilerStep*` 的 call 数（25）归一到每迭代。
- **DB 对齐**：[scripts/analyze_phase397l_db_align.py](../../scripts/analyze_phase397l_db_align.py)
  跑 cb_sim `run_static(mode="static_gen", bs=128, isl=9000)`，dump `generation_latency_dict`，
  与实测逐类对比。
- **窗口自洽性**：profiler 顶层 marker 为 `execute_context_0(0)_generation_128(128)`
  （0 prefill token + 128 decode token = **纯 decode step**），确认窗口落在稳态 decode。

## 3. 实测逐算子分解（每 decode 迭代）

### 3.1 tp8ep8-8k2k（主，bs=128，kv≈9000）

来源：[K2.5-tp8ep8-8k2k/op_breakdown.csv](phase397l_decode_profile/K2.5-tp8ep8-8k2k/op_breakdown.csv)

| 类别 | ms/iter | % GPU-busy | 说明 |
|---|---|---|---|
| attention | 16.90 | 42.9% | flash MLA + concat_and_cache_mla + scheduler_metadata |
| moe_expert_gemm | 8.02 | 20.4% | `marlin_moe_wna16`（专家 grouped GEMM，wna16 量化） |
| **allreduce** | **5.89** | **15.0%** | **`multimem_all_reduce`（TP/EP combine，vLLM 自定义对称内存 kernel，非 NCCL）** |
| proj_gemm | 4.04 | 10.3% | `nvjet_*` cublas（QKV/O/router/shared 投影） |
| moe_aux | 1.51 | 3.8% | grouped_topk / moe_align / count_and_sort / act_and_mul |
| norm/elementwise | 1.12 | 2.8% | RMSNorm / add / silu triton |
| allgather | 0.29 | 0.7% | logits/采样相关 |
| all2all | **0.00** | **0.0%** | **无任何 all2all/dispatch/combine 算子** |
| other | 1.62 | 4.1% | reduce/memcpy/argmax 等 |
| **合计 GPU-busy** | **39.39** | 100% | |
| decode 墙钟（ProfilerStep cpu-total 均值） | **36.85** | | |
| **overlap = GPU-busy / 墙钟** | **1.07** | | 只重叠 ~7%，近乎全串行 |
| **通信合计（allreduce+allgather+all2all）** | **6.18** | **15.7%** | |

### 3.2 tp4dp2ep8-8k2k（DP 对照，bs=69/副本，kv≈9000）

来源：[K2.5-tp4ep8dp2-8k2k/op_breakdown.csv](phase397l_decode_profile/K2.5-tp4ep8dp2-8k2k/op_breakdown.csv)

| 类别 | ms/iter | % GPU-busy |
|---|---|---|
| moe_expert_gemm | 14.16 | 28.6% |
| attention | 13.65 | 27.6% |
| proj_gemm | 6.17 | 12.5% |
| allreduce | 6.34 | 12.8% |
| allgather | 5.66 | 11.5% |
| moe_aux / norm / other | 3.45 | 7.0% |
| all2all | **0.00** | **0.0%** |
| **合计 GPU-busy** | **49.42** | 100% |
| decode 墙钟 | **46.68** | |
| overlap 比 | **1.06** | |
| 通信合计 | **12.00** | 24.3% |

- DP2 相比 tp8ep8 多出一段 **allgather 5.66 ms**（DP 副本间同步），所以通信占比更高（24.3% vs 15.7%），
  但**仍无 all2all**，通信仍是 reduce/gather 家族。
- 仍近乎全串行（overlap 1.06）。

## 4. NCCL 交叉验证：没有 all2all

原始 NCCL COLL 日志（[K2.5-tp8ep8-8k2k/nccl_logs/](phase397l_decode_profile/K2.5-tp8ep8-8k2k/nccl_logs/)）grep 结果：

- `AllToAll` / `Send:` / `Recv:` / `SendRecv`：**0 条**。
- 仅有 `AllReduce` / `AllGather` / `Broadcast`。

> 注意：decode 主通信 `multimem_all_reduce` / `cross_device_reduce_2stage` 是 vLLM 的**自定义对称内存
> allreduce kernel，不走 NCCL**，因此不出现在 NCCL 日志里——**profiler 才是通信时延的权威来源**，
> NCCL 日志只用于确认"collective 类型里没有 all2all"。
> `analyze_vllm_nccl_trace_phase39.py` 的 `OP_RE` 目前也不含 `AllToAll`（本次无需扩展，因为确实没有）。

**含义**：ep8 与 tp8 同卡共置（ep_size == tp_size == 8，单节点）时，MoE 的 combine 走的是
**all-reduce**，而不是跨节点的 all2all dispatch/combine。`ep8_per_iteration_overhead_ms=90` 想替身的
"EP all2all 通信"在这个部署形态下**不存在**。

## 5. cb_sim(0.19 DB) 逐算子 vs 实测（5x 拆解）

来源：[K2.5-tp8ep8-8k2k/db_align.csv](phase397l_decode_profile/K2.5-tp8ep8-8k2k/db_align.csv)（bs=128，kv=9000）

| 类别 | cb_sim ms | 实测 ms | sim/实测 | 判定 |
|---|---|---|---|---|
| attention | 31.89 | 16.90 | **1.89x** | DB attention 偏高 ~2x |
| **moe** | **48.88** | **9.53** | **5.13x** | **最大单点误差；~39 ms 超算** |
| proj_gemm + norm | 4.16 | 5.16 | 0.81x | 基本吻合（略低） |
| comm | 2.03 | 6.18 | 0.33x | cb_sim 把通信建成 2 ms dispatch，实测是 6.2 ms allreduce |
| **串行和** | **86.96** | 39.39（GPU-busy） | **2.21x** | |
| | | 36.85（墙钟） | **2.36x** | |
| **串行和 + 90 ms 常数** | **176.96** | 36.85（墙钟） | **4.80x** | ← 这就是那个 ~5x |

**5x 的完整归因（预测 176.96 ms，实测墙钟 36.85 ms，超出 140 ms）：**

| 来源 | 超算量 | 占超出比 |
|---|---|---|
| `ep8_per_iteration_overhead_ms = 90` 假通信 | +90.0 ms | 64% |
| `generation_moe` 5.13x 超算 | +39.4 ms | 28% |
| `generation_attention` 1.89x 超算 | +15.0 ms | 11% |
| comm 低估（allreduce 未建） | −4.2 ms | −3% |
| proj/norm 略低 | −1.0 ms | −1% |

> **关键推翻**：这 5x 里 **0% 来自"缺 overlap 建模"**（实测 overlap 只有 ~7%，cb_sim 全串行反而更接近真实结构）。
> 真正的两大来源是"90 ms 假数"和"MoE 表 5x 超算"。

tp4dp2ep8 同向但幅度略小（moe 3.33x、串行和 1.56x、+90 后 3.58x），见
[K2.5-tp4ep8dp2-8k2k/db_align.csv](phase397l_decode_profile/K2.5-tp4ep8dp2-8k2k/db_align.csv)。

## 6. all2all 时延隔离（替换 90 ms 的候选数）

- **不存在独立 all2all**。ep8-共置形态下 decode 的真实"EP/TP 通信"= `multimem_all_reduce`
  **6.18 ms/iter**（tp8ep8）/ allreduce 6.34 + allgather 5.66 = **12.0 ms/iter**（tp4dp2）。
- 因此 **替换 `ep8_per_iteration_overhead_ms=90` 的物理正确目标不是 90，而是 ~6 ms（tp8ep8）/ ~12 ms（tp4dp2）
  的 all-reduce**，且这段本应由 DB 的 `custom_allreduce_perf` 表在 attention/MoE 之间自然计入，
  而不是加一个迭代级常数。

## 7. 给后续 phase 的具体输入

- **A（最高优先，修 MoE）**：`generation_moe` 在 decode bs=128 高估 5.13x（cb_sim 48.9 ms vs 实测 8.0 ms
  专家 GEMM + 1.5 ms aux）。需查 `moe_perf` 表在该 decode shape 的查值，以及 cb_sim 对 MoE token 数的假设
  （疑似按稠密/全专家计费，未按 topk 路由后每专家实际 token 计费）。实测锚点：
  **专家 GEMM 8.02 ms + aux 1.51 ms = 9.53 ms/iter @ bs128**。
- **B（替换 90 ms）**：把 `ep8_per_iteration_overhead_ms=90` 移除，改由 DB all-reduce 表贡献。实测锚点：
  **allreduce 5.89 ms/iter（tp8ep8）**、**allreduce 6.34 + allgather 5.66 ms/iter（tp4dp2）**。
- **attention 校准**：`generation_attention` 高估 1.89x（cb_sim 31.9 ms vs 实测 16.9 ms @ bs128/kv9000）。
  次优先，需查 `generation_mla_perf` 在该 shape 的查值。
- **overlap 模型（低优先）**：实测 overlap 仅 ~7%，当前 pure-decode 分支的全串行求和（`overlap_factor` 不参与）
  已经接近真实结构，**不需要为 decode 引入激进重叠**；把上面 A/B 修对后 overlap 只需 ~1.07 的小修正。
- **C（重冻结）**：修完 A/B 后再把 `MULTI_CONFIG_DATA` 基线与 acceptance gate 冻结到 0.19。

## 8. 自洽性检查

- profiler 顶层 marker = 纯 decode step（`context_0(0)_generation_128`），窗口不含 prefill/warmup。✅
- NCCL 分类无 all2all；出现的是 allreduce/allgather，与 profiler 一致。✅
- 逐算子 self-CUDA 求和（39.39 ms）≈ 实测单迭代墙钟（36.85 ms），overlap 1.07，物理自洽（GPU 近乎打满）。✅

## 9. 产物与变更边界

- 新增采集脚本：`collector/vllm/run_phase397l_decode_profile.sh`
- 新增分析脚本：`scripts/analyze_phase397l_op_breakdown.py`、`scripts/analyze_phase397l_db_align.py`
- 原始 trace / 分析产物：`docs/iter_gap_investigation/phase397l_decode_profile/`
  （每点：`prof/*.pt.trace.json.gz`（8 rank）+ `prof/profiler_out_*.txt`、`nccl_logs/`、
  `op_breakdown.csv`、`db_align.csv`、`bench_result.json`、`meta.json`、`serve.log`）
- **未改动**：`src/aiconfigurator/sdk/backends/vllm_backend.py`（90 ms 常数原样保留）、
  任何 DB 表、`scripts/validate_cb_simulator.py` 的 gate 常数；未动 `_get_cb_efficiency_factor`
  与 `_get_ttft_cb_correction`（默认路径本就不用）。
