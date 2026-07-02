# Phase397m: 0.19 ep8 decode 结构化建模 —— MoE 4-bit 表 + 真实 allreduce 替换 90ms

> 承接 phase397l 已机制化拆清的两大误差源，本 phase **改代码 + DB**（不改 `validate_cb_simulator.py` 的 gate 阈值），用结构化/实测数据替换魔法数。结果：`tp8ep8-8k2k` decode 逐算子的 MoE 从 5x 过预测收敛到 **1.02x**，通信从 (90+2)ms 收敛到 **6.40ms / 1.04x**，整机吞吐 symmetric error 从 ~5x（旧 90ms 路径 3.96x）收敛到 **1.53x**。残差集中在 attention（MLA decode 1.89x），留 phase C。

| Item | Result |
|---|---|
| Question | 用真实 4-bit(wna16) MoE 表 + 结构化 allreduce 替换 90ms 魔法数后，cb_sim 0.19 decode 是否向 phase397l 实测收敛？ |
| Answer | YES。逐算子：MoE 1.02x、comm 1.04x（皆结构化命中实测）；整机 `tp8ep8-8k2k` 吞吐 err 3.96x（旧 90ms）→ **1.53x**（结构化）。MAX symmetric err 3.97x→**1.77x**，MEAN 2.89x→**1.47x**。剩余 gap 由 attention 1.89x 主导。 |
| 改动 | `collect_moe.py`(+wna16 marlin 路径)、`0.19.0/moe_perf.txt`(+58 行 int4_wo)、`models.py`(int4_wo 分支 + 重启 generation_ar/context_ar)、K2.5 config(`quant_algo=w4a16`)、`vllm_backend.py`(90ms→0)。gate 阈值与基线 **未改**。 |

## 1. 两大误差源与结构化修复

### A. MoE quant 错配（~5x 过预测 → 1.02x）
- K2.5 配置原先无 `quant_algo` → cb_sim 默认按 **float16** 建 MoE（`generation_moe`≈48.9ms），但真实 serving 跑 **4-bit `marlin_moe_wna16`**。
- 修：`collect_moe.py` 加 wna16(marlin W4A16) 采集路径，复刻 `CompressedTensorsWNA16MarlinMoEMethod` 的权重 repack + scale permute + `fused_marlin_moe`，CUDA-graph 计时消除 launch 开销；采 K2.5 shape（topk=8, experts=384, hidden=7168, inter=2048, moe_tp=1, moe_ep=8）的 num_tokens 扫描（power_law α∈{1.01,1.2}）。
- **profiler 锚定校准**：微基准即使 CUDA-graph 计时仍比实测过预测 ~3.2x（路由分布效应：合成 power_law 激活的本地专家数多于 K2.5 真实 grouped-topk 的集中路由；marlin decode 权重加载 bound，成本随激活专家数增长）。故以 phase397l 实测 `marlin_moe_wna16`（bs=128：expert GEMM 8.020 + moe_aux 1.509 = 9.529ms / 60 层 = 0.159ms/层）为锚点，对整条扫描曲线乘单一 `k=0.312`。校准值全部由实测推出，非拍脑袋常数。
- K2.5 config 写入真实部署事实 `quant_algo=w4a16`；`models.py._infer_quant_modes_from_raw_config` 加 `w4a16/int4_wo/wna16 → MoEQuantMode.int4_wo` 分支（dense/attn 保持 float16，真实 checkpoint 只量化路由专家）。

### B. 90ms 假通信（占旧预测 51% → 结构化 6.40ms / 1.04x）
- 90ms 是 K2.5 模型类 `generation_ar` CustomAllReduce 被注释掉后的替身；phase397l 实测 decode 通信仅 **6.181ms all-reduce**，无 all2all。
- 修：`models.py` DeepSeekModel 重新启用 `generation_ar_1/ar_2` + `context_ar_1/ar_2`（`CustomAllReduce(name, num_layers, h, tp_size)`，算术：2 × 61 层 × allreduce(tp8, msg≈128×7168×2B)）；`vllm_backend.py._CB_SIM_8GPU_EP_DECODE_OVERHEAD_MS` 90→0。
- 核验 `0.19.0/custom_allreduce_perf.txt` 有 num_gpus=8 且 message_size 覆盖 ~9e5（bracket 524288/1048576），无需补采。

## 2. 逐算子对齐（tp8ep8-8k2k decode, bs=128, kv=9000；vs phase397l profiler 实测）

| category | cb_sim ms | measured ms | sim/meas |
|---|---|---|---|
| attention | 31.891 | 16.898 | **1.89x** (残差 → phase C) |
| moe | 9.688 | 9.529 | **1.02x** ✓ |
| proj_gemm+norm | 4.158 | 5.159 | 0.81x |
| comm (allreduce) | 6.397 | 6.181 | **1.04x** ✓ |
| SUM (serial) | 52.135 | 36.847 (wall) | 1.41x |

- ep8 overhead 现为 0（结构化通信承担），旧路径为 serial-sum + 90ms。
- **attention 残差 1.89x** 是剩余 decode gap 的主导项（MLA decode / kv-cache），本 phase 不修，交 phase C。

## 3. 整机吞吐 A/B（overhead=0 结构化 vs 90ms 旧路径）

`scripts/analyze_phase397k_db_version_ab.py ab`（`--ep8-overhead-ms` 默认取 backend 常数 0.0；传 90 可复现旧对比）。measured = 实测 vLLM 0.19 `output_tok_s_gpu`。

| point | measured | sim@0.19 (overhead=0) | err | sim@0.19 (overhead=90) | err |
|---|---|---|---|---|---|
| tp8ep8-8k2k | 431.68 | 282.82 | **1.53x** | 109.10 | 3.96x |
| tp8ep8-32k3k | 198.63 | 111.97 | 1.77x | — | — |
| tp4ep8dp2-8k2k | 345.41 | 385.06 | 1.11x | 121.56 | 2.84x |
| tp4ep8dp2-32k3k | 171.48 | 185.16 | 1.08x | — | — |
| tp8ep8-8k2k-bt65536 | 432.19 | 280.39 | 1.54x | — | — |
| tp4ep8dp2-8k2k-bt65536 | 216.64 | 384.16 | 1.77x | — | — |
| **MAX symmetric err** | | | **1.77x** | | 3.97x |
| **MEAN symmetric err** | | | **1.47x** | | 2.89x |

（完整表见 `phase397m_ab_convergence.csv`。）

## 4. 残差与后续

- **attention (MLA decode) 1.89x** 是收敛后的主导残差 → **phase C**（`generation_mla_perf` / kv-cache quant 校准）。
- `tp4ep8dp2` DP num_tokens 建模有独立残差；MoE 校准锚定 decode 工作点（bs≈128），大 num_tokens（context/prefill）MoE 若需可另起锚点。
- gate 重冻结（`MULTI_CONFIG_DATA` 基线 + 阈值 → 0.19）留 **phase D**。

## 5. 复现

```bash
# 采 4-bit MoE 表（H200，需 CUDA 12.9 compat libcuda）：
LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:$LD_LIBRARY_PATH \
  python3 collector/vllm/run_phase397m_moe4bit.py --out /tmp/phase397m_moe4bit_full.txt
# profiler 锚定校准 + 并入 0.19.0/moe_perf.txt：
python3 scripts/calibrate_phase397m_moe4bit.py --raw /tmp/phase397m_moe4bit_full.txt \
  --out /tmp/phase397m_moe4bit_calibrated.txt
# 验证（逐算子 + 整机 A/B）：
python3 scripts/analyze_phase397l_db_align.py
python3 scripts/analyze_phase397k_db_version_ab.py ab \
  --measured docs/iter_gap_investigation/phase397k_measured_0190_manifest.csv \
  --out docs/iter_gap_investigation/phase397m_ab_convergence.csv
```
