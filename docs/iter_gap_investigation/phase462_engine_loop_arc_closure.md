# Phase462 Step 2c-11 引擎环弧线收口

结论：默认路径只保留 vLLM 0.19 每个 BlockPool 预留 1 个 null block 的容量语义；引擎环核心状态机 parked 且默认关闭，到达/暴露建模剔出默认路径。`null-only` 六点结果定为新的官方基线，计分板保持 3/6，`Default AIC=No-Go`。

## 官方基线

| 场景 | real tok/s/GPU | sim tok/s/GPU | error | 15% gate |
|---|---:|---:|---:|---|
| K2.5-tp8ep8-8k2k | 146.639701 | 165.984493 | 1.131921 | PASS |
| K2.5-tp8ep8-32k3k | 50.111595 | 49.502685 | 1.012301 | PASS |
| K2.5-tp4ep8dp2-8k2k | 162.948516 | 154.048354 | 1.057775 | PASS |
| K2.5-tp4ep8dp2-32k3k | 53.277429 | 63.706816 | 1.195756 | FAIL |
| K2.5-tp8ep8-8k2k-bt65536 | 124.016988 | 155.869852 | 1.256843 | FAIL |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 109.499911 | 131.404059 | 1.200038 | FAIL |

通过集合固定为 TP8-8k2k、DP2-8k2k、TP8-32k3k；动态主线目标固定为 DP2-32k3k、TP8-bt65536、DP2-bt65536。

## 产物定位

| 产物 | 收口定位 | 默认路径 |
|---|---|---|
| null block 容量修复 | 源码正确的 vLLM 0.19 容量语义；其暴露的残差移交动态主线 | 保留 |
| EngineCore 引擎环状态机 | oracle 取证成立，但未在六场景证明可迁移收益 | parked / off |
| tokenizer 到达与 workload 暴露 | client、HTTP、api-server、harness 特定 | 剔出 |
| 三点边界日志 | scheduler 入参、结果、future 完成的 `diagnostic-only` 资产 | 不参与预测 |
| first-divergence 分析器族 | 容量、时序、抢占首分叉定位的 `diagnostic-only` 资产 | 不参与预测 |
| 释放链回溯与门判卷器 | 事件级因果审计的 `diagnostic-only` 资产 | 不参与默认验收 |

本弧线的交付收益是 null block 修复；事件级工具链保留为诊断资产，不再以默认建模能力计分。

## 移交

1. 版本语义 profile v1 只收纳源码可核的 backend-version 语义，必须保持默认六点输出逐字节不变。
2. DP Step 3 在本表的 null-only 基线上重做 dp2-bt65536 相位/成本分解，并复查 Phase461 sim-only 大 bucket。
3. Step 4 先解释 dp2-32k3k 从 1.173318 到 1.195756 的暴露面，再做六点验收。

边界：未改 gate 或 PerfDB；未跑 GPU；6/6 前不收紧门限、不翻转 Default AIC。
