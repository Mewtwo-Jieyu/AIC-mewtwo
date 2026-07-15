# Phase462 Step 2c-10 引擎环倒退分诊

结论：因果拆分成立，但默认采用条件不成立：三个 TP 场景 core-only 与 full 完全一致，arrival/exposure 贡献为零；核心状态机仍把组合计分板从 3/6 拉到 2/6。引擎环继续关闭，只保留 null block。

Step 2c-9 的“A/B 唯一差异是引擎环”审计失败：Phase461 baseline 产物来自 `01b88c72`，早于 null block `55073d9d`；candidate TP 是 `null block + engine loop`，candidate DP 是 `null block only`。因此五个倒退不能统一归因到 tokenizer 原语。

## 对等审计

| 项目 | 结果 |
|---|---|
| only engine-loop difference | `false` |
| status | `fail_confounded` |
| confounds | `["null_block_semantics"]` |
| 同 HEAD 重建 | `null_only / core_only / full`；core-only 用即时暴露，无时间常数、无拟合参数 |

## 拆分裁决

预注册定义：“保留大部分改善”=`core improvement / full improvement > 0.5`。核心半边包含 null block 与 EngineCore 状态机；这是多数定义，不按输出调阈值。

| baseline | null-only | core-only | full | null 改善 | core 增量 | arrival 增量 | 核心占比 | 裁决 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1.103595 | 1.012301 | 1.005709 | 1.005709 | +0.091295 | +0.006591 | +0.000000 | 100.00% | split_supported |

三个 TP 场景 `core-only == full`：`true`。这直接证伪 Step 2c-9 的“测量原语未迁移”解释；TP8-8k2k 与 TP8-bt65536 的额外倒退来自核心状态机，三个 DP 变化来自 null block。

## 六点分解

`null_only`、`core_only`、`full` 均在当前 HEAD 上离线重跑；DP 引擎环仍显式拒绝，所以 DP 只有 `null_only`。

| 场景 | Phase461 error | null-only | core-only | full | 机制判定 |
|---|---:|---:|---:|---:|---|
| K2.5-tp8ep8-8k2k | 1.132524 | 1.131921 | 1.168856 | 1.168856 | null=-0.000603; core=+0.036936; arrival=+0.000000 |
| K2.5-tp8ep8-32k3k | 1.103595 | 1.012301 | 1.005709 | 1.005709 | null=-0.091295; core=-0.006591; arrival=+0.000000 |
| K2.5-tp4ep8dp2-8k2k | 1.041312 | 1.057775 | - | - | null block delta=+0.016463; Step2c-9 did not run the engine loop on DP |
| K2.5-tp4ep8dp2-32k3k | 1.173318 | 1.195756 | - | - | null block delta=+0.022438; Step2c-9 did not run the engine loop on DP |
| K2.5-tp8ep8-8k2k-bt65536 | 1.234521 | 1.256843 | 1.318934 | 1.318934 | null=+0.022321; core=+0.062091; arrival=+0.000000 |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 1.184544 | 1.200038 | - | - | null block delta=+0.015494; Step2c-9 did not run the engine loop on DP |

## 动态签名

| 场景 | 变体 | iterations | preempt/self/repeat | recompute tokens | mixed steps | avg/peak decode | avg/peak tokens |
|---|---|---:|---:|---:|---:|---:|---:|
| K2.5-tp8ep8-8k2k | null_only | 14065 | 148/12/97 | 1244091 | 640 (4.5503%) | 54.576/67 | 361.208/8000 |
| K2.5-tp8ep8-8k2k | core_only | 14074 | 83/1/38 | 723681 | 516 (3.6663%) | 54.536/67 | 324.173/8000 |
| K2.5-tp8ep8-8k2k | full | 14074 | 83/1/38 | 723681 | 516 (3.6663%) | 54.536/67 | 324.173/8000 |
| K2.5-tp8ep8-32k3k | null_only | 87048 | 68/5/6 | 2199860 | 578 (0.6640%) | 13.230/14 | 179.665/32000 |
| K2.5-tp8ep8-32k3k | core_only | 87076 | 31/1/0 | 1013986 | 474 (0.5444%) | 13.225/14 | 165.988/32000 |
| K2.5-tp8ep8-32k3k | full | 87076 | 31/1/0 | 1013986 | 474 (0.5444%) | 13.225/14 | 165.988/32000 |
| K2.5-tp4ep8dp2-8k2k | null_only | 16116 | -/-/- | - | - | 47.631/57 | 297.880/8000 |
| K2.5-tp4ep8dp2-32k3k | null_only | 132048 | -/-/- | - | - | 8.721/10 | 123.598/32000 |
| K2.5-tp8ep8-8k2k-bt65536 | null_only | 22005 | 149/8/96 | 1254235 | 173 (0.7862%) | 34.884/42 | 231.486/65536 |
| K2.5-tp8ep8-8k2k-bt65536 | core_only | 22015 | 92/0/43 | 802482 | 106 (0.4815%) | 34.864/42 | 210.861/65536 |
| K2.5-tp8ep8-8k2k-bt65536 | full | 22015 | 92/0/43 | 802482 | 106 (0.4815%) | 34.864/42 | 210.861/65536 |
| K2.5-tp4ep8dp2-8k2k-bt65536 | null_only | 55476 | -/-/- | - | - | 13.837/16 | 96.940/65536 |

历史 real/sim 抢占计数来自 N512 panorama，仅作方向对照；本表三层重跑是 N384 验证协议，不能直接比较绝对值。

| 场景 | real N512 | legacy sim N512 |
|---|---:|---:|
| K2.5-tp8ep8-8k2k | 111 | 186 |
| K2.5-tp8ep8-32k3k | 42 | 194 |
| K2.5-tp4ep8dp2-8k2k | 101 | 192 |
| K2.5-tp4ep8dp2-32k3k | 56 | 136 |
| K2.5-tp8ep8-8k2k-bt65536 | 117 | 219 |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 121 | 277 |

## 常数归置

每个常数只落一类；完整逐行证据见 `phase462_engine_loop_constant_provenance.csv`。

| 分类 | 数量 |
|---|---:|
| version_semantic_profile | 11 |
| physical_scaling | 1 |
| scoped_measurement | 4 |
| remove_from_default_path | 5 |

`null-only` 计分板 `3/6`；`core-only TP + null-only DP` 计分板 `2/6`。版本 profile 字段与分发点位见 `phase462_version_semantic_profile_design.md`。版本 profile 可作为不启用行为的框架小步单独评审；动态主线回到 DP Step 3 与 bt65536 分解。

边界：report-only；未改 runtime、门或 PerfDB；未跑 GPU；`Default AIC=No-Go`；归档行重审与 DP Step 3 顺延。
