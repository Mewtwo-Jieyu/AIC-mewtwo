# Phase461 Step4a-2 未覆盖 cell 分诊

结论：Step4a 的 62.6%/20.7% 是 latency cache 首次查询口径；本步截取每次真实 sim 调度调用后，按 `查询频次 × 当前 sim 迭代墙钟` 重新加权。`sim_only_for_phase458_reference` 只表示当前 Phase458 参考日志没有出现该精确 cell，不宣称真实系统永远不会产生。全程未改 runtime、PerfDB 或 gate，Default AIC 维持 No-Go。

| 场景 | 调度查询 | 未覆盖查询 | 查询覆盖 | 墙钟覆盖 | 可达覆盖 | real-observed 缺口 | reference 未观察缺口 |
|---|---:|---:|---:|---:|---:|---:|---:|
| K2.5-tp8ep8-32k3k | 708 | 53 | 92.5% | 99.4% | 99.4% | 0 | 46 |
| K2.5-tp8ep8-8k2k-bt65536 | 185 | 128 | 30.8% | 57.9% | 57.9% | 0 | 119 |

## 下一步分流

| 场景 | 动作 | 原因 |
|---|---|---|
| K2.5-tp8ep8-32k3k | anchor_calibration_only | 加权覆盖已过 95%，不补覆盖点；只设计跨 session 校准锚点 |
| K2.5-tp8ep8-8k2k-bt65536 | phase462_dynamics_first | 当前 reference 未观察到全部缺口，补采无法达到 95%；先移交 Phase462 修 dynamics |

## 高权重处置

| 场景 | cell | 权重 | real 样本 | 分类 | 决策 |
|---|---|---:|---:|---|---|
| K2.5-tp8ep8-8k2k-bt65536 | 64153/28 | 1.27% | 0 | sim_only_for_phase458_reference | phase462_dynamics |
| K2.5-tp8ep8-8k2k-bt65536 | 60458/34 | 1.20% | 0 | sim_only_for_phase458_reference | phase462_dynamics |
| K2.5-tp8ep8-8k2k-bt65536 | 59252/34 | 1.18% | 0 | sim_only_for_phase458_reference | phase462_dynamics |
| K2.5-tp8ep8-8k2k-bt65536 | 59106/28 | 1.17% | 0 | sim_only_for_phase458_reference | phase462_dynamics |
| K2.5-tp8ep8-8k2k-bt65536 | 8978/34 | 1.07% | 0 | sim_only_for_phase458_reference | phase462_dynamics |
| K2.5-tp8ep8-8k2k-bt65536 | 51511/35 | 1.03% | 0 | sim_only_for_phase458_reference | phase462_dynamics |
| K2.5-tp8ep8-8k2k-bt65536 | 42565/30 | 0.85% | 0 | sim_only_for_phase458_reference | phase462_dynamics |
| K2.5-tp8ep8-8k2k-bt65536 | 42058/36 | 0.84% | 0 | sim_only_for_phase458_reference | phase462_dynamics |
| K2.5-tp8ep8-8k2k-bt65536 | 8308/36 | 0.75% | 0 | sim_only_for_phase458_reference | phase462_dynamics |
| K2.5-tp8ep8-8k2k-bt65536 | 26503/31 | 0.69% | 0 | sim_only_for_phase458_reference | phase462_dynamics |
| K2.5-tp8ep8-8k2k-bt65536 | 34041/31 | 0.69% | 0 | sim_only_for_phase458_reference | phase462_dynamics |
| K2.5-tp8ep8-8k2k-bt65536 | 34000/37 | 0.69% | 0 | sim_only_for_phase458_reference | phase462_dynamics |
| K2.5-tp8ep8-8k2k-bt65536 | 8078/37 | 0.49% | 0 | sim_only_for_phase458_reference | phase462_dynamics |
| K2.5-tp8ep8-8k2k-bt65536 | 8064/38 | 0.49% | 0 | sim_only_for_phase458_reference | phase462_dynamics |
| K2.5-tp8ep8-8k2k-bt65536 | 8051/39 | 0.48% | 0 | sim_only_for_phase458_reference | phase462_dynamics |
| K2.5-tp8ep8-8k2k-bt65536 | 8050/39 | 0.48% | 0 | sim_only_for_phase458_reference | phase462_dynamics |
| K2.5-tp8ep8-8k2k-bt65536 | 18004/39 | 0.48% | 0 | sim_only_for_phase458_reference | phase462_dynamics |
| K2.5-tp8ep8-8k2k-bt65536 | 17796/35 | 0.48% | 0 | sim_only_for_phase458_reference | phase462_dynamics |
| K2.5-tp8ep8-8k2k-bt65536 | 17701/36 | 0.47% | 0 | sim_only_for_phase458_reference | phase462_dynamics |
| K2.5-tp8ep8-8k2k-bt65536 | 17649/35 | 0.47% | 0 | sim_only_for_phase458_reference | phase462_dynamics |

## 口径

| 项目 | 定义 |
|---|---|
| 墙钟权重 | 当前 sim 每次 mixed 调度实际返回的 iteration latency；通过离线 monkeypatch 采集，不改源码行为 |
| real-observed | Phase458 N=512 serve.log 中存在完全相同的 bucket/decode cell |
| sim-only for reference | 当前 Phase458 reference 未观察到；移交 Phase462 验证 dynamics 修复后是否消失 |
| 95% 门 | 本报告提出的下一轮采集预算门，不修改正式 gate |
