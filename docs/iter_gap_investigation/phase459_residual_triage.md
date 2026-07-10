# Phase459 residual triage

结论：`dp2` KV 数值接线无漂移。`tp8-32k3k` 主因是 decode 单步高计费；抢占过发是并存但尚未独立分离的次级构成误差。两个 `bt65536` 点都由 mixed step 低计费主导，但 TP8 是表缺失，DP2 是跨 regime 命中旧行。Phase459 不改 runtime/PerfDB/gate。

## KV 接线审计

| 场景 | Phase454 真值 | validate | 结果 | 备注 |
|---|---:|---:|---|---|
| K2.5-tp4ep8dp2-8k2k | 458128 | 458128 | pass | validate_source=docs/iter_gap_investigation/phase446_b2b_event_timing/overhead_gate_20260708_075726/overhead_on/K2.5-tp4ep8dp2-8k2k/serve.log; numeric_drift=False |
| K2.5-tp4ep8dp2-32k3k | 320800 | 320800 | pass | validate_source=docs/iter_gap_investigation/phase397k_measured_0190/K2.5-tp4ep8dp2-32k3k/serve.log; numeric_drift=False |

## Track A：tp8-32k3k

| 指标 | 数值 | 含义 |
|---|---:|---|
| total_gap | 1.3970814466948587 | three-class representative wall-share replay; preemption work factor is reported separately |
| cost_replay_factor | 1.4222674569307618 | three-class representative wall-share replay; preemption work factor is reported separately |
| cost_replay_residual | 0.9822916497785484 | three-class representative wall-share replay; preemption work factor is reported separately |
| decode_real_total_ms | 21.0 |  |
| decode_generation_attention_ms | 31.02908534321159 | attention term alone exceeds the real whole decode step |
| preemption_ratio | 4.619047619047619 | three-class representative wall-share replay; preemption work factor is reported separately |
| prompt_work_factor | 1.3815907592773438 | three-class representative wall-share replay; preemption work factor is reported separately |
| mixed_step_real_over_sim | 1.8884790734363006 | three-class representative wall-share replay; preemption work factor is reported separately |
| decode_step_sim_over_real | 1.8605565621035314 | three-class representative wall-share replay; preemption work factor is reported separately |
| preemption_role | secondary_unresolved_composition_error | three-class representative wall-share replay; preemption work factor is reported separately |
| verdict | decode_cost_overcharge_dominates | three-class representative wall-share replay; preemption work factor is reported separately |

| 步型 | real 墙钟占比 | sim/real 代表步 |
|---|---:|---:|
| mixed | 0.3285008547373921 | 0.5295266514022775 |
| prefill | 0.0006683274045228861 | 0.2982799073495962 |
| decode | 0.6708305332526259 | 1.8605565621035314 |

`tp8-32k3k` 中 decode 占 67% real 墙钟，sim 单步却高计费约 1.86x；其中 generation attention 一项约 31ms，已经高于 real decode 整步 21ms。mixed 单步低计费约 1.89x，形成明显误差抵消。三类代表步按 real 墙钟份额重放得到 1.42x，闭合 1.397x 总 gap。抢占 194 vs 42 仍是模型错误，但不是本点吞吐残差的第一修复靶。

## Track B：bt65536

| 场景 | observed sim wall/real | cost replay | replay 残差 | modal mixed sim/real | 结论 |
|---|---:|---:|---:|---:|---|
| K2.5-tp8ep8-8k2k-bt65536 | 0.810030457258567 | 0.7245309676450806 | 1.1180066738781127 | 0.366824595833783 | mixed_step_undercharge_table_missing |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 0.8570494425762037 | 0.8855705027272672 | 0.9677935748049106 | 0.50172279199478 | mixed_step_undercharge_regime_hit_mismatch |

TP8 大 mixed step real/sim=2.7260985532527484，查询为 table missing。DP2 大 mixed step已命中且基本匹配，但占墙钟更多的 modal mixed 步命中后仍只有 real 的 50.2%。当前共享表混有 max_bt=8000,32000,65536 三套行，而 DP2 查询未带 max_bt 键，属于跨 regime 错误命中，不是简单的网格外 miss。

Phase458 iteration 日志只能定位整步，不能拆 kernel 类别。`dp2-bt65536` 已有 Phase454 B2b 原始数据，下一相可先做 max_bt regime 键控重放；`tp8-bt65536` 没有同 regime B2b 证据，不能把 bt8000 行直接迁移。

## 后续分相

| 相位 | 靶点 | 状态 |
|---|---|---|
| phase460_target | tp8_32k_mla_decode_cost_audit | proposed |
| phase461_target | bt65536_regime_scoped_mixed_step_rows | proposed |
| phase459_verdict | residuals_triaged_gate_unchanged | fail |

6/6 达标前不收紧 gate，不进入 fork 交接。`diagnostic_only=true valid_for_default=false perf_database=false`；Default AIC 维持 No-Go。
