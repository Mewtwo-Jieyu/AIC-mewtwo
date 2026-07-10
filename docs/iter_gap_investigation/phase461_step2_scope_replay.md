# Phase461 Step2 scope replay

结论: topology scope 已完整隔离;ISL band 没有因果证据,不加。dp2-32k mixed 行已对齐,不补窗口。dp2-bt65536 的 0.50x modal 缺口来自同 cell 的跨 rank/相位分裂,不是 max_bt/ISL 漏键;Step3 只补诊断复现窗,不得把新单值行直接入库。

## 六点 scope 审计

| scenario | topology | max_bt | hits | misses / reasons |
|---|---|---:|---:|---|
| K2.5-tp8ep8-8k2k | tp8ep8 | 8000 | 0 | misses=1746;table_missing=1746 |
| K2.5-tp8ep8-32k3k | tp8ep8 | 32000 | 0 | misses=996;table_missing=996 |
| K2.5-tp4ep8dp2-8k2k | tp4dp2ep8 | 8000 | 150 | misses=252;bucket_below_range=152;decode_batch_above_range=2;decode_batch_below_range=16;hit=150;interpolation_gap=35;table_missing=47 |
| K2.5-tp4ep8dp2-32k3k | tp4dp2ep8 | 32000 | 24 | misses=288;bucket_below_range=100;decode_batch_below_range=15;hit=24;interpolation_gap=10;table_missing=163 |
| K2.5-tp8ep8-8k2k-bt65536 | tp8ep8 | 65536 | 0 | misses=1092;table_missing=1092 |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | 65536 | 41 | misses=923;bucket_above_range=12;bucket_below_range=9;decode_batch_above_range=3;decode_batch_below_range=73;hit=41;interpolation_gap=511;table_missing=315 |

## DP2 离线重放

| scenario | modal mixed sim/real | target | status |
|---|---:|---:|---|
| K2.5-tp4ep8dp2-32k3k | 0.9998 | 0.85..1.15 | pass |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 0.5017 | 0.85..1.15 | fail |

| check | value | decision |
|---|---:|---|
| full 65k mixed sim/real | 0.9660 | base large-step cost is in range |
| 6486/15 rank median spread | 8.51x | scalar row cannot represent DP phase |
| same-run modal sim/real | 0.5016 | cross-run drift excluded |
| 14484/14 event busy / wall | 0.9997 | spike is real in-run, independently repeat in Step3 |
| ISL band | reject_isl_band_no_causal_evidence | do not extend schema |

## Step3 GPU 最终清单

| item | decision | measurement | coverage | reason |
|---|---|---|---|---|
| mla_decode_grid | collect | kernel_microbench | heads8; batch=8,16; KV=16384,32768,65536; >=3 repeats | replace confirmed SILICON bad row without smoothing |
| tp8_bt65536_mixed | collect | serving_state_window | tp8ep8/max_bt65536 mixed working set | no TP8 serving-state rows exist for this regime |
| tp8_32k3k_mixed | collect | serving_state_window | tp8ep8/max_bt32000 mixed working set | decode bad-row fix exposes the measured mixed undercharge |
| dp2_32k3k_mixed | skip | serving_state_window | tp4dp2ep8/max_bt32000 mixed working set | existing modal sim/real=0.9998 |
| dp2_bt65536_mixed | collect_diagnostic_repeat | paired_rank_event_plus_iteration_wall | 14484/14 spike; modal rank-conditioned cells; both DP ranks | modal sim/real=0.5017; rank spread=8.51x; spike_reachable=True; diagnostic only until lockstep semantics are resolved |

## 边界

| item | decision |
|---|---|
| tp8 window reuse for dp2 | 禁止; topology key 不同 |
| dp2-32k window | 不采;现有 modal mixed 误差小于 0.1% |
| dp2-bt65536 window | 只做 paired-rank + iteration-wall 诊断复现;在 lockstep 语义定案前不直接入库 |
| Default AIC | No-Go |
| runtime / PerfDB / gate | 本步均未修改 |

## 验证

| check | result |
|---|---|
| analyzer tests | 20 passed |
| py_compile / diff check / CRLF | passed |
| GPU / SSH | not used |
