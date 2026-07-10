# Phase461 Step 3 GPU 成本补采

## 结论

三类采集门全部通过。MLA 坏行已由精确网格重采坐实；两个 TP8 serving-state 工作区已覆盖；DP2 65k 的跨 rank 差异再次复现且 busy/wall 闭合。Step 3 只交测量件，不入库，Default AIC 维持 No-Go。

## 采集门

| 场景 | 采集开销 | event 行 | 状态 |
|---|---:|---:|---|
| K2.5-tp8ep8-8k2k-bt65536 | 0.0307 | 224184 | pass |
| K2.5-tp8ep8-32k3k | 0.0000 | 936760 | pass |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 0.0000 | 303328 | pass |

## MLA 精确网格

| 格点 | 延迟 (ms) | 重复性 |
|---|---:|---|
| b8_kv16384 | 0.077429 | samples=3;spread=1.037734 |
| b8_kv32768 | 0.134677 | samples=3;spread=1.064424 |
| b8_kv65536 | 0.239755 | samples=3;spread=1.052027 |
| b16_kv16384 | 0.134624 | samples=3;spread=1.010039 |
| b16_kv32768 | 0.241291 | samples=3;spread=1.050555 |
| b16_kv65536 | 0.444368 | samples=3;spread=1.001922 |

确认坏行 `batch=8, KV=32768`：旧值 `0.958645 ms`，重采中位数 `0.134677 ms`，旧值高 `7.12x`。Step 4 只能用本次实测值替换，禁止平滑或 clamp。

## Mixed 覆盖

| 场景 | mixed 步 | bucket 范围 | decode batch 范围 |
|---|---:|---:|---:|
| K2.5-tp8ep8-8k2k-bt65536 | 138 | 446..65536 | 1..41 |
| K2.5-tp8ep8-32k3k | 634 | 26..32000 | 1..13 |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 168 | 6486..65536 | 1..15 |

## DP 诊断边界

| 指标 | 值 | 说明 |
|---|---:|---|
| perf_database | False | diagnostic_only required |
| max_same_cell_rank_spread | 8.540214161763164 | bucket=6486;batch=15;medians={'0': 4795.55908203125, '1': 561.5267944335938} |
| max_busy_over_wall | 0.9997121252363487 | bucket=6486;batch=15;samples=8 |
| transition_decision | retain_measured_row_diagnostic_only_until_phase_model |  |

DP2 同一构成 cell 的跨 rank spread 为 `8.54x`，且最大 CUDA busy / iteration wall 为 `0.9997`。这是可重复的真实执行态，不是坏行。过渡处置拍板为：保留实测行与诊断 provenance；在 DP phase/lockstep 机制模型落地前，不隔离、不删除，也不转成单值修正。

D 收尾时 8 个 worker 超时未退出，runner 按既有清理协议强杀；各 session 与整批最终 GPU/process residual 均为空。

## 下一步

Step 4 可入库 MLA replacement 与两个 TP8 scoped mixed 行并跑六点 `--ab`。DP2-bt65536 只保留诊断数据，转 Phase462 做 per-rank 构成成本 + lockstep max；dp2-32k3k 的 1.173x 继续单独分诊。
