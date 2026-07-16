# Phase462 TP8-bt65536 8宽 mixed first-divergence

结论：首个分叉在 sim iteration `1`，首个连续 8-new-request mixed 聚簇从 `2` 开始并持续 `4` 步。real 也存在 decode 数 `+8` 且 context requests 相同的等价波次：`True`；因此裁决为 `initial_admission_wave_phase_offset`，不是 sim 独有的 chunk 配额规则。`Default AIC=No-Go`。

| 首波 | context requests | context tokens | decode requests |
|---|---:|---:|---:|
| real iteration 0 | 1 | 8000 | 0 |
| sim iteration 1 | 9 | 65536 | 0 |

| 指标 | 首个 8-new-request 聚簇 |
|---|---:|
| prefill requests/tokens | 9/65528 |
| decode requests | 8 |
| 新 admit | 8 |
| admit 前 token budget | 59064 |
| schedule 后剩余 budget | 0 |
| 下一请求所需 token | 8000 |
| max_num_seqs 命中 | False |
| block capacity 命中 | False |
| partial-prefill stop | True |
| real 有等价 +8 波次 | True |
| real 有完全相同宏观 cell | False |

判卷只使用 scheduler 入参和结果重建预算，没有修改 scheduler。sim 的单步机械链是“剩余 token budget 形成 fresh partial → partial-prefill stop → 下一步 1 continued + 8 fresh”；real 聚合轨迹同样出现 9 个 context 且 decode 每步增加 8。两侧不同的是首波暴露/admission：real 首步 1 个 context，sim 首步 9 个，随后 decode 相位相差 7，才形成 sim-only 精确 cell。配套 CSV 保存该窗口请求 ID 与 chunk 构成；本步不授权修改 token budget、chunk 规则或暴露模型。

本步全离线，不改 runtime、PerfDB 或 gate。
