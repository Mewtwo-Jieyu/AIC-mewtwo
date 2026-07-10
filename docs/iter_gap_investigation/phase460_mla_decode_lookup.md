# Phase460 Step 1: TP8 32k MLA decode lookup audit

结论：31ms 高计费来自 `generation_mla_perf.txt` 的单个反单调实测行，不是 context 长度语义、外推或 lookup 公式。当前查询在表内做双线性插值；`batch=8, kv=32768` 的 `0.958645ms/layer` 比同 batch 上下文趋势高 7.19x，并且比 `batch=16` 同 KV 更慢。Phase460 不应改 runtime lookup，也不能用自动平滑修表；正确动作是把精确网格加入 Phase461 GPU 补采批次。

| 查询项 | 值 |
|---|---:|
| decode batch | 13 |
| 输入平均 KV | 33500 |
| 512-token bucket | 33792 |
| 最终 PerfDB KV 查询 | 33793 |
| local heads | 8 |
| layer scale | 61 |
| 查询方式 | SILICON / bilinear |

## 插值四角

| 行 | ms/layer |
|---|---:|
| batch8_kv32768_ms_per_layer | 0.958645 |
| batch8_kv65536_ms_per_layer | 0.244603 |
| batch16_kv32768_ms_per_layer | 0.244939 |
| batch16_kv65536_ms_per_layer | 0.473616 |

`batch=8, kv=32768` 是唯一直接破坏该插值矩形单调性的角：它为 0.958645ms，而 `batch=16` 同 KV 仅 0.244939ms，`batch=8, kv=65536` 也仅 0.244603ms。该行已存在于最初的 H200 vLLM 0.19 PerfDB 对象；Phase431 只追加 8k 行，没有修到此点。

## 贡献量化

| 项目 | 结果 |
|---|---:|
| 当前 attention | 31.029 ms |
| 当前 decode 总步 | 39.072 ms |
| real decode 总步 | 21.000 ms |
| 32k..35k context 扫描造成的 attention 波动 | 0.521 ms |
| 邻点反事实 attention | 12.739 ms |
| 邻点反事实 decode 总步 | 20.782 ms |
| 邻点反事实 sim/real | 0.990x |

邻点反事实仅用于证明归因：用同 batch 的 16k/64k 行线性趋势替换异常角后，代表步回到 0.990x。它不是可入库的替代值。生产修复必须重采。

## Phase461 合并采集提案

| 维度 | 格点 | 目的 |
|---|---|---|
| local heads | 8 | TP8 K2.5 实际查询口径 |
| batch | 8, 16 | 覆盖代表 batch=13 的插值两侧 |
| KV | 16384, 32768, 65536 | 验证上下文单调性并替换 32k 异常行 |
| 重复 | 每点至少 3 次 | 阻止单次测量异常再次入库 |

Step 2 分支因此明确为 `PerfDB 精确网格补采`，并入 Phase461 GPU 单批；不需要离线修改 lookup/context 公式。补采与入库前等待用户确认。

## 边界

- 本阶段 report-only，未修改 runtime、PerfDB、gate，也未使用 GPU。
- mixed step 低计费、bt65536 regime scope、抢占残差继续按 Phase459 存档，不在本报告中混修。
- Default AIC 维持 No-Go；6/6 进入 1.15 前不收紧门限。
