# Phase462 Step 3f 度量保真修复与日志字段审计

结论：度量修复后计分板仍为 `3/6`。TP8 三点逐字节不动；DP2-bt65536 从 `1.200038` 收敛到 `1.164056`，但仍未过 `1.15`。Phase458 日志可复现 coarse cell spread，不能完成 exact composition / undercharge 判卷。

| 场景 | 修复前 error | 修复后 error | delta | 1.15 门 |
|---|---:|---:|---:|---|
| K2.5-tp8ep8-8k2k | 1.131921 | 1.131921 | +0.000000 | pass |
| K2.5-tp8ep8-32k3k | 1.012301 | 1.012301 | +0.000000 | pass |
| K2.5-tp4ep8dp2-8k2k | 1.057775 | 1.000764 | -0.057011 | pass |
| K2.5-tp4ep8dp2-32k3k | 1.195756 | 1.202183 | +0.006426 | fail |
| K2.5-tp8ep8-8k2k-bt65536 | 1.256843 | 1.256843 | +0.000000 | fail |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 1.200038 | 1.164056 | -0.035982 | fail |

DP2 采用与 real bench 相同的全局 N512、warmup=0、首个 arrival 到最后 completion 的完整墙钟；TP8 不进入新组装路径。历史 `65.02%/34.98%` 只对 Step 3d 的固定切换顺序成立，不可交换，也不是唯一因果比例。

## 现有字段

| 项 | 结果 |
|---|---|
| TP8 iteration rows | 28025 |
| DP2 iteration rows | 75277；rank0=38021；rank1=37256 |
| DP2 mixed rows | 100 |
| `(6486,15)` samples | rank0=1；rank1=7 |
| `(6486,15)` median elapsed | rank0=4791.130 ms；rank1=561.970 ms；spread=8.525597x |
| 已有 | `dp_rank`, `iteration`, `context_requests`, `context_tokens`, `generation_requests`, `generation_tokens`, `elapsed_ms` |
| 缺失 | `context_chunk_tokens_multiset`, `context_state_token_counts`, `decode_kv_token_sum`, `cudagraph_mode` |

相同聚合 cell 内仍有 8.53x 差异，所以已有字段只能复现问题，不能把差异归给 chunk 构成或其他隐藏维。当前 sim 成本查询还需要 decode KV 长度，日志也没有按 fresh/recompute/resume 分开的 context 状态；用常数补齐会造成现场拟合，禁止执行。

## 轻量采集 v2（已否决）

| 项 | 设计 |
|---|---|
| 触发 | 仅 mixed step；不做周期 K 采样 |
| 原因 | 目标 cell 在 rank0 只有 1 次，任何 K>1 都可能漏样 |
| 输出规模 | 100/75277 rows（0.1328%）；字节数须由确定的序列化格式再算，不伪造上界 |
| 新字段 | context chunk token multiset；fresh/recompute/resume token counts；decode KV token sum；cudagraph mode |
| 明确不采 | request id、arrival 链、逐请求生命周期、每步同步写文件 |
| 开销结论 | 离线只能证明输出量很小，不能证明 <=2%；下一次仍须独立 off/on GPU 硬门 |

后续 GPU 硬门已经完成：v2/v3/v4 的 off/on 绝对差分别为 `13.8486%`、`8.7915%`、`8.2043%`，均未通过 `<=2%` 门。当前协议路线关闭，没有可采信的构成级 GPU 数据，不得重跑同类采集或据此进入 Step 4；详见 [Phase462 bt65536 度量与构成取证合并判卷](../phase462_bt65536_metric_and_composition_audit/phase462_bt65536_metric_and_composition_audit.md)。

本步未改 simulator runtime、PerfDB 或 gate；Default AIC 维持 No-Go，Step 4 继续顺延。
