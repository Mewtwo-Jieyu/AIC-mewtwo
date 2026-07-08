# Phase447 DP phase model closeout

结论: Phase447 在 Step3 指纹门停止。真实 B2b 指纹要求 8k2k mixed decode batch 落在 34-52;当前 sim 修完 running 顺序后仍是 p50=62,所以不合入 runtime DP lockstep。

## What landed

| 项 | 结果 |
|---|---|
| Step0 真实指纹 | 已生成 `phase447_sequence_fingerprint.{csv,md}` |
| Step1 分歧审计 | 已生成 `phase447_scheduler_divergence.md` |
| Step2c running 顺序 | 已按 vLLM v1 改为 running queue 单 pass;红测覆盖 prefill 插在 decode 中间的场景 |
| Step3 指纹门 | FAIL: sim p50=62, real p10-p90=34-52 |
| Step4 lockstep | 未合入 |

## Why lockstep stopped

Phase446 已证明“实测序列 + max 耦合”能重建墙钟,但当前 runtime sim 还不能生成真实序列。真实序列的高代价 mixed 步主体是 `(bucket=8000, decode_batch=41-44)`;当前 sim 仍生成 `(bucket=8000, decode_batch≈62)`。这说明缺口还在 DP global admission / stale score routing / closed-loop wave,不是 lockstep 公式本身。

## Validate

| scenario | real output tok/s/gpu | sim tok/s/gpu | ratio | gate |
|---|---:|---:|---:|---|
| K2.5-tp8ep8-8k2k | 133.5 | 167.0 | 1.25x | pass |
| K2.5-tp8ep8-32k3k | 52.5 | 56.1 | 1.07x | pass |
| K2.5-tp4ep8dp2-8k2k | 137.7 | 291.7 | 2.12x | fail |
| K2.5-tp4ep8dp2-32k3k | 53.3 | 58.3 | 1.09x | pass |
| K2.5-tp8ep8-8k2k-bt65536 | 138.5 | 155.1 | 1.12x | pass |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 113.9 | 140.5 | 1.23x | pass |

MULTI_CONFIG max=2.12x, mean=1.31x. Default AIC remains No-Go.

## Next target

The next phase must build DP global admission before lockstep:

| 缺口 | 需要建模 |
|---|---|
| request placement | score=`waiting*4+running`, 100ms stale stats, local optimistic waiting increment |
| closed-loop wave | global concurrency pool, request completion triggers routed replacement |
| step composition | new prefill position inside running list must naturally produce decode_batch 41-44 |

No manual phase offset should be introduced.
