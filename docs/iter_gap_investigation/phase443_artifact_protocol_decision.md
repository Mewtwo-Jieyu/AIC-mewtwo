# Phase443-B Burst Artifact Protocol Decision

结论: 当前 dp2 的 N=128 闭集参考混入 burst artifact,不适合作为唯一验收口径。推荐后续执行候选 (i):把 dp2 两个参考改成 N=512 持续到达协议重采;过渡期报告保留 artifact/steady 拆列。

## Evidence

| item | existing evidence | implication |
|---|---|---|
| 32k3k arrival sweep | Phase412: N=512 后 request split 拉平,overall penalty 从 N=128 的 1.887 降到 1.741,steady-window 为 1.358 | N=128 burst 会放大 dp2 gap |
| 8k2k arrival sweep | Phase425: N=512 后 per-engine request split 从 1.723 拉到 1.008,artifact 约 1.30x | 当前最大点 dp2-8k2k 的 2.118x 不是纯模型差 |
| steady decomposition | Phase426: 8k2k steady gap 独立于 burst tail,仍需模型侧处理 | 协议修正不能替代模型修复 |

## Options

| option | action | expected impact | GPU cost | risk |
|---|---|---:|---:|---|
| (i) N=512 reference recollect | dp2-8k2k、dp2-32k3k 两个 reference 改用 N=512/C=128 持续到达协议 | dp2-8k2k 2.118 -> about 1.63; dp2-32k3k 1.255 -> <=1.15 | 2 full bench runs | 与其余 4 点 N=128 协议不一致;需声明或全表统一 |
| (ii) steady-window acceptance | validate 改成稳态窗口吞吐对比 | artifact 从指标中消失 | 0 | 改了验收语义,不再是端到端吞吐 |
| (iii) report split only | 保持现有 reference,报告拆 artifact/steady 两列 | gate 数字不变 | 0 | 15% 判据仍口径模糊 |

## Recommendation

选 (i)。理由是它仍保留端到端吞吐,只是把 benchmark 从一次性突发改成持续到达,更接近服务稳态。若要完全避免混口径,可以后续把 6 点全表统一重采 N=512;成本高但一次性清掉 reference protocol 变量。

短期过渡采用 (iii):所有报告同时列 `overall_ratio` 与 `artifact_adjusted_ratio`,但 Default AIC 仍按当前 validate 全表保持 No-Go。

## Boundary

- This document is decision-only.
- No runtime, PerfDB, gate, or reference data is changed here.
- `valid_for_default=false`; Default AIC remains No-Go until the selected protocol is executed and validate passes.
