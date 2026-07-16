# Phase462 Step 2c-7 amplification first-divergence

结论：落入预注册分支 **(b) `workload_boundary_preemption_recompute_compound`**。核心假设“sim 把已完成释放多量化一拍”被证伪；两侧都是 batch 完成并释放后，队深 2 的下一个 schedule 立即看见容量。额外一拍由 workload 暴露滞后与同绝对步 KV 抢占复合产生，不是 scheduler、victim 或 future 回填顺序错误。

## mismatch 本体

`new` 表示请求从 `WAITING` 首次进入 running/prefill 的 admission 段，不是某个 phase 字段。request 110 的 `new` admit 为 real/sim `8893/8895`；首个不全等发生在相对第 `502` 步，即 real/sim `9395/9397`。变化字段是 `num_computed_tokens, num_output_placeholders, block_count`：real `{'num_computed_tokens': 32500, 'num_output_placeholders': 1, 'block_count': 2032}`，sim `{'num_computed_tokens': 0, 'num_output_placeholders': 0, 'block_count': 0}`。原因是两侧都在绝对 schedule `9396` 抢占 request 110；按 admit 对齐后，sim 已被移回 waiting，而 real 对齐位置仍在抢占前。

## +1 注入位置

| victim | 作用 | 抢占 real/sim | trigger real/sim | computed real/sim | token 差 | new admit lag | running/waiting 顺序相同 |
|---:|---|---|---|---|---:|---:|---|
| 13 | request 27 链控制项 | 944/944 | 12/12 | 32927/32927 | 0 | 0 | True/True |
| 82 | 放大注入 | 6955/6955 | 78/78 | 32906/32905 | 1 | 1 | True/True |
| 96 | 下游传播 | 8176/8176 | 94/94 | 32729/32727 | 2 | 2 | True/True |
| 110 | 下游传播 | 9396/9396 | 109/109 | 32501/32499 | 2 | 2 | True/True |

精确链条：request 82 首次 admit 的 real/sim lag 是 1；两侧在绝对 schedule `6955` 以同一 trigger `78`、同一 tail victim `82` 抢占，但 sim 因晚 admit 一步少完成 1 token。resume 仍为 `7150/7151`；首段 recompute chunk 同为 `31988`，第二段为 real/sim `919/918`，所以 recompute target 差 1。最终 release 与 request 96 admit 从 `7445/7447` 变成 lag 2；96 release 到 110 admit 保持 lag 2，没有再次放大。

request 13 是 request 27 链的控制项：绝对 schedule `944` 抢占时 computed 两侧相等，resume chunk 两侧均为 `(31988, 940)`，completion-to-schedule 关系也相同，因此其 release lag 只继承 workload 的 1 步，不增加。

## 步内事件排序

| 侧别 | 请求 | 完成 batch | 释放可见 schedule | batch 间隔 | 释放到 schedule(ns) |
|---|---:|---:|---:|---:|---:|
| real 放大链 | 82 | 7443 | 7445 | 2 | 1080207 |
| sim 放大链 | 82 | 7445 | 7447 | 2 | 0 |
| real 控制链 | 13 | 1474 | 1476 | 2 | 1010848 |
| sim 控制链 | 13 | 1475 | 1477 | 2 | 0 |

real request 82 的原始顺序是 `future_complete(7443, ts=5989955939846932) -> finish_remove_running(82, ts=5989955940201995) -> scheduler_input/admit(7445, ts=5989955941282202)`；sim 是 `complete(7445) -> on_complete 释放(82) -> schedule/admit(7447)`，后两者处在同一模拟时钟。两侧 batch 间隔都是 2，符合 queue depth 2。resume 到 release 窗口的邻近抢占计数 real/sim 为 `0/0`；真正参与的是更早的 schedule 6955 抢占及随后 919/918 的 chunk 差。

## 裁决

分支 (a) 排除：没有引擎环步内排序缺口，不应改预测 runtime。分支 (b) 成立：复合效应可以机械表达为“初始暴露 lag -> 同绝对步同 victim 抢占 -> sampled/computed deficit 等于 admit lag -> resume recompute 工作量不同 -> release lag 放大”。但这是同一门的第二次修正候选，必须单独提交更强 real 证据和过拟合风险评审；本步不改门。

当前状态：`blocked_second_gate_amendment_requires_stronger_review`；下一步：`review_mechanical_compound_boundary_rule_and_overfit_risk`。证据 SHA256：`931594934fc90341d822f4a6645a664f2200d4ec50e9552b561c694056c5e1ca`。

边界：`diagnostic_only=true`、`valid_for_default=false`、`perf_database=false`、未改 runtime/gate/PerfDB、未跑 `--ab`、`six_point_ab_allowed=false`、`Default AIC=No-Go`。若后续批准二次门修正，判据必须机械覆盖上述整链，不得按 request ID 或现有 sim 输出写特例。
