# Phase462 Step2b: 抢占决策点真实观测

结论：Step2b 观测与开销门均通过，但没有找到 allocation 阈值差。real 与 sim 都在差 1 个 block 时进入抢占，victim policy 也都是 running tail。首次分歧出现在请求相位：real 首次由一个请求触发、抢另一个落后 1 token 的 tail，10 次 victim 全不重复；sim 首次 trigger 就是 tail victim 本人，2 个 local iteration 后再次抢同一请求。修复靶点收窄为 per-request decode phase evolution / re-entry dynamics，Step2c 在源码语义钉死前继续 blocked。Default AIC 维持 No-Go。

| 项 | real | sim |
|---|---:|---:|
| 抢占次数 | 10 | 26 |
| unique victim | 10 | 22 |
| repeat victim event | 0 | 4 |
| self-preemption | 0 | 5 |

| 门 | 结果 |
|---|---|
| 字段完整性 | 10/10 complete，pass |
| logging 开销 | 0.146%（阈值 2%），pass |
| GPU/process residual | empty |
| patch restore | scheduler/kv SHA 与采集前一致，已恢复 |

源码边界：vLLM `scheduler.py:384-518` 逐请求计算 `num_tokens_with_spec + num_output_placeholders - num_computed_tokens` 后依次申请 block；cb_sim `scheduler.py:183-205` 固定为每个 decode request 申请 1 token，`simulator.py:621-624` 再统一推进 1 token。这个差异与实测 phase skew 同方向，但尚未证明哪一条状态更新导致首次 victim 关系分岔，因此不能直接改代码。

下一步只做源码对照：解释 vLLM 如何让同 cohort 请求产生 1-token phase skew，以及 cb_sim 为何在首次 block boundary 形成 self-victim + immediate repeat。不能修改 block threshold，也不能加去同步随机数。

本批数据 `diagnostic_only=true / valid_for_default=false / perf_database=false`，不进参考与 PerfDB。
