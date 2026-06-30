# Phase382 Runtime Binding Probe Sufficiency Gate

Phase382 separates two questions: op-level binding is sufficient, but full-model exact lookup remains blocked.
It does not SSH, run GPU, write new PerfDatabase data, or open Default AIC.

| Gate | Verdict |
|---|---|
| module table | 14 exact keys: 2 modules x 7 buckets |
| current EP8 formula | max(1, raw_tokens//4) |
| current FusedMoE formula | max(1, raw_tokens//4)*2 |
| FusedMoE reachable buckets | 16/1808/2048/8192 |
| FusedMoE unreachable buckets | 1/15/241 |
| paired full-model candidates | 0 |
| op-level runtime binding | sufficient |
| full-model exact lookup | blocked |
| Default AIC | No-Go |

max(1, ...) 不改变当前 paired gap，只修正公式表达。

The next allowed phase is `phase383_runtime_bucket_semantics_decision_spec`.
It must decide whether to change runtime bucket semantics or design paired-bucket data expansion.
