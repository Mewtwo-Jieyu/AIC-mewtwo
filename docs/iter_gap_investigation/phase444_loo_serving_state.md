# Phase444-B Serving-State LOO

LOO 不支持本 phase 改 query 规则: interior median 可接受,但 p90 超过 20% 的分层仍存在。

结论按 `(phase, row_kind, category, fold)` 分层;只有 interior 计入门。

| phase | row_kind | category | fold | n | median err % | p90 err % | gate |
|---|---|---|---|---:|---:|---:|---|
| decode | category | collective_other | frontier | 2 |  |  | not_counted |
| decode | category | collective_other | interpolation_gap | 5 |  |  | not_counted |
| decode | category | ep_a2a | frontier | 2 |  |  | not_counted |
| decode | category | ep_a2a | interpolation_gap | 5 |  |  | not_counted |
| decode | category | moe_gemm_or_aux | frontier | 2 |  |  | not_counted |
| decode | category | moe_gemm_or_aux | interpolation_gap | 5 |  |  | not_counted |
| decode | category | other_cuda | frontier | 2 |  |  | not_counted |
| decode | category | other_cuda | interpolation_gap | 5 |  |  | not_counted |
| mixed_prefill | category | collective_other | frontier | 3 |  |  | not_counted |
| mixed_prefill | category | ep_a2a | frontier | 4 |  |  | not_counted |
| mixed_prefill | category | ep_a2a | interior | 32 | 6.056824 | 25.246530 | failed |
| mixed_prefill | category | ep_a2a | interpolation_gap | 5 |  |  | not_counted |
| mixed_prefill | category | moe_gemm_or_aux | frontier | 4 |  |  | not_counted |
| mixed_prefill | category | moe_gemm_or_aux | interior | 32 | 6.444805 | 22.891349 | failed |
| mixed_prefill | category | moe_gemm_or_aux | interpolation_gap | 5 |  |  | not_counted |
| mixed_prefill | category | other_cuda | frontier | 4 |  |  | not_counted |
| mixed_prefill | category | other_cuda | interior | 32 | 5.984922 | 24.642132 | failed |
| mixed_prefill | category | other_cuda | interpolation_gap | 5 |  |  | not_counted |

## Boundary

- diagnostic_only=true; valid_for_default=false.
- This phase measures interpolation error only; it does not change PerfDB query behavior.
- Frontier and interpolation-gap rows are reported but not counted in the LOO gate.
