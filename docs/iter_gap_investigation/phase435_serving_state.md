# Phase435 Serving-State Cost Curves

## Verdict

- serving-state category measurements are admitted as scoped PerfDB rows, not as a mechanism model.
- wait/wire/imbalance attribution remains unresolved; the row provenance records that boundary.
- extracted curve rows: 28. consistency gates failed: 0.
- Default AIC remains No-Go until A/B validation closes the validation table.

## Consistency Gates

| scenario | phase | target missing ms | reconstructed ms | error % | gate |
|---|---:|---:|---:|---:|---|
| K2.5-tp4ep8dp2-32k3k | mixed_prefill | 2900.317 | 2751.036 | 5.15 | passed |
| K2.5-tp4ep8dp2-8k2k | decode | 16.494 | 16.494 | 0.00 | passed |

## Scope

- key scope: K2.5 / tp4dp2ep8 / int4_wo / H200 / vLLM 0.19.0.
- query behavior: exact scope match plus in-grid interpolation only; out-of-grid returns None.
- categories replaced: ep_a2a, moe_gemm_or_aux, other_cuda, collective_other.
- MLA and dense GEMM remain on the existing计费链.
