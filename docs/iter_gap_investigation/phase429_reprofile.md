# Phase429 Guaranteed Kernel Reprofile

Phase429 uses GPU only for measurement. It does not modify runtime, PerfDatabase, gate, or Default AIC readiness.

## Verdict

- mechanism: `phase429_reprofile_window_hits_attributed`
- prefill: `prefill_serialization_bubble_unmodeled`
- decode: `decode_mla_attention_slope_dominates`
- reconstruction gate: `passed`
- Phase430 runtime target: `apply_named_undercharge_or_bubble_fix`
- Phase430 PerfDB target: `patch_or_recollect_named_perfdb_component`

## Window Checks

| window | target concurrency | gate | retry count | mean decode batch |
|---|---:|---|---:|---:|
| w0_prefill |  | passed | 1 |  |
| w1_decode_c16 | 16 | passed | 0 | 8.000000 |
| w2_decode_c64 | 64 | passed | 0 | 36.000000 |
| w3_decode_c128 | 128 | passed | 0 | 52.000000 |

## Coverage

| prefill steps | decode steps | decode batch min | decode batch max | span | steady penalty |
|---:|---:|---:|---:|---:|---:|
| 37 | 103 | 8 | 52 | 44 | 1.516977 |

## Decode Slope

| category | mean ms/rank | slope ms/request |
|---|---:|---:|
| mla_attention | 7.841973 | 0.215499 |
| moe_gemm_or_aux | 8.203102 | 0.164715 |
| ep_a2a | 6.299386 | 0.070353 |
| dense_gemm | 4.706812 | 0.017480 |
| other_cuda | 1.970539 | 0.016603 |
| tp_or_dp_allreduce | 0.664403 | 0.010351 |
| memcpy_memset | 0.108530 | 0.002244 |

## Boundary

- GPU/SSH were used only for measurement.
- Runtime, PerfDatabase, and gate were not modified.
- Phase405 penalty was not read.
- Default AIC remains No-Go.
