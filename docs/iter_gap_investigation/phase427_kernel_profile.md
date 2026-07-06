# Phase427 Kernel Profile

Phase427 只做 GPU profiler 测量和离线聚合；不改 runtime、PerfDatabase 或 gate。

## Verdict

- verdict: `kernel_profile_collected_prefill_and_decode`
- prefill verdict: `mixed_mla_attention_dominates`
- decode verdict: `decode_mla_attention_dominates`
- Phase428 target: `classify_prefill_execution_and_decode_slope_from_kernel_profile`
- reconstruction gate: `passed`

## Kernel Categories

| window | category | cuda ms | share of self cuda |
|---|---|---:|---:|
| mixed | dense_gemm | 1480.100000 | 0.149143 |
| mixed | ep_a2a | 2141.367000 | 0.215777 |
| mixed | memcpy_memset | 44.953137 | 0.004530 |
| mixed | mla_attention | 3293.870530 | 0.331910 |
| mixed | moe_gemm_or_aux | 2123.602000 | 0.213986 |
| mixed | other_cuda | 772.484674 | 0.077840 |
| mixed | tp_or_dp_allreduce | 245.376372 | 0.024726 |
| decode | dense_gemm | 1476.667000 | 0.128239 |
| decode | ep_a2a | 2173.489000 | 0.188753 |
| decode | memcpy_memset | 47.381127 | 0.004115 |
| decode | mla_attention | 3508.115106 | 0.304656 |
| decode | moe_gemm_or_aux | 3468.585000 | 0.301223 |
| decode | other_cuda | 772.592828 | 0.067094 |
| decode | tp_or_dp_allreduce | 243.739468 | 0.021167 |

## Phase426 Context

| steady penalty | prefill share | decode gap share | peer-stall share |
|---:|---:|---:|---:|
| 1.516977 | 0.595460 | 0.357492 | 0.047048 |

## Boundary

- GPU/SSH: used only for Phase427 measurement artifacts.
- Runtime/PerfDatabase/gate: not modified.
- Phase405 penalty: not read.
- Default AIC: No-Go.
