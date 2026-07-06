# Phase424 bt65536 recollect

- verdict: `dp2_bt65536_reference_replaced_converged`
- default_readiness: `No-Go`
- boundary: measurement/reference replacement only; no PerfDB or runtime charge change.

## Capacity

| field | value |
|---|---:|
| max_model_len | 131072 |
| kv_cache_tokens | 131664 |
| num_gpu_blocks | 8229 |
| old_dirty_kv_cache_tokens | 25744 |
| capacity_vs_old_dirty | 5.114357 |
| output_tok_s_gpu | 113.910186 |

## Validation Delta

| scenario | Phase422 error | Phase424 error | class |
|---|---:|---:|---|
| K2.5-tp4ep8dp2-32k3k | 1.734090 | 1.734090 | unchanged |
| K2.5-tp4ep8dp2-8k2k | 2.553139 | 2.553139 | unchanged |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 5.642576 | 1.127078 | reference_replaced_improved |
| K2.5-tp8ep8-32k3k | 1.068795 | 1.068795 | unchanged |
| K2.5-tp8ep8-8k2k | 1.250461 | 1.250461 | unchanged |
| K2.5-tp8ep8-8k2k-bt65536 | 1.119881 | 1.119881 | unchanged |

## Large Step

| field | value |
|---|---:|
| max_context_tokens_real_trace | 65535 |
| trace_initial_prefill_tokens_sim | 65536 |
| trace_context_non_attention_ms | 1740.707002 |
| combined_64k_lower_bound_ms | 1320.316060 |
| charge_status | combined_non_attention_above_64k_lower_bound |

## Next

`phase425_return_to_main_dp2_residual`. Default AIC stays No-Go.
