# Phase445-A Three-Way Data Audit

- Verdict: `no_runtime_coefficient_schema_alignment_blocked`.
- No runtime coefficient was generated; this is a schema and evidence audit only.
- `moe_gemm_or_aux` is only partially comparable. EP a2a and residual CUDA categories lack a direct upstream table.
- TP8 out-of-sample validation is blocked until a category map and layer normalization make the DP2 serving rows transferable.

## Alignment Summary
| status | count |
| --- | --- |
| missing_direct_upstream_table | 4 |
| partial_collective_scope_mismatch | 2 |
| partial_requires_layer_normalization | 2 |

## Category Alignment
| phase | category | serving_rows | bucket_min | bucket_max | upstream_file | alignment_status |
| --- | --- | --- | --- | --- | --- | --- |
| decode | collective_other | 7 | 2 | 64 | custom_allreduce_perf.parquet | partial_collective_scope_mismatch |
| decode | ep_a2a | 7 | 2 | 64 | none | missing_direct_upstream_table |
| decode | moe_gemm_or_aux | 7 | 2 | 64 | moe_perf.parquet | partial_requires_layer_normalization |
| decode | other_cuda | 7 | 2 | 64 | none | missing_direct_upstream_table |
| mixed_prefill | collective_other | 3 | 8000 | 32000 | custom_allreduce_perf.parquet | partial_collective_scope_mismatch |
| mixed_prefill | ep_a2a | 41 | 4016 | 64001 | none | missing_direct_upstream_table |
| mixed_prefill | moe_gemm_or_aux | 41 | 4016 | 64001 | moe_perf.parquet | partial_requires_layer_normalization |
| mixed_prefill | other_cuda | 41 | 4016 | 64001 | none | missing_direct_upstream_table |

## Upstream Inventory
| file | status | rows | filtered_rows | token_min | token_max |
| --- | --- | --- | --- | --- | --- |
| context_attention_perf.parquet | available | 49952 | 0 |  |  |
| custom_allreduce_perf.parquet | available | 138 | 0 |  |  |
| generation_attention_perf.parquet | available | 54347 | 0 |  |  |
| mla_context_module_perf.parquet | available | 3886 | 0 |  |  |
| mla_generation_module_perf.parquet | available | 5888 | 0 |  |  |
| moe_perf.parquet | available | 48195 | 162 | 1 | 16384 |

## Next Gate

Before any mechanism coefficient can enter runtime, build an explicit category map that states exactly how a serving-step category maps to module/kernel rows. For MoE, that map must include layer count and prove the serving category is not mixing non-MoE auxiliary work. For EP a2a, a direct EP8 all-to-all table or serving-derived primitive is still required.
