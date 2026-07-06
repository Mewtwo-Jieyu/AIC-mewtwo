# Phase430 bubble / MLA fix

## Verdict

- phase430_fix_a_applied_fix_b_requires_gpu_recollect.
- Fix A landed only the safe runtime correction: pure prefill now charges context non-attention plus context attention independent of overlap_factor.
- Mixed prefill was already a serial sum in cb_sim; Phase429's bubble is an execution-state gap, not a remaining max-overlap branch.
- Fix B is not safe to patch offline: MLA slope is close, while MoE decode SOL has the wrong slope sign and existing small-token table is not a valid replacement.
- Default AIC remains No-Go.

## vLLM Source Basis

| file | lines | conclusion |
|---|---:|---|
| `vllm/model_executor/layers/fused_moe/modular_kernel.py` | 1072-1158,1160-1221,1223-1303,1356-1381 | prepare_then_fused_experts_then_finalize_call_order |
| `vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ll.py` | 363-389,391-465 | deepep_prepare_waits_for_hook_receiver_and_finalize_combines |
| `vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py` | 103-141,143-165 | naive_dp_ep_dispatch_then_weight_reduce_then_combine |

## Decode Slope Audit

| category | real ms/request | sim ms/request | gap | verdict |
|---|---:|---:|---:|---|
| dense_gemm | 0.017480 | 0.004317 | 0.013163 | decode_slope_ok |
| ep_a2a | 0.070353 | 0.003887 | 0.066466 | ep_a2a_decode_slope_undercharged_secondary |
| memcpy_memset | 0.002244 | 0.000000 | 0.002244 | decode_slope_ok |
| mla_attention | 0.215499 | 0.197075 | 0.018424 | mla_slope_near_real_but_joint_recollect_recommended |
| moe_gemm_or_aux | 0.164715 | -0.147130 | 0.311845 | moe_decode_sol_slope_wrong_sign_recollect_required |
| other_cuda | 0.016603 | 0.001874 | 0.014729 | decode_slope_ok |
| tp_or_dp_allreduce | 0.010351 | 0.015279 | -0.004928 | decode_slope_ok |

## Validate After

- MULTI_CONFIG after Phase430: max=2.55, mean=1.48; verdict=multi_config_still_fails_dp2_8k2k.

| scenario | real out tok/s/GPU | sim out tok/s/GPU | ratio | verdict |
|---|---:|---:|---:|---|
| K2.5-tp8ep8-8k2k | 133.5 | 167.0 | 1.25x | pass |
| K2.5-tp8ep8-32k3k | 52.5 | 56.1 | 1.07x | pass |
| K2.5-tp4ep8dp2-8k2k | 137.7 | 351.6 | 2.55x | fail |
| K2.5-tp4ep8dp2-32k3k | 53.3 | 92.4 | 1.73x | fail |
| K2.5-tp8ep8-8k2k-bt65536 | 138.5 | 155.1 | 1.12x | pass |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 113.9 | 101.1 | 0.89x | pass |

## Phase431 Recollect List

| category | required point |
|---|---|
| mla_attention | `gpu_recollect_generation_mla_heads16_batch8_16_32_64_128_kv8192_float16` |
| moe_gemm_or_aux | `gpu_recollect_int4_wo_marlin_decode_batch8_16_32_64_128_tp4dp2ep8` |
| ep_a2a | `gpu_profile_decode_ep_a2a_latency_floor_batch_sweep` |

## Boundaries

- Runtime changed: true, limited to pure prefill serial accounting in `iteration_latency.py`.
- PerfDatabase changed: false.
- Gate/default changed: false; `valid_for_default=false`, `default_readiness=No-Go`.
