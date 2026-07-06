# Phase433 NSYS Prefill Forensics

Phase433 is measurement-only. It does not modify runtime, PerfDatabase, or gate.

## Verdict

- profiler mode: `torch_profiler_cpu_fallback`
- precision boundary: `torch_profiler_fallback_cannot_split_nccl_wait_from_transfer`
- mechanism: `collective_or_transfer_dominates_torch_fallback_cannot_split_wait_vs_transfer`
- reconstruction gate: `passed` (4.313116% error)
- Phase434 target: `separate_nccl_wait_vs_transfer_or_build_serving_state_ep_model`

## Four-Way Decomposition

| item | ms/step |
|---|---:|
| collective wait or transfer | 1526.141775 |
| serving slow kernel excess | 1224.894025 |
| CPU/host or unobserved gap | 19.979791 |
| explicit memcpy/memset transfer | 4.207238 |
| reconstructed | 2775.222829 |
| target missing | 2900.316853 |

## Per-GPU Busy

| rank | busy ms | ratio to mean |
|---:|---:|---:|
| dp0_rank0 | 4503.836260 | 0.997852 |
| dp0_rank1 | 4503.824369 | 0.997849 |
| dp0_rank2 | 4503.939349 | 0.997875 |
| dp0_rank3 | 4502.296243 | 0.997511 |
| dp1_rank0 | 4523.359805 | 1.002178 |
| dp1_rank1 | 4522.515155 | 1.001990 |
| dp1_rank2 | 4524.655851 | 1.002465 |
| dp1_rank3 | 4523.825127 | 1.002281 |

## Kernel Categories

| category | real ms/step | sim ms/step | excess ms/step |
|---|---:|---:|---:|
| collective_other | 348.179960 | 242.290304 | 105.889656 |
| dense_gemm | 238.167994 | 255.654474 | 0.000000 |
| ep_a2a | 1917.743178 | 497.491058 | 1420.252120 |
| memcpy_memset | 4.207238 | 0.000000 | 4.207238 |
| mla_attention | 366.395312 | 402.019077 | 0.000000 |
| moe_gemm_or_aux | 1143.462225 | 352.717448 | 790.744777 |
| other_cuda | 495.698383 | 61.549135 | 434.149248 |

## Boundary

- `nsys` was not available when `trace_precision=torch_profiler_fallback`; NCCL wait and wire transfer remain a combined bucket.
- The output is diagnostic-only and not valid for Default AIC.
- Runtime/PerfDatabase/gate: not modified.
- Default AIC: No-Go.
