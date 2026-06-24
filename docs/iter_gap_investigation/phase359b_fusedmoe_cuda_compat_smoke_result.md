# Phase359b FusedMoE CUDA Compat Smoke Result

| Item | Result |
|---|---|
| Verdict | runner-boundary smoke pass |
| Default AIC | No-Go |
| PerfDatabase | not written |
| Evidence status | diagnostic-only runner smoke |

## Result

- VLLM_ENABLE_CUDA_COMPATIBILITY=1 is effective when the direct smoke process starts with `/usr/local/cuda-12.9/compat` before `/usr/local/nvidia/lib64` in `LD_LIBRARY_PATH`.
- The direct smoke is not equivalent to vLLM serve worker subprocess environment: `vllm serve` can propagate the compat setting before worker startup, while direct Python already has libcuda binding pressure at process start.
- Each direct `FusedMoE.forward()` call must run inside `set_forward_context`, with `static_all_moe_layers` registered for the same runner layer.
- `kernel_source=CompressedTensorsWNA16MarlinMoEMethod:Marlin` is measurement metadata only; kernel_source stays measurement metadata and is not a PerfDatabase lookup key.
- The latency value proves this runner boundary can execute, but it is not a PerfDatabase row and is not default AIC evidence.

## Final Smoke Row

- ok: `true`
- latency_ms_median: `1.039261`
- output_shape: `128x7168`
- output_all_finite: `true`
- diagnostic_only: `true`
- valid_for_default: `false`
- perf_database: `false`
