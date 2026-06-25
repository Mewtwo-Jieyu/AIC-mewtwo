# Phase367 FusedMoE Runner Shape Sweep Spec

| Item | Result |
|---|---|
| Verdict | FusedMoE runner shape sweep spec only |
| Default AIC | No-Go |
| PerfDatabase | not written |
| GPU | not run in Phase367 |

## Spec

- Measurement boundary is `FusedMoE.forward() runner level`.
- Shape sweep buckets are fixed to `1`, `15`, `16`, `241`, `1808`, `2048`, `8192`.
- `128` remains smoke-only and is excluded from the shape sweep.
- Runtime quant method must be `CompressedTensorsWNA16MarlinMoEMethod`.
- The kernel source is metadata only and is not a PerfDatabase lookup key.
- Direct smoke requires `LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:/usr/local/nvidia/lib64` and `VLLM_ENABLE_CUDA_COMPATIBILITY=1` before process start.
- Stop on PTX failure, forward-context failure, quant runtime drift, or GPU/process residue.
- Phase368 may run the minimal FusedMoE runner shape sweep on the recorded worker only.

## Future GPU Entry

- Recorded only: `ssh -CAXY ws-faaf0de74ef9a14d-worker-gn6kz.zhaojieyu+root.ailab-sys.pod@h.pjlab.org.cn`
- Phase367 does not permit SSH or GPU execution.
