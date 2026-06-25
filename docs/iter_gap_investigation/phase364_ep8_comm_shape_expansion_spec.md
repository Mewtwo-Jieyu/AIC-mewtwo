# Phase364 EP8 Comm Shape Expansion Spec

| Item | Result |
|---|---|
| Verdict | shape expansion spec only |
| Default AIC | No-Go |
| PerfDatabase | not written |
| GPU | not run in Phase364 |

## Gate Result

- Phase363 proved the EP8 single point is insufficient for a PerfDatabase curve.
- `128` remains a runner/comm smoke point and is excluded from real coverage.
- Real bucket coverage is fixed to Phase124 buckets: `1`, `15`, `16`, `241`, `1808`, `2048`, `8192`.
- Every future GPU bucket must runtime capture `backend` and `manager`.
- Direct smoke requires `LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:/usr/local/nvidia/lib64` and `VLLM_ENABLE_CUDA_COMPATIBILITY=1` before process start.
- Stop on PTX failure, forward-context failure, unknown backend, GPU/process residue, or backend drift away from `allgather_reducescatter`.
- The next allowed phase is Phase365 minimal shape sweep, not a full GPU matrix.

## Future GPU Entry

- Recorded only: `ssh -CAXY ws-faaf0de74ef9a14d-worker-gn6kz.zhaojieyu+root.ailab-sys.pod@h.pjlab.org.cn`
- Phase364 does not permit SSH or GPU execution.
