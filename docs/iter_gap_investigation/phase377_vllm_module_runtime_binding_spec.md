# Phase377 vLLM Module Runtime Binding Spec

Phase377 defines how `vllm_module_perf.txt` may be bound into runtime later. It does not modify `operations.py`.

| Item | Decision |
|---|---|
| Phase376 table | 14 exact keys required |
| MoE binding | `MoE.query(...)` -> `fusedmoe_runner_compute` |
| EP8 comm binding | `MoEDispatch.query(...)` -> `ep8_comm_dispatch_combine` |
| Hardware source | `database.system`, must be `h200_sxm` |
| vLLM version source | `database.version`, must be `0.19.0` |
| Topology | `tp4dp2ep8` |
| Quant runtime | `CompressedTensorsWNA16MarlinMoEMethod` |
| Buckets | `1/15/16/241/1808/2048/8192` |
| Rejected bucket | bucket `128` remains rejected |
| Bucket policy | exact only; no nearest bucket, interpolation, or extrapolation |
| Kernel source | metadata only, not a lookup key |
| operations.py | unchanged |
| Default AIC | No-Go |

Phase378 may make a minimal `operations.py` binding. That implementation must fail fast when a runtime token bucket is not one of `1/15/16/241/1808/2048/8192`.
