# Phase375 vLLM Module Perf Materialization Spec

Phase375 defines how Phase366 EP8 comm rows and Phase369 FusedMoE runner rows may be materialized into `vllm_module_perf.txt` later. It does not write the data file.

| Item | Decision |
|---|---|
| Candidate rows | 14 candidate rows |
| Module boundaries | `fusedmoe_runner_compute` and `ep8_comm_dispatch_combine` |
| Buckets | `1/15/16/241/1808/2048/8192` |
| Smoke bucket | bucket `128` remains excluded |
| Target file | `vllm_module_perf.txt` |
| Hardware key | `h200_sxm` |
| Key columns | `model;hardware;vllm_version;topology;bucket_tokens;module_boundary;quant_runtime` |
| Lookup policy | exact lookup only |
| Kernel source | kernel source remains metadata only |
| Power default | `0.0`; energy is `power*latency` |
| Duplicate key policy | fail fast |
| Missing exact key policy | fail fast |
| Real data write | blocked until Phase376 |
| Default AIC | No-Go |
| PerfDatabase | not written |

## Candidate Rows

| module_boundary | rows | source |
|---|---:|---|
| `fusedmoe_runner_compute` | 7 | `phase369_fusedmoe_runner_shape_sweep_result` |
| `ep8_comm_dispatch_combine` | 7 | `phase366_ep8_comm_minimal_shape_sweep_result` |

## Boundary

Phase375 is a local diagnostic spec. It does not use SSH or GPU, does not create `systems/.../vllm_module_perf.txt`, does not interpolate, does not extrapolate, and does not create a PerfDatabase curve.
