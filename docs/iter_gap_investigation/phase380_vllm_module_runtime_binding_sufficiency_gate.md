# Phase380 vLLM Module Runtime Binding Sufficiency Gate

Phase380 records that runtime binding implemented, but it is not default AIC evidence.

| Gate | Verdict |
|---|---|
| runtime binding | implemented |
| exact lookup guard | model;hardware;vllm_version;topology;bucket_tokens;module_boundary;quant_runtime |
| runtime model | moonshotai/Kimi-K2.5 |
| PerfDB model key | kimi-k2.5 |
| topology | tp4dp2ep8 |
| buckets | 1/15/16/241/1808/2048/8192; bucket `128` remains rejected |
| integration level | unit + validator evidence only |
| validator | 1.50x/1.47x/1.79x |
| PerfDatabase | no new rows, no schema change |
| GPU / SSH | not allowed |
| Default AIC | No-Go |

Phase381 is the next allowed phase: local end-to-end exact-bucket probe only.
It must not open default AIC, fit curves, interpolate, extrapolate, or write new PerfDatabase data.
