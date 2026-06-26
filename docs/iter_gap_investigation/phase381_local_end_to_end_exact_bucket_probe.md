# Phase381 Local End-to-End Exact-Bucket Probe

Phase381 runs a local BaseBackend.run_static probe through the real MoE.query and MoEDispatch.query vLLM module binding.
It proves simulator runtime binding local exact lookup, not vLLM GPU runtime evidence.
It is local evidence only: no GPU, no SSH, no new PerfDatabase rows, and no Default AIC enablement.

| Gate | Verdict |
|---|---|
| FusedMoE runner probe | pass: raw_tokens 32 maps to bucket_tokens 16; exact key `kimi-k2.5|h200_sxm|0.19.0|tp4dp2ep8|16|fusedmoe_runner_compute|CompressedTensorsWNA16MarlinMoEMethod`, latency 0.239764 ms |
| EP8 dispatch/combine probe | pass: raw_tokens 64 maps to bucket_tokens 16; exact key `kimi-k2.5|h200_sxm|0.19.0|tp4dp2ep8|16|ep8_comm_dispatch_combine|CompressedTensorsWNA16MarlinMoEMethod`, pre latency 0.083776 ms, post latency 0.000000 ms |
| bucket 128 | fail-fast inside Kimi h200_sxm vLLM 0.19.0 tp4dp2ep8 scope |
| non-scope | non-Kimi and non-tp4dp2ep8 do not use vLLM module table |
| combined full model | blocked-by-bucket-mapping: same raw_tokens maps to ep8_bucket=raw//4 and fusedmoe_bucket=raw//4*2; not treated as failure and not bypassed |
| PerfDatabase | no new rows |
| GPU / SSH | not allowed |
| Default AIC | No-Go |

Phase382 may decide whether this local probe evidence is sufficient for a larger exact-bucket runtime gate.
It must not turn this probe into default AIC readiness.
