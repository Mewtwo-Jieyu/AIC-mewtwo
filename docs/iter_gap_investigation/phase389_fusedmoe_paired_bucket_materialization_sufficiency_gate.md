# Phase389 FusedMoE Paired Bucket Materialization Sufficiency Gate

| Item | Result |
|---|---|
| Verdict | materialization sufficiency gate passed |
| existing rows | 14 |
| planned added rows | 7 |
| future row count | 21 |
| Default AIC | No-Go |
| PerfDatabase | not written |

## Decision

- Phase388 has seven diagnostic FusedMoE paired bucket rows with finite output and clean cleanup.
- The existing 14 rows are preserved: seven FusedMoE original rows and seven EP8 comm rows.
- Phase390 may add only the seven FusedMoE paired rows for buckets `2`, `30`, `32`, `482`, `3616`, `4096`, `16384`.
- Bucket `128` remains forbidden.
- Lookup stays exact-only; no nearest lookup, interpolation, or extrapolation is allowed.
- Kernel source remains metadata only and is not part of the lookup key.
- Phase389 does not write `vllm_module_perf.txt`, write PerfDatabase rows, use GPU, SSH, or open Default AIC.
