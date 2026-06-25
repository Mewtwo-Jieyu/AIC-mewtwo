# Phase366 EP8 Comm Minimal Shape Sweep Result

| Item | Result |
|---|---|
| Verdict | Phase365 EP8 comm minimal shape sweep passed |
| Default AIC | No-Go |
| PerfDatabase | not written |
| Evidence status | diagnostic-only shape sweep |

## Result

- Phase365 artifact: `/mnt/shared-storage-user/ailab-sys/zhaojieyu/aic/phase365_ep8_comm_minimal_shape_sweep_d422088`
- `128` is not included; it remains a smoke-only point from Phase361/364.
- All 7 Phase124 real buckets passed with backend `allgather_reducescatter`.
- Runtime manager was captured as `AgRsAll2AllManager` for every bucket.
- Rank errors are `0`, cleanup is `true`, and GPU/process residue is `false` for every row.
- These latencies are diagnostic only and cannot be interpolated or extrapolated into a PerfDatabase curve.
- They are not default AIC evidence.

## Bucket Results

| bucket_tokens | latency_ms | backend | manager | rank_error |
|---:|---:|---|---|---:|
| 1 | 0.157088 | allgather_reducescatter | AgRsAll2AllManager | 0 |
| 15 | 0.109920 | allgather_reducescatter | AgRsAll2AllManager | 0 |
| 16 | 0.083776 | allgather_reducescatter | AgRsAll2AllManager | 0 |
| 241 | 0.288960 | allgather_reducescatter | AgRsAll2AllManager | 0 |
| 1808 | 1.192704 | allgather_reducescatter | AgRsAll2AllManager | 0 |
| 2048 | 1.237760 | allgather_reducescatter | AgRsAll2AllManager | 0 |
| 8192 | 2.237024 | allgather_reducescatter | AgRsAll2AllManager | 0 |
