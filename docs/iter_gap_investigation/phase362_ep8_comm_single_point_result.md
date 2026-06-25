# Phase362 EP8 Comm Single Point Result

| Item | Result |
|---|---|
| Verdict | EP8 comm single point pass |
| Default AIC | No-Go |
| PerfDatabase | not written |
| Evidence status | diagnostic-only single point |

## Result

- Phase361 artifact: `/mnt/shared-storage-user/ailab-sys/zhaojieyu/aic/phase361_ep8_comm_single_point_c68cfa1`
- Phase361 pass only proves the runner-level EP comm single point is measurable.
- `alltoall` is only the route label; the actual backend is `allgather_reducescatter`.
- Runtime manager was captured as `AgRsAll2AllManager`.
- `0.590688` ms is not a PerfDatabase row and is not default AIC evidence.
- The next decision should be shape expansion or a sufficiency gate, not direct e2e/default AIC promotion.

## Captured Row

- ok: `true`
- worker: `worker-892rz`
- vllm_version: `0.19.0`
- source_root: `/usr/local/lib/python3.12/dist-packages/vllm`
- measurement_boundary: `vllm_ep_group_dispatch_router_logits_plus_combine`
- backend: `allgather_reducescatter`
- manager: `AgRsAll2AllManager`
- latency_ms: `0.590688`
- cleanup: `true`
- diagnostic_only: `true`
- valid_for_default: `false`
- perf_database: `false`
