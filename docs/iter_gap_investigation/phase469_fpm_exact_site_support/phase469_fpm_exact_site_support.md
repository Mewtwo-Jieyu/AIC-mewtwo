# Phase469A Exact Site Support Contract

Status: `READY_FOR_NATIVE_GRID_INPUT`.

| Topology | Phase | Queries | Uses | Sites | Singleton sites | Curve points |
|---|---|---:|---:|---:|---:|---|
| tp4pp1dp2moetp1ep8cp1 | decode | 57916 | 231784 | 57 | 1 | 1-25306 |
| tp4pp1dp2moetp1ep8cp1 | prefill | 498 | 1859 | 324 | 298 | 1-68 |
| tp8pp1dp1moetp1ep8cp1 | decode | 58833 | 116192 | 67 | 0 | 2-19659 |
| tp8pp1dp1moetp1ep8cp1 | prefill | 802 | 1394 | 575 | 541 | 1-97 |

No native grid was supplied, so collection coverage is not evaluated.
Native-grid identity must bind both the Phase469B preflight and Phase469C canary manifests.
Phase469B records only the expected quant runtime; Phase469C must attest the loaded quant runtime.
The runtime grid digest is the opaque Dynamo value extracted by the pinned native reader.
Coordinate and measurement digests are separate order-independent integrity checks.
The guard permits only exact coordinates or a strict bracket inside the same site.
It does not calculate latency; interpolation remains delegated to the pinned upstream implementation.
The seven repository artifacts total approximately 17 KiB; no full descriptor CSV is stored.

`diagnostic_only=true`, `valid_for_default=false`, `perf_database=false`, `Default AIC=No-Go`.
