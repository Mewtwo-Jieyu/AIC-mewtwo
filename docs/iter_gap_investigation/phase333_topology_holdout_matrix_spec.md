# Phase333 Topology Holdout Matrix Spec

| Item | Decision |
|---|---|
| Candidate | topology_specific_cadence_boundary_candidate |
| Default AIC | No-Go |
| Runtime integration | Not allowed in this phase |
| PerfDatabase | Not written |
| Run status | not_run |

This file defines the next validation matrix only. It does not run any scenario and does not create evidence artifacts.

The error threshold is must_be_defined_before_gpu_run. That label is not a pass threshold and cannot be treated as validation success.

Failure is explicit: missing_artifact, worker_payload_divergence, posthoc_threshold, unknown_key, or direction_conflict_unresolved.

| Topology | Shape | Control | Holdout |
|---|---|---|---|
| tp8_dp1_ep8 | isl12000_osl2000_batch128 | tp8ep8-12k2k-bt12000 | tp8ep8-12k2k-bt65536 |
| tp4_dp2_ep8 | isl12000_osl2000_batch128 | tp4dp2ep8-12k2k-bt12000 | tp4dp2ep8-12k2k-bt65536 |
| tp8_dp1_ep8 | isl4000_osl2000_batch128 | tp8ep8-4k2k-bt4000 | tp8ep8-4k2k-bt65536 |
| tp4_dp2_ep8 | isl4000_osl2000_batch128 | tp4dp2ep8-4k2k-bt4000 | tp4dp2ep8-4k2k-bt65536 |

| Flag | Value |
|---|---|
| diagnostic_only | true |
| valid_for_default | false |
| perf_database | false |
