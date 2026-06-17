# Phase303 Deeper Trace Topology Family

This family audit combines the Phase287 tp8 diagnostic and the Phase301 tp4dp2 diagnostic.

| Item | Value |
|---|---|
| Family | deeper_trace_12k2k_topology_family |
| tp8 ratio | 0.969300 (holdout_slower) |
| tp4dp2 ratio | 1.321996 (holdout_faster) |
| Budget ceiling rejected | true |
| Default AIC | No-Go |
| Diagnostic only | true |

Both topology traces reject configured max_num_batched_tokens as a linear runtime cost: the high-budget holdouts do not fill the aggregate budget. The throughput direction is topology-dependent: tp8 is slightly slower while tp4dp2 is clearly faster.

This means the evidence cannot support a global correction, interpolation, or extrapolation. The next modeling layer should stay topology-specific and focus on cadence and boundary mechanisms.

Flags: diagnostic_only=true, valid_for_default=false, perf_database=false.
