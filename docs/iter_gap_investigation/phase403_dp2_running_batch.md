# Phase403 DP2 running batch

Phase403 recollects only the two DP2 clean points with prefix caching off and /metrics polling. It does not change scheduler, runtime, PerfDatabase, or gates.

## Verdict

| scenario | real tok/s/gpu | sim tok/s/gpu | error | real running max | sim running global | verdict |
|---|---:|---:|---:|---:|---:|---|
| K2.5-tp4ep8dp2-8k2k | 139.063168 | 271.958961 | 1.955651 | 103.000000 | 112.000000 | dp_ep_comm_or_sync_under_modeled |
| K2.5-tp4ep8dp2-32k3k | 49.063908 | 92.590110 | 1.887133 | 20.000000 | 20.000000 | dp_ep_comm_or_sync_under_modeled |

Interpretation:

- If real running were far below sim running, the next target would be capacity/occupancy.
- When real running is near or reaches the sim DP2 running target while throughput is still slower, the next target is DP/EP communication or synchronization not represented in the current composition.

Boundary:

- gpu_allowed=true and ssh_allowed=true because this phase recollects evidence.
- runtime_modified=false, perf_database=false, valid_for_default=false, diagnostic_only=true, default_readiness=No-Go.

Next: Phase404 should implement only the mechanism selected by this attribution result.
