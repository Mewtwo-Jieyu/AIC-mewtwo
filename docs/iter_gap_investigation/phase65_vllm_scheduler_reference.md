# Phase 65 vLLM Scheduler Reference

## Source

Phase65 复用 Phase62 artifact，不跑远端、不重新采集：

`docs/iter_gap_investigation/phase62_scheduler_alignment_10k2k_b32/scheduler_alignment_rows_dedup_by_alignment_dp.csv`

| 字段 | 值 |
|---|---|
| scenario | `10k2k_b32_bt8192` |
| topology | `tp4dp2moetp1ep8` |
| row key | `(alignment_key, dp_rank)` |
| key type | `engine_core_dp_step` |
| total rows | `4003` |
| output qualification | `valid_for_default=false`, `perf_database=false`, `diagnostic_only=true` |

## DP Step Shape

| DP | rows | phase distribution | step range |
|---:|---:|---|---|
| 0 | `2001` | prefill `2`, mixed `0`, pure_decode `1999` | `0..2000` |
| 1 | `2002` | prefill `2`, mixed `1`, pure_decode `1999` | `0..2001` |

## Key Rows

| key | phase | context tokens | decode tokens | context reqs | decode reqs | forward regime |
|---|---|---:|---:|---:|---:|---|
| `engine_dp:0:step:0` | prefill | `8192` | `0` | `1` | `0` | `NONE:8192` |
| `engine_dp:0:step:1` | prefill | `2048` | `0` | `16` | `0` | `NONE:2048` |
| `engine_dp:0:step:2` | pure_decode | `0` | `16` | `0` | `16` | `FULL:16` |
| `engine_dp:1:step:0` | prefill | `8192` | `0` | `1` | `0` | `NONE:8192` |
| `engine_dp:1:step:1` | prefill | `1808` | `0` | `1` | `0` | `NONE:1808` |
| `engine_dp:1:step:2` | mixed | `240` | `1` | `15` | `1` | `NONE:248` |
| `engine_dp:1:step:2001` | pure_decode | `0` | `15` | `0` | `15` | `FULL:16` |

## Reference Semantics

The Phase62 row is a DP-local EngineCore scheduler descriptor joined to the worker runtime row. It includes the scheduler split, request split, forward padded token count, forward regime, and cudagraph runtime mode for the same DP step.

It is not comparable to a cb_sim row unless cb_sim can produce the same DP-local scheduling semantics. A matching key string alone is insufficient.

## Remote Note

No remote or GPU was used in Phase65. If a later phase needs to re-capture descriptor-only rows, the current cluster entry is:

`ssh -CAXY ws-faaf0de74ef9a14d-worker-tc88x.zhaojieyu+root.ailab-sys.pod@h.pjlab.org.cn`

That future run must remain descriptor-only and must restore the GPU occupancy script after use.
