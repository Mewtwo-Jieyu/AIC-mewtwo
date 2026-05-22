# Phase 66 vLLM Scheduler Requirements

## Reference Artifact

Phase66 reuses Phase62 output as the fixed first-evidence target:

`docs/iter_gap_investigation/phase62_scheduler_alignment_10k2k_b32/scheduler_alignment_rows_dedup_by_alignment_dp.csv`

No remote run was used in Phase66.

## Required Shape Sequence

| DP | rows | required sequence |
|---:|---:|---|
| 0 | `2001` | `8192` prefill, `2048` prefill, then `1999` pure decode rows of `16` |
| 1 | `2002` | `8192` prefill, `1808` prefill, `240+1` mixed padded to `248`, then pure decode with final `15 -> FULL:16` tail |

## Required Key Rows

| key | phase | context tokens | decode tokens | context reqs | decode reqs | forward regime |
|---|---|---:|---:|---:|---:|---|
| `engine_dp:0:step:1` | prefill | `2048` | `0` | `16` | `0` | `NONE:2048` |
| `engine_dp:1:step:1` | prefill | `1808` | `0` | `1` | `0` | `NONE:1808` |
| `engine_dp:1:step:2` | mixed | `240` | `1` | `15` | `1` | `NONE:248` |
| `engine_dp:1:step:2001` | pure_decode | `0` | `15` | `0` | `15` | `FULL:16` |

## Required Boundaries

| Field | Requirement |
|---|---|
| `alignment_key` | `engine_dp:<dp_rank>:step:<engine_step_id>` |
| `alignment_key_type` | `engine_core_dp_step` |
| `scheduled_*` | DP-local scheduler split |
| `forward_token_count` | runtime padded forward shape |
| `cudagraph_runtime_mode` | concrete `NONE` or `FULL` |
| `valid_for_default` | `false` |
| `perf_database` | `false` |
| `diagnostic_only` | `true` |

## Rejected Shortcuts

| Shortcut | Reason |
|---|---|
| use phase ordinal | not DP-local |
| use intersection-only compare | hides missing keys |
| copy `CBIterationTraceRow` | source semantics mismatch |
| emit `AIC_UNSET` graph mode | not runtime-aligned |
| add timing or residual fields | descriptor-only line |
