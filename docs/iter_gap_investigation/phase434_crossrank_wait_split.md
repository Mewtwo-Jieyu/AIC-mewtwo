# Phase434 Cross-Rank Wait Split

Phase434 is report-only. It reuses Phase433 torch-profiler traces and does not modify runtime, PerfDatabase, or gate.

## Verdict

- verdict: `crossrank_split_low_confidence_expert_imbalance_not_confirmed`
- dominant mechanism: `crossrank_alignment_low_confidence`
- clock alignment gate: `low_confidence` (p95 abs residual 5334.809249 ms)
- dual imbalance gate: `failed`
- precision boundary: `torch_profiler_crossrank_end_alignment_estimate_not_nsys_wire_split`
- Phase435 target: `separate_nccl_wait_vs_transfer_or_build_serving_state_ep_model`

## Collective Split

| category | wait ms/step | transfer ms/step | matched ms/step | coverage pct | observed real ms/step | sim ms/step | excess ms/step |
|---|---:|---:|---:|---:|---:|---:|---:|
| all_collective | 6326.291874 | 120.483373 | 260.551348 | 11.498684 | 2265.923137 | 739.781362 | 1526.141775 |
| ep_a2a | 566.877976 | 20.510524 | 27.369490 | 1.427172 | 1917.743178 | 497.491058 | 1420.252120 |
| collective_other | 3770.874493 | 75.209534 | 170.198907 | 48.882454 | 348.179960 | 242.290304 | 105.889656 |

## Expert Imbalance Check

| item | value |
|---|---:|
| MoE mean ms/step | 1143.462225 |
| MoE hot ms/step | 1378.906294 |
| hot/mean ratio | 1.215359 |
| hot excess ms/step | 235.444069 |
| wait vs hot-excess error pct | 96.278324 |
| MoE excess from ratio ms | 75.960740 |
| MoE excess ratio error pct | 90.393773 |

## Clock Offsets

| rank | offset ms |
|---|---:|
| dp0_rank0 | 0.000000 |
| dp0_rank1 | 0.006905 |
| dp0_rank2 | -0.000026 |
| dp0_rank3 | 0.000005 |
| dp1_rank0 | 1120.699897 |
| dp1_rank1 | 1120.785719 |
| dp1_rank2 | 1120.878777 |
| dp1_rank3 | 1120.998704 |

## Boundary

- `nsys` was not available in Phase433; this split is based on torch-profiler cross-rank end-time alignment.
- DP replicas are phase-skewed in the captured window, so matched collective rows are a subset and the observed real column is the bucket-safe value.
- NCCL wait and wire transfer remain an estimate, not an nsys-grade split.
- Default AIC: No-Go.
