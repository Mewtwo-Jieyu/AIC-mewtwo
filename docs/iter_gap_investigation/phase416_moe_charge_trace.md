# Phase416 MoE Charge Trace

Phase416 只做离线查询链追踪；不改 runtime、PerfDatabase 或 gate。

## Verdict

- verdict: `moe_ep_charge_fixed`
- Phase417 target: `revalidate_phase400_phase403_anchor_scenarios`
- roofline-lifted mixed step: `1818.637996` ms
- Default AIC: `No-Go`

## Query Trace

| component | phase | input tokens | scaled tokens | expected tokens | path | table | returned ms | roofline ms |
|---|---|---:|---:|---:|---|---|---:|---:|
| moe_compute | mixed_prefill | 32006 | 32000 | 32000 | moe_perf_lookup | moe_perf | 352.717448 | 170.822563 |
| ep_dispatch_combine | mixed_prefill | 32006 | 32000 | 32000 | ep8_alltoall_roofline | roofline | 497.491058 | 489.335467 |
| moe_compute | decode | 9 | 18 | 18 | phase397v_int4_wo_calibrated_sol | moe_perf_not_used | 3.055296 | 0.000000 |
| ep_dispatch_combine | decode | 9 | 2 | 9 | fallback_tp_dp_collectives | custom_allreduce+nccl | 0.031093 | 0.000000 |

## PerfDB Coverage

| component | table | min | max | contains scaled | contains expected |
|---|---|---:|---:|---|---|
| moe_compute | moe_perf | 1 | 32768 | true | true |
| ep_dispatch_combine | vllm_module_perf | 1 | 8192 | false | false |

## Diagnosis

- MoE: mixed prefill now queries `moe_perf.txt` at the expected EP-local 32k token-expert rows instead of the Phase397v decode SOL path.
- EP dispatch/combine: mixed prefill exact-key miss now uses the EP8 all-to-all byte model instead of TP/DP collective fallback.
- Component roofline gate is now clean; remaining end-to-end gap needs anchor revalidation before any default-readiness claim.

## Boundary

- GPU/SSH: not used in Phase416.
- Runtime/PerfDatabase/gate: not modified.
- Phase405 penalty: not read.
- Default AIC: No-Go.
