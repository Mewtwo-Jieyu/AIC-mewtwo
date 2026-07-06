# Phase417 MoE/EP Charge Fix

Phase417 修改 runtime 计费路径，但不改 PerfDatabase 数据、不改 gate、不使用 GPU。

## Verdict

- verdict: `phase417_moe_ep_charge_fixed`
- next phase: `anchor_revalidate_phase400_phase403`
- anchor revalidation: `required_before_default`
- Default AIC: `No-Go`

## Component Result

| component | before path | after path | before scaled | after scaled | expected | before ms | after ms | roofline ms | remaining ms | status |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| moe_compute | phase397v_int4_wo_calibrated_sol | moe_perf_lookup | 16002 | 32000 | 32000 | 52.706517 | 352.717448 | 170.822563 | 0.000000 | fixed |
| ep_dispatch_combine | fallback_tp_dp_collectives | ep8_alltoall_roofline | 8001 | 32000 | 32000 | 136.511523 | 497.491058 | 489.335467 | 0.000000 | fixed |

## Summary

- mixed step current charge: `1818.637996` ms
- mixed step roofline lower-bound per-step: `1045.553790` ms
- remaining below-roofline gap: `0.000000` ms
- EP dispatch/combine is fixed in this phase.
- MoE now uses the measured large-context table and clears the corrected Phase415 physical lower-bound check.
- Anchor revalidation is required before any Default AIC readiness claim.

## Boundary

- GPU/SSH: not used.
- PerfDatabase data file: not modified.
- Gate/default readiness: not modified.
- Phase405 penalty: not read.
