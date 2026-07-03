# Phase416 MoE Charge Trace

Phase416 只做离线查询链追踪；不改 runtime、PerfDatabase 或 gate。

## Verdict

- verdict: `moe_sol_prefill_misapplied_plus_ep_dispatch_fallback_undercharge`
- Phase417 target: `split_prefill_moe_charge_and_restore_ep8_alltoall_charge`
- roofline-lifted mixed step: `1913.521971` ms
- Default AIC: `No-Go`

## Query Trace

| component | phase | input tokens | scaled tokens | expected tokens | path | table | returned ms | roofline ms |
|---|---|---:|---:|---:|---|---|---:|---:|
| moe_compute | mixed_prefill | 32006 | 16002 | 32000 | phase397v_int4_wo_calibrated_sol | moe_perf_not_used | 52.706517 | 455.757015 |
| ep_dispatch_combine | mixed_prefill | 32006 | 8001 | 32000 | fallback_tp_dp_collectives | custom_allreduce+nccl | 136.511523 | 489.335467 |
| moe_compute | decode | 9 | 18 | 18 | phase397v_int4_wo_calibrated_sol | moe_perf_not_used | 3.055296 | 0.000000 |
| ep_dispatch_combine | decode | 9 | 2 | 9 | fallback_tp_dp_collectives | custom_allreduce+nccl | 2.445050 | 0.000000 |

## PerfDB Coverage

| component | table | min | max | contains scaled | contains expected |
|---|---|---:|---:|---|---|
| moe_compute | moe_perf | 1 | 32768 | true | true |
| ep_dispatch_combine | vllm_module_perf | 1 | 8192 | false | false |

## Diagnosis

- MoE: int4_wo 在 h200_sxm/vLLM 0.19.0 下提前走 Phase397v decode 锚定 calibrated-SOL；`moe_perf.txt` 有 32k 覆盖，但这次没有被查。
- EP dispatch/combine: 32k mixed prefill 的 exact module bucket 不存在，fallback 只按 TP/DP collectives 的 scaled token 计费，不是 EP8 all-to-all token-expert 字节口径。
- decode 侧也走同类路径，但 token 很小，只表现为小的固定缺口，不解释 32k prefill 主 gap。

## Boundary

- GPU/SSH: not used in Phase416.
- Runtime/PerfDatabase/gate: not modified.
- Phase405 penalty: not read.
- Default AIC: No-Go.
