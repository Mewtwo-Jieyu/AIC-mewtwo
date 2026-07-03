# Phase415 Prefill Charge Components

Phase415 只做离线组件审计；不改 runtime、PerfDatabase 或 gate。

## Verdict

- verdict: `moe_compute_undercharged`
- Phase416 target: `audit_moe_compute_prefill_charge`
- consistency gate: `passed`
- reconstructed missing ms: `1182454.006820` / target `1244249.221128`

## Mixed Prefill Components

| component | sim ms | sim mean ms | roofline lower ms | roofline gap ms | status |
|---|---:|---:|---:|---:|---|
| decode_mla_attention | 2296.278253 | 6.916501 | 0.000000 | 0.000000 | ok |
| ep_dispatch_combine | 45321.825620 | 136.511523 | 162459.374933 | 117137.549313 | below_roofline |
| gemm_other | 84877.285210 | 255.654474 | 0.000000 | 0.000000 | ok |
| moe_compute | 17498.563759 | 52.706517 | 151311.328892 | 133812.765133 | below_roofline |
| other | 20434.312776 | 61.549135 | 0.000000 | 0.000000 | ok |
| prefill_mla_attention | 133470.333539 | 402.019077 | 107305.513653 | 0.000000 | ok |
| tp_comm | 80440.380936 | 242.290304 | 20645.878898 | 0.000000 | ok |

## Decode Components

| component | sim ms | note |
|---|---:|---|
| decode_ep_dispatch_combine | 2.445050 | not_checked_fixed_gap_candidate |
| decode_gemm_other | 3.601032 | not_checked_fixed_gap_candidate |
| decode_mla_attention | 8.951647 | not_checked_fixed_gap_candidate |
| decode_moe_compute | 3.055296 | not_checked_fixed_gap_candidate |
| decode_other | 0.568870 | not_checked_fixed_gap_candidate |
| decode_tp_comm | 0.717473 | not_checked_fixed_gap_candidate |

## Roofline Assumptions

- hidden=7168, inter=2048, topk=8, moe_layers=60, attention_layers=61.
- H200 tensor roofline=1979.0 TFLOP/s, NVLink budget=900.0 GB/s.
- lower_bound 是物理下界，只用于找明显不可能的 undercharge，不代表真实耗时。

## Boundary

- GPU/SSH: not used in Phase415.
- Runtime/PerfDatabase/gate: not modified.
- Phase405 penalty: not read.
- Default AIC: No-Go.
