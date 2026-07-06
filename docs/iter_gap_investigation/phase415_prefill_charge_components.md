# Phase415 Prefill Charge Components

Phase415 只做离线组件审计；不改 runtime、PerfDatabase 或 gate。

## Verdict

- verdict: `offline_roofline_inconclusive`
- Phase416 target: `gpu_profile_single_prefill_iteration`
- consistency gate: `passed`
- reconstructed missing ms: `1182454.006820` / target `1244249.221128`

## Mixed Prefill Components

| component | sim ms | sim mean ms | roofline lower ms | roofline gap ms | status |
|---|---:|---:|---:|---:|---|
| decode_mla_attention | 2296.278253 | 6.916501 | 0.000000 | 0.000000 | ok |
| ep_dispatch_combine | 165167.031182 | 497.491058 | 162459.374933 | 0.000000 | ok |
| gemm_other | 84877.285210 | 255.654474 | 0.000000 | 0.000000 | ok |
| moe_compute | 117102.192633 | 352.717448 | 56713.090886 | 0.000000 | ok |
| other | 20434.312776 | 61.549135 | 0.000000 | 0.000000 | ok |
| prefill_mla_attention | 133470.333539 | 402.019077 | 107305.513653 | 0.000000 | ok |
| tp_comm | 80440.380936 | 242.290304 | 20645.878898 | 0.000000 | ok |

## Decode Components

| component | sim ms | note |
|---|---:|---|
| decode_ep_dispatch_combine | 0.031093 | not_checked_fixed_gap_candidate |
| decode_gemm_other | 3.601032 | not_checked_fixed_gap_candidate |
| decode_mla_attention | 8.951647 | not_checked_fixed_gap_candidate |
| decode_moe_compute | 3.055296 | not_checked_fixed_gap_candidate |
| decode_other | 0.568870 | not_checked_fixed_gap_candidate |
| decode_tp_comm | 0.717473 | not_checked_fixed_gap_candidate |

## Roofline Assumptions

- hidden=7168, inter=2048, topk=8, moe_layers=60, moe_ep=8, attention_layers=61.
- attention roofline=1979.0 TFLOP/s, MoE WNA16 conservative roofline=990.0 TFLOP/s, NVLink budget=900.0 GB/s.
- MoE lower bound uses EP-local token-expert rows and 6hi gated MLP work.
- lower_bound 是物理下界，只用于找明显不可能的 undercharge，不代表真实耗时。

## Boundary

- GPU/SSH: not used in Phase415.
- Runtime/PerfDatabase/gate: not modified.
- Phase405 penalty: not read.
- Default AIC: No-Go.
