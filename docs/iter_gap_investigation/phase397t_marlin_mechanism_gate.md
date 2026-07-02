# Phase397t Marlin Mechanism Gate

Phase397t H200 mechanism gate failed. No route reached the no-k serve anchor band.

| Route | Latency ms/layer | Ratio vs anchor | Gate |
|---|---:|---:|---|
| power_law | 0.450844 | 3.372852x | route_gate_failed |
| power_law_rank0_compact | 0.488430 | 3.654046x | route_gate_failed |
| power_law_eplb | 0.429909 | 3.216239x | route_gate_failed |
| balanced | 0.565736 | 4.232386x | route_gate_failed |
| power_law_batched_rank0 | 0.749925 | 5.610339x | route_gate_failed |

Decision: `mechanism_gate_failed_report_only`.

Do not append `moe_perf.txt`, do not retire `k=0.312`, and do not persist validate 0.19.0 repoint.

Default AIC remains No-Go.
