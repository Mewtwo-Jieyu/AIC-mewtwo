# Phase397s Int4 WO Route Gate

Phase397s stopped report-only. The route gate did not justify retiring the Phase397m calibrated MoE path.

| Route | Latency ms/layer | Ratio vs 0.1588 ms anchor | Gate |
|---|---:|---:|---|
| power_law | 0.470627 | 2.963337x | route_gate_failed |
| balanced | 0.566186 | 3.565026x | route_gate_failed |
| power_law_eplb | 0.426923 | 2.688151x | route_gate_failed |

Decision: `route_gate_failed_report_only`.

Do not append `moe_perf.txt` rows, do not persist the validate 0.19.0 repoint as a passing state, and do not relax thresholds.
The missing family remains `moonshotai/Kimi-K2.5:int4_wo:moe_tp16:moe_ep1`.

Default AIC remains No-Go.
