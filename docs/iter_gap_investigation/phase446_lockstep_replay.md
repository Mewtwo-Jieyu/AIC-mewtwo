# Phase446 Lockstep Replay

- Verdict: `lockstep_busy_replay_passes`.
- Coupling granularity: event rows group into `4` TP rows per DP0 step and `4` TP rows per DP1 step.
- Event grouping mismatches: `0`; event/wall key mismatches: `0`.
- Primary gate: `steady_decode_ge_40` must reconstruct real window wall within 10%.
- Source basis: vLLM 0.19.0 `vllm/v1/worker/dp_utils.py:78-90` pads each DP rank to the max token count; `dp_utils.py:153-159` enables that padding when cudagraph/ubatching requires it; `gpu_model_runner.py:3615-3639` coordinates DP ranks each step and re-dispatches cudagraph with the padded token count.

## Replay Windows
| window | cycles | replay_busy_ms | real_wall_ms | error_pct | sum_max_wall_ms | sum_max_error_pct | gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all_paired | 11610 | 787258.623959 | 787149.730000 | 0.013834 | 826659.380000 | -4.766263 | passed |
| steady_decode_ge_40 | 10006 | 713680.552242 | 709595.330000 | 0.575712 | 747886.040000 | -4.573623 | passed |
| high_decode_ge_45 | 9806 | 493163.370243 | 492638.470000 | 0.106549 | 526441.510000 | -6.321337 | passed |
| peak_decode_ge_56 | 405 | 14856.658161 | 16025.440000 | -7.293290 | 16123.560000 | -7.857457 | passed |

## Fixture Seed Cycles
| cycle | e0_ctx | e0_decode | e1_ctx | e1_decode | e0_busy | e1_busy | coupled_busy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6042 | 7959 | 41 | 7958 | 42 | 1134.536255 | 1136.847046 | 1136.847046 |
| 6041 | 7959 | 41 | 7958 | 42 | 1136.810181 | 1135.398315 | 1136.810181 |
| 32 | 7969 | 31 | 7970 | 30 | 1135.030518 | 1135.612061 | 1135.612061 |
| 31 | 7970 | 30 | 7971 | 29 | 1135.491699 | 1124.573608 | 1135.491699 |
| 6068 | 7953 | 47 | 7953 | 47 | 1134.803467 | 1135.430664 | 1135.430664 |
| 6039 | 7959 | 41 | 7958 | 42 | 1135.258667 | 1135.367798 | 1135.367798 |
| 6038 | 7959 | 41 | 7958 | 42 | 1135.350586 | 1130.349731 | 1135.350586 |
| 6040 | 7959 | 41 | 7958 | 42 | 1135.319580 | 1135.203613 | 1135.319580 |
| 6067 | 7954 | 46 | 7954 | 46 | 1135.313354 | 1133.889282 | 1135.313354 |
| 33 | 7968 | 32 | 7969 | 31 | 1135.146118 | 1135.245850 | 1135.245850 |

## Next Gate

Step 0 passes only the offline mechanism gate. It does not make Phase442 event rows ingestible because that run failed the overhead gate. Runtime DP lockstep coupling may proceed to a red/green implementation only if the same fixture-style max coupling is used and the next B2b run passes its overhead and reproducibility gates.
