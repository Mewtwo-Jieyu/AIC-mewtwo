# Phase457 TP8 protocol residual

Verdict: `tp8_protocol_unification_closes_gate_with_secondary_preemption_residual`.

## Protocol Projection

| metric | value | status | note |
|---|---:|---|---|
| n128_reference_real_output_tok_s_gpu | 133.528 |  | current validate reference; N=128 closed-set |
| current_sim_output_tok_s_gpu | 166.1170248292807 |  | Phase456 current sim after TP8 capacity wiring |
| n512_proxy_overhead_off_output_tok_s_gpu | 146.37653533465672 |  | N=512; C=128; B2b overhead-off proxy, not a replacement reference |
| n512_proxy_overhead_on_output_tok_s_gpu | 146.31592065302334 |  | used only as patch-overhead cross-check |
| error_vs_n128_reference | 1.2440613566389125 | fail | existing mixed-protocol validation point |
| error_vs_n512_proxy_overhead_off | 1.134861024340355 | pass | predicts the vanilla N=512 recollect payoff |
| error_vs_n512_proxy_overhead_on | 1.1353311662044905 | pass | patch on/off agrees if close to overhead_off |
| n512_protocol_gain_vs_n128 | 1.0962235286580846 | pass | real reference throughput gain from N=128 to N=512 proxy |

## Dynamics Check

| metric | value | status | note |
|---|---:|---|---|
| real_mixed_decode_batch_p50 | 51.0 |  | count=4192; mean=50.213740; p10=50.000000; p90=60.000000 |
| sim_avg_decode_batch | 54.576 | pass | relative_delta=0.070 |
| real_decode_batch_p50 | 59.0 |  | count=140600; mean=56.744865; p10=32.000000; p90=65.000000 |
| real_running_p50 | 56.0 |  | p10=52.000000; p90=64.300000; n=88 |
| real_preemption_delta | 110.0 | informational | engine 0: 110 |
| sim_preemptions | 189.0 | warn | real_preemptions=110; ratio=1.718; sim_recompute_tokens=1588814; sim_trace_tput=169.156 |

## Decision

| metric | value | status | note |
|---|---|---|---|
| phase457_verdict | tp8_protocol_unification_closes_gate_with_secondary_preemption_residual | pass | N=512 protocol unification is enough to reach <=15% by proxy; sim preemption remains higher than real and is stored as a secondary residual. |

The Phase454 N=512 overhead-off run is a proxy, not a clean reference. It is only used to decide whether Phase457 Step2 is needed before vanilla recollection.
`diagnostic_only=true valid_for_default=false perf_database=false`; Default AIC stays No-Go until the vanilla N=512 TP8 reference is collected and validate is rerun.
