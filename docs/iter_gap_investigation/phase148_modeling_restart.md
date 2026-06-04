# Phase148: Modeling Restart

## Decision

| Item | Result |
|---|---|
| Scope | Local cb_sim modeling restart |
| Remote run | No-Go |
| Default AIC | No-Go |
| PerfDatabase | No-Go |
| PR / main merge | No-Go |

Phase148 restarts modeling from the current validated cb_sim baseline. It does not turn diagnostic evidence into default latency data.

## Baseline Gate

| Metric | Display gate | Exact baseline constant |
|---|---:|---:|
| throughput max error | `1.50x` | `1.4989592822599629` |
| multi-config max error | `1.47x` | `1.4696362054928367` |
| TTFT max error | `1.79x` | `1.790056498134038` |

The exact constants keep the current baseline from regressing. The two-decimal values are the public validation readout.

## Gap Matrix

| Area | Current readout | Modeling action |
|---|---|---|
| prefill / TTFT | validation threshold path reaches `1.79x` max error | keep as baseline gate; do not add a new constant-only term |
| mixed | worker/device evidence shows `cuda_forward_ms ~= engine_result_wait_ms` for mixed, about `99.8%` of result wait | model only through runtime shape / topology / source keys |
| pure_decode | worker/device evidence shows much smaller `pure_decode` wait, about `26.8ms` mean on `10k2k_b32` | keep as guardrail; not the first modeling target |
| EP8 multi-config | multi-config gate reaches `1.47x`, with EP8 topologies in the max-error set | prioritize EP8 comm / All2All candidate key before changing model behavior |

## EP8 Comm Candidate

| Field | Value |
|---|---|
| source key | `phase148_h200_vllm_ep8_all2all_decode_candidate` |
| topology key example | `tp8dp1moetp1ep8` |
| shape key example | `decode_present:isl8000:osl2000:bs128:ctx8000:max_bt8000:max_seqs256` |
| diagnostic_only | `true` |
| valid_for_default | `false` |
| perf_database | `false` |

The existing `90ms` decode-present overhead is now attached to an explicit source, topology, and shape key in `cb_sim_scheduling`. This still does not make it clean performance data.

## MoE WNA16 Timing Gate

| Item | Gate |
|---|---|
| exact buckets | `1 / 15 / 16 / 241 / 1808 / 2048 / 8192` |
| source | Phase127 diagnostic exact-shape manifest |
| required next evidence | clean timing with exact bucket, loaded weights, fallback false |
| forbidden | default AIC, PerfDatabase, interpolation, extrapolation |

The Phase127 buckets can seed a future timing design, but they remain descriptor evidence until a separate timing phase accepts clean timing rows.

## Top-K Validation Plan

| Candidate | Required check |
|---|---|
| `tp16dp1` | baseline throughput / TTFT gate stays within current constants |
| `tp8ep8` | EP8 candidate key is present; no default model claim |
| `tp4dp2ep8` | EP8 candidate key is present; multi-config rank is compared against real near-best data |

Top-K validation must compare ordering against real vLLM measurements. A candidate with only a residual or unkeyed millisecond constant is rejected.
