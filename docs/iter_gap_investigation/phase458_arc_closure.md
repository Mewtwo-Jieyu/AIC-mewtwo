# Phase458 N=512 unified acceptance

Verdict: `fail`.

## Reference updates
| scenario | metric | value | status | note |
|---|---:|---:|---|---|
| K2.5-tp8ep8-8k2k | old_real_output_tok_s_gpu | 133.528000 |  | reference before Phase458 N=512 unification |
| K2.5-tp8ep8-8k2k | new_real_output_tok_s_gpu | 146.639701 | pass | ok=512; fail=0 |
| K2.5-tp8ep8-8k2k | protocol_gain_vs_old_reference | 1.098194 |  | new N=512 output throughput divided by old reference throughput |
| K2.5-tp8ep8-32k3k | old_real_output_tok_s_gpu | 52.469143 |  | reference before Phase458 N=512 unification |
| K2.5-tp8ep8-32k3k | new_real_output_tok_s_gpu | 50.111595 | pass | ok=512; fail=0 |
| K2.5-tp8ep8-32k3k | protocol_gain_vs_old_reference | 0.955068 |  | new N=512 output throughput divided by old reference throughput |
| K2.5-tp8ep8-8k2k-bt65536 | old_real_output_tok_s_gpu | 138.470000 |  | reference before Phase458 N=512 unification |
| K2.5-tp8ep8-8k2k-bt65536 | new_real_output_tok_s_gpu | 124.016988 | pass | ok=512; fail=0 |
| K2.5-tp8ep8-8k2k-bt65536 | protocol_gain_vs_old_reference | 0.895624 |  | new N=512 output throughput divided by old reference throughput |
| K2.5-tp4ep8dp2-8k2k-bt65536 | old_real_output_tok_s_gpu | 113.910186 |  | reference before Phase458 N=512 unification |
| K2.5-tp4ep8dp2-8k2k-bt65536 | new_real_output_tok_s_gpu | 109.499911 | pass | ok=512; fail=0 |
| K2.5-tp4ep8dp2-8k2k-bt65536 | protocol_gain_vs_old_reference | 0.961283 |  | new N=512 output throughput divided by old reference throughput |

## KV capacity checks
| scenario | metric | value | status | note |
|---|---:|---:|---|---|
| K2.5-tp8ep8-8k2k | kv_cache_tokens | 546144 | pass | raw truth to wire into validate capacity if changed |
| K2.5-tp8ep8-32k3k | kv_cache_tokens | 461200 | pass | raw truth to wire into validate capacity if changed |
| K2.5-tp8ep8-8k2k-bt65536 | kv_cache_tokens | 343584 | pass | raw truth to wire into validate capacity if changed |
| K2.5-tp4ep8dp2-8k2k-bt65536 | kv_cache_tokens | 131664 | pass | raw truth to wire into validate capacity if changed |

## Six-point acceptance
| scenario | metric | value | status | note |
|---|---:|---:|---|---|
| K2.5-tp8ep8-8k2k | current_error_ratio | 1.132524 | pass | improved |
| K2.5-tp4ep8dp2-8k2k | current_error_ratio | 1.041312 | pass | unchanged |
| K2.5-tp8ep8-8k2k-bt65536 | current_error_ratio | 1.234521 | fail | regressed |
| K2.5-tp4ep8dp2-8k2k-bt65536 | current_error_ratio | 1.166794 | fail | improved |
| K2.5-tp4ep8dp2-32k3k | current_error_ratio | 1.056381 | pass | unchanged |
| K2.5-tp8ep8-32k3k | current_error_ratio | 1.397081 | fail | regressed |

## Archive notes
- Phase458 did not close the 15% gate: tp8-32k3k, tp8-8k2k-bt65536, and dp2-8k2k-bt65536 remain above target in the unified N=512 reference table.
- The N=512 protocol fixed tp8-8k2k, but exposed model or coverage residuals in the 32k and bt65536 regimes.
- Phase454 dp2-8k2k clean reference was already collected; Phase458 also wires that value into validate to avoid stale-reference false positives.
- vLLM 0.19 API-server route stats overwrite remains a documented non-modeled boundary.
- B2b TP8 rows remain archived unless a future scoped serving-state table can prevent cross-regime pollution.
- Next task: residual triage before any gate tightening or fork handoff.
