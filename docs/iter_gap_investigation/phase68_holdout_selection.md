# Phase 68 Holdout Selection

## Conclusion

No holdout can be selected from existing artifacts. Phase68 is therefore No-Go for generator generalization beyond the Phase67 first-evidence case.

| candidate | intended check | artifact status | decision |
|---|---|---|---|
| `3k3k_b128` | short ISL, high concurrency | no vLLM scheduler alignment descriptor | missing evidence |
| `32k1k_b16` | long ISL, low concurrency, tail behavior | no vLLM scheduler alignment descriptor | missing evidence |
| `10k2k_b32` different budget | budget sensitivity | no vLLM scheduler alignment descriptor | missing evidence |
| different topology | topology sensitivity | intentionally deferred | out of scope |

## Selection Rule Applied

Only an artifact with all of the following fields is acceptable:

| field group | required |
|---|---|
| alignment | `alignment_key`, `alignment_key_type=engine_core_dp_step`, `engine_step_id`, `dp_rank` |
| scheduler split | context/decode tokens and context/decode requests |
| runtime link | `forward_token_count`, `forward_regime`, `cudagraph_runtime_mode` |
| topology | `tp`, `dp`, `moe_tp`, `moe_ep`, `topology_key` |
| boundary | `valid_for_default=false`, `perf_database=false`, `diagnostic_only=true` |

The only existing artifact satisfying these requirements is Phase62 `10k2k_b32`, already used for Phase66/67.

## Why Not Generate Holdout Rows Anyway

Generating cb_sim-side rows without real vLLM scheduler rows would only test the generator against itself. It would not prove generalization and would risk baking in assumptions about vLLM DP splitting, padding, tail rows, or graph mode.

## Future Holdout Requirements

Before Phase69 can generalize the generator, at least one new descriptor-only vLLM holdout should be collected and deduped by `(alignment_key, dp_rank)`.

| preferred holdout | reason |
|---|---|
| `3k3k_b128` | exercises short prefill and higher per-DP request count |
| `32k1k_b16` | exercises long prefill, low per-DP request count, and decode tail |
| `10k2k_b32` with different `max_num_batched_tokens` | isolates token budget effects |

No latency or profiler data is needed for these holdouts.
