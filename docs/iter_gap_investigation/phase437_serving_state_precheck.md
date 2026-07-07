# Phase437 Serving-State Precheck

## Verdict

- verdict: `serving_state_grid_collection_can_proceed`.
- hit count: 191.
- GPU-grid-required miss count: 429.
- expected fallback miss count: 224.
- No serving-state runtime, PerfDB, or gate logic was changed.
- Default AIC remains No-Go.

## Miss Classification

| classification | weighted count | meaning |
|---|---:|---|
| `covered` | 191 | already hits current measured grid |
| `expected_pure_prefill_fallback` | 8 | pure prefill remains outside serving-state table |
| `expected_ramp_or_tail_fallback` | 216 | low ramp/tail query remains on baseline fallback |
| `in_scope_grid_gap` | 429 | Phase437 must collect measured grid points |

## Raw Miss Reasons

| miss reason | weighted count |
|---|---:|
| `bucket_above_range` | 60 |
| `bucket_below_range` | 216 |
| `decode_batch_above_range` | 169 |
| `hit` | 191 |
| `interpolation_gap` | 200 |
| `table_missing` | 8 |

## Collection Requirements

| scenario | phase | category | axis | bucket range | decode batch range | count |
|---|---|---|---|---:|---:|---:|
| K2.5-tp4ep8dp2-32k3k | decode | collective_other | rectangular_grid_density | 9-10 | 9-10 | 8 |
| K2.5-tp4ep8dp2-32k3k | decode | ep_a2a | rectangular_grid_density | 9-10 | 9-10 | 8 |
| K2.5-tp4ep8dp2-32k3k | decode | moe_gemm_or_aux | rectangular_grid_density | 9-10 | 9-10 | 8 |
| K2.5-tp4ep8dp2-32k3k | decode | other_cuda | rectangular_grid_density | 9-10 | 9-10 | 8 |
| K2.5-tp4ep8dp2-32k3k | mixed_prefill | collective_other | decode_batch_high | 32000-32000 | 9-9 | 1 |
| K2.5-tp4ep8dp2-32k3k | mixed_prefill | ep_a2a | decode_batch_high | 32000-32000 | 9-9 | 1 |
| K2.5-tp4ep8dp2-32k3k | mixed_prefill | moe_gemm_or_aux | decode_batch_high | 32000-32000 | 9-9 | 1 |
| K2.5-tp4ep8dp2-32k3k | mixed_prefill | other_cuda | decode_batch_high | 32000-32000 | 9-9 | 1 |
| K2.5-tp4ep8dp2-8k2k | decode | collective_other | rectangular_grid_density | 9-51 | 9-51 | 42 |
| K2.5-tp4ep8dp2-8k2k | decode | collective_other | token_or_decode_bucket_high | 53-64 | 53-64 | 15 |
| K2.5-tp4ep8dp2-8k2k | decode | ep_a2a | rectangular_grid_density | 9-51 | 9-51 | 42 |
| K2.5-tp4ep8dp2-8k2k | decode | ep_a2a | token_or_decode_bucket_high | 53-64 | 53-64 | 15 |
| K2.5-tp4ep8dp2-8k2k | decode | moe_gemm_or_aux | rectangular_grid_density | 9-51 | 9-51 | 42 |
| K2.5-tp4ep8dp2-8k2k | decode | moe_gemm_or_aux | token_or_decode_bucket_high | 53-64 | 53-64 | 15 |
| K2.5-tp4ep8dp2-8k2k | decode | other_cuda | rectangular_grid_density | 9-51 | 9-51 | 42 |
| K2.5-tp4ep8dp2-8k2k | decode | other_cuda | token_or_decode_bucket_high | 53-64 | 53-64 | 15 |
| K2.5-tp4ep8dp2-8k2k | mixed_prefill | collective_other | decode_batch_high | 8000-8000 | 2-63 | 66 |
| K2.5-tp4ep8dp2-8k2k | mixed_prefill | ep_a2a | decode_batch_high | 8000-8000 | 35-63 | 33 |
| K2.5-tp4ep8dp2-8k2k | mixed_prefill | moe_gemm_or_aux | decode_batch_high | 8000-8000 | 35-63 | 33 |
| K2.5-tp4ep8dp2-8k2k | mixed_prefill | other_cuda | decode_batch_high | 8000-8000 | 35-63 | 33 |

## Phase437 Acceptance List

- Mixed prefill token axis must cover real measured points up to the 8k and 32k working points, with decode-batch brackets beyond the observed 8k2k batch 34 and 32k3k batch 9.
- Decode grid must be rectangular enough to bracket the 8k2k steady decode queries through batch 64; diagonal-only measurements are not sufficient under the current inner-only query policy.
- Pure prefill `table_missing` rows are expected fallback rows, not a table-structure blocker for this phase.
- Low-batch ramp/tail rows below batch 8 are recorded but not required for this acceptance gate.
