# Phase442 Serving-State Floor Ingest

## Verdict

`serving_state_floor_ingested_zero_regression`.

Step 1 only: no GPU was used. `vllm_serving_state_perf.txt` was updated with the Phase438 union-merge floor. This is the floor, not the final fix; Default AIC remains No-Go.

## Axis Decisions

| phase | category | decode-batch spread % | bucket spread % | decision |
|---|---|---:|---:|---|
| decode | collective_other | nan | nan | keep_2d |
| decode | ep_a2a | nan | nan | keep_2d |
| decode | moe_gemm_or_aux | nan | nan | keep_2d |
| decode | other_cuda | nan | nan | keep_2d |
| mixed_prefill | collective_other | 0.00 | nan | reduce_decode_batch_axis |
| mixed_prefill | ep_a2a | 70.54 | 188.10 | keep_2d |
| mixed_prefill | moe_gemm_or_aux | 56.42 | 166.67 | keep_2d |
| mixed_prefill | other_cuda | 56.54 | 183.64 | keep_2d |

## Merge Conflicts

| count | action |
|---:|---|
| 5 | exact-key candidate conflicts are reported; existing accepted row is kept |

## Hit Audit

| metric | count |
|---|---:|
| bucket_below_range | 148 |
| decode_batch_above_range | 169 |
| hit | 219 |
| interpolation_gap | 300 |
| table_missing | 156 |

## Validate A/B

| scenario | baseline ratio | candidate ratio | classification |
|---|---:|---:|---|
| K2.5-tp8ep8-8k2k | 1.250 | 1.250 | unchanged |
| K2.5-tp8ep8-32k3k | 1.069 | 1.069 | unchanged |
| K2.5-tp4ep8dp2-8k2k | 2.600 | 2.118 | improved |
| K2.5-tp4ep8dp2-32k3k | 1.255 | 1.255 | unchanged |
| K2.5-tp8ep8-8k2k-bt65536 | 1.120 | 1.120 | unchanged |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 1.249 | 1.252 | unchanged |

## Remaining Window List

| window | scenario | reason | count | note |
|---|---|---|---:|---|
| W-A | K2.5-tp4ep8dp2-8k2k | 8k mixed prefill in-scope coverage miss | 213 | steady trigger required; extract all observed steps, not only high decode_batch rows |
| W-B | K2.5-tp4ep8dp2-32k3k | 32k mixed prefill bucket coverage missing | 33 | only needed if union merge does not preserve Phase433/436 32k rows |

## Boundary

- Inner-only query semantics stay unchanged.
- No clamp, extrapolation, or silent overwrite is introduced.
- B2 event timing is a separate step and is not included in this floor commit.
- Default AIC remains No-Go.
