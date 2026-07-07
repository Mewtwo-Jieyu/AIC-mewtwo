# Phase436 Serving-State Hit Audit and Step Grid

## Verdict

- verdict: `grid_still_out_of_range_no_extrapolation`.
- serving curve rows: 123.
- 8k2k still misses the serving-state table at runtime: decode is only measured at 8/36/52, while the sim reaches batch 64.
- query policy remains inner-only: out-of-grid returns None; no clamp, fake rectangle, or extrapolation was added.
- Default AIC remains No-Go until the validation table is clean and the burst-artifact protocol is settled.

## Grid Coverage

| scenario | phase | decode_batch min | max | points |
|---|---|---:|---:|---:|
| K2.5-tp4ep8dp2-32k3k | mixed_prefill | 0 | 8 | 2 |
| K2.5-tp4ep8dp2-8k2k | decode | 8 | 52 | 3 |
| K2.5-tp4ep8dp2-8k2k | mixed_prefill | 1 | 34 | 34 |

## Consistency Gates

| scenario | phase | target ms | reconstructed ms | error % | gate |
|---|---|---:|---:|---:|---|
| K2.5-tp4ep8dp2-32k3k | mixed_prefill | 2900.317 | 2751.036 | 5.15 | passed |
| K2.5-tp4ep8dp2-8k2k | decode | 16.494 | 16.494 | 0.00 | passed |

## Hit Audit

| audit | hit | decode_batch_above_range | bucket_above_range | other miss |
|---|---:|---:|---:|---:|
| before | 224 | 136 | 60 | 424 |
| after | 191 | 169 | 60 | 424 |

## Validate A/B

| scenario | real out tok/s/gpu | Phase435 sim | Phase435 ratio | Phase436 sim | Phase436 ratio | verdict |
|---|---:|---:|---:|---:|---:|---|
| K2.5-tp8ep8-8k2k | 133.528 | 166.972 | 1.250x | 166.972 | 1.250x | unchanged |
| K2.5-tp8ep8-32k3k | 52.469 | 56.079 | 1.069x | 56.079 | 1.069x | unchanged |
| K2.5-tp4ep8dp2-8k2k | 137.716 | 358.061 | 2.600x | 358.061 | 2.600x | still failing: grid miss |
| K2.5-tp4ep8dp2-32k3k | 53.277 | 66.840 | 1.255x | 66.840 | 1.255x | unchanged |
| K2.5-tp8ep8-8k2k-bt65536 | 138.470 | 155.070 | 1.120x | 155.070 | 1.120x | unchanged |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 113.910 | 142.301 | 1.249x | 142.301 | 1.249x | unchanged |

MULTI_CONFIG remains FAIL: max error is 2.600x on `K2.5-tp4ep8dp2-8k2k`.

## Phase437 Grid Collection Design

- Goal: collect a serving-state table that is dense enough for arbitrary workload shapes within the same K2.5/tp4dp2ep8/H200/vLLM0.19 scope.
- Token axis: 2k, 4k, 8k, 16k, 32k, 64k prefill chunks plus decode-only token buckets 8, 16, 32, 48, 64, 96, 128.
- Decode-batch axis: 0, 8, 16, 32, 48, 64, 96, 128, clipped by max_num_seqs and KV capacity.
- Collection protocol: run real serving with profiler windows at each grid point, extract ep_a2a, moe_gemm_or_aux, other_cuda, collective_other per step, and store only measured grid rows.
- Boundary: this fixes workload interpolation inside the scoped table; a different model, topology, card, or vLLM version needs its own serving-state table.
- Mechanism note: wait/wire/imbalance remains unresolved; the grid is an empirical serving-state PerfDB surface.
