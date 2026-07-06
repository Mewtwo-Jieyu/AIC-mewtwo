# Phase431 PerfDB Recollect

## Verdict

- phase431_perfdb_recollect_ingested_default_no_go.
- GPU raw grids were collected from committed collector code and ingested as measured PerfDB rows.
- Default AIC remains No-Go.

## Slope Recheck

| category | real ms/request | phase431 ms/request | ratio | verdict |
|---|---:|---:|---:|---|
| mla_attention | 0.215499 | 0.197388 | 0.916 | phase431_slope_matches_real |
| moe_gemm_or_aux | 0.164715 | 0.136387 | 0.828 | phase431_slope_matches_real |
| ep_a2a | 0.070353 | 0.012944 | 0.184 | phase431_ep_floor_ingested_but_slope_still_low |

## Route Checks

| category | metric | bucket | raw ms | routed ms | verdict |
|---|---|---:|---:|---:|---|
| mla_attention | generation_mla_query | 64 | 0.234671990077 | 0.234671990077 | phase431_route_matches_raw |
| moe_gemm_or_aux | query_moe_phase431_decode_distribution | 64 | 0.381834716797 | 0.381834716797 | phase431_route_matches_raw |
| ep_a2a | query_vllm_ep8_a2a_decode | 64 | 0.085200000000 | 0.085200000000 | phase431_route_matches_raw |

## Validate A/B

- MULTI_CONFIG after Phase431: max=2.60x, mean=1.50x; verdict=multi_config_still_fails_dp2_8k2k.

| scenario | real out tok/s/GPU | sim before | sim after | before | after | delta | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| K2.5-tp8ep8-8k2k | 133.5 | 167.0 | 167.0 | 1.25x | 1.25x | +0.00x | pass |
| K2.5-tp8ep8-32k3k | 52.5 | 56.1 | 56.1 | 1.07x | 1.07x | +0.00x | pass |
| K2.5-tp4ep8dp2-8k2k | 137.7 | 351.6 | 358.1 | 2.55x | 2.60x | +0.05x | fail |
| K2.5-tp4ep8dp2-32k3k | 53.3 | 92.4 | 92.4 | 1.73x | 1.73x | +0.00x | fail |
| K2.5-tp8ep8-8k2k-bt65536 | 138.5 | 155.1 | 155.1 | 1.12x | 1.12x | +0.00x | pass |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 113.9 | 101.1 | 142.5 | 0.89x | 1.25x | +0.36x | pass |

## Phase426 Replay

- steady penalty remains 1.517x; verdict=prefill_dominant_with_decode_gap_secondary.
- decode mean real/sim = 42.849884 / 30.001457 ms (1.432x).
- attribution share: prefill=0.646, decode_gap=0.302, peer_stall=0.051.

## Boundaries

- PerfDatabase changed: true, measured rows only.
- Runtime route changed: true, limited to Phase431 measured decode coverage.
- Gate/default changed: false; `valid_for_default=false`, `default_readiness=No-Go`.
