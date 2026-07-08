# Phase446 B2b forward_total candidate rows

结论: B2b event 行可生成 `forward_total` 候选表。`bucket_tokens` 使用 vLLM 实际 forward token 数 `num_tokens_unpadded`; `num_tokens_unpadded > generation_requests` 的步按 mixed/prefill 计,避免把 chunked-prefill 误放进 decode。

decode 入表只接受 `peer_decode` 且 `cudagraph_mode=FULL` 的纯 decode 步。`peer_prefill` 下的 decode 步和非 FULL decode 步交给 runtime lockstep 耦合计费,不进入 intrinsic decode 表,避免双计。

## Summary

| scenario | event steps | candidate rows | mixed rows | decode rows | excluded peer-prefill decode steps | excluded non-FULL decode steps |
|---|---:|---:|---:|---:|---:|---:|
| K2.5-tp4ep8dp2-8k2k | 94132 | 157 | 101 | 56 | 470 | 36 |
| K2.5-tp4ep8dp2-32k3k | 360244 | 36 | 27 | 9 | 464 | 28 |

## Candidate Rows

| scenario | phase | bucket | decode batch | median ms | samples | peer phase |
|---|---|---:|---:|---:|---:|---|
| K2.5-tp4ep8dp2-32k3k | decode | 1 | 1 | 15.299 | 4158 | peer_decode |
| K2.5-tp4ep8dp2-32k3k | decode | 2 | 2 | 16.128 | 19792 | peer_decode |
| K2.5-tp4ep8dp2-32k3k | decode | 3 | 3 | 17.092 | 7 | peer_decode |
| K2.5-tp4ep8dp2-32k3k | decode | 4 | 4 | 17.144 | 6 | peer_decode |
| K2.5-tp4ep8dp2-32k3k | decode | 5 | 5 | 20.409 | 6 | peer_decode |
| K2.5-tp4ep8dp2-32k3k | decode | 6 | 6 | 20.612 | 6 | peer_decode |
| K2.5-tp4ep8dp2-32k3k | decode | 8 | 8 | 21.054 | 121 | peer_decode |
| K2.5-tp4ep8dp2-32k3k | decode | 9 | 9 | 25.691 | 330388 | peer_decode |
| K2.5-tp4ep8dp2-32k3k | decode | 10 | 10 | 25.557 | 3896 | peer_decode |
| K2.5-tp4ep8dp2-32k3k | mixed_prefill | 13 | 6 | 25.047 | 8 | any_peer |
| K2.5-tp4ep8dp2-32k3k | mixed_prefill | 32 | 9 | 29.902 | 16 | any_peer |
| K2.5-tp4ep8dp2-32k3k | mixed_prefill | 38 | 9 | 32.913 | 8 | any_peer |
| K2.5-tp4ep8dp2-32k3k | mixed_prefill | 42 | 9 | 37.827 | 8 | any_peer |
| K2.5-tp4ep8dp2-32k3k | mixed_prefill | 44 | 9 | 38.144 | 8 | any_peer |
| K2.5-tp4ep8dp2-32k3k | mixed_prefill | 45 | 9 | 37.693 | 8 | any_peer |
| K2.5-tp4ep8dp2-32k3k | mixed_prefill | 46 | 9 | 43.808 | 8 | any_peer |
| K2.5-tp4ep8dp2-32k3k | mixed_prefill | 47 | 9 | 37.322 | 8 | any_peer |
| K2.5-tp4ep8dp2-32k3k | mixed_prefill | 49 | 9 | 47.925 | 8 | any_peer |
| K2.5-tp4ep8dp2-32k3k | mixed_prefill | 58 | 9 | 43.824 | 8 | any_peer |
| K2.5-tp4ep8dp2-32k3k | mixed_prefill | 61 | 9 | 43.355 | 8 | any_peer |
| K2.5-tp4ep8dp2-32k3k | mixed_prefill | 63 | 9 | 2817.672 | 4 | any_peer |
| K2.5-tp4ep8dp2-32k3k | mixed_prefill | 64 | 9 | 44.721 | 4 | any_peer |
| K2.5-tp4ep8dp2-32k3k | mixed_prefill | 67 | 9 | 41.333 | 104 | any_peer |
| K2.5-tp4ep8dp2-32k3k | mixed_prefill | 73 | 9 | 1442.171 | 8 | any_peer |
| K2.5-tp4ep8dp2-32k3k | mixed_prefill | 77 | 9 | 1438.711 | 8 | any_peer |
| K2.5-tp4ep8dp2-32k3k | mixed_prefill | 80 | 8 | 54.849 | 4 | any_peer |
| K2.5-tp4ep8dp2-32k3k | mixed_prefill | 81 | 9 | 2824.397 | 4 | any_peer |
| K2.5-tp4ep8dp2-32k3k | mixed_prefill | 32000 | 1 | 4702.674 | 16 | any_peer |
| K2.5-tp4ep8dp2-32k3k | mixed_prefill | 32000 | 2 | 4704.478 | 8 | any_peer |
| K2.5-tp4ep8dp2-32k3k | mixed_prefill | 32000 | 3 | 4703.476 | 8 | any_peer |
| K2.5-tp4ep8dp2-32k3k | mixed_prefill | 32000 | 4 | 4704.624 | 8 | any_peer |
| K2.5-tp4ep8dp2-32k3k | mixed_prefill | 32000 | 5 | 4701.830 | 8 | any_peer |
| K2.5-tp4ep8dp2-32k3k | mixed_prefill | 32000 | 6 | 4716.126 | 632 | any_peer |
| K2.5-tp4ep8dp2-32k3k | mixed_prefill | 32000 | 7 | 4718.195 | 224 | any_peer |
| K2.5-tp4ep8dp2-32k3k | mixed_prefill | 32000 | 8 | 4719.627 | 112 | any_peer |
| K2.5-tp4ep8dp2-32k3k | mixed_prefill | 32000 | 9 | 4722.889 | 112 | any_peer |
| K2.5-tp4ep8dp2-8k2k | decode | 1 | 1 | 14.057 | 1238 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 2 | 2 | 14.600 | 1288 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 3 | 3 | 15.153 | 1336 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 4 | 4 | 14.940 | 2212 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 5 | 5 | 17.398 | 2276 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 6 | 6 | 17.593 | 1576 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 7 | 7 | 17.556 | 2872 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 8 | 8 | 16.717 | 712 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 9 | 9 | 21.353 | 8 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 10 | 10 | 21.420 | 8 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 11 | 11 | 21.198 | 8 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 12 | 12 | 20.838 | 8 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 13 | 13 | 20.889 | 8 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 14 | 14 | 20.896 | 8 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 15 | 15 | 20.335 | 8 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 16 | 16 | 20.198 | 8 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 17 | 17 | 24.182 | 8 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 18 | 18 | 23.714 | 8 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 19 | 19 | 23.491 | 8 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 20 | 20 | 23.417 | 8 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 21 | 21 | 22.490 | 8 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 22 | 22 | 22.717 | 8 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 23 | 23 | 22.452 | 8 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 24 | 24 | 24.003 | 8 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 25 | 25 | 24.809 | 8 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 26 | 26 | 24.805 | 8 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 27 | 27 | 25.337 | 8 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 28 | 28 | 28.470 | 8 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 29 | 29 | 28.374 | 8 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 30 | 30 | 28.128 | 8 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 31 | 31 | 25.645 | 8 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 32 | 32 | 25.251 | 8 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 33 | 33 | 29.419 | 8 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 34 | 34 | 32.198 | 8 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 35 | 35 | 32.109 | 8 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 36 | 36 | 31.969 | 8 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 37 | 37 | 31.998 | 8 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 38 | 38 | 31.586 | 8 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 39 | 39 | 31.529 | 7 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 40 | 40 | 31.546 | 6 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 41 | 41 | 35.567 | 5 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 42 | 42 | 35.185 | 4 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 44 | 44 | 33.353 | 21 | peer_decode |
| K2.5-tp4ep8dp2-8k2k | decode | 45 | 45 | 37.704 | 547 | peer_decode |
| ... | ... | ... | ... | ... | ... | 113 more rows |
