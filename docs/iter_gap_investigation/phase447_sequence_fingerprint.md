# Phase447 DP sequence fingerprint

结论: 真实 B2b 序列的指纹门必须先于 runtime 修复。门的核心不是均值,而是 mixed 构成、跨 engine 相位联合和 mixed 波大小分布。

## Engine Summary

| scenario | dp | total steps | mixed steps | mixed freq | decode run p50 | decode run p90 | decode run max |
|---|---:|---:|---:|---:|---:|---:|---:|
| K2.5-tp4ep8dp2-8k2k | 0 | 47068 | 1380 | 0.0293 | 0.0 | 0.0 | 1942 |
| K2.5-tp4ep8dp2-8k2k | 1 | 47064 | 1380 | 0.0293 | 0.0 | 0.0 | 1943 |
| K2.5-tp4ep8dp2-32k3k | 0 | 180124 | 680 | 0.0038 | 0.0 | 465.0 | 3000 |
| K2.5-tp4ep8dp2-32k3k | 1 | 180120 | 680 | 0.0038 | 0.0 | 465.0 | 2999 |

## Phase Joint

| scenario | phase pair | count | share |
|---|---|---:|---:|
| K2.5-tp4ep8dp2-8k2k | decode+decode | 45445 | 0.9656 |
| K2.5-tp4ep8dp2-8k2k | decode+mixed_prefill | 232 | 0.0049 |
| K2.5-tp4ep8dp2-8k2k | decode+prefill | 3 | 0.0001 |
| K2.5-tp4ep8dp2-8k2k | mixed_prefill+decode | 235 | 0.0050 |
| K2.5-tp4ep8dp2-8k2k | mixed_prefill+mixed_prefill | 1145 | 0.0243 |
| K2.5-tp4ep8dp2-8k2k | prefill+mixed_prefill | 3 | 0.0001 |
| K2.5-tp4ep8dp2-8k2k | prefill+prefill | 1 | 0.0000 |
| K2.5-tp4ep8dp2-32k3k | decode+decode | 179204 | 0.9949 |
| K2.5-tp4ep8dp2-32k3k | decode+mixed_prefill | 229 | 0.0013 |
| K2.5-tp4ep8dp2-32k3k | decode+prefill | 3 | 0.0000 |
| K2.5-tp4ep8dp2-32k3k | mixed_prefill+decode | 232 | 0.0013 |
| K2.5-tp4ep8dp2-32k3k | mixed_prefill+mixed_prefill | 448 | 0.0025 |
| K2.5-tp4ep8dp2-32k3k | prefill+mixed_prefill | 3 | 0.0000 |
| K2.5-tp4ep8dp2-32k3k | prefill+prefill | 1 | 0.0000 |

## Mixed Composition Top Rows

| scenario | dp | bucket | decode batch | count | share | busy p50 ms | busy p90 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| K2.5-tp4ep8dp2-8k2k | 0 | 8000 | 43 | 272 | 0.0986 | 1130.520 | 1133.250 |
| K2.5-tp4ep8dp2-8k2k | 1 | 8000 | 43 | 272 | 0.0986 | 1130.462 | 1133.347 |
| K2.5-tp4ep8dp2-8k2k | 0 | 8000 | 42 | 224 | 0.0812 | 1127.430 | 1130.985 |
| K2.5-tp4ep8dp2-8k2k | 1 | 8000 | 42 | 224 | 0.0812 | 1127.625 | 1131.145 |
| K2.5-tp4ep8dp2-8k2k | 0 | 8000 | 41 | 144 | 0.0522 | 1130.936 | 1133.150 |
| K2.5-tp4ep8dp2-8k2k | 1 | 8000 | 41 | 144 | 0.0522 | 1130.964 | 1133.162 |
| K2.5-tp4ep8dp2-8k2k | 0 | 8000 | 44 | 124 | 0.0449 | 1099.655 | 1125.865 |
| K2.5-tp4ep8dp2-8k2k | 1 | 8000 | 44 | 124 | 0.0449 | 1099.701 | 1125.574 |
| K2.5-tp4ep8dp2-8k2k | 0 | 8000 | 47 | 40 | 0.0145 | 1125.735 | 1134.010 |
| K2.5-tp4ep8dp2-8k2k | 1 | 8000 | 47 | 40 | 0.0145 | 1125.760 | 1133.059 |
| K2.5-tp4ep8dp2-8k2k | 0 | 8000 | 45 | 36 | 0.0130 | 1127.496 | 1130.252 |
| K2.5-tp4ep8dp2-8k2k | 0 | 8000 | 46 | 36 | 0.0130 | 1127.376 | 1136.223 |
| K2.5-tp4ep8dp2-8k2k | 1 | 8000 | 45 | 36 | 0.0130 | 1127.612 | 1130.303 |
| K2.5-tp4ep8dp2-8k2k | 1 | 8000 | 46 | 36 | 0.0130 | 1127.485 | 1136.259 |
| K2.5-tp4ep8dp2-8k2k | 0 | 8000 | 50 | 32 | 0.0116 | 1129.396 | 1133.773 |
| K2.5-tp4ep8dp2-8k2k | 0 | 8000 | 52 | 32 | 0.0116 | 1130.227 | 1132.627 |
| K2.5-tp4ep8dp2-8k2k | 1 | 8000 | 50 | 32 | 0.0116 | 1129.626 | 1134.115 |
| K2.5-tp4ep8dp2-8k2k | 1 | 8000 | 52 | 32 | 0.0116 | 1130.455 | 1132.701 |
| K2.5-tp4ep8dp2-8k2k | 0 | 8000 | 49 | 28 | 0.0101 | 1129.813 | 1133.120 |
| K2.5-tp4ep8dp2-8k2k | 0 | 8000 | 51 | 28 | 0.0101 | 1131.176 | 1133.117 |
| K2.5-tp4ep8dp2-32k3k | 0 | 32000 | 6 | 316 | 0.2324 | 4716.058 | 4718.786 |
| K2.5-tp4ep8dp2-32k3k | 1 | 32000 | 6 | 316 | 0.2324 | 4716.261 | 4719.158 |
| K2.5-tp4ep8dp2-32k3k | 0 | 32000 | 7 | 112 | 0.0824 | 4718.035 | 4724.750 |
| K2.5-tp4ep8dp2-32k3k | 1 | 32000 | 7 | 112 | 0.0824 | 4718.195 | 4724.742 |
| K2.5-tp4ep8dp2-32k3k | 0 | 32000 | 8 | 56 | 0.0412 | 4719.563 | 4725.215 |
| K2.5-tp4ep8dp2-32k3k | 0 | 32000 | 9 | 56 | 0.0412 | 4722.877 | 4726.367 |
| K2.5-tp4ep8dp2-32k3k | 1 | 32000 | 8 | 56 | 0.0412 | 4719.695 | 4725.298 |
| K2.5-tp4ep8dp2-32k3k | 1 | 32000 | 9 | 56 | 0.0412 | 4722.909 | 4726.285 |
| K2.5-tp4ep8dp2-32k3k | 0 | 67 | 9 | 52 | 0.0382 | 41.308 | 41.829 |
| K2.5-tp4ep8dp2-32k3k | 1 | 67 | 9 | 52 | 0.0382 | 41.358 | 41.856 |
| K2.5-tp4ep8dp2-32k3k | 0 | 32 | 9 | 8 | 0.0059 | 29.893 | 30.020 |
| K2.5-tp4ep8dp2-32k3k | 0 | 32000 | 1 | 8 | 0.0059 | 4702.563 | 4704.062 |
| K2.5-tp4ep8dp2-32k3k | 1 | 32 | 9 | 8 | 0.0059 | 29.917 | 30.053 |
| K2.5-tp4ep8dp2-32k3k | 1 | 32000 | 1 | 8 | 0.0059 | 4702.674 | 4704.046 |
| K2.5-tp4ep8dp2-32k3k | 0 | 13 | 6 | 4 | 0.0029 | 25.023 | 25.027 |
| K2.5-tp4ep8dp2-32k3k | 0 | 38 | 9 | 4 | 0.0029 | 32.897 | 32.898 |
| K2.5-tp4ep8dp2-32k3k | 0 | 42 | 9 | 4 | 0.0029 | 37.817 | 37.817 |
| K2.5-tp4ep8dp2-32k3k | 0 | 44 | 9 | 4 | 0.0029 | 38.140 | 38.140 |
| K2.5-tp4ep8dp2-32k3k | 0 | 45 | 9 | 4 | 0.0029 | 37.668 | 37.670 |
| K2.5-tp4ep8dp2-32k3k | 0 | 46 | 9 | 4 | 0.0029 | 42.964 | 42.999 |

## Gate Tolerances

| scenario | metric | target low | target high | tolerance |
|---|---|---:|---:|---|
| K2.5-tp4ep8dp2-8k2k | mixed_decode_batch_p10_p90 | 34.0 | 52.0 | sim p50 must fall inside real p10-p90; distribution EMD <= 0.20 |
| K2.5-tp4ep8dp2-8k2k | mixed_bucket_tokens_p10_p90 | 8000.0 | 8000.0 | sim p50 must fall inside real p10-p90; no manual phase offset |
| K2.5-tp4ep8dp2-8k2k | phase_joint_share | 0.0 | 1.0 | each major phase-pair share within +/-0.10 absolute |
| K2.5-tp4ep8dp2-32k3k | mixed_decode_batch_p10_p90 | 6.0 | 9.0 | sim p50 must fall inside real p10-p90; distribution EMD <= 0.20 |
| K2.5-tp4ep8dp2-32k3k | mixed_bucket_tokens_p10_p90 | 67.0 | 32000.0 | sim p50 must fall inside real p10-p90; no manual phase offset |
| K2.5-tp4ep8dp2-32k3k | phase_joint_share | 0.0 | 1.0 | each major phase-pair share within +/-0.10 absolute |
