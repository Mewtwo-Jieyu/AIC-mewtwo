# Phase448 DP admission fingerprint gate

结论: 指纹门未通过。`runtime_lockstep_allowed=false`。

| scenario | metric | sim | target low | target high | passed |
|---|---|---:|---:|---:|---|
| K2.5-tp4ep8dp2-8k2k | mixed_decode_batch_p50 | 62.000 | 34.000 | 52.000 | False |
| K2.5-tp4ep8dp2-8k2k | mixed_bucket_tokens_p50 | 8000.000 | 8000.000 | 8000.000 | True |
| K2.5-tp4ep8dp2-32k3k | mixed_decode_batch_p50 | 8.000 | 6.000 | 9.000 | True |
| K2.5-tp4ep8dp2-32k3k | mixed_bucket_tokens_p50 | 32000.000 | 67.000 | 32000.000 | True |

## Phase Joint Snapshot

| scenario | phase pair | share |
|---|---|---:|
| K2.5-tp4ep8dp2-8k2k | decode+decode | 0.9672 |
| K2.5-tp4ep8dp2-8k2k | mixed_prefill+decode | 0.0005 |
| K2.5-tp4ep8dp2-8k2k | mixed_prefill+mixed_prefill | 0.0321 |
| K2.5-tp4ep8dp2-8k2k | pure_prefill+pure_prefill | 0.0002 |
| K2.5-tp4ep8dp2-32k3k | decode+decode | 0.9950 |
| K2.5-tp4ep8dp2-32k3k | mixed_prefill+mixed_prefill | 0.0049 |
| K2.5-tp4ep8dp2-32k3k | pure_prefill+pure_prefill | 0.0000 |
