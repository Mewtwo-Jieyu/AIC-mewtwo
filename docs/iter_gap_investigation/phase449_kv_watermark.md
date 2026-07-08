# Phase449 KV watermark

结论: `MULTI_CONFIG` 过门。Phase449 把 DP2 8k2k 的 KV 容量从旧脏值 `672128` tokens 改为 Phase446 真值 `458128` tokens,8k2k 指纹从 `mixed_decode_batch_p50=62` 收到 `46`,随后只在已过指纹门的 one-chunk DP regime 合入 multi-replica lockstep。完整 validate: `max=1.42x`, `mean=1.19x`。

| check | result | decision |
|---|---:|---|
| 8k2k real KV capacity | 458128 tokens / 28633 blocks | use Phase446 serve.log lines 209/212 |
| validate capacity | 458128 tokens / 28633 blocks | pass |
| 8k2k fingerprint | p50 46, target 34..52 | pass |
| 32k3k fingerprint | p50 8, target 6..9 | pass |
| bt65536 default path | legacy per-replica | kept, because large-bt fingerprint was not gated |
| MULTI_CONFIG gate | max 1.42x <= 1.50x | pass |

## Validate A/B

| scenario | before | after | status |
|---|---:|---:|---|
| K2.5-tp8ep8-8k2k | 1.250x | 1.250x | unchanged |
| K2.5-tp8ep8-32k3k | 1.070x | 1.069x | unchanged |
| K2.5-tp4ep8dp2-8k2k | 2.120x | 1.416x | improved |
| K2.5-tp4ep8dp2-32k3k | 1.090x | 1.071x | improved |
| K2.5-tp8ep8-8k2k-bt65536 | 1.120x | 1.120x | unchanged |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 1.230x | 1.234x | unchanged within the protected legacy path |

## What Changed

| area | change |
|---|---|
| `validate_cb_simulator.py` | `K2.5-tp4ep8dp2-8k2k` capacity now uses Phase446 clean log value |
| `CBSimulator.run_multi_replica` | added optional lockstep mode: DP replicas advance on `max(replica_step_ms)` |
| `VLLMBackend._run_agg_cb_sim` | dp>1 uses multi-replica lockstep only when `ctx_tokens == isl` |
| tests | added capacity red/green checks, lockstep clock alignment check, and DP routing guard |

## Boundary

`bt65536` did not enter the new default path. A direct branch audit showed `old=140.52 tok/s/gpu`, `multi=142.14`, `lockstep=143.14`; that is a small unvalidated regression against the prior `1.23x` point. The safe rule is: only regimes with a passed composition fingerprint can switch to multi-replica lockstep. Large-bt remains queued for its own fingerprint gate.

Legacy throughput and TTFT surfaces still print `legacy_skip`; this phase only closes the 0.19 `MULTI_CONFIG` gate. Default readiness should be judged from that scoped gate, not from legacy surfaces.
