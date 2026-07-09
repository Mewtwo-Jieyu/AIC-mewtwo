# Phase455 TP8 scope collision

Verdict: `tp8_scope_required_max_num_batched_tokens`.

## Candidate Surface

| phase | rows | bucket range | decode batch range |
|---|---:|---:|---:|
| decode | 66 | 1..67 | 1..67 |
| mixed_prefill | 127 | 122..8000 | 1..66 |

## No-Scope Temporary Ingest

| scenario | baseline | no-scope | delta | hits | status |
|---|---:|---:|---:|---:|---|
| K2.5-tp8ep8-8k2k | 1.439x | 1.455x | +0.016 | 145 | regressed |
| K2.5-tp8ep8-32k3k | 1.163x | 1.703x | +0.540 | 34 | regressed |
| K2.5-tp8ep8-8k2k-bt65536 | 1.120x | 1.117x | -0.002 | 32 | unchanged |

## Scope Decision

| scenario | scope |
|---|---|
| K2.5-tp8ep8-32k3k | `max_num_batched_tokens=32000` |
| K2.5-tp8ep8-8k2k | `max_num_batched_tokens=8000` |
| K2.5-tp8ep8-8k2k-bt65536 | `max_num_batched_tokens=65536` |

Max no-scope error: `1.703x`.

`max_num_batched_tokens` is configuration-derived and separates the three TP8 regimes in the six-point table.
