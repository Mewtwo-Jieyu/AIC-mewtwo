# Phase360 EP8 Alltoall Single Point Spec

| Item | Result |
|---|---|
| Verdict | run spec only |
| Default AIC | No-Go |
| PerfDatabase | not written |
| GPU | not run in Phase360 |

## Scope

- Phase360 is a run spec, not performance evidence.
- Phase361 is the first allowed GPU single-point run, and only if the exact runtime inputs are available.
- WideEP schema is not reused; the run must measure the vLLM 0.19.0 runner-level EP comm path.
- FusedMoE total latency is not the EP8 comm boundary.
- The route name may say alltoall, but the actual runtime backend must be captured.
- The single-point artifact may produce one diagnostic CSV row only; it must not become a PerfDatabase row or default AIC evidence.

## Stop Rules

- Stop on PTX failure.
- Stop on forward/context setup failure.
- Stop when the backend cannot be identified.
- Stop when GPU or process residue remains after cleanup.
- Do not retry, and do not fall back to a bare kernel measurement.
