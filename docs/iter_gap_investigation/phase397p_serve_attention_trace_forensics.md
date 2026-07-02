# Phase397p Serve Attention Trace Forensics

| Item | Result |
|---|---|
| Scope | offline serve-trace forensics only; no runtime/DB/gate change |
| Verdict | confirmed_fa3_split_kv_scheduler_harness_gap |
| Default AIC | No-Go |

## Serve Trace Summary

| Config | heads/GPU | mean FA3 main us/call | spread | calls/rank | DB 8192 anchor us | DB s9001 query us | Phase397o graph us | graph/serve | split/scheduler evidence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| tp8ep8-8k2k | 8 | 260.256000 | 0.874908% | 1583 | 486.000000 | 523.000000 | 481.504000 | 1.850117x | FA3 main + combine + get_scheduler_metadata + prepare_varlen |
| tp4ep8dp2-8k2k | 16 | 207.211125 | 37.529838% | 1584 | 453.000000 | 290.000000 | 245.824000 | 1.186346x | FA3 main + combine + get_scheduler_metadata + prepare_varlen |

## Root Cause

- serve trace uses FA3 main plus combine and scheduler/varlen metadata while DB and Phase397o single-kernel microbench remain about 1.85x slower.
- tp4ep8dp2 has a rank split (rank 0-3 around 245us, rank 4-7 around 169us), so heads=16 needs follow-up trace interpretation; it is not the primary q acceptance target.
- Phase397o already showed CUDA graph replay and randomize_blocks do not explain the gap.
- The missing piece is harness fidelity: current DB/microbench path does not reproduce serve FA3 varlen split-KV scheduler metadata.

## Phase397q Acceptance

- Recollect generation_mla with FA3 varlen split-KV scheduler metadata and block_size=64.
- heads=8, batch=128, s=9001 must land near 260.256000 us/call before any table edit.
- Recollect float16 flash and fp8 flashmla as separate rows; do not use a scalar multiplier.
- q may touch collector and generation_mla_perf rows only after evidence; it must not touch gate/models runtime.
- Default AIC remains No-Go.
