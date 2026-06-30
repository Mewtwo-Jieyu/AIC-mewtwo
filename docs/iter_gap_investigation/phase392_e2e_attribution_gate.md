# Phase392 End-to-End Attribution Gate (offline, verdict-only)

| Item | Result |
|---|---|
| Workload | 10k2k_b32 / bt8192 / tp4dp2ep8 / Kimi-K2.5 / vLLM 0.19.0 |
| Ground truth | `compare_10k2k_b32_dp0.csv` (vllm_iter_lat_ms, per phase) |
| Module binding | triggers (MoE + EP8 comm), but exact-only unreachable on mixed/decode |
| Verdict | **mixed** — Route A validated for prefill/mixed; decode regresses; gating blocker is scheduler token-shape, not GPU buckets |
| Default AIC | No-Go |
| Runtime / PerfDatabase | not modified |

## Per-phase bare error (median ms, error_ratio = max(p/r, r/p))

| Phase | Ground truth | Baseline 0.12 query_moe | err | Module-bound 0.19 (diag snap, oh=0) | err |
|---|---|---|---|---|---|
| prefill | 1054.15 | 334.92 | 3.15x | 789.73 | 1.33x |
| mixed | 976.54 | 333.92 | 2.92x | 789.46 | 1.24x |
| pure_decode | 24.82 | 21.15 | 1.17x | 12.34 | 2.01x |

- `+90ms` decode overhead overshoots: pure_decode 102.34ms vs ground truth 24.82ms (4.12x). The 90ms fudge was calibrated for a different operating point.

## Reachability (exact-only vs cb_sim scheduler)

- Module binding fires for both `fusedmoe_runner_compute` and `ep8_comm_dispatch_combine`.
- Exact-bucket hits: ep8 19/201; fusedmoe 11/201 of 201 module calls — only the pure-prefill (`decode_bs=0`) iterations land on a materialized bucket (8192 -> 2048).
- cb_sim emits a **continuous** token spread (ep8 1..2500; fusedmoe 2..5000) because mixed-phase prefill chunk = `max_num_batched_tokens - decode_bs` (8191, 8190, ...), whereas the materialized buckets (ep8 7; fusedmoe 14) come from the real vLLM scheduler's discrete token counts. Under exact-only lookup cb_sim fails fast on the first mixed iteration.
- All module-bound numbers above used a DIAGNOSTIC nearest-bucket snap (max relative bucket error 0.18); they are directional, not certified. The model lookup contract stays exact-only.

## Attribution

- **prefill/mixed**: the ~3x gap was dominated by MoE compute being under-measured in the legacy 0.12 `moe_perf.txt`. The measured 0.19 module table (~2x the 0.12 MoE) recovers most of it (3.15x -> 1.33x prefill, 2.92x -> 1.24x mixed). This refutes the earlier "residual belongs to no module" worry for prefill.
- **decode**: the 0.12 path was already accurate (1.17x). The 0.19 module table makes decode worse (2.01x) because decode-phase small buckets are under-valued and EP8 all2all comm in the decode loop is not captured — exactly what the 90ms hand-fudge was masking.

## Verdict and next step

- Route A (module-level perf table) is **validated for the prefill/mixed compute path**, the original big gap.
- The gating blocker is **not** missing GPU buckets (measuring `8191/8190/...` would be futile artifacts). It is the cb_sim scheduler emitting token shapes that do not match vLLM's discrete buckets. Fix is **Route B scheduler token-shape alignment** (or a documented bucketization/quantization lookup contract), not naive bucket/topology expansion.
- Phase393 GPU work should target the **decode path** (EP8 all2all comm + small-bucket MoE), not prefill.

## No-Go discipline held

- exact-only model contract unchanged; nearest/interpolation/extrapolation not enabled in runtime.
- no fudge tuning used to declare success; 90ms shown as a diagnostic column only.
- runtime (`operations.py` / `vllm_backend.py`) and PerfDatabase files not modified; no GPU/SSH used.
- Default AIC remains No-Go.
