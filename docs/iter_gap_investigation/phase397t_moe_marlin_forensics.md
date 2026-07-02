# Phase397t MoE Marlin Forensics

Phase397t is report-only until the H200 mechanism gate passes.

| Item | Value |
|---|---:|
| Serve mean marlin us/call | 64.305125 |
| Serve ms/layer from per-call | 0.128610 |
| Phase397l anchor ms/layer | 0.133668 |
| Phase397s microbench ms/layer | 0.470627 |
| Microbench / anchor | 3.520859x |

| Rank | Mechanism | Decision |
|---:|---|---|
| 1 | activation_format_or_effective_m_mismatch | test_on_h200 |
| 2 | marlin_block_size_or_template_selection | test_on_h200 |
| 3 | route_distribution_only | ruled_out_by_phase397s |

Next gate: on H200, instrument actual M, padded rows, block_size_m/template, and per-call us.
Only a no-k structured collector variant within 0.1337 ms/layer +/-15% may unlock table rewrites.

Default AIC remains No-Go.
