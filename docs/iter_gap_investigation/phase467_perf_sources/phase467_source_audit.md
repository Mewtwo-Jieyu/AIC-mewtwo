# Phase467 Performance Source Audit

| Check | Result |
|---|---:|
| Six-scenario numeric invariance | 6/6 PASS |
| Charged operators total | 279 |
| Charged operators with sources | 279 |
| Source coverage | 100.0% |
| Source records | 9477 |
| Charge ledger entries | 1007 |
| Missing sources | 0 |
| Unknown sources | 0 |
| Unapproved sources | 0 |
| Reconciliation failures | 0 |
| Numeric baseline commit | `1a24a99edd6fb21d8918bc07c2307dc46266653b` |
| Default AIC | No-Go |

## Source Types

| Type | Count |
|---|---:|
| calibrated | 12 |
| measured_exact | 170 |
| measured_interp | 2087 |
| structural | 7208 |

## Calibration Debt

| Source | Anchor | Scale | Status |
|---|---|---:|---|
| phase397v_int4_wo_moe_calibrated_roofline | phase397l_tp8ep8_decode_moe_expert_gemm_8.0201ms_60layers_bs128 | 0.606277008472 | technical_debt_existing_phase397v |

## Audit Failures

No missing, unknown, unapproved, or unreconciled charges.

`diagnostic_only=true`, `valid_for_default=false`, `perf_database=false`.
