# Phase119 Minimal Delivery Manifest

## Decision

The minimal delivery package is limited to the Phase116-118 experimental interface and the small evidence files needed to review it. Do not stage the research artifact tree as a whole.

## Required Files

| Type | File |
|---|---|
| Fit audit script | `scripts/fit_moe_wna16_diagnostic_phase116.py` |
| Table builder script | `scripts/build_moe_wna16_experimental_table_phase117.py` |
| Query API script | `scripts/query_moe_wna16_experimental_table_phase118.py` |
| Fit audit test | `tests/unit/scripts/test_fit_moe_wna16_diagnostic_phase116.py` |
| Table builder test | `tests/unit/scripts/test_build_moe_wna16_experimental_table_phase117.py` |
| Query API test | `tests/unit/scripts/test_query_moe_wna16_experimental_table_phase118.py` |
| Fit audit output | `docs/iter_gap_investigation/phase116_moe_wna16_fit_audit.csv` |
| Holdout error output | `docs/iter_gap_investigation/phase116_moe_wna16_holdout_errors.csv` |
| Fit report | `docs/iter_gap_investigation/phase116_moe_wna16_fit_report.md` |
| Phase116 decision | `docs/iter_gap_investigation/phase116_go_nogo.md` |
| Exact-key table | `docs/iter_gap_investigation/phase117_moe_wna16_experimental_table.csv` |
| Query contract | `docs/iter_gap_investigation/phase117_moe_wna16_query_contract.md` |
| TRT-LLM method alignment | `docs/iter_gap_investigation/phase117_trtllm_methodology_alignment.md` |
| Phase117 decision | `docs/iter_gap_investigation/phase117_go_nogo.md` |
| Query example key | `docs/iter_gap_investigation/phase118_query_key_tokens128.json` |
| Query example result | `docs/iter_gap_investigation/phase118_query_result_tokens128.csv` |
| Query fail-fast record | `docs/iter_gap_investigation/phase118_query_failfast_cases.md` |
| Phase118 decision | `docs/iter_gap_investigation/phase118_go_nogo.md` |
| Phase119 closeout | `docs/iter_gap_investigation/phase119_experimental_interface_closeout.md` |
| Phase119 manifest | `docs/iter_gap_investigation/phase119_minimal_delivery_manifest.md` |
| Phase119 review checklist | `docs/iter_gap_investigation/phase119_review_checklist.md` |
| Phase119 decision | `docs/iter_gap_investigation/phase119_go_nogo.md` |

## Optional Evidence For Reviewer Context

| File | Use |
|---|---|
| `docs/iter_gap_investigation/phase114_shape_trend_summary.csv` | Source summary used by Phase116/117 scripts |
| `docs/iter_gap_investigation/phase115_moe_wna16_fit_dataset.md` | Dataset gate before Phase116 |
| `docs/iter_gap_investigation/phase115_model_form_selection.md` | Explains why the linear fit stayed diagnostic |
| `docs/iter_gap_investigation/phase115_holdout_error_policy.md` | Defines Phase116 holdout policy |
| `docs/iter_gap_investigation/phase115_go_nogo.md` | Authorizes Phase116 only as experimental fit audit |

## Excluded By Default

| Excluded | Reason |
|---|---|
| `collector/vllm/*` | Remote collection and runner research scripts |
| Phase90-107 tuning generation logs | Config provenance research artifacts |
| Phase110/112/114 raw timing CSVs | Useful evidence, but larger diagnostic research artifacts |
| Historical profiler/NCCL/sync outputs | Forbidden as model inputs |
| Residual bucket artifacts | Not a physical module model |
| `PerfDatabase` updates | Default path remains No-Go |

## Staging Rule

Phase120 must use `git add -n -- <whitelist>` first. Do not use `git add .`.
