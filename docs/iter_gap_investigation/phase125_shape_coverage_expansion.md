# Phase125: MoE WNA16 Shape Coverage Expansion

## Decision

| Item | Result |
|---|---|
| Phase125 scope | Local exact-shape manifest |
| Input | `phase124_moe_token_bucket_coverage.csv` |
| Output | `phase125_moe_shape_coverage_manifest.csv` |
| Phase126 time gate | No-Go until this manifest is accepted |
| Default AIC | No-Go |
| PerfDatabase | No-Go |

## Input Summary

| Item | Value |
|---|---|
| Phase124 marker data rows | `967680` |
| Phase124 observed buckets | `1 / 15 / 16 / 241 / 1808 / 2048 / 8192` |
| Phase117 table buckets | `128 / 248 / 512 / 1024` |
| Covered by Phase117 | `0 / 7` |

## Manifest Summary

| tokens_actual | occurrence_count | Phase125 action |
|---:|---:|---|
| 1 | 6960 | `needs_exact_shape_evidence` |
| 15 | 240 | `needs_exact_shape_evidence` |
| 16 | 959280 | `needs_exact_shape_evidence` |
| 241 | 240 | `needs_exact_shape_evidence` |
| 1808 | 240 | `needs_exact_shape_evidence` |
| 2048 | 240 | `needs_exact_shape_evidence` |
| 8192 | 480 | `needs_exact_shape_evidence` |

Total occurrence count: `967680`.

## Rule

Phase117 is exact-key only. Since none of the Phase124 buckets equal `128 / 248 / 512 / 1024`, Phase125 marks every observed bucket as `needs_exact_shape_evidence`.

No bucket is skipped by count. No bucket is predicted from a nearby Phase117 row.

## Boundary

| Boundary | Status |
|---|---|
| Remote run | Not done |
| Time data collection | Not done |
| PerfDatabase write | Not done |
| Default AIC path change | Not done |
| Current marker runner reuse | Not done |

## Validation

| Check | Result |
|---|---|
| `python3 -m pytest tests/unit/scripts/test_build_moe_wna16_shape_coverage_phase125.py -q` | Blocked: local Python has no pytest |
| `pytest tests/unit/scripts/test_build_moe_wna16_shape_coverage_phase125.py::test_timing_field_fails -q` | Blocked: pytest command not found |
| `python3 -m py_compile scripts/build_moe_wna16_shape_coverage_phase125.py` | PASS |
| `python3 -m py_compile tests/unit/scripts/test_build_moe_wna16_shape_coverage_phase125.py` | PASS |
| Timing header guard smoke | PASS |
| Manifest generation | PASS: 7 rows, total `967680` |
| Forbidden-field scan | PASS: no output |
| `git diff --check` | PASS |
| Staged area | Empty |
