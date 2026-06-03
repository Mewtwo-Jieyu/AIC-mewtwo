# Phase126: Exact-Shape Capture Blocker

## Decision

| Item | Result |
|---|---|
| Capture attempt | `capture1` |
| Traffic health | PASS |
| Parser | PASS |
| Exact coverage acceptance | FAIL |
| Cleanup | PASS |
| Next full capture | No-Go until blocker is reviewed |

## Remote Output

| Item | Path |
|---|---|
| Output dir | `/mnt/nvme1n1/ml_research/jieyu/aic/docs/iter_gap_investigation/phase126_moe_activation_10k2k_b32_bt8192_p18000_capture1` |
| Bench result | `bench_result.json` |
| Rows CSV | `moe_activation_rows.csv` |
| Coverage CSV | `moe_token_bucket_coverage.csv` |

Large raw files were left on the target machine and not copied into the local worktree.

## Acceptance Check

| Check | Expected | Actual | Result |
|---|---:|---:|---|
| `ok_requests` | 32 | 32 | PASS |
| `failed_requests` | 0 | 0 | PASS |
| marker rows | 967680 | 963578 | FAIL |
| coverage rows | 7 | 7 | PASS |
| all covered flags false | true | true | PASS |
| original source marker | absent | absent | PASS |
| vLLM serve process | absent | absent | PASS |
| GPU compute residual | absent | absent | PASS |

## Coverage Diff

| tokens_actual | expected_count | actual_count | Result |
|---:|---:|---:|---|
| 1 | 6960 | 2858 | FAIL |
| 15 | 240 | 240 | PASS |
| 16 | 959280 | 959280 | PASS |
| 241 | 240 | 240 | PASS |
| 1808 | 240 | 240 | PASS |
| 2048 | 240 | 240 | PASS |
| 8192 | 480 | 480 | PASS |

Actual total: `963578`.

## Actual Coverage CSV

```csv
tokens_actual,occurrence_count,phase117_exact_key_covered
1,2858,False
15,240,False
16,959280,False
241,240,False
1808,240,False
2048,240,False
8192,480,False
```

## Blocker

The fixed capture reproduced all target buckets, but bucket `1` count did not match Phase124. This violates the exact-shape acceptance rule.

Do not reinterpret the missing count, do not use benchmark metrics as evidence, and do not start another full capture until the blocker is reviewed.
