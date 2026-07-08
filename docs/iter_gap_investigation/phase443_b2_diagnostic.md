# Phase443-A B2 event timing diagnostic

B2b 候选: event busy 已明显高于 sim 计费,需要继续测低开销 busy。

| metric | value |
|---|---:|
| b2b_gate | recommended |
| primary_gap | busy_over_sim |
| bucket_rows | 136 |
| wall_gap_share | 0.111121 |
| busy_gap_share | 0.898347 |
| wall_gap_buckets | 0 |
| busy_gap_buckets | 7 |
| mixed_buckets | 17 |
| low_confidence_buckets | 111 |

## Largest Busy-Sim Buckets

| ctx_tokens | decode_batch | event rows | wall rows | sim ms | busy ms | wall ms | class |
|---:|---:|---:|---:|---:|---:|---:|---|
| 7958 | 42 | 556 | 139 | 413.190328 | 1131.170837 | 1135.840000 | busy_gap_dominates_b2b_candidate |
| 7959 | 41 | 408 | 102 | 413.019601 | 1131.856384 | 1136.245000 | busy_gap_dominates_b2b_candidate |
| 0 | 44 | 256 | 64 | 28.680258 | 1072.376648 | 1077.175000 | busy_gap_dominates_b2b_candidate |
| 7957 | 43 | 264 | 66 | 413.361055 | 1131.437927 | 1136.110000 | busy_gap_dominates_b2b_candidate |
| 0 | 49 | 7724 | 1931 | 29.762383 | 39.993328 | 42.690000 | busy_gap_dominates_b2b_candidate |
| 0 | 50 | 7672 | 1918 | 29.966939 | 39.638800 | 42.310000 | busy_gap_dominates_b2b_candidate |
| 0 | 51 | 7256 | 1814 | 30.171494 | 38.893841 | 41.580000 | mixed_or_inconclusive |
| 0 | 46 | 8356 | 2089 | 29.118372 | 36.630272 | 39.400000 | mixed_or_inconclusive |
| 0 | 43 | 52 | 13 | 28.459859 | 1114.825195 | 1118.960000 | low_confidence |
| 0 | 47 | 8508 | 2127 | 29.337429 | 35.943104 | 38.680000 | mixed_or_inconclusive |
| 0 | 48 | 8108 | 2027 | 29.557828 | 35.092720 | 37.790000 | mixed_or_inconclusive |
| 0 | 53 | 6840 | 1710 | 30.581947 | 37.107489 | 39.790000 | mixed_or_inconclusive |

## Largest Wall-Busy Buckets

| ctx_tokens | decode_batch | event rows | wall rows | sim ms | busy ms | wall ms | class |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 47 | 8508 | 2127 | 29.337429 | 35.943104 | 38.680000 | mixed_or_inconclusive |
| 0 | 46 | 8356 | 2089 | 29.118372 | 36.630272 | 39.400000 | mixed_or_inconclusive |
| 0 | 48 | 8108 | 2027 | 29.557828 | 35.092720 | 37.790000 | mixed_or_inconclusive |
| 0 | 49 | 7724 | 1931 | 29.762383 | 39.993328 | 42.690000 | busy_gap_dominates_b2b_candidate |
| 0 | 50 | 7672 | 1918 | 29.966939 | 39.638800 | 42.310000 | busy_gap_dominates_b2b_candidate |
| 0 | 52 | 7264 | 1816 | 37.368430 | 37.829201 | 40.540000 | no_material_gap |
| 0 | 54 | 7392 | 1848 | 30.786503 | 36.339567 | 39.000000 | mixed_or_inconclusive |
| 0 | 51 | 7256 | 1814 | 30.171494 | 38.893841 | 41.580000 | mixed_or_inconclusive |
| 0 | 53 | 6840 | 1710 | 30.581947 | 37.107489 | 39.790000 | mixed_or_inconclusive |
| 0 | 55 | 6040 | 1510 | 30.991058 | 35.438223 | 38.120000 | mixed_or_inconclusive |
| 0 | 6 | 2184 | 546 | 19.910502 | 17.759456 | 20.480000 | mixed_or_inconclusive |
| 0 | 56 | 2080 | 520 | 31.196956 | 34.395424 | 37.080000 | mixed_or_inconclusive |

## Scope

- diagnostic_only=true; valid_for_default=false; perf_database=false.
- event rows are grouped by `(ctx_tokens, generation_requests)` because event rows are approximately 4x serve iteration rows.
- The 5.676% event timing overhead gate failure is recorded as measurement risk; no data is ingested into PerfDB.
