# Phase454 GPU batch acceptance

结论: GPU 批已收包。dp2 两个 N=512 清洁参考都可用;dp2 bt65536 的非冲突 mixed 行可入库并改善 A/B;TP8 B2b 行因缺 KV/ISL scope 轴导致跨场景回归,本轮阻断不入库。

## Summary

| Section | Scenario | Metric | Value | Target | Status | Note |
|---|---|---|---:|---|---|---|
| clean_reference | K2.5-tp4ep8dp2-8k2k | clean_error_ratio | 1.041312 | <=1.15 | pass | N=512 C=128 |
| clean_reference | K2.5-tp4ep8dp2-8k2k | clean_real_output_tok_s_gpu | 162.948516 |  | clean_reference_collected |  |
| clean_reference | K2.5-tp4ep8dp2-32k3k | clean_error_ratio | 1.057492 | <=1.15 | pass | N=512 C=64 |
| clean_reference | K2.5-tp4ep8dp2-32k3k | clean_real_output_tok_s_gpu | 53.221414 |  | clean_reference_collected |  |
| clean_validate | K2.5-tp8ep8-8k2k | error_ratio | 1.438569 | <=1.15 | fail | existing_reference |
| clean_validate | K2.5-tp4ep8dp2-8k2k | error_ratio | 1.041312 | <=1.15 | pass | phase454_n512_clean |
| clean_validate | K2.5-tp8ep8-8k2k-bt65536 | error_ratio | 1.119881 | <=1.15 | pass | existing_reference |
| clean_validate | K2.5-tp4ep8dp2-8k2k-bt65536 | error_ratio | 1.233619 | <=1.15 | fail | existing_reference |
| clean_validate | K2.5-tp8ep8-32k3k | error_ratio | 1.163036 | <=1.15 | fail | existing_reference |
| clean_validate | K2.5-tp4ep8dp2-32k3k | error_ratio | 1.057492 | <=1.15 | pass | phase454_n512_clean |
| clean_validate | all | max_error_ratio | 1.438569 | <=1.15 | fail | max_scenario=K2.5-tp8ep8-8k2k |
| gpu_overhead_gate | K2.5-tp8ep8-8k2k | overhead_pct | 0.041410 | <=2.0 | pass | off=1171.0122826772538 on=1170.5273652241867 |
| gpu_overhead_gate | K2.5-tp4ep8dp2-8k2k-bt65536 | overhead_pct | 1.444566 | <=2.0 | pass | off=883.000989801678 on=870.2454619600386 |
| b2b_candidate | K2.5-tp4ep8dp2-8k2k-bt65536 | active_rows | 104 | >0 | pass |  |
| b2b_candidate | K2.5-tp4ep8dp2-8k2k-bt65536 | blocked_rows | 15 | reported, not overwritten | reported |  |
| b2b_candidate | K2.5-tp8ep8-8k2k | active_rows | 0 | >0 | blocked |  |
| b2b_candidate | K2.5-tp8ep8-8k2k | blocked_rows | 193 | reported, not overwritten | reported |  |
| perfdb_candidate | all | active_total | 104 | accepted or already ingested | ready | appendable=0 already_ingested=104 |
| perfdb_candidate | all | blocked_total | 208 | no silent overwrite | pass | Duplicate keys require schema/regime decision. |
| gpu_residual | K2.5-tp4ep8dp2-8k2k | process_residual_after_bytes | 0 | 0 | pass | /Users/mewtwo/2026/work/codebase/AIC-mewtwo/.worktrees/feature-pr403/docs/iter_gap_investigation/phase454_gpu_batch/recollect_dp2_8k2k/process_residual_after.txt |
| gpu_residual | K2.5-tp8ep8-8k2k | process_residual_after_bytes | 0 | 0 | pass | /Users/mewtwo/2026/work/codebase/AIC-mewtwo/.worktrees/feature-pr403/docs/iter_gap_investigation/phase454_gpu_batch_scope/b2b_tp8_8k2k/overhead_on/process_residual_after.txt |
| gpu_residual | K2.5-tp4ep8dp2-8k2k-bt65536 | process_residual_after_bytes | 0 | 0 | pass | /Users/mewtwo/2026/work/codebase/AIC-mewtwo/.worktrees/feature-pr403/docs/iter_gap_investigation/phase454_gpu_batch_scope/b2b_dp2_8k2k_bt65536/overhead_on/process_residual_after.txt |
| verdict | phase454 | clean_acceptance_verdict | gpu_batch_collected_dp2_bt_ingested_tp8_blocked_by_scope | clean reference + safe B2b ingestion evidence | no_go | TP8 B2b rows require a KV/ISL scope key before PerfDB ingestion. |

## Candidate Row Preview

| Scenario | Topology | Phase | Bucket | Batch | Median ms | Samples | Status |
|---|---|---|---:|---:|---:|---:|---|
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | decode | 1 | 1 | 16.785776 | 5044 | blocked_duplicate_key |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | decode | 2 | 2 | 16.646688 | 2144 | blocked_duplicate_key |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | decode | 3 | 3 | 16.796192 | 3624 | blocked_duplicate_key |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | decode | 4 | 4 | 16.816032 | 101 | blocked_duplicate_key |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | decode | 5 | 5 | 17.423680 | 4 | blocked_duplicate_key |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | decode | 6 | 6 | 16.862017 | 3 | blocked_duplicate_key |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | decode | 8 | 8 | 17.403760 | 10 | blocked_duplicate_key |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | decode | 9 | 9 | 21.559856 | 32 | blocked_duplicate_key |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | decode | 10 | 10 | 21.306017 | 15 | blocked_duplicate_key |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | decode | 11 | 11 | 20.830960 | 4 | blocked_duplicate_key |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | decode | 12 | 12 | 20.842384 | 118 | blocked_duplicate_key |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | decode | 13 | 13 | 20.624096 | 85429 | blocked_duplicate_key |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | decode | 14 | 14 | 20.412832 | 94099 | blocked_duplicate_key |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | decode | 15 | 15 | 19.919231 | 87747 | blocked_duplicate_key |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | decode | 16 | 16 | 19.371103 | 18031 | blocked_duplicate_key |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 6486 | 15 | 561.312408 | 32 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 8015 | 15 | 1118.641235 | 16 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 8078 | 15 | 671.527527 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 8079 | 15 | 678.821808 | 8 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 8093 | 15 | 672.048248 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 8094 | 15 | 673.032928 | 12 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 8103 | 15 | 674.980743 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 8108 | 15 | 665.553192 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 8109 | 15 | 675.491699 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 8110 | 15 | 673.925262 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 8111 | 15 | 673.797791 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 8119 | 15 | 675.613464 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 8125 | 15 | 1131.062500 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 8126 | 15 | 1131.077393 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 8141 | 15 | 676.362976 | 8 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 8142 | 15 | 676.960022 | 8 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 8208 | 14 | 680.279724 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 8210 | 14 | 684.899933 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 8432 | 13 | 683.668030 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 8433 | 13 | 692.850769 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 8549 | 5 | 698.332275 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 8700 | 14 | 704.909119 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 8701 | 14 | 713.168976 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 8716 | 14 | 704.015350 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 8717 | 14 | 713.950928 | 12 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 8733 | 14 | 713.811798 | 8 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 8767 | 2 | 2445.371216 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 8891 | 13 | 723.951172 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 8893 | 13 | 721.614624 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 8945 | 13 | 1211.437927 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 8947 | 13 | 1211.461182 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 9056 | 13 | 726.768402 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 9065 | 13 | 734.114014 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 9210 | 13 | 741.894867 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 9213 | 13 | 737.195465 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 9266 | 13 | 743.024658 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 9275 | 13 | 742.493042 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 9372 | 13 | 1005.499695 | 8 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 9373 | 13 | 746.869507 | 16 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 9388 | 13 | 743.417480 | 12 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 9389 | 13 | 750.256775 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 9403 | 13 | 751.029663 | 8 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 9404 | 13 | 742.504791 | 8 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 9405 | 13 | 755.875305 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 14484 | 14 | 5855.418945 | 32 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 16078 | 15 | 1290.335571 | 8 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 16079 | 15 | 1288.080627 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 16412 | 4 | 1289.156311 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 16816 | 15 | 1314.064331 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 16825 | 15 | 1329.572510 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 17950 | 14 | 1396.384460 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 17960 | 14 | 1386.689148 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 18031 | 14 | 2423.638672 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 18038 | 14 | 2423.461426 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 18077 | 14 | 2410.429932 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 18078 | 14 | 1907.465210 | 8 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 18089 | 14 | 1403.959412 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 22482 | 13 | 6525.164307 | 12 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 24328 | 3 | 1903.533630 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 24854 | 3 | 2444.778687 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 30480 | 12 | 7011.966309 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 32146 | 3 | 2527.269287 | 8 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 32161 | 3 | 2528.898804 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 38478 | 11 | 7522.151611 | 4 | already_ingested |
| K2.5-tp4ep8dp2-8k2k-bt65536 | tp4dp2ep8 | mixed_prefill | 40001 | 1 | 3141.027588 | 4 | already_ingested |
| ... | ... | ... | ... | ... | ... | ... | 232 more rows |

Default AIC 仍为 No-Go: dp2 bt65536 已有零回归 A/B;TP8 仍需带 KV/ISL scope 的后续采集或 schema 扩展。
