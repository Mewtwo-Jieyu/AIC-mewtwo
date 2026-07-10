# Phase461 Step 0: H200 vLLM 0.19 PerfDB anomaly scan

结论：扫描 10 张表、69,543 行，按严格局部极值规则得到 67 个可疑行，其中 33 行被两个运行轴同时命中，16 行落在当前 K2.5 六点语义范围。Phase460 已确认的 MLA 坏行被正确检出并列为 mandatory。其余结果仍是 candidate，不得自动删除、平滑或替换。

## 扫描规则

| 规则 | 约束 |
|---|---|
| 语义隔离 | 除被扫描运行轴外，其余列必须完全相同 |
| 邻域 | 目标点必须有左右两个实测格点 |
| 偏离门 | 实测/邻点线性趋势的对称比 > 3.0x |
| 极值门 | 同时高于两邻点 3.0x，或低于两邻点 3.0x |
| 输出语义 | candidate only；kernel 切换等合法 cliff 需 provenance 或重采确认 |

## 表级汇总

| 表 | 行数 | 扫描轴 | 检出记录 | 可疑行 | 双轴命中 | 六点范围 |
|---|---:|---|---:|---:|---:|---:|
| context_attention_perf.txt | 10120 | batch_size,isl | 0 | 0 | 0 | 0 |
| context_mla_perf.txt | 1543 | batch_size,isl | 31 | 22 | 9 | 8 |
| custom_allreduce_perf.txt | 138 | message_size | 0 | 0 | 0 | 0 |
| gemm_perf.txt | 26040 | m,n,k | 0 | 0 | 0 | 0 |
| generation_attention_perf.txt | 10870 | batch_size,effective_seq | 0 | 0 | 0 | 0 |
| generation_mla_perf.txt | 2901 | batch_size,effective_seq | 63 | 39 | 24 | 6 |
| moe_perf.txt | 17463 | num_tokens | 4 | 4 | 0 | 0 |
| vllm_ep8_a2a_decode_perf.txt | 5 | bucket_tokens | 0 | 0 | 0 | 0 |
| vllm_module_perf.txt | 21 | bucket_tokens | 0 | 0 | 0 | 0 |
| vllm_serving_state_perf.txt | 442 | bucket_tokens,decode_batch | 2 | 2 | 0 | 2 |

## 当前六点范围候选

| 表:行 | shape | 轴 | 方向 | 实测 ms | 邻点趋势 ms | 偏离 | GPU 处置 |
|---|---|---|---|---:|---:|---:|---|
| vllm_serving_state_perf.txt:324 | topology=tp4dp2ep8;max_bt=32000;phase=mixed_prefill;kind=forward_total;category=forward_total;bucket=63;decode=9 | bucket_tokens | spike | 2817.671753 | 44.265359 | 63.65x | gpu_batch_candidate_review |
| generation_mla_perf.txt:73 | heads=8;batch=16;seq=2 | batch_size | spike | 1.659456 | 0.054892 | 30.23x | gpu_batch_candidate_review |
| context_mla_perf.txt:312 | heads=8;batch=2;seq=32 | batch_size,isl | spike | 0.934411 | 0.053497 | 17.47x | gpu_batch_candidate_review |
| context_mla_perf.txt:473 | heads=16;batch=4;seq=64 | batch_size,isl | spike | 0.967024 | 0.058055 | 16.66x | gpu_batch_candidate_review |
| generation_mla_perf.txt:88 | heads=16;batch=32;seq=2 | batch_size | spike | 0.804859 | 0.056302 | 14.30x | gpu_batch_candidate_review |
| generation_mla_perf.txt:1640 | heads=8;batch=8;seq=1024 | batch_size,effective_seq | spike | 0.486048 | 0.053682 | 9.05x | gpu_batch_candidate_review |
| generation_mla_perf.txt:1659 | heads=8;batch=16;seq=1024 | batch_size | valley | 0.052352 | 0.418878 | 8.00x | gpu_batch_candidate_review |
| generation_mla_perf.txt:2521 | heads=8;batch=8;seq=32768 | batch_size,effective_seq | spike | 0.958645 | 0.133260 | 7.19x | mandatory_recollect |
| context_mla_perf.txt:1059 | heads=8;batch=1;seq=2048 | isl | spike | 0.766085 | 0.145161 | 5.28x | gpu_batch_candidate_review |
| generation_mla_perf.txt:1678 | heads=8;batch=32;seq=1024 | batch_size,effective_seq | spike | 0.284539 | 0.055113 | 5.16x | gpu_batch_candidate_review |
| vllm_serving_state_perf.txt:384 | topology=tp4dp2ep8;max_bt=65536;phase=mixed_prefill;kind=forward_total;category=forward_total;bucket=14484;decode=14 | bucket_tokens | spike | 5855.418945 | 1139.706924 | 5.14x | gpu_batch_candidate_review |
| context_mla_perf.txt:915 | heads=16;batch=8;seq=1024 | isl | valley | 0.240523 | 1.111224 | 4.62x | gpu_batch_candidate_review |
| context_mla_perf.txt:854 | heads=16;batch=64;seq=512 | isl | spike | 7.883226 | 2.123129 | 3.71x | gpu_batch_candidate_review |
| context_mla_perf.txt:843 | heads=16;batch=32;seq=512 | isl | spike | 3.801499 | 1.075141 | 3.54x | gpu_batch_candidate_review |
| context_mla_perf.txt:832 | heads=16;batch=16;seq=512 | isl | spike | 1.915968 | 0.550747 | 3.48x | gpu_batch_candidate_review |
| context_mla_perf.txt:823 | heads=16;batch=8;seq=512 | isl | spike | 0.966651 | 0.290101 | 3.33x | gpu_batch_candidate_review |

Phase460 根因行在 `generation_mla_perf.txt:2521`，同时被 batch 与 effective-sequence 两轴检出；scanner 对已知故障的召回门通过。

## GPU 批次影响

| 类型 | 数量 | 处理 |
|---|---:|---|
| 已确认坏行 | 1 | 必采；沿用 Phase460 的 heads=8 × batch 8/16 × KV 16k/32k/64k 网格 |
| 六点范围其他候选 | 15 | 先做 provenance/运行命中审核；确认可达后并入对应 microbench 或 serving window |
| 六点范围外候选 | 51 | 不塞入本轮 GPU 批；归档为全库数据维护清单 |

不能把全部 candidate 无差别塞进一次 GPU：context/generation MLA 是 kernel microbench，serving-state 是在位 workload 窗口，测量原语不同。Step 3 清单应在 Step 1/2 的实际查询命中审计后按原语拆分，但仍可共用同一次 GPU 分配窗口。

## 后续门

- Step 1 才允许修改 serving-state schema/runtime；本阶段没有代码路径或数据表改动。
- Step 1 必须带 `max_num_batched_tokens` 精确键，miss 走解析模型，禁止跨 regime 静默复用。
- 抢占残差、路由竞态、tp8-bt65536 replay 残差 1.118、LOO 清单继续存档。
- Default AIC 维持 No-Go；6/6 进入 1.15 前不收紧门限。

完整逐轴邻域证据见 `docs/iter_gap_investigation/phase461_perfdb_anomaly_scan.csv`。
