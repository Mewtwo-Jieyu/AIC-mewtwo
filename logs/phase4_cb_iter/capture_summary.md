# Phase 4 cb_sim Iteration Profiling Capture

Service: `kimi-k2.5`, vLLM `0.19.0`, `tensor_parallel_size=4`, `data_parallel_size=2`, `enable_expert_parallel=True`, `max_num_batched_tokens=8192`, `enable_logging_iteration_details=True`.

Remote log source: `/mnt/shared-storage-user/ailab-sys/caikun/code/configs/service/accuracy/vllm_kimi_k2_5_v2.log`

## Captured Data

| Scenario | Global Requests | Per-Engine Concurrency | Local Directory | Iter Log Lines | Iteration Rows |
| --- | ---: | ---: | --- | ---: | ---: |
| `3k-3k b=128` | 128 | 64 | `logs/phase4_cb_iter/3k3k_b128` | 6311 | 6021 |
| `10k-2k b=32` | 32 | 16 | `logs/phase4_cb_iter/10k2k_b32` | 4158 | 4044 |
| `32k-1k b=16` | 16 | 8 | `logs/phase4_cb_iter/32k1k_b16` | 2132 | 2056 |

Each directory contains `summary.json`, `records.jsonl`, `metrics_before.txt`, `metrics_after.txt`, `log_start.txt`, `log_end.txt`, and `iter.log`.

## Benchmark Summary

| Scenario | Success | Wall Time s | Mean Latency ms | p99 Latency ms | Tail Avg Tokens/Iter | Tail Avg Iter Latency ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `3k-3k b=128` | 128/128 | 158.53 | 158351.28 | 158488.63 | 31.93 | 13.17 |
| `10k-2k b=32` | 32/32 | 70.33 | 69711.93 | 70321.44 | 7.95 | 6.41 |
| `32k-1k b=16` | 16/16 | 51.67 | 51450.31 | 51663.26 | 3.99 | 5.91 |

The `/metrics` tail fields are coarse service-level signals. Use `iter.log` as the primary source for per-iteration latency modeling.

## DP0 Diagnostic Summary

Command template:

```bash
conda run -n aic python scripts/diagnose_cb_iter_latency.py \
  --isl <isl> --osl <osl> --concurrency <global_b/2> \
  --tp 4 --dp 2 --moe-tp 1 --moe-ep 8 --max-num-batched-tokens 8192 \
  --vllm-log <scenario>/iter.log --vllm-engine-id DP0
```

| Scenario | Phase | cb_sim Mean ms | vLLM Mean ms | Mean Overhead ms | vLLM Iterations |
| --- | --- | ---: | ---: | ---: | ---: |
| `3k-3k b=128` | prefill | 359.37 | 297.72 | -61.65 | 1 |
| `3k-3k b=128` | mixed | 345.60 | 867.12 | 508.20 | 8 |
| `3k-3k b=128` | pure_decode | 31.11 | 49.61 | 14.83 | 3001 |
| `10k-2k b=32` | prefill | 381.03 | 1054.15 | 673.12 | 2 |
| `10k-2k b=32` | mixed | 373.75 | 946.31 | 572.81 | 19 |
| `10k-2k b=32` | pure_decode | 23.14 | 24.65 | 1.32 | 2001 |
| `32k-1k b=16` | prefill | 441.63 | 1185.70 | 744.07 | 4 |
| `32k-1k b=16` | mixed | 432.97 | 1053.65 | 612.21 | 23 |
| `32k-1k b=16` | pure_decode | 21.60 | 21.96 | 0.37 | 1001 |

## Immediate Readout

Pure decode overhead is strongly batch-size dependent in this service:

- b=128 global, per-engine decode batch about 64: overhead about `14.83 ms/iter`.
- b=32 global, per-engine decode batch about 16: overhead about `1.32 ms/iter`.
- b=16 global, per-engine decode batch about 8: overhead about `0.37 ms/iter`.

Prefill and mixed iterations show large positive gaps in the two long-prompt scenarios, so a decode-only overhead model is not enough to explain all observed latency.
