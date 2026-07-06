# Phase418 Roofline Reconcile

Phase418 只复核报告门和 provenance；不加 runtime clamp、不改 PerfDatabase 数据。

## Verdict

- verdict: `phase415_roofline_gate_wrong_not_moe_perf_table`
- next phase: `regenerate_phase415_416_417_and_anchor_revalidate`
- remaining mixed-prefill gap: real `4718.954849` ms vs fixed sim `1818.637996` ms = `2.594774`x
- Default AIC: `No-Go`

## Roofline Replay

| component | old bound | fp8 bound | bf16 bound | selected bound | measured | old gate | corrected gate | MFU |
|---|---:|---:|---:|---:|---:|---|---|---:|
| moe_compute | 455.757015 | 85.454440 | 170.822563 | 170.822563 | 352.717448 | failed | passed | 0.484304 |
| ep_dispatch_combine | 489.335467 |  |  | 489.335467 | 497.491058 | passed | passed |  |

## Provenance

- committed collector semantics: `num_tokens_token_rows_single_layer_ep_expert_map_no_dispatch`
- dirty collector drift: `dirty_collect_moe_diff_present`
- provenance: `aligned`

## Boundary

- GPU/SSH: not used.
- Runtime/PerfDatabase/gate: not modified.
- Phase405 penalty: not read.
- Default AIC: No-Go.
