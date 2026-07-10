# Phase461 max_bt scope

Verdict: `max_bt_exact_keying_enabled_step2_pending`.

## Six-point A/B

| scenario | before | after | delta | class | attribution |
|---|---:|---:|---:|---|---|
| K2.5-tp8ep8-8k2k | 1.133x | 1.133x | +0.000 | unchanged | TP8 already used exact max_bt before Step 1 |
| K2.5-tp4ep8dp2-8k2k | 1.041x | 1.041x | +0.000 | unchanged | exact bt8000 reproduces the prior effective charge |
| K2.5-tp8ep8-8k2k-bt65536 | 1.235x | 1.235x | +0.000 | unchanged | TP8 already used exact max_bt before Step 1 |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 1.167x | 1.185x | +0.018 | regressed | correct bt65536 rows replace the cross-regime unscoped table; 14484/14 spike reached |
| K2.5-tp4ep8dp2-32k3k | 1.056x | 1.173x | +0.117 | regressed | correct bt32000 rows replace the cross-regime unscoped table; 63/9 spike not reached |
| K2.5-tp8ep8-32k3k | 1.397x | 1.397x | +0.000 | unchanged | TP8 already used exact max_bt before Step 1 |

## Spike reachability

| scenario | cell | exact queries | influenced hits | status |
|---|---|---:|---:|---|
| K2.5-tp4ep8dp2-32k3k | 32000/63/9 | 0 | 0 | not_reached_current_six |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 65536/14484/14 | 2 | 2 | reachable_recollect |

## Scope decision

| item | decision |
|---|---|
| DP2 query key | Exact configured `max_num_batched_tokens`; absent scope returns `None` and the existing analytic path is used. |
| Regression handling | Retain source-correct exact scope. The two DP2 movements expose prior cross-regime error cancellation and are not silently rolled back. |
| Phase455 TP8 rows | `retain_pending_isl_replay`: safe from collisions in the current six points, but max_bt alone does not prove generic cross-ISL safety. |
| ISL band | Not added in Step 1; Step 2 replay remains the decision gate. |
| Default AIC | No-Go. |

## Verification

| check | result |
|---|---|
| Step1 targeted pytest | 17 passed |
| Broader two-file pytest | 115 passed, 3 pre-existing unrelated failures |
| py_compile / diff check / CRLF | passed |
