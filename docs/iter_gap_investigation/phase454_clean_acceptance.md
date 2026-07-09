# Phase454 clean acceptance

## Decision record

| Item | Decision | Status | Note |
|---|---|---|---|
| Zero-regression rule | global_source_backed_semantic_fix_may_expose_error_cancellation | recorded | Do not silently revert when a global vLLM semantic fix reveals prior offsetting errors. |
| Error-cancellation warning | dp2_8k2k_1.136_is_not_closed | recorded | Current N=128 reference can hide steady-state model error through burst artifact. |

## Reserve scope audit

| Metric | Value | Target | Status | Note |
|---|---|---|---|---|
| waiting_gate | `and not self.kv_cache_manager.can_fit_full_sequence(` |  | source_located | /Users/mewtwo/2026/work/codebase/vllm-0.19.0/vllm/v1/core/sched/scheduler.py:734 |
| can_fit_input | `full_num_tokens = min(request.num_tokens, self.max_model_len)` | request.num_tokens | source_located | /Users/mewtwo/2026/work/codebase/vllm-0.19.0/vllm/v1/core/kv_cache_manager.py:244 |
| request_num_tokens | `return len(self._all_token_ids)` | current prompt + generated tokens | source_located | /Users/mewtwo/2026/work/codebase/vllm-0.19.0/vllm/v1/request.py:227 |
| max_tokens_not_in_num_tokens | `self.max_tokens = sampling_params.max_tokens` | not referenced by can_fit_full_sequence | source_located | /Users/mewtwo/2026/work/codebase/vllm-0.19.0/vllm/v1/request.py:103 |
| async_placeholder_not_num_tokens | `request.num_output_placeholders += 1 + cur_num_spec_tokens` | separate counter | source_located | /Users/mewtwo/2026/work/codebase/vllm-0.19.0/vllm/v1/core/sched/async_scheduler.py:32 |
| prompt_plus_max_tokens_hypothesis | `current_sequence_not_prompt_plus_max_tokens` | prompt+max_tokens if source proves it | rejected_prompt_plus_max_tokens_hypothesis | No runtime change: vLLM gate reserves current full sequence, not final max sequence. |

Conclusion: the vLLM gate reserves the current full sequence (`request.num_tokens`). It does not reserve `prompt + max_tokens`, so Phase454 does not change cb_sim on this hypothesis.

## Residual preemption

| Metric | Value | Target | Status | Note |
|---|---:|---|---|---|
| preemption_events | 192 | ~24 real observed victims | partial | Residual events remain after Phase453 headroom gate. |
| classification_coverage | 1.0 | >=0.80 | pass | Phase451D classification covers all residual events. |
| admission_induced | 6 |  |  |  |
| decode_growth_pressure | 66 |  |  |  |
| thrash_repeat_victim | 120 |  |  |  |
| runtime_fix_gate | blocked_pending_unique_semantic_fix | unique source-backed semantic fix | blocked | The prompt+max reserve hypothesis does not provide that unique fix. |

## Acceptance state

| Metric | Value | Target | Status | Note |
|---|---:|---|---|---|
| dp2_8k2k_error_ratio | 1.1362789195250411 | <=1.15 target | pass_current_but_not_clean | N=128 reference may still contain artifact cancellation. |
| max_error_ratio | 1.4385687849872588 | <=1.50 current; <=1.15 target | pass_default_open_15pct | max_scenario=K2.5-tp8ep8-8k2k |

## GPU batch plan

| Task | Protocol | Purpose | Status |
|---|---|---|---|
| dp2_8k2k_recollect | N=512/C=128 vanilla | clean steady reference | pending_gpu |
| dp2_32k3k_recollect | N=512/C=64 vanilla | clean steady reference | pending_gpu |
| tp8_b2b | tp8ep8-8k2k B2b | cover TP8 serving-state scope | pending_gpu |
| dp2_bt65536_b2b | tp4dp2ep8-8k2k-bt65536 B2b | large-bt regime coverage | pending_gpu |

Final verdict: `reserve_prompt_plus_max_rejected_gpu_batch_required`. Default AIC remains No-Go until the clean-reference GPU batch and final validate table close.
