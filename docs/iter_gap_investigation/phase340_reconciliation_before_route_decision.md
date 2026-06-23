# Phase340 Reconciliation Before Route Decision

| Item | Decision |
|---|---|
| Default AIC | No-Go |
| GPU run allowed | false |
| Purpose | Resolve two specification drift risks before discussing A/B route |

## Issue 1: Phase333 4k control scenario mismatch

Phase333 holdout matrix uses `tp8ep8-4k2k-bt12000` and `tp4dp2ep8-4k2k-bt12000` as control scenarios for isl4000 rows.

Phase258 actual scheduled token family API uses `control_bt4000` for isl4000 rows (matching the pattern: control budget = ISL).

| Source | isl4000 control | isl12000 control |
|---|---|---|
| Phase258 API evidence | bt4000 | bt12000 |
| Phase333 holdout matrix (before fix) | bt12000 | bt12000 |

Decision: Phase333 must align with Phase258. The control scenario for isl4000 rows is `bt4000`, not `bt12000`. The holdout matrix is corrected in this phase.

Rationale: the control budget should match the ISL (tight budget = actual prefill tokens fit in one iteration), which is the mechanism tested by the family evidence. Using bt12000 for isl4000 would be a different experimental condition with no committed evidence.

## Issue 2: Phase335 numeric_error_threshold_contract ambiguity

The value `blocked_until_numeric_model_form_exists` in the threshold column could be misread by downstream parsers as a pass threshold rather than a blocking precondition.

Decision: add explicit `threshold_type` field with value `blocking_precondition` to Phase335 CSV, making it unambiguous that no threshold has been defined and no run is authorized.

## After reconciliation

Both issues are specification-only fixes. No GPU run, no new evidence, no model integration. After this phase, the next discussion is A/B route selection:

- Route A: module-level vLLM MoE perf table (Phase 123 continuation)
- Route B: cb_sim runtime shape key integration into IterationLatencyCalculator (Phase 10 continuation)

Both routes must first answer: what are the pre-run inputs, outputs, formula form, and error threshold for a default model? Without that, running more traces or holdouts only accumulates diagnostic evidence without producing a model.

| Flag | Value |
|---|---|
| diagnostic_only | true |
| valid_for_default | false |
| perf_database | false |
