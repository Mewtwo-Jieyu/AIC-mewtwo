# Phase329 Default Model Form Candidates

This diagnostic audit lists possible default model candidate forms and rejects routes that are already contradicted by committed evidence.

| Item | Value |
|---|---|
| Default AIC | No-Go |
| PerfDatabase | No-Go |
| diagnostic_only | true |
| valid_for_default | false |
| perf_database | false |
| retained candidate | topology_specific_cadence_boundary_candidate |
| retained status | candidate_needs_more_evidence |

| Candidate | Model form | Status | Reason or next validation |
|---|---|---|---|
| global_correction | single_multiplier_or_global_penalty | rejected | Phase303 topology directions conflict: tp8 slower while tp4dp2 faster |
| budget_ceiling_model | configured_max_num_batched_tokens_as_linear_cost | rejected | Phase258 and Phase303 reject configured budget ceiling as actual cost |
| diagnostic_lookup_as_model | reuse_exact_key_lookup_for_default_prediction | rejected | diagnostic lookup is not a prediction API |
| lifecycle_canary_as_evidence | treat_worker_canary_ratio_as_model_evidence | rejected | Phase323 proves runner lifecycle only, not model behavior |
| topology_specific_cadence_boundary_candidate | topology_shape_budget_exact_model_with_cadence_boundary_features | candidate_needs_more_evidence | independent_topology_specific_holdout_with_error_threshold |

The global correction is rejected because Phase303 has opposite throughput directions across topology. The budget ceiling model is rejected because Phase258 and Phase303 show that configured max_num_batched_tokens is a ceiling, not actual scheduled-token cost. The diagnostic exact-key lookup cannot be reused as a default prediction API. Phase323 is runner lifecycle evidence only.

Only topology_specific_cadence_boundary_candidate remains as a future diagnostic candidate, and it still needs independent topology-specific holdout validation plus a defined error threshold before any default AIC discussion. Default readiness remains No-Go.
