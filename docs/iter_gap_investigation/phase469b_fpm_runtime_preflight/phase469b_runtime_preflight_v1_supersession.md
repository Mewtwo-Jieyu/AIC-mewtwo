# Phase469B v1 Supersession

- Status: `SUPERSEDED_INVALID_CONFIG_COMPARISON`
- Original result: `phase469b_runtime_preflight_v1_original.json`
- Original SHA256: `a727d141621f3918edd22156b30ff8b69bb88e29d950d4defce975a65aec76eb`
- Replacement schema: `phase469b_fpm_runtime_preflight_v2`

V1 compared the AIC flattened simulator config hash with the official multimodal checkpoint wrapper config hash. These files have different responsibilities, so that mismatch was not a valid model identity failure.

The original v1 result remains unchanged for audit. It is not valid evidence for Phase469C, model attribution, PerfDatabase, or Default AIC readiness.
