# Phase 75 Minimal Delivery Manifest

## Conclusion

The minimal delivery package should contain only experimental scheduler descriptor code, strict-alignment utilities, tests, and the Phase63-74 decision docs. It must not stage collector scripts, runner scripts, raw logs, or old residual/profiler/NCCL artifacts.

## Staging Whitelist

| type | file |
|---|---|
| schema/export | `src/aiconfigurator/sdk/backends/cb_simulator/forward_descriptor.py` |
| schema/export | `src/aiconfigurator/sdk/backends/cb_simulator/__init__.py` |
| CLI | `scripts/diagnose_cb_iter_latency.py` |
| CLI | `scripts/validate_cb_simulator.py` |
| parser | `scripts/analyze_vllm_scheduler_descriptor_phase62.py` |
| test | `tests/unit/test_forward_descriptor_scheduler_alignment.py` |
| test | `tests/unit/test_forward_descriptor_vllm_like_scheduler.py` |
| test | `tests/unit/scripts/test_analyze_vllm_scheduler_descriptor_phase62.py` |
| docs | `docs/iter_gap_investigation/phase63_scheduler_alignment_descriptor_schema.md` |
| docs | `docs/iter_gap_investigation/phase63_scheduler_alignment_go_nogo.md` |
| docs | `docs/iter_gap_investigation/phase64_scheduler_alignment_compare_schema.md` |
| docs | `docs/iter_gap_investigation/phase64_scheduler_alignment_compare_go_nogo.md` |
| docs | `docs/iter_gap_investigation/phase65_cb_scheduler_semantics_audit.md` |
| docs | `docs/iter_gap_investigation/phase65_vllm_scheduler_reference.md` |
| docs | `docs/iter_gap_investigation/phase65_scheduler_semantics_go_nogo.md` |
| docs | `docs/iter_gap_investigation/phase66_cb_scheduler_source_audit.md` |
| docs | `docs/iter_gap_investigation/phase66_vllm_scheduler_requirements.md` |
| docs | `docs/iter_gap_investigation/phase66_vllm_like_scheduler_descriptor_design.md` |
| docs | `docs/iter_gap_investigation/phase66_go_nogo.md` |
| docs | `docs/iter_gap_investigation/phase67_scheduler_compare_closeout.md` |
| docs | `docs/iter_gap_investigation/phase67_go_nogo.md` |
| docs | `docs/iter_gap_investigation/phase68_available_descriptor_artifacts.md` |
| docs | `docs/iter_gap_investigation/phase68_holdout_selection.md` |
| docs | `docs/iter_gap_investigation/phase68_scheduler_shape_generalization_summary.md` |
| docs | `docs/iter_gap_investigation/phase69_holdout_capture_plan.md` |
| docs | `docs/iter_gap_investigation/phase69_scheduler_holdout_go_nogo.md` |
| docs | `docs/iter_gap_investigation/phase70_short_isl_scheduler_generator.md` |
| docs | `docs/iter_gap_investigation/phase71_32k1k_b16_generator_fail_fast.txt` |
| docs | `docs/iter_gap_investigation/phase71_artifact_inventory.md` |
| docs | `docs/iter_gap_investigation/phase71_scheduler_generalization_go_nogo.md` |
| docs | `docs/iter_gap_investigation/phase72_scheduler_regime_taxonomy.md` |
| docs | `docs/iter_gap_investigation/phase72_32k1k_scheduler_rule_design.md` |
| docs | `docs/iter_gap_investigation/phase72_go_nogo.md` |
| docs | `docs/iter_gap_investigation/phase73_32k1k_scheduler_branch_closeout.md` |
| docs | `docs/iter_gap_investigation/phase73_go_nogo.md` |
| docs | `docs/iter_gap_investigation/phase74_scheduler_descriptor_coverage_closeout.md` |
| docs | `docs/iter_gap_investigation/phase74_scheduler_descriptor_review_checklist.md` |
| evidence CSV | `docs/iter_gap_investigation/phase73_32k1k_b16_vllm_like_descriptor.csv` |
| evidence CSV | `docs/iter_gap_investigation/phase73_32k1k_b16_scheduler_shape_gap.csv` |
| phase75 docs | `docs/iter_gap_investigation/phase75_minimal_delivery_manifest.md` |
| phase75 docs | `docs/iter_gap_investigation/phase75_staging_dry_run.md` |

## Explicit Exclusions

| exclude | reason |
|---|---|
| `collector/vllm/*` | remote research collection scripts |
| `collector/vllm/run_*.sh` | remote runner scripts |
| `scripts/analyze_vllm_cuda_forward_*.py` | earlier envelope/profiler research parsers |
| `scripts/analyze_scheduler_semantics_phase65.py` | gap-audit helper, not runtime descriptor interface |
| `docs/iter_gap_investigation/phase69_3k3k_b128_scheduler_alignment/` | raw marker/log directory |
| `docs/iter_gap_investigation/phase71_32k1k_b16_scheduler_alignment/` | raw marker/log directory |
| `docs/iter_gap_investigation/phase65_scheduler_semantics_gap.csv` | large audit CSV |
| `docs/iter_gap_investigation/phase66_*.csv` | intermediate generated/gap CSV |
| `docs/iter_gap_investigation/phase67_*.csv` | intermediate generated/gap CSV |
| `docs/iter_gap_investigation/phase70_*.csv` | intermediate generated/gap CSV |
| Phase27-39 profiler/NCCL/sync artifacts | diagnostic-only stopped path |
| residual bucket artifacts | stopped path, not physical model evidence |

## Delivery Boundary

This package delivers scheduler descriptor input semantics. It does not deliver a latency model, does not write `PerfDatabase`, and does not modify `run_static` or `IterationLatencyCalculator`.
