# Phase397y Mixed-Step Sum Report

Phase397y fixes one accounting bug: mixed iterations now charge the serial sum of merged non-attention, context attention, and decode attention. It does not change DB rows, thresholds, int4 constants, or Default AIC gates.

## Decision

| Item | Result |
|---|---|
| Runtime change | `iteration_latency.py` mixed branch `max()` overlap changed to serial sum |
| Pure decode | unchanged, already serial sum |
| Pure prefill | unchanged in this phase |
| GPU/SSH | not used |
| Default AIC | No-Go |

## Validation

| Metric | Phase397x | Phase397y | Delta | Result |
|---|---:|---:|---:|---|
| MULTI_CONFIG max error | 3.57x | 3.28x | -0.29x | improved but failed |
| MULTI_CONFIG mean error | 2.57x | 2.46x | -0.11x | improved but failed |
| Acceptance threshold | 1.50x | 1.50x | 0.00x | still failed |

## Scenario Results

| Scenario | Real tok/s/GPU | Phase397x sim | Phase397x error | Phase397y sim | Phase397y error | Delta |
|---|---:|---:|---:|---:|---:|---:|
| K2.5-tp8ep8-8k2k | 133.528 | 292.900 | 2.194x | 283.919 | 2.126x | -0.067x |
| K2.5-tp8ep8-32k3k | 52.469 | 113.809 | 2.169x | 108.437 | 2.067x | -0.102x |
| K2.5-tp4ep8dp2-8k2k | 137.716 | 396.569 | 2.880x | 384.887 | 2.795x | -0.085x |
| K2.5-tp4ep8dp2-32k3k | 53.277 | 190.056 | 3.567x | 174.757 | 3.280x | -0.287x |
| K2.5-tp8ep8-8k2k-bt65536 | 138.470 | 289.609 | 2.091x | 286.152 | 2.067x | -0.025x |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 155.952 | 394.418 | 2.529x | 386.549 | 2.479x | -0.050x |

## Remaining Gap

The bug fix moves the right direction, but the active 0.19-real 8-card MULTI_CONFIG gate is still not close to 1.50x. The worst row remains `K2.5-tp4ep8dp2-32k3k` at 3.28x. The next phase should treat the remaining gap as a separate modeling issue, not tune this fix.

## Checks

| Check | Result |
|---|---|
| Red test | `test_mixed_serial_sum_is_independent_of_overlap_factor` failed before the fix with 15.0ms vs 30.0ms |
| Target pytest | `tests/unit/sdk/backends/test_cb_simulator.py` passed |
| validate | `scripts/validate_cb_simulator.py` completed; MULTI_CONFIG still failed |
