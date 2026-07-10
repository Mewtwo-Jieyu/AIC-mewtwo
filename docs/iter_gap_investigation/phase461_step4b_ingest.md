# Phase461 Step 4b: measured-cost ingestion

## Verdict

The staged ingestion passed. The measured MLA replacement and the qualified TP8
32k mixed rows move only `K2.5-tp8ep8-32k3k`; the other five validation points
are bit-for-bit unchanged. The scorecard is now 3/6 within 15%, so Default AIC
remains No-Go and the gate is not tightened.

## Staged A/B

| Scenario | Baseline | MLA only | MLA + TP8 32k mixed | Final change |
|---|---:|---:|---:|---:|
| K2.5-tp8ep8-8k2k | 1.1325x | 1.1325x | 1.1325x | 0.0000 |
| K2.5-tp4ep8dp2-8k2k | 1.0413x | 1.0413x | 1.0413x | 0.0000 |
| K2.5-tp8ep8-8k2k-bt65536 | 1.2345x | 1.2345x | 1.2345x | 0.0000 |
| K2.5-tp4ep8dp2-8k2k-bt65536 | 1.1845x | 1.1845x | 1.1845x | 0.0000 |
| K2.5-tp4ep8dp2-32k3k | 1.1733x | 1.1733x | 1.1733x | 0.0000 |
| K2.5-tp8ep8-32k3k | 1.3971x | 1.1385x | 1.1036x | -0.2935 |

The MLA-only result formally validates the Phase460 counterfactual: replacing
the single measured bad row is sufficient to remove the decode overcharge. The
mixed rows then close the exposed mixed-step undercharge without cross-regime
movement.

## PerfDB changes

| Change | Scope | Measurement source | Guard |
|---|---|---|---|
| Replace MLA decode row | heads=8, batch=8, KV=32768 | phase461 GPU microbenchmark, 0.134677330653 ms/layer | exact-key data-integrity test |
| Add 54 `forward_total` rows | tp8ep8, mixed_prefill, max_bt=32000 | phase461 B2b event timing | max_bt/topology exact scope |
| Exclude cell | bucket=92, decode=13 | 69.18% anchor drift | row absent by test |
| Exclude regime | tp8ep8, max_bt=65536 | only 57.9% weighted coverage; missing cells are sim-only for the Phase458 reference | no phase461 65k rows by test |

The excluded `92/13` measurement is not stored. Current 2D interpolation has no
cell-level mask, so the 18 queries marked as influenced during the anchor audit
remain outside the 99.27% qualified-coverage claim; no clamp, extrapolation, or
ad-hoc replacement was added.

## Remaining work

Phase462 owns the dynamics family: TP8 bt65536 sim-only giant-bucket steps,
DP bt65536 cross-rank phase/lockstep composition, DP 32k3k residual attribution,
and the preemption excess. The primary joint test is whether recompute-associated
giant-bucket query weight falls together with the preemption count.

Artifacts: `phase461_step4b_baseline.csv`, `phase461_step4b_mla_ab.csv`,
`phase461_step4b_mixed_ab.csv`, and `phase461_step4b_final_ab.csv`.
