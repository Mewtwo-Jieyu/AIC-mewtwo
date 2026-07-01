"""Phase397b tp16 num_heads=4 decode MLA-vs-KV single-GPU remeasure (Route beta).

Phase397a concluded -- by OFFLINE elimination -- that the standing tp16 ~1.5x
over-count is concentrated in the decode(=attention) component, and named the
0.12.0 decode MLA-latency-vs-KV table values (num_heads=4) as the likely locus,
pending a fresh single-GPU measurement. Phase397b performs that measurement and
compares it point-wise to the shipped 0.12.0 table.

This is a confirm-or-refute gate:
  * CONFIRM (Route beta): fresh ~= stored / 1.5 (ratio ~0.66) across the decode
    band -> the 0.12.0 values are ~1.5x too high -> a later phase updates them.
  * REFUTE: fresh ~= stored (or higher) -> the per-KV kernel values are NOT the
    locus; the ~1.5x lives in the accumulation/composition layer instead.

Measurement (single GPU, H200, driver 570.133.20):
  * vLLM 0.19.0 (only version on the node). The MLA decode kernel is
    memory-bandwidth bound at the operating KV (~11k), so version drift is not
    ~1.5x; and, more importantly, the real 89.5 tok/s/gpu benchmark we calibrate
    against was run on a modern vLLM, so current-kernel (0.19.0) is the correct
    reference for the sim, while the sim currently READS a stale 0.12.0 table.
  * Faithful path: the collector's own run_attention_torch setup (config, paged
    KV cache, metadata) + log_perf, with only the decode call adapted to 0.19.0
    (FlashAttnMLAImpl.forward_mqa; the monolithic .forward was removed). Kept
    provenance identical to the stored rows: global num_heads=128, tp_size=32 ->
    local heads 4; float16 KV -> vllm_flash_attn_mla.
  * CAVEAT: forward_mqa measures the decode MQA attention over the paged latent
    cache; it EXCLUDES the small q-absorption GEMM + kv-cache write that the old
    monolithic 0.12.0 forward folded in. So fresh is a LOWER bound on the true
    current-kernel decode cost. This only strengthens a refutation (fresh >=
    stored despite excluding overhead => the table is not too high).

Result: at the gate scenario (10k3k_b128, the worst over-count 1.517x) the fresh
current-kernel value is ~1.4x HIGHER than the stored 0.12.0 value at the decode
operating point -- the OPPOSITE of the Route-beta prediction. At the b32
scenarios fresh is only ~0.81x stored (partly stored noise at high KV), nowhere
near the ~0.66 that "table 1.5x too high" would require. No consistent ~0.66
ratio exists.

Verdict: REFUTE Route beta. The decode MLA-vs-KV per-point table values are NOT
the source of the standing tp16 ~1.5x over-count. Because Phase397a already
localized the over-count to the decode component AND this measurement clears the
per-point table values, the ~1.5x must live in the decode ACCUMULATION /
COMPOSITION layer: (a) the KV-growth trapezoid decode-iteration count / effective
KV trajectory, (b) missing decode overlap (real vLLM overlaps memory-bound MLA
attention under compute-bound MoE/dense GEMMs; the sim serial-sums with
overlap_factor=0), and/or (c) missing MTP / speculative multi-token acceptance
(the real Kimi K2.5 config carries num_nextn_predict_layers=1 while the sim runs
nextn=0 / mtp_scale=1.0, so the real system emits >1 token per decode step). Next
route (delta) investigates the accumulation/composition, NOT the table. Route
gamma (freeze legacy 0.12.0 as diagnostic) remains valid since the table is not
the bug. This file only encodes the measured values + the verdict; it does not
modify runtime, does not write the shipped PerfDatabase table, and does not open
Default AIC.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase397b_tp16_mla_decode_vs_kv_remeasure.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase397b_tp16_mla_decode_vs_kv_remeasure.md"
)
RAW_MEASUREMENT_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase397b_mla_num_heads4_raw_measurement.csv"
)

SOURCE = "phase397b_tp16_mla_decode_vs_kv_remeasure"
MODEL = "kimi-k2.5"
HARDWARE = "h200_sxm"
STORED_VLLM_VERSION = "0.12.0"
FRESH_VLLM_VERSION = "0.19.0"
TOPOLOGY = "tp16dp1"
TRUE = "true"
FALSE = "false"
DEFAULT_READINESS = "No-Go"
NEXT_PHASE = "phase397c_route_delta_decode_accumulation_composition"

LOCAL_HEADS_TP16 = 4  # global num_heads 64 // tp16 (Kimi K2.5)

# Fresh single-GPU measurement, num_heads=4, float16 KV, vllm_flash_attn_mla,
# forward_mqa on vLLM 0.19.0 (H200). Keyed (batch, step) -> latency ms/layer.
# Copied verbatim from phase397b_mla_num_heads4_raw_measurement.csv.
FRESH = {
    (16, 511): 0.03874133278926214,
    (16, 1023): 0.04561600089073181,
    (16, 2047): 0.04423466821511587,
    (16, 4095): 0.048895999789237976,
    (16, 8191): 0.07409066458543141,
    (16, 16383): 0.1272053321202596,
    (16, 32767): 0.23085866371790567,
    (16, 65535): 0.4366186857223511,
    (32, 511): 0.04293866455554962,
    (32, 1023): 0.04273599882920583,
    (32, 2047): 0.04978133241335551,
    (32, 4095): 0.07497600217660268,
    (32, 8191): 0.12732266386349997,
    (32, 16383): 0.23045865694681802,
    (32, 32767): 0.4383466641108195,
    (32, 65535): 0.8515413602193197,
    (64, 511): 0.04602666695912679,
    (64, 1023): 0.048954665660858154,
    (64, 2047): 0.0727040022611618,
    (64, 4095): 0.12545599540074667,
    (64, 8191): 0.22904000679651895,
    (64, 16383): 0.4371039867401123,
    (64, 32767): 0.8496426741282145,
    (64, 65535): 1.6825920740763347,
    (128, 511): 0.052442664901415505,
    (128, 1023): 0.07860266665617625,
    (128, 2047): 0.13074666261672974,
    (128, 4095): 0.23384000857671103,
    (128, 8191): 0.4405440092086792,
    (128, 16383): 0.8552746772766113,
    (128, 32767): 1.681418736775716,
    (128, 65535): 3.3354454040527344,
}

# Shipped 0.12.0 generation_mla_perf.txt, num_heads=4, float16 KV,
# vllm_flash_attn_mla. Keyed (batch, step) -> latency ms/layer.
STORED = {
    (16, 511): 0.16311466693878174,
    (16, 1023): 0.3009066581726074,
    (16, 2047): 0.1304800013701121,
    (16, 4095): 0.17839999993642172,
    (16, 8191): 0.1267146666844686,
    (16, 16383): 0.17585599422454834,
    (16, 32767): 0.2755413254102071,
    (16, 65535): 0.48315731684366864,
    (32, 511): 0.15780267119407654,
    (32, 1023): 0.15428800384203592,
    (32, 2047): 0.1586666703224182,
    (32, 4095): 0.16416000326474509,
    (32, 8191): 0.16859199603398642,
    (32, 16383): 0.2593013246854146,
    (32, 32767): 1.1695306301116943,
    (32, 65535): 0.9234933058420817,
    (64, 511): 0.1551466683546702,
    (64, 1023): 0.1553813318411509,
    (64, 2047): 0.1318666636943817,
    (64, 4095): 0.17378133535385132,
    (64, 8191): 0.25597866376241046,
    (64, 16383): 0.6857706705729166,
    (64, 32767): 0.8988640308380127,
    (64, 65535): 1.7716107368469238,
    (128, 511): 0.15612266461054483,
    (128, 1023): 0.12261866529782613,
    (128, 2047): 0.1634666621685028,
    (128, 4095): 0.18664532899856567,
    (128, 8191): 0.3282080094019572,
    (128, 16383): 0.5960640112559,
    (128, 32767): 1.1498986879984539,
    (128, 65535): 2.2344160079956055,
}

# Decode-band point-wise comparison anchors (grid steps that bracket the sim's
# decode KV trajectory), reported per batch.
POINTWISE = [
    (128, 8191),
    (128, 16383),
    (128, 32767),
    (32, 8191),
    (32, 16383),
]

# Operating points from Phase397a scenarios: (batch_curve, kv, overcount).
# decode_bs 126 -> b128 curve; decode_bs 28 -> b32 curve. KV = decode-band mid.
OPERATING_POINTS = {
    "10k3k_b128": (128, 11500, 1.517),
    "10k2k_b32": (32, 11000, 1.458),
    "16k2k_b32": (32, 17000, 1.341),
}
GATE_SCENARIO = "10k3k_b128"

# If Route beta were correct (table ~1.5x too high) fresh/stored would be ~0.66.
ROUTE_BETA_EXPECTED_RATIO = round(1.0 / 1.5, 3)  # 0.667
# A "confirm" would need the gate scenario near 0.66; observed is the opposite.
CONFIRM_UPPER_BOUND = 0.80

FIELDNAMES = [
    "source",
    "row_type",
    "scenario",
    "metric",
    "fresh_value",
    "stored_value",
    "ratio",
    "verdict",
    "model",
    "hardware",
    "stored_vllm_version",
    "fresh_vllm_version",
    "topology",
    "next_allowed_phase",
    "runtime_modified",
    "nearest_lookup_allowed",
    "interpolation_allowed",
    "extrapolation_allowed",
    "fudge_factor_tuning_used",
    "scope_gating_used",
    "write_real_data_file",
    "gpu_allowed",
    "ssh_allowed",
    "default_aic_allowed",
    "default_readiness",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]

COMMON_FIELDS = {
    "source": SOURCE,
    "scenario": "",
    "metric": "",
    "fresh_value": "",
    "stored_value": "",
    "ratio": "",
    "verdict": "",
    "model": MODEL,
    "hardware": HARDWARE,
    "stored_vllm_version": STORED_VLLM_VERSION,
    "fresh_vllm_version": FRESH_VLLM_VERSION,
    "topology": TOPOLOGY,
    "next_allowed_phase": "",
    "runtime_modified": FALSE,
    "nearest_lookup_allowed": FALSE,
    # We interpolate on the measured curve to reach the operating KV (exactly how
    # the sim reads the table); this is analysis, not a modeling shortcut.
    "interpolation_allowed": TRUE,
    "extrapolation_allowed": FALSE,
    "fudge_factor_tuning_used": FALSE,
    "scope_gating_used": FALSE,
    # Route A (this phase): the shipped generation_mla_perf.txt is NOT written.
    "write_real_data_file": FALSE,
    "gpu_allowed": TRUE,
    "ssh_allowed": TRUE,
    "default_aic_allowed": FALSE,
    "default_readiness": DEFAULT_READINESS,
    "diagnostic_only": TRUE,
    "valid_for_default": FALSE,
    "perf_database": FALSE,
}


def _row(row_type: str, **overrides: str) -> dict[str, str]:
    row = dict(COMMON_FIELDS)
    row["row_type"] = row_type
    row.update(overrides)
    return row


def _interp(table: dict[tuple[int, int], float], batch: int, kv: int) -> float:
    """Linear interpolation on step for a fixed batch curve (as the sim reads)."""
    steps = sorted(s for (b, s) in table if b == batch)
    if not steps:
        raise ValueError(f"no rows for batch {batch}")
    if kv <= steps[0]:
        return table[(batch, steps[0])]
    if kv >= steps[-1]:
        return table[(batch, steps[-1])]
    lo = max(s for s in steps if s <= kv)
    hi = min(s for s in steps if s >= kv)
    if lo == hi:
        return table[(batch, lo)]
    y_lo = table[(batch, lo)]
    y_hi = table[(batch, hi)]
    return y_lo + (y_hi - y_lo) * (kv - lo) / (hi - lo)


def analyze_phase397b() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = [
        _row(
            "measurement_provenance",
            metric="fresh_kernel;num_heads;dtype;kernel_source;caveat",
            fresh_value=(
                f"vllm={FRESH_VLLM_VERSION};forward_mqa;num_heads={LOCAL_HEADS_TP16};"
                "float16;vllm_flash_attn_mla;single_gpu_h200_driver570.133.20"
            ),
            stored_value=(
                f"vllm={STORED_VLLM_VERSION};num_heads={LOCAL_HEADS_TP16};float16;"
                "vllm_flash_attn_mla;shipped_generation_mla_perf.txt"
            ),
            verdict=(
                "provenance_matched_global128_tp32_local4; fresh_forward_mqa_"
                "EXCLUDES_absorption_gemm_and_kv_write -> fresh_is_LOWER_bound; "
                "memory_bound_at_operating_kv_so_version_drift_not_1.5x"
            ),
        ),
        _row(
            "route_beta_expectation",
            metric="confirm_ratio_if_table_1.5x_too_high",
            ratio=f"{ROUTE_BETA_EXPECTED_RATIO}",
            verdict=(
                "confirm_would_need_fresh_over_stored_~0.66_across_decode_band_"
                "especially_gate_scenario"
            ),
        ),
    ]

    for batch, step in POINTWISE:
        fresh = FRESH[(batch, step)]
        stored = STORED[(batch, step)]
        ratio = fresh / stored
        rows.append(
            _row(
                "pointwise_ratio",
                scenario=f"b{batch}_kv{step}",
                metric="fresh_over_stored_num_heads4_decode_band",
                fresh_value=f"{fresh:.5f}",
                stored_value=f"{stored:.5f}",
                ratio=f"{ratio:.3f}",
                verdict=(
                    "fresh_higher" if ratio > 1.0 else "fresh_lower"
                )
                + "_than_stored",
            )
        )

    for name, (batch, kv, overcount) in OPERATING_POINTS.items():
        fresh = _interp(FRESH, batch, kv)
        stored = _interp(STORED, batch, kv)
        ratio = fresh / stored
        rows.append(
            _row(
                "operating_point",
                scenario=name,
                metric=f"fresh_vs_stored_interp_b{batch}_kv{kv}",
                fresh_value=f"{fresh:.5f}",
                stored_value=f"{stored:.5f}",
                ratio=f"{ratio:.3f}",
                verdict=(
                    f"overcount={overcount}x; "
                    + (
                        "fresh_HIGHER_than_stored_opposite_of_route_beta"
                        if ratio > 1.0
                        else "fresh_lower_but_far_from_0.66_table_not_1.5x_high"
                    )
                ),
            )
        )

    gate_batch, gate_kv, _ = OPERATING_POINTS[GATE_SCENARIO]
    gate_ratio = _interp(FRESH, gate_batch, gate_kv) / _interp(
        STORED, gate_batch, gate_kv
    )

    rows.extend(
        [
            _row(
                "ruled_out",
                metric="route_beta_table_1.5x_too_high",
                ratio=f"{gate_ratio:.3f}",
                verdict=(
                    "REFUTED: gate_scenario_fresh_over_stored="
                    f"{gate_ratio:.3f}_(fresh_higher); no_consistent_0.66_ratio; "
                    "b32_scenarios_~0.81_partly_stored_noise; decode_mla_per_kv_"
                    "table_values_are_NOT_the_locus"
                ),
            ),
            _row(
                "verdict",
                metric="standing_overcount_locus",
                verdict=(
                    "decode_mla_vs_kv_table_values_CLEARED; overcount_lives_in_"
                    "decode_accumulation_composition: kv_growth_trapezoid_iter_"
                    "count_or_kv_trajectory; missing_decode_overlap_overlap_"
                    "factor0_serial_sum; missing_mtp_speculative_multitoken_"
                    "nextn0_vs_config_num_nextn_predict_layers1"
                ),
                next_allowed_phase=NEXT_PHASE,
            ),
            _row(
                "next_phase",
                metric="route_delta_target",
                verdict=(
                    "route_delta_phase397c: investigate_decode_accumulation_"
                    "composition_NOT_the_table; enumerate_trapezoid_iter_count_"
                    "overlap_and_mtp; route_gamma_freeze_legacy_0.12.0_still_valid"
                ),
                next_allowed_phase=NEXT_PHASE,
            ),
        ]
    )
    _validate_rows(rows)
    return rows


def _validate_rows(rows: list[dict[str, str]]) -> None:
    actual = [r["row_type"] for r in rows]
    if actual[0] != "measurement_provenance":
        raise ValueError(
            f"Phase397b must start with measurement_provenance: {actual[0]}"
        )
    if actual[-2:] != ["verdict", "next_phase"]:
        raise ValueError(
            f"Phase397b must end with verdict,next_phase: {actual[-2:]}"
        )

    required_types = {
        "measurement_provenance",
        "route_beta_expectation",
        "pointwise_ratio",
        "operating_point",
        "ruled_out",
        "verdict",
        "next_phase",
    }
    if required_types - set(actual):
        raise ValueError(
            f"Phase397b missing row types: {required_types - set(actual)}"
        )

    if sum(1 for r in rows if r["row_type"] == "pointwise_ratio") != len(POINTWISE):
        raise ValueError("pointwise_ratio row count mismatch")
    if sum(1 for r in rows if r["row_type"] == "operating_point") != len(
        OPERATING_POINTS
    ):
        raise ValueError("operating_point row count mismatch")

    for row in rows:
        label = row["row_type"]
        if set(row) != set(FIELDNAMES):
            raise ValueError(f"{label} fields do not match FIELDNAMES")
        if row["source"] != SOURCE:
            raise ValueError(f"{label} bad source")
        for guard in (
            "runtime_modified",
            "nearest_lookup_allowed",
            "extrapolation_allowed",
            "fudge_factor_tuning_used",
            "scope_gating_used",
            "write_real_data_file",
            "default_aic_allowed",
            "valid_for_default",
            "perf_database",
        ):
            if row[guard] != FALSE:
                raise ValueError(f"{label} {guard} must be {FALSE}")
        if row["diagnostic_only"] != TRUE:
            raise ValueError(f"{label} diagnostic_only must be {TRUE}")
        # This is a GPU/SSH measurement phase.
        if row["gpu_allowed"] != TRUE or row["ssh_allowed"] != TRUE:
            raise ValueError(f"{label} gpu_allowed/ssh_allowed must be {TRUE}")
        if row["default_readiness"] != DEFAULT_READINESS:
            raise ValueError(f"{label} default_readiness must be {DEFAULT_READINESS}")

    # Load-bearing: the gate scenario must REFUTE (fresh higher than stored).
    gate = next(
        r
        for r in rows
        if r["row_type"] == "operating_point" and r["scenario"] == GATE_SCENARIO
    )
    gate_ratio = float(gate["ratio"])
    if gate_ratio <= 1.0:
        raise ValueError(
            f"gate scenario ratio {gate_ratio} must be >1.0 (fresh higher) to refute"
        )

    # No operating point may sit near the ~0.66 that "table 1.5x too high" needs.
    for row in rows:
        if row["row_type"] == "operating_point":
            if float(row["ratio"]) < ROUTE_BETA_EXPECTED_RATIO + 0.05:
                raise ValueError(
                    f"{row['scenario']} ratio {row['ratio']} too close to 0.66; "
                    "would not refute Route beta"
                )

    ruled_out = next(r for r in rows if r["row_type"] == "ruled_out")
    if "REFUTED" not in ruled_out["verdict"]:
        raise ValueError("ruled_out must record REFUTED")

    verdict = next(r for r in rows if r["row_type"] == "verdict")
    if "decode_mla_vs_kv_table_values_CLEARED" not in verdict["verdict"]:
        raise ValueError("verdict must clear the decode MLA-vs-KV table values")
    if "accumulation_composition" not in verdict["verdict"]:
        raise ValueError("verdict must point at accumulation/composition")
    if verdict["next_allowed_phase"] != NEXT_PHASE:
        raise ValueError("verdict next phase mismatch")


def write_phase397b_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_phase397b_md(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    gate_batch, gate_kv, gate_oc = OPERATING_POINTS[GATE_SCENARIO]
    gate_ratio = _interp(FRESH, gate_batch, gate_kv) / _interp(
        STORED, gate_batch, gate_kv
    )
    lines = [
        "# Phase397b tp16 num_heads=4 Decode MLA-vs-KV Single-GPU Remeasure "
        "(Route beta)",
        "",
        "| Item | Result |",
        "|---|---|",
        "| Question | are the 0.12.0 decode MLA-vs-KV values (num_heads=4) "
        "~1.5x too high? |",
        "| Answer | **No -- REFUTED.** fresh current-kernel is >= stored at the "
        "worst-over-count point |",
        f"| Gate scenario ({GATE_SCENARIO}) | fresh/stored = "
        f"**{gate_ratio:.3f}** at b{gate_batch}, KV~{gate_kv} (fresh HIGHER) |",
        "| Shipped table / runtime / Default AIC | not modified (Route A: "
        "measure+verdict only) |",
        "",
        "## Measurement",
        "",
        f"- Single GPU (H200, driver 570.133.20), vLLM **{FRESH_VLLM_VERSION}** "
        f"(only version on node), `num_heads={LOCAL_HEADS_TP16}`, float16 KV, "
        "`vllm_flash_attn_mla`.",
        "- Faithful path: collector `run_attention_torch` setup + `log_perf`, "
        "decode call adapted to 0.19.0 `FlashAttnMLAImpl.forward_mqa` (monolithic "
        "`.forward` removed). Provenance identical to stored rows (global "
        "num_heads=128, tp_size=32 -> local 4).",
        "- CAVEAT: `forward_mqa` measures decode MQA attention over the paged "
        "latent cache; it EXCLUDES the small q-absorption GEMM + kv-write that "
        "the old monolithic 0.12.0 `forward` folded in. So fresh is a LOWER bound "
        "on the true current-kernel cost -- which only strengthens a refutation.",
        "- Why 0.19.0 is the right reference: the real 89.5 tok/s/gpu benchmark "
        "was run on a modern vLLM; the sim currently READS a stale 0.12.0 table. "
        "Decode MLA is memory-bandwidth bound at the operating KV, so version "
        "drift is not ~1.5x.",
        "",
        "## Point-wise decode-band comparison (num_heads=4, float16)",
        "",
        "| batch | KV(step) | fresh 0.19.0 | stored 0.12.0 | fresh/stored |",
        "|---|---|---|---|---|",
    ]
    for batch, step in POINTWISE:
        fresh = FRESH[(batch, step)]
        stored = STORED[(batch, step)]
        lines.append(
            f"| {batch} | {step} | {fresh:.5f} | {stored:.5f} | "
            f"{fresh / stored:.3f} |"
        )
    lines += [
        "",
        "## Operating points (interpolated as the sim reads the table)",
        "",
        "| Scenario | over-count | b/KV | fresh | stored | fresh/stored |",
        "|---|---|---|---|---|---|",
    ]
    for name, (batch, kv, overcount) in OPERATING_POINTS.items():
        fresh = _interp(FRESH, batch, kv)
        stored = _interp(STORED, batch, kv)
        lines.append(
            f"| {name} | {overcount}x | b{batch}/KV~{kv} | {fresh:.5f} | "
            f"{stored:.5f} | {fresh / stored:.3f} |"
        )
    lines += [
        "",
        f"- Route beta would need fresh/stored ~= "
        f"{ROUTE_BETA_EXPECTED_RATIO} (stored 1.5x too high) across the decode "
        "band, especially at the gate scenario. Observed: the gate scenario is "
        f"**{gate_ratio:.3f}** (fresh HIGHER, opposite sign), and the b32 "
        "scenarios are ~0.81 (partly stored high-KV noise) -- nowhere near 0.66.",
        "",
        "## Verdict -- REFUTE Route beta",
        "",
        "- The decode MLA-vs-KV per-point table values are **NOT** the source of "
        "the standing tp16 ~1.5x over-count. If anything the shipped 0.12.0 "
        "values are slightly LOW versus the current kernel.",
        "- Phase397a already localized the over-count to the decode component; "
        "this measurement clears the per-point values, so the ~1.5x lives in the "
        "decode **accumulation / composition** layer:",
        "  1. the KV-growth trapezoid decode-iteration count / effective KV "
        "trajectory;",
        "  2. missing decode overlap (real vLLM overlaps memory-bound MLA "
        "attention under compute-bound MoE/dense GEMMs; the sim serial-sums with "
        "`overlap_factor=0`);",
        "  3. missing MTP / speculative multi-token acceptance (real Kimi K2.5 "
        "config carries `num_nextn_predict_layers=1`, but the sim runs `nextn=0` "
        "/ `mtp_scale=1.0`, so the real system emits >1 token per decode step).",
        f"- Next: Phase397c (`{NEXT_PHASE}`) -- Route delta investigates the "
        "accumulation/composition, NOT the table. Route gamma (freeze legacy "
        "0.12.0 as diagnostic) remains valid since the table is not the bug.",
        "",
        "## No-Go discipline held",
        "",
        "- Route A measure+verdict only: shipped `generation_mla_perf.txt` NOT "
        "written; runtime / operations / PerfDatabase not modified; no fudge "
        "tuning; no scope gating; Default AIC remains No-Go.",
        "- Raw single-GPU measurement retained at "
        f"`{RAW_MEASUREMENT_CSV.relative_to(REPO_ROOT)}`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    rows = analyze_phase397b()
    write_phase397b_csv(args.output_csv, rows)
    write_phase397b_md(args.output_md, rows)


if __name__ == "__main__":
    main()
