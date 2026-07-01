"""Phase397c tp16 decode accumulation/composition attribution (Route delta).

Phase397b REFUTED "the 0.12.0 decode MLA-vs-KV per-point table values are ~1.5x
too high" (fresh current-kernel was >= stored), and named the decode ACCUMULATION
/ COMPOSITION layer as the remaining locus. Phase397c decomposes that layer OFFLINE
against the canonical run_agg cb_sim path (the same path validate_cb_simulator /
_run_agg_cb_sim use) for the three tp16 gate scenarios and pinpoints the root cause.

This is verdict-only: it does NOT modify runtime, does NOT write the shipped
PerfDatabase, does NOT open Default AIC, and needs no GPU/SSH.

Findings (all measured with `aic` env against the pr403 worktree; raw evidence in
phase397c_canonical_wall_split_raw.csv):

  step0 -- candidates ruled out up front (tp16 pure decode):
    * per_iteration_overhead_ms = 0 (tp16 uses _CB_SIM_DEFAULT_DECODE_OVERHEAD_MS
      = 0.0; the 90.0 ms overhead is only for 8-GPU EP8). overhead is NOT it.
    * overlap_factor = 0.0 (_CB_SIM_DEFAULT_OVERLAP_FACTOR) and
      _combine_with_overlap(a,b) = max(a,b) + factor*min(a,b) -> pure-decode
      iter_lat = max(gen_non_attn, gen_attn); MoE is NOT serial-summed, it is
      max'd. "missing overlap (serial sum)" is NOT it.
    * MTP: the in-repo Kimi-K2.5 config has NO num_nextn_predict_layers (nextn=0,
      mtp_scale_factor=1.0); the only in-repo real Kimi vLLM trace shows
      speculative_config=None, 1 gen token/step. Phase397b's note that "real Kimi
      config carries num_nextn_predict_layers=1" was WRONG -- it came from the
      collector's DeepSeek-V3 fake config, not the sim's Kimi config. RETRACTED.
      MTP is NOT it.

  step1 -- canonical run() steady wall is 90-95% decode-skip-trapezoid:
    the single pure-decode iters contribute ~30 ms total; essentially all of the
    decode wall is the batch-skip trapezoid. The pure-decode single-iter breakdown
    is gen_attn-dominated (gen_attn >> gen_non_attn), and gen_non_attn (decode MoE,
    already /tp_size scaled) is fully masked by max() at overlap_factor=0.

  step2 -- reconciliation: the canonical trapezoid decode wall (e.g. 754175 ms at
    10k3k_b128) is ~3.2x the expand-skips per-iteration decode sum (236647 ms) for
    the SAME decode iterations. The earlier "~135 ms/iter" figure was just
    steady_wall / steady_iters -- a counting artifact, not a per-iter latency.

  step3 -- ROOT CAUSE: simulator.py _estimate_decode_skip_latency is seeded with
    first_iter_latency_ms = iter_lat, where iter_lat is the latency of the
    iteration that TRIGGERS the skip. That triggering iteration is the one that
    just finished the last prefill chunk, so it is a MIXED/prefill iteration whose
    latency (~231 ms at 10k3k) includes the ~isl-token prefill compute -- NOT a
    pure-decode-at-start-KV latency (~24 ms). The trapezoid then averages 231 ms
    down to the pure-decode end latency (~30 ms) across ALL ~2869 skipped
    pure-decode iters, giving an effective ~79 ms/iter instead of the true
    ~27 ms/iter. Two big skip events carry 99.99% of the decode wall.

    This trapezoid over-count (~3.2x) is PARTIALLY compensated by a per-iteration
    decode UNDER-count: gen_non_attn (decode MoE) is divided by tp_size in
    _scale_generation_non_attention and then hidden under attention by
    overlap_factor=0 max(), so the per-iter decode (~27 ms) is ~3x below the
    real-implied decode iteration (~85 ms). The two opposing errors nearly cancel,
    leaving the standing ~1.5x. Neither can be fixed alone: correcting only the
    trapezoid seed swings the prediction ~2.6x UNDER real; correcting only the
    per-iter composition swings it OVER.

Verdict: the standing tp16 ~1.5x is a RUNTIME accumulation/composition defect
(skip-trapezoid seeding with the triggering mixed-iteration latency) net of a
compensating per-iteration decode under-count -- NOT the decode table (cleared by
397b), NOT decode MoE as a standalone term, NOT MTP, NOT overhead. Next route
(phase397d) is a GATED runtime fix that must correct BOTH the trapezoid seed AND
the per-iteration decode composition together, then re-validate offline.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase397c_tp16_decode_accumulation_composition.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase397c_tp16_decode_accumulation_composition.md"
)
RAW_EVIDENCE_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase397c_canonical_wall_split_raw.csv"
)

SOURCE = "phase397c_tp16_decode_accumulation_composition"
MODEL = "kimi-k2.5"
HARDWARE = "h200_sxm"
VLLM_DB_VERSION = "0.12.0"
TOPOLOGY = "tp16dp1"
TRUE = "true"
FALSE = "false"
DEFAULT_READINESS = "No-Go"
NEXT_PHASE = "phase397d_decode_skip_trapezoid_seed_and_periter_composition_fix"
GATE_SCENARIO = "10k3k_b128"


@dataclass(frozen=True)
class Scenario:
    name: str
    real_tok_s_gpu: float
    sim_tok_s_gpu: float
    steady_wall_ms: float
    decode_wall_ms: float
    decode_wall_pct: float
    mixed_wall_pct: float
    expand_periter_decode_wall_ms: float
    real_implied_wall_ms: float
    puredecode_gen_attn_ms: float
    puredecode_gen_non_attn_ms: float
    skip_first_lat_mixed_ms: float
    skip_end_lat_puredecode_ms: float
    skip_eff_per_iter_ms: float
    puredecode_single_iter_ms: float


# Measured with the `aic` env against the pr403 worktree (canonical run() split +
# skip-event instrumentation + expand-skips per-iter sum). See raw evidence CSV.
SCENARIOS: dict[str, Scenario] = {
    "10k3k_b128": Scenario(
        name="10k3k_b128",
        real_tok_s_gpu=89.5,
        sim_tok_s_gpu=59.01,
        steady_wall_ms=796040.0,
        decode_wall_ms=754175.0,
        decode_wall_pct=94.74,
        mixed_wall_pct=5.26,
        expand_periter_decode_wall_ms=236647.0,
        real_implied_wall_ms=524812.0,
        puredecode_gen_attn_ms=27.12,
        puredecode_gen_non_attn_ms=6.885,
        skip_first_lat_mixed_ms=231.54,
        skip_end_lat_puredecode_ms=30.23,
        skip_eff_per_iter_ms=79.20,
        puredecode_single_iter_ms=27.12,
    ),
    "10k2k_b32": Scenario(
        name="10k2k_b32",
        real_tok_s_gpu=49.8,
        sim_tok_s_gpu=34.16,
        steady_wall_ms=496487.0,
        decode_wall_ms=458200.0,
        decode_wall_pct=92.29,
        mixed_wall_pct=7.71,
        expand_periter_decode_wall_ms=163721.0,
        real_implied_wall_ms=340553.0,
        puredecode_gen_attn_ms=12.35,
        puredecode_gen_non_attn_ms=5.282,
        skip_first_lat_mixed_ms=99.00,
        skip_end_lat_puredecode_ms=13.05,
        skip_eff_per_iter_ms=40.31,
        puredecode_single_iter_ms=12.35,
    ),
    "16k2k_b32": Scenario(
        name="16k2k_b32",
        real_tok_s_gpu=41.7,
        sim_tok_s_gpu=31.09,
        steady_wall_ms=545446.0,
        decode_wall_ms=495461.0,
        decode_wall_pct=90.84,
        mixed_wall_pct=9.16,
        expand_periter_decode_wall_ms=237963.0,
        real_implied_wall_ms=406704.0,
        puredecode_gen_attn_ms=18.75,
        puredecode_gen_non_attn_ms=5.282,
        skip_first_lat_mixed_ms=99.00,
        skip_end_lat_puredecode_ms=22.76,
        skip_eff_per_iter_ms=43.45,
        puredecode_single_iter_ms=18.75,
    ),
}

FIELDNAMES = [
    "source",
    "row_type",
    "scenario",
    "metric",
    "value_a",
    "value_b",
    "ratio",
    "verdict",
    "model",
    "hardware",
    "vllm_db_version",
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
    "value_a": "",
    "value_b": "",
    "ratio": "",
    "verdict": "",
    "model": MODEL,
    "hardware": HARDWARE,
    "vllm_db_version": VLLM_DB_VERSION,
    "topology": TOPOLOGY,
    "next_allowed_phase": "",
    "runtime_modified": FALSE,
    "nearest_lookup_allowed": FALSE,
    # We integrate the measured per-KV table exactly as the sim reads it; analysis
    # interpolation is faithful, not a modeling shortcut.
    "interpolation_allowed": TRUE,
    "extrapolation_allowed": FALSE,
    "fudge_factor_tuning_used": FALSE,
    "scope_gating_used": FALSE,
    "write_real_data_file": FALSE,
    # Pure offline decomposition: no GPU, no SSH.
    "gpu_allowed": FALSE,
    "ssh_allowed": FALSE,
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


def analyze_phase397c() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    # --- step0: retraction + up-front ruled-out mechanisms ---
    rows.append(
        _row(
            "correction_retraction",
            metric="phase397b_mtp_claim",
            value_a="claimed:real_config_num_nextn_predict_layers=1",
            value_b="actual:kimi_config_has_no_nextn;nextn=0;mtp_scale=1.0;"
            "only_real_trace_speculative_config=None",
            verdict=(
                "RETRACTED: the num_nextn_predict_layers=1 came from the collector "
                "DeepSeek-V3 fake config, NOT the sim Kimi config; MTP is not a "
                "current sim-side inflation source"
            ),
        )
    )
    rows.append(
        _row(
            "mechanism_ruled_out",
            metric="tp16_pure_decode_composition",
            value_a="per_iteration_overhead_ms=0 (default; 90ms only for 8-GPU EP8)",
            value_b="overlap_factor=0 -> iter_lat=max(gen_non_attn,gen_attn) "
            "(NOT serial sum)",
            verdict=(
                "overhead_ruled_out; overlap_serial_sum_ruled_out; "
                "mtp_ruled_out_no_repo_evidence"
            ),
        )
    )

    # --- step1: canonical wall split (decode is trapezoid-dominated) ---
    for name, sc in SCENARIOS.items():
        rows.append(
            _row(
                "canonical_wall_split",
                scenario=name,
                metric="decode_wall_pct_vs_mixed_wall_pct_of_steady_wall",
                value_a=f"decode={sc.decode_wall_ms:.0f}ms({sc.decode_wall_pct:.2f}%)",
                value_b=f"mixed={sc.mixed_wall_pct:.2f}%",
                ratio=f"{sc.decode_wall_pct / 100.0:.4f}",
                verdict=(
                    "canonical steady wall is decode-skip-trapezoid dominated "
                    "(single pure-decode iters ~negligible)"
                ),
            )
        )

    # --- step1: decode iter is gen_attn-bound, gen_non_attn masked ---
    for name, sc in SCENARIOS.items():
        rows.append(
            _row(
                "decode_iter_composition",
                scenario=name,
                metric="puredecode_single_iter_gen_attn_vs_gen_non_attn",
                value_a=f"gen_attn={sc.puredecode_gen_attn_ms:.3f}ms",
                value_b=f"gen_non_attn={sc.puredecode_gen_non_attn_ms:.3f}ms(/tp_size)",
                ratio=f"{sc.puredecode_gen_attn_ms / sc.puredecode_gen_non_attn_ms:.3f}",
                verdict=(
                    "iter_lat=max()=gen_attn; gen_non_attn(decode MoE) masked and "
                    "already /tp_size scaled -> per-iter decode is under-counted"
                ),
            )
        )

    # --- step2: trapezoid vs expand-skips per-iter sum reconciliation ---
    for name, sc in SCENARIOS.items():
        ratio = sc.decode_wall_ms / sc.expand_periter_decode_wall_ms
        rows.append(
            _row(
                "trapezoid_vs_periter_reconciliation",
                scenario=name,
                metric="canonical_trapezoid_decode_wall_over_expand_periter_sum",
                value_a=f"trapezoid={sc.decode_wall_ms:.0f}ms",
                value_b=f"expand_periter_sum={sc.expand_periter_decode_wall_ms:.0f}ms",
                ratio=f"{ratio:.3f}",
                verdict=(
                    "trapezoid over-counts the SAME decode iters vs the true "
                    "per-iteration sum; the ~135ms/iter figure was steady_wall/"
                    "steady_iters (a counting artifact)"
                ),
            )
        )

    # --- step3: root cause -- trapezoid seeded with triggering MIXED iter latency ---
    for name, sc in SCENARIOS.items():
        rows.append(
            _row(
                "root_cause_skip_trapezoid_seed",
                scenario=name,
                metric="skip_first_iter_latency_ms_is_mixed_not_puredecode",
                value_a=f"first_lat(mixed)={sc.skip_first_lat_mixed_ms:.2f}ms",
                value_b=f"end_lat(puredecode)={sc.skip_end_lat_puredecode_ms:.2f}ms; "
                f"eff_per_iter={sc.skip_eff_per_iter_ms:.2f}ms vs "
                f"true_puredecode={sc.puredecode_single_iter_ms:.2f}ms",
                ratio=f"{sc.skip_first_lat_mixed_ms / sc.puredecode_single_iter_ms:.3f}",
                verdict=(
                    "_estimate_decode_skip_latency seeds first_iter_latency_ms with "
                    "the triggering MIXED/prefill iter latency (~isl-token prefill), "
                    "not a pure-decode-at-start-KV latency -> ~3x decode wall inflation"
                ),
            )
        )
    rows.append(
        _row(
            "compensating_error",
            metric="periter_decode_under_count_vs_trapezoid_over_count",
            value_a="per_iter_decode~27ms (gen_non_attn /tp_size + masked by max)",
            value_b="real_implied_decode_iter~85ms; trapezoid inflates to ~79-131ms",
            verdict=(
                "the trapezoid over-count (~3.2x) is partially compensated by the "
                "per-iteration decode under-count (~3x); the two opposing errors "
                "nearly cancel and leave the standing ~1.5x -- neither is fixable "
                "alone (trapezoid-only fix swings ~2.6x UNDER real)"
            ),
        )
    )

    # --- step3: attribution -- decode_wall vs real-implied total wall ---
    for name, sc in SCENARIOS.items():
        over = sc.real_tok_s_gpu / sc.sim_tok_s_gpu
        dec_over_real = sc.decode_wall_ms / sc.real_implied_wall_ms
        rows.append(
            _row(
                "attribution",
                scenario=name,
                metric="over_count_vs_decode_wall_over_real_implied",
                value_a=f"over_count={over:.4f}",
                value_b=f"decode_wall/real_implied_wall={dec_over_real:.4f}",
                ratio=f"{dec_over_real / over:.4f}",
                verdict=(
                    "the decode-skip-trapezoid wall alone already exceeds the real "
                    "TOTAL wall -> the standing over-count lives in the decode "
                    "accumulation, not prefill/mixed"
                ),
            )
        )

    # --- ruled_out summary ---
    rows.append(
        _row(
            "ruled_out",
            metric="non_root_causes",
            verdict=(
                "decode_mla_vs_kv_table_values(cleared_by_397b); decode_moe_as_"
                "standalone_term(masked_by_overlap0_max); decode_iter_count(osl_"
                "correct); mtp(no_repo_evidence); per_iteration_overhead(0_for_tp16); "
                "overlap_serial_sum(overlap_factor=0_is_max)"
            ),
        )
    )

    # --- verdict + next phase ---
    rows.append(
        _row(
            "verdict",
            metric="standing_overcount_root_cause",
            verdict=(
                "RUNTIME_accumulation_composition_defect: skip_trapezoid seeded with "
                "triggering_mixed_iteration_latency (simulator.py "
                "_estimate_decode_skip_latency first_iter_latency_ms=iter_lat) net of "
                "a compensating per_iteration_decode_under_count (gen_non_attn /tp_size "
                "+ overlap_factor0 max); NOT the table, NOT MoE-standalone, NOT MTP"
            ),
            next_allowed_phase=NEXT_PHASE,
        )
    )
    rows.append(
        _row(
            "next_phase",
            metric="route_delta_followon_target",
            verdict=(
                "phase397d GATED runtime fix: correct BOTH the skip-trapezoid seed "
                "(use pure-decode-at-start-KV latency) AND the per-iteration decode "
                "composition (gen_non_attn scaling + overlap treatment) TOGETHER, "
                "then re-validate offline; no table write, no GPU required for the fix"
            ),
            next_allowed_phase=NEXT_PHASE,
        )
    )

    _validate_rows(rows)
    return rows


def _validate_rows(rows: list[dict[str, str]]) -> None:
    actual = [r["row_type"] for r in rows]
    if actual[0] != "correction_retraction":
        raise ValueError(
            f"Phase397c must start with correction_retraction: {actual[0]}"
        )
    if actual[-2:] != ["verdict", "next_phase"]:
        raise ValueError(f"Phase397c must end with verdict,next_phase: {actual[-2:]}")

    required_types = {
        "correction_retraction",
        "mechanism_ruled_out",
        "canonical_wall_split",
        "decode_iter_composition",
        "trapezoid_vs_periter_reconciliation",
        "root_cause_skip_trapezoid_seed",
        "compensating_error",
        "attribution",
        "ruled_out",
        "verdict",
        "next_phase",
    }
    if required_types - set(actual):
        raise ValueError(f"Phase397c missing row types: {required_types - set(actual)}")

    n = len(SCENARIOS)
    for per_scenario_type in (
        "canonical_wall_split",
        "decode_iter_composition",
        "trapezoid_vs_periter_reconciliation",
        "root_cause_skip_trapezoid_seed",
        "attribution",
    ):
        if sum(1 for r in rows if r["row_type"] == per_scenario_type) != n:
            raise ValueError(f"{per_scenario_type} must have {n} rows")

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
            "gpu_allowed",
            "ssh_allowed",
            "default_aic_allowed",
            "valid_for_default",
            "perf_database",
        ):
            if row[guard] != FALSE:
                raise ValueError(f"{label} {guard} must be {FALSE}")
        if row["diagnostic_only"] != TRUE:
            raise ValueError(f"{label} diagnostic_only must be {TRUE}")
        if row["default_readiness"] != DEFAULT_READINESS:
            raise ValueError(f"{label} default_readiness must be {DEFAULT_READINESS}")

    # Load-bearing: canonical decode wall is trapezoid-dominated (>90%).
    for row in rows:
        if row["row_type"] == "canonical_wall_split":
            if float(row["ratio"]) < 0.90:
                raise ValueError(
                    f"{row['scenario']} decode_wall_pct {row['ratio']} must be >=0.90"
                )

    # Load-bearing: the trapezoid over-counts the decode iters vs the per-iter sum.
    for row in rows:
        if row["row_type"] == "trapezoid_vs_periter_reconciliation":
            if float(row["ratio"]) <= 2.0:
                raise ValueError(
                    f"{row['scenario']} trapezoid/periter {row['ratio']} must be >2.0"
                )

    # Load-bearing: the skip trapezoid is seeded with a MIXED (>> pure-decode) latency.
    for row in rows:
        if row["row_type"] == "root_cause_skip_trapezoid_seed":
            if float(row["ratio"]) <= 3.0:
                raise ValueError(
                    f"{row['scenario']} first_lat/puredecode {row['ratio']} must be >3.0"
                )

    # Load-bearing: gate scenario decode wall alone exceeds the real total wall.
    gate = next(
        r
        for r in rows
        if r["row_type"] == "attribution" and r["scenario"] == GATE_SCENARIO
    )
    if float(gate["value_b"].split("=")[1]) <= 1.0:
        raise ValueError("gate decode_wall/real_implied must exceed 1.0")

    retraction = next(r for r in rows if r["row_type"] == "correction_retraction")
    if "RETRACTED" not in retraction["verdict"]:
        raise ValueError("correction_retraction must record RETRACTED")

    verdict = next(r for r in rows if r["row_type"] == "verdict")
    if "skip_trapezoid" not in verdict["verdict"]:
        raise ValueError("verdict must name the skip_trapezoid seed defect")
    if "compensating" not in verdict["verdict"]:
        raise ValueError("verdict must record the compensating per-iter under-count")
    if verdict["next_allowed_phase"] != NEXT_PHASE:
        raise ValueError("verdict next phase mismatch")


def write_phase397c_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_phase397c_md(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    gate = SCENARIOS[GATE_SCENARIO]
    gate_over = gate.real_tok_s_gpu / gate.sim_tok_s_gpu
    lines = [
        "# Phase397c tp16 Decode Accumulation/Composition Attribution (Route delta)",
        "",
        "| Item | Result |",
        "|---|---|",
        "| Question | after 397b cleared the decode MLA-vs-KV table, WHERE does the "
        "standing tp16 ~1.5x live in the decode accumulation/composition? |",
        "| Answer | the **skip-trapezoid** in `simulator.py "
        "_estimate_decode_skip_latency`, seeded with the TRIGGERING mixed/prefill "
        "iteration latency, net of a compensating per-iteration decode under-count |",
        f"| Gate ({GATE_SCENARIO}) | over_count={gate_over:.3f}x; canonical steady "
        f"wall is {gate.decode_wall_pct:.1f}% decode-skip-trapezoid; decode wall "
        f"alone ({gate.decode_wall_ms:.0f} ms) > real-implied TOTAL wall "
        f"({gate.real_implied_wall_ms:.0f} ms) |",
        "| Runtime / table / Default AIC | not modified (verdict-only, offline, "
        "no GPU/SSH) |",
        "",
        "## Ruled out up front (tp16 pure decode)",
        "",
        "- `per_iteration_overhead_ms = 0` for tp16 (`_CB_SIM_DEFAULT_DECODE_"
        "OVERHEAD_MS = 0.0`; the `90.0` ms overhead applies only to 8-GPU EP8).",
        "- `overlap_factor = 0.0` and `_combine_with_overlap(a,b) = max(a,b) + "
        "factor*min(a,b)` -> pure-decode `iter_lat = max(gen_non_attn, gen_attn)`; "
        "decode MoE is max'd, NOT serial-summed. \"Missing overlap serial sum\" is "
        "not the bug.",
        "- MTP: the in-repo Kimi-K2.5 config has **no** `num_nextn_predict_layers` "
        "(`nextn=0`, `mtp_scale_factor=1.0`); the only in-repo real Kimi vLLM trace "
        "shows `speculative_config=None`, 1 token/step. Phase397b's "
        "`num_nextn_predict_layers=1` note (from the collector's DeepSeek-V3 fake "
        "config) is **RETRACTED**.",
        "",
        "## Canonical run() steady-wall decomposition",
        "",
        "| Scenario | over_count | steady_wall | decode(trapezoid) | mixed | "
        "trapezoid/expand-periter | decode_wall/real_implied |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, sc in SCENARIOS.items():
        over = sc.real_tok_s_gpu / sc.sim_tok_s_gpu
        trap_over_expand = sc.decode_wall_ms / sc.expand_periter_decode_wall_ms
        dec_over_real = sc.decode_wall_ms / sc.real_implied_wall_ms
        lines.append(
            f"| {name} | {over:.3f}x | {sc.steady_wall_ms:.0f} ms | "
            f"{sc.decode_wall_ms:.0f} ms ({sc.decode_wall_pct:.1f}%) | "
            f"{sc.mixed_wall_pct:.1f}% | {trap_over_expand:.2f}x | "
            f"{dec_over_real:.3f}x |"
        )
    lines += [
        "",
        "## Root cause -- skip trapezoid seeded with the triggering mixed iteration",
        "",
        "`simulator.py::_estimate_decode_skip_latency` computes the skipped "
        "pure-decode wall as `(first_iter_latency_ms + end_iter_latency_ms) * "
        "skip_iters / 2`. The caller passes `first_iter_latency_ms = iter_lat`, but "
        "`iter_lat` is the latency of the iteration that TRIGGERS the skip -- the one "
        "that just finished the last prefill chunk. That is a MIXED/prefill iteration "
        "carrying the ~isl-token prefill compute, not a pure-decode-at-start-KV "
        "latency.",
        "",
        "| Scenario | skip first_lat (mixed) | skip end_lat (pure decode) | "
        "trapezoid eff/iter | true pure-decode/iter |",
        "|---|---|---|---|---|",
    ]
    for name, sc in SCENARIOS.items():
        lines.append(
            f"| {name} | {sc.skip_first_lat_mixed_ms:.1f} ms | "
            f"{sc.skip_end_lat_puredecode_ms:.1f} ms | "
            f"{sc.skip_eff_per_iter_ms:.1f} ms | "
            f"{sc.puredecode_single_iter_ms:.1f} ms |"
        )
    lines += [
        "",
        "At 10k3k_b128 two skip events (bs=128, skip~2869, first_lat~232 ms) carry "
        "99.99% of the decode wall; averaging 232 ms down to 30 ms across ~2869 "
        "*pure-decode* iters yields an effective ~79 ms/iter versus the true "
        "~27 ms/iter -> ~3.2x decode-wall inflation.",
        "",
        "## Compensating per-iteration decode under-count",
        "",
        "- The trapezoid over-count (~3.2x) is partially compensated by a "
        "per-iteration decode UNDER-count: `gen_non_attn` (decode MoE) is divided by "
        "`tp_size` in `_scale_generation_non_attention` and then hidden under "
        "attention by `overlap_factor=0` `max()`, so the per-iter decode (~27 ms) is "
        "~3x below the real-implied decode iteration (~85 ms).",
        "- The two opposing errors nearly cancel and leave the standing ~1.5x. "
        "**Neither is fixable alone**: correcting only the trapezoid seed swings the "
        "prediction ~2.6x UNDER real; correcting only the per-iter composition swings "
        "it OVER.",
        "",
        "## Verdict -- Route delta",
        "",
        "- The standing tp16 ~1.5x is a **runtime accumulation/composition defect** "
        "(skip-trapezoid seeded with the triggering mixed-iteration latency) net of a "
        "compensating per-iteration decode under-count. It is **NOT** the decode "
        "table (cleared by 397b), **NOT** decode MoE as a standalone term (masked by "
        "`overlap_factor=0` max), **NOT** MTP, **NOT** per-iteration overhead.",
        f"- Next: **{NEXT_PHASE}** -- a GATED runtime fix that corrects BOTH the "
        "trapezoid seed AND the per-iteration decode composition TOGETHER, then "
        "re-validates offline. No table write and no GPU required for the fix.",
        "",
        "## No-Go discipline held",
        "",
        "- Verdict-only, offline: runtime / operations / PerfDatabase not modified; "
        "no fudge tuning; no scope gating; no GPU/SSH; Default AIC remains No-Go.",
        f"- Raw canonical wall-split + skip-event evidence at "
        f"`{RAW_EVIDENCE_CSV.relative_to(REPO_ROOT)}`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    rows = analyze_phase397c()
    write_phase397c_csv(args.output_csv, rows)
    write_phase397c_md(args.output_md, rows)


if __name__ == "__main__":
    main()
