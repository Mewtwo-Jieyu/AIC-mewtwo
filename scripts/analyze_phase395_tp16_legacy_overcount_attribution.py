"""Phase395 tp16 legacy over-count attribution (offline, verdict-only).

Phase394 landed merged-batch granularity globally and observed that the legacy
tp16 / vLLM-0.12.0 throughput gate moved 1.50x -> 1.52x while the tp4dp2ep8
module path improved 1.47x -> 1.43x. The Phase394 hypothesis was that the old
split granularity UNDER-counted mixed non-attention and cancelled a separate
masked OVER-count, which merged then exposed.

Phase395 tests that hypothesis with an offline per-component attribution and
LARGELY REFUTES it. It does not modify runtime, write PerfDatabase rows, use
GPU/SSH, add fudge factors, or open Default AIC.

Harness (offline, scripts/diagnose_cb_iter_latency.py --cb-trace-out, tp16 dp1
moe_tp16 moe_ep1, overlap_factor=0 per_iteration_overhead_ms=0,
max_num_batched_tokens=isl, num_requests/warmup matching validate_cb_simulator):
three over-predicting throughput scenarios, each run twice -- merged (HEAD) and
split (parent d8827534 iteration_latency.py) -- aggregating steady-state
per-phase time and per-mixed-iter component means.

Measured findings (steady state):
  * Pure prefill and pure decode paths are byte-identical merged vs split
    (identical pure_decode time; identical mixed ctx_attn / gen_attn).
  * Merged's only per-mixed-iter effect on tp16 is +decode_bs tokens on
    ctx_non_attn (+0.2%..+0.7%); gen_non_attn folds 5-7ms -> 0 but is INVISIBLE
    because with overlap_factor=0 the mixed total = max(...) is dominated by
    ctx_non_attn, and gen_non_attn was never the max.
  * Trace-implied steady throughput ratio merged/split = 0.9993 / 0.9975 /
    0.9995 (-0.05%..-0.25%), far smaller than the validate gate move.
  * The ~1.5x tp16 error is STANDING: split already scores 0.67x (=1.49x
    symmetric) on 10k-3k b=128; merged 0.66x (=1.52x). Merged did not create a
    large over-count; it marginally amplified a pre-existing one.
  * Steady time is 65%..75% pure_decode (unchanged by merged), so the tp16
    error is a broad legacy-path calibration gap, NOT localized to mixed
    non-attention.

Verdict: the Phase394 "masked over-count exposed by merged" hypothesis is
largely refuted. tp16's ~1.5x error predates merged and spans the decode-heavy
steady state; merged is negligible on tp16. The right Phase396 target is the
standing legacy-path calibration error and unifying the two perf backends onto
the measured module-boundary schema, NOT scope-gating merged or patching a
merged artifact.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase395_tp16_legacy_overcount_attribution.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase395_tp16_legacy_overcount_attribution.md"
)

SOURCE = "phase395_tp16_legacy_overcount_attribution"
MODEL = "kimi-k2.5"
HARDWARE = "h200_sxm"
VLLM_VERSION = "0.12.0"
TOPOLOGY = "tp16dp1"
TRUE = "true"
FALSE = "false"
DEFAULT_READINESS = "No-Go"

NEXT_PHASE = "phase396_unify_perf_backends_onto_module_boundary_schema"

# Measured per-scenario aggregates (steady state). See module docstring.
# name: (split_tok_s, merged_tok_s, split_validate_ratio, merged_validate_ratio,
#         mixed_ctx_non_attn_split, mixed_ctx_non_attn_merged, mixed_gen_non_attn_split,
#         pure_decode_time_share_pct)
SCENARIOS = {
    "10k2k_b32": (117.99, 117.91, "0.70x", "0.69x", 300.76, 301.47, 5.19, 71.7),
    "10k3k_b128": (201.05, 200.55, "0.67x", "0.66x", 312.52, 314.77, 6.82, 65.0),
    "16k2k_b32": (80.92, 80.88, "0.76x", "0.75x", 392.88, 393.59, 5.19, 74.7),
}

FIELDNAMES = [
    "source",
    "row_type",
    "scenario",
    "metric",
    "split_value",
    "merged_value",
    "delta_pct",
    "verdict",
    "model",
    "hardware",
    "vllm_version",
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
    "split_value": "",
    "merged_value": "",
    "delta_pct": "",
    "verdict": "",
    "model": MODEL,
    "hardware": HARDWARE,
    "vllm_version": VLLM_VERSION,
    "topology": TOPOLOGY,
    "next_allowed_phase": "",
    "runtime_modified": FALSE,
    "nearest_lookup_allowed": FALSE,
    "interpolation_allowed": FALSE,
    "extrapolation_allowed": FALSE,
    "fudge_factor_tuning_used": FALSE,
    "scope_gating_used": FALSE,
    "write_real_data_file": FALSE,
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


def _pct(split: float, merged: float) -> str:
    if split == 0:
        return "n/a"
    return f"{100.0 * (merged / split - 1.0):+.2f}%"


def analyze_phase395_tp16_legacy_overcount_attribution() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = [
        _row(
            "pure_paths_unchanged",
            verdict=(
                "pure_prefill_and_pure_decode_byte_identical_merged_vs_split; "
                "identical_pure_decode_time_and_mixed_ctx_attn_gen_attn"
            ),
        ),
    ]

    # Per-scenario steady throughput delta (merged vs split).
    for name, vals in SCENARIOS.items():
        s_tok, m_tok = vals[0], vals[1]
        rows.append(
            _row(
                "steady_throughput_delta",
                scenario=name,
                metric="steady_tok_s_gpu",
                split_value=f"{s_tok:.2f}",
                merged_value=f"{m_tok:.2f}",
                delta_pct=_pct(s_tok, m_tok),
                verdict="merged_effect_on_tp16_steady_throughput_is_negligible",
            )
        )

    # Per-scenario mixed component change (the only thing merged touches).
    for name, vals in SCENARIOS.items():
        s_ctx, m_ctx, s_gen = vals[4], vals[5], vals[6]
        rows.append(
            _row(
                "mixed_component_change",
                scenario=name,
                metric="mixed_ctx_non_attn_ms_per_iter",
                split_value=f"{s_ctx:.2f}",
                merged_value=f"{m_ctx:.2f}",
                delta_pct=_pct(s_ctx, m_ctx),
                verdict=(
                    f"only_change_is_plus_decode_bs_tokens; gen_non_attn_folds_"
                    f"{s_gen:.1f}ms_to_0_but_invisible_under_overlap0_max"
                ),
            )
        )

    rows.extend(
        [
            _row(
                "standing_error_precedes_merged",
                scenario="10k3k_b128",
                metric="validate_symmetric_error",
                split_value="1.49x",
                merged_value="1.52x",
                verdict=(
                    "split_already_0.67x_the_1.5x_error_is_standing_not_created_"
                    "by_merged"
                ),
            ),
            _row(
                "decode_dominates_steady",
                metric="pure_decode_time_share_pct",
                split_value="65..75",
                merged_value="65..75",
                verdict=(
                    "steady_time_is_65_to_75pct_pure_decode_unchanged_by_merged;"
                    "_tp16_error_is_broad_not_mixed_localized"
                ),
            ),
            _row(
                "hypothesis_status",
                verdict=(
                    "phase394_masked_overcount_exposed_by_merged_hypothesis_"
                    "largely_refuted; merged_negligible_on_tp16; error_predates_"
                    "merged_and_spans_decode"
                ),
            ),
            _row(
                "verdict",
                verdict=(
                    "tp16_1.5x_is_a_standing_broad_legacy_0120_calibration_gap; "
                    "do_not_scope_gate_merged; target_standing_error_and_unify_"
                    "perf_backends_onto_measured_module_boundary_schema"
                ),
                next_allowed_phase=NEXT_PHASE,
            ),
            _row(
                "next_phase",
                verdict=(
                    "phase396_unify_legacy_0120_interpolated_path_onto_module_"
                    "boundary_schema_and_target_standing_tp16_error"
                ),
                next_allowed_phase=NEXT_PHASE,
            ),
        ]
    )
    _validate_rows(rows)
    return rows


def _validate_rows(rows: list[dict[str, str]]) -> None:
    expected_prefix = ["pure_paths_unchanged"]
    actual = [r["row_type"] for r in rows]
    if actual[:1] != expected_prefix:
        raise ValueError(f"unexpected Phase395 first row: {actual[:1]}")
    if actual[-2:] != ["verdict", "next_phase"]:
        raise ValueError(f"Phase395 must end with verdict,next_phase: {actual[-2:]}")

    required_types = {
        "pure_paths_unchanged",
        "steady_throughput_delta",
        "mixed_component_change",
        "standing_error_precedes_merged",
        "decode_dominates_steady",
        "hypothesis_status",
        "verdict",
        "next_phase",
    }
    if required_types - set(actual):
        raise ValueError(f"Phase395 missing row types: {required_types - set(actual)}")

    for row in rows:
        label = row["row_type"]
        if set(row) != set(FIELDNAMES):
            raise ValueError(f"{label} fields do not match FIELDNAMES")
        if row["source"] != SOURCE:
            raise ValueError(f"{label} bad source")
        # verdict-only phase: observes but does not change anything.
        for guard in (
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
            "valid_for_default",
            "perf_database",
        ):
            if row[guard] != FALSE:
                raise ValueError(f"{label} {guard} must be {FALSE}")
        if row["diagnostic_only"] != TRUE:
            raise ValueError(f"{label} diagnostic_only must be {TRUE}")
        if row["default_readiness"] != DEFAULT_READINESS:
            raise ValueError(f"{label} default_readiness must be {DEFAULT_READINESS}")

    # The whole verdict rests on merged being negligible on tp16 steady tput.
    for row in rows:
        if row["row_type"] == "steady_throughput_delta":
            pct = float(row["delta_pct"].rstrip("%"))
            if abs(pct) > 1.0:
                raise ValueError(
                    f"steady throughput delta {pct}% too large to call negligible"
                )
    by_type = {r["row_type"]: r for r in rows if r["row_type"] in (
        "verdict", "hypothesis_status")}
    if "refuted" not in by_type["hypothesis_status"]["verdict"]:
        raise ValueError("hypothesis_status must record the refutation")
    if by_type["verdict"]["next_allowed_phase"] != NEXT_PHASE:
        raise ValueError("verdict next phase mismatch")


def write_phase395_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_phase395_md(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase395 tp16 Legacy Over-count Attribution (offline, verdict-only)",
        "",
        "| Item | Result |",
        "|---|---|",
        "| Question | is the tp16 1.50->1.52 move a masked over-count exposed "
        "by merged? |",
        "| Answer | **largely NO** -- merged is negligible on tp16; the ~1.5x "
        "error is standing and broad |",
        "| Runtime / PerfDatabase / GPU | not touched (verdict-only) |",
        "| Default AIC | No-Go |",
        "",
        "## Harness",
        "",
        "`scripts/diagnose_cb_iter_latency.py --cb-trace-out`, tp16 dp1 "
        "moe_tp16 moe_ep1, `overlap_factor=0` `per_iteration_overhead_ms=0`, "
        "`max_num_batched_tokens=isl`, num_requests/warmup matching "
        "`validate_cb_simulator`. Three over-predicting throughput scenarios, "
        "each run merged (HEAD) and split (parent `d8827534`), aggregating "
        "steady-state per-phase time and per-mixed-iter component means.",
        "",
        "## Measured findings",
        "",
        "| Scenario | steady tok/s/gpu split->merged | delta | validate "
        "Sim/Out split->merged |",
        "|---|---|---|---|",
    ]
    for name, vals in SCENARIOS.items():
        s_tok, m_tok = vals[0], vals[1]
        lines.append(
            f"| {name} | {s_tok:.2f} -> {m_tok:.2f} | {_pct(s_tok, m_tok)} | "
            f"{vals[2]} -> {vals[3]} |"
        )
    lines += [
        "",
        "- **Pure prefill and pure decode paths are byte-identical** merged vs "
        "split (identical pure_decode time; identical mixed ctx_attn / "
        "gen_attn).",
        "- Merged's ONLY per-mixed-iter effect on tp16 is `+decode_bs` tokens "
        "on ctx_non_attn (+0.2%..+0.7%). gen_non_attn folds 5-7 ms -> 0 but is "
        "INVISIBLE: with `overlap_factor=0` the mixed total is "
        "`max(...)`-dominated by ctx_non_attn, and gen_non_attn was never the "
        "max.",
        "- Trace-implied steady throughput moves only -0.05%..-0.25%, far "
        "smaller than the validate gate move (metric/steady-state "
        "sensitivity).",
        "- The ~1.5x tp16 error is **standing**: split already scores 0.67x "
        "(=1.49x) on 10k-3k b=128; merged 0.66x (=1.52x). Merged did not create "
        "it.",
        "- Steady time is **65%..75% pure_decode** (unchanged by merged), so "
        "the tp16 error is a broad legacy-path calibration gap, not localized "
        "to mixed non-attention.",
        "",
        "## Verdict",
        "",
        "- The Phase394 'masked over-count exposed by merged' hypothesis is "
        "**largely refuted**. tp16's ~1.5x error predates merged and spans the "
        "decode-heavy steady state; merged is negligible on tp16.",
        "- **Do NOT scope-gate merged** (it barely matters on tp16) and do not "
        "chase a merged artifact.",
        f"- Phase396 (`{NEXT_PHASE}`): target the standing legacy-path "
        "calibration error and unify the two perf backends onto the measured "
        "module-boundary schema.",
        "",
        "## No-Go discipline held",
        "",
        "- verdict-only: runtime / operations / PerfDatabase not modified; no "
        "fudge tuning; no scope gating; no GPU/SSH; Default AIC remains No-Go.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    rows = analyze_phase395_tp16_legacy_overcount_attribution()
    write_phase395_csv(args.output_csv, rows)
    write_phase395_md(args.output_md, rows)


if __name__ == "__main__":
    main()
