"""Phase396 tp16 standing over-count attribution (offline, verdict-only).

Route alpha follow-up to Phase395. Phase395 showed the tp16 / vLLM-0.12.0 legacy
throughput error (~1.5x) is STANDING (present in split too) and that merged
granularity is negligible on tp16, but did not locate the over-count in absolute
ms. Phase396 does, offline, and reaches a clear verdict.

It does not modify runtime, write PerfDatabase rows, use GPU/SSH, add fudge
factors, or open Default AIC.

Step 0 -- metric reconciliation (was a blocking precondition):
  The Phase395 diagnose `--cb-trace-out` harness reported ~201 tok/s/gpu steady
  for 10k-3k b=128, but validate_cb_simulator / run_agg reports 59.01. Root
  cause found and closed: the canonical basis is
  `tokens/s/gpu = steady_output_tokens / (steady_time_ms/1000) / num_gpus`
  (num_gpus = tp*pp*dp = 16), and run_agg's real steady run is
  steady_iters=5874, steady_time_ms=796039.6ms, steady_output_tokens~=751531 ->
  59.01. The Phase395 trace under-counted the decode wall ~6.6x because its
  `--expand-skips` path expands pure-decode skips at low/near-constant latency,
  whereas the real sim charges the KV-growth trapezoid
  (`_estimate_decode_skip_latency`). Confirmed: diagnose `--no-expand-skips`
  reproduces a same-order steady decode wall (4e5..1.3e6 ms). Phase395's
  RELATIVE merged-vs-split conclusions still hold; only its ABSOLUTE trace
  throughput was on the wrong basis, corrected here.

Measured attribution (canonical run_agg basis, 0.12.0, tp16 dp1):
  * Steady state is ~pure decode: avg_prefill_reqs_per_iter = 0.08. So the
    standing over-count lives in the DECODE path, not prefill. (No real TTFT
    split needed -- steady throughput gap == decode wall gap.)
  * Each decode iteration is 100% attention-bound: gen_attn (24-30 ms @ kv
    10k-13k for 10k-3k b=128) >> gen_non_attn (6.82 ms). With overlap_factor=0
    the pure-decode total = max(...) = gen_attn; MoE/GEMM (gen_non_attn) is
    masked. Same shape for 10k-2k b=32 and 16k-2k b=32 (attn_share 100%).
  * Over-count magnitude: sim steady wall 796039 ms vs real-implied steady wall
    524810 ms (751531 tok / (89.5 tok/s/gpu * 16)) = 1.517x -- exactly the
    throughput gate.

Verdict: the standing tp16 ~1.5x over-count is DECODE-side and ATTENTION-bound
-- it is the 0.12.0 decode attention latency vs KV (gen_attn), accumulated over
~3000 decode steps via the KV-growth trapezoid. It is NOT in MoE/GEMM, NOT in
prefill, and NOT created by merged. This is a narrow, GPU-measurable module
boundary -> Route beta with a specific target: a measured decode-attention-vs-KV
table for Kimi tp16 on the module-boundary (0.19.0) schema, unifying the two
perf backends (0.12.0 over-predicts, 0.19.0 measured under-predicts: pure data
magnitude, not a shared composition bug). Route gamma (freeze legacy 0.12.0 as
diagnostic) is the fallback if that table cannot be measured.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT / "docs/iter_gap_investigation/phase396_tp16_decode_attribution.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT / "docs/iter_gap_investigation/phase396_tp16_decode_attribution.md"
)

SOURCE = "phase396_tp16_decode_attribution"
MODEL = "kimi-k2.5"
HARDWARE = "h200_sxm"
VLLM_VERSION = "0.12.0"
TOPOLOGY = "tp16dp1"
TRUE = "true"
FALSE = "false"
DEFAULT_READINESS = "No-Go"
NEXT_PHASE = "phase397_route_beta_measured_decode_attention_vs_kv_table"

# Canonical run_agg steady basis for 10k-3k b=128 (validate_cb_simulator).
RUNAGG_STEADY_ITERS = 5874
RUNAGG_STEADY_WALL_MS = 796039.6
RUNAGG_STEADY_OUTPUT_TOKENS = 751531
RUNAGG_TOK_S_GPU = 59.01
NUM_GPUS = 16
AVG_PREFILL_REQS_PER_ITER = 0.08
DIAGNOSE_EXPAND_SKIPS_STEADY_WALL_MS = 119711.0  # phase395 basis (under-counts)

# Real anchor + over-count.
REAL_OUT_TOK_S_GPU_10K3K = 89.5
REAL_IMPLIED_STEADY_WALL_MS = 524810.0  # 751531 / (89.5*16) * 1000
OVERCOUNT_RATIO = round(RUNAGG_STEADY_WALL_MS / REAL_IMPLIED_STEADY_WALL_MS, 3)  # 1.517

# Per-scenario decode-iter component split (measured via get_last_breakdown,
# mid-KV point). name: (decode_bs, gen_attn_ms, gen_non_attn_ms).
DECODE_COMPONENTS = {
    "10k3k_b128": (126, 27.17, 6.82),
    "10k2k_b32": (28, 11.48, 5.15),
    "16k2k_b32": (28, 17.25, 5.15),
}

FIELDNAMES = [
    "source",
    "row_type",
    "scenario",
    "metric",
    "sim_value",
    "real_value",
    "ratio_or_share",
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
    "sim_value": "",
    "real_value": "",
    "ratio_or_share": "",
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


def analyze_phase396_tp16_decode_attribution() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = [
        _row(
            "metric_reconciliation",
            scenario="10k3k_b128",
            metric="steady_tok_s_gpu_canonical_run_agg",
            sim_value=f"{RUNAGG_TOK_S_GPU:.2f}",
            ratio_or_share=(
                f"steady_iters={RUNAGG_STEADY_ITERS};steady_wall_ms="
                f"{RUNAGG_STEADY_WALL_MS:.0f};num_gpus={NUM_GPUS}"
            ),
            verdict=(
                "canonical_basis=steady_output_tokens/steady_wall/num_gpus; "
                "phase395_diagnose_expand_skips_undercounts_decode_wall_6.6x "
                "(low_latency_skip_expansion_vs_kv_growth_trapezoid)"
            ),
        ),
        _row(
            "phase395_correction",
            metric="phase395_absolute_throughput_basis",
            sim_value="~201_tok_s_gpu_diagnose_expand_skips",
            real_value=f"{RUNAGG_TOK_S_GPU:.2f}_tok_s_gpu_run_agg",
            verdict=(
                "phase395_absolute_trace_throughput_was_wrong_basis; "
                "its_relative_merged_vs_split_conclusions_still_hold"
            ),
        ),
        _row(
            "steady_is_pure_decode",
            scenario="10k3k_b128",
            metric="avg_prefill_reqs_per_iter",
            sim_value=f"{AVG_PREFILL_REQS_PER_ITER}",
            verdict=(
                "steady_state_is_essentially_pure_decode; overcount_lives_in_"
                "decode_path_not_prefill; no_real_ttft_split_needed"
            ),
        ),
    ]

    for name, (dbs, gen_attn, gen_non_attn) in DECODE_COMPONENTS.items():
        share = 100.0 if gen_attn >= gen_non_attn else (
            100.0 * gen_non_attn / (gen_attn + gen_non_attn)
        )
        rows.append(
            _row(
                "decode_is_attention_bound",
                scenario=name,
                metric=f"decode_iter_gen_attn_vs_gen_non_attn_ms_bs{dbs}",
                sim_value=f"gen_attn={gen_attn:.2f};gen_non_attn={gen_non_attn:.2f}",
                ratio_or_share=f"attn_share={share:.0f}%",
                verdict=(
                    "pure_decode_total=max(...)=gen_attn_under_overlap0; "
                    "moe_gemm_gen_non_attn_masked"
                ),
            )
        )

    rows.extend(
        [
            _row(
                "overcount_magnitude",
                scenario="10k3k_b128",
                metric="steady_decode_wall_ms_sim_vs_real",
                sim_value=f"{RUNAGG_STEADY_WALL_MS:.0f}",
                real_value=f"{REAL_IMPLIED_STEADY_WALL_MS:.0f}",
                ratio_or_share=f"{OVERCOUNT_RATIO}x",
                verdict=(
                    "sim_over_predicts_decode_wall_1.517x = the_throughput_gate; "
                    "accumulated_over_~3000_decode_steps_via_kv_growth_trapezoid"
                ),
            ),
            _row(
                "verdict",
                metric="standing_overcount_locus",
                verdict=(
                    "decode_side_and_attention_bound: 0.12.0_decode_attention_"
                    "gen_attn_vs_kv; NOT_moe_gemm, NOT_prefill, NOT_created_by_"
                    "merged"
                ),
                next_allowed_phase=NEXT_PHASE,
            ),
            _row(
                "next_phase",
                metric="route_beta_target",
                verdict=(
                    "route_beta: measure_decode_attention_vs_kv_table_for_kimi_"
                    "tp16_on_module_boundary_0.19.0_and_unify_backends; "
                    "gamma_freeze_legacy_0.12.0_is_fallback"
                ),
                next_allowed_phase=NEXT_PHASE,
            ),
        ]
    )
    _validate_rows(rows)
    return rows


def _validate_rows(rows: list[dict[str, str]]) -> None:
    actual = [r["row_type"] for r in rows]
    if actual[0] != "metric_reconciliation":
        raise ValueError(f"Phase396 must start with metric_reconciliation: {actual[0]}")
    if actual[-2:] != ["verdict", "next_phase"]:
        raise ValueError(f"Phase396 must end with verdict,next_phase: {actual[-2:]}")

    required_types = {
        "metric_reconciliation",
        "phase395_correction",
        "steady_is_pure_decode",
        "decode_is_attention_bound",
        "overcount_magnitude",
        "verdict",
        "next_phase",
    }
    if required_types - set(actual):
        raise ValueError(f"Phase396 missing row types: {required_types - set(actual)}")

    for row in rows:
        label = row["row_type"]
        if set(row) != set(FIELDNAMES):
            raise ValueError(f"{label} fields do not match FIELDNAMES")
        if row["source"] != SOURCE:
            raise ValueError(f"{label} bad source")
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

    # decode-attention-bound is the load-bearing claim of the verdict.
    for row in rows:
        if row["row_type"] == "decode_is_attention_bound":
            if "attn_share=100%" not in row["ratio_or_share"]:
                raise ValueError(f"{row['scenario']} not attention-bound as claimed")
    verdict = next(r for r in rows if r["row_type"] == "verdict")
    if "attention_bound" not in verdict["verdict"]:
        raise ValueError("verdict must name decode attention as the locus")
    if verdict["next_allowed_phase"] != NEXT_PHASE:
        raise ValueError("verdict next phase mismatch")


def write_phase396_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_phase396_md(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase396 tp16 Standing Over-count Attribution (offline, verdict-only)",
        "",
        "| Item | Result |",
        "|---|---|",
        "| Question | where (absolute ms) is the standing tp16 ~1.5x over-count? |",
        "| Answer | **decode-side and attention-bound** (0.12.0 gen_attn vs KV) |",
        "| Runtime / PerfDatabase / GPU | not touched (verdict-only) |",
        "| Default AIC | No-Go |",
        "",
        "## Step 0: metric reconciliation (3.4x gap closed)",
        "",
        f"- Canonical basis: `tokens/s/gpu = steady_output_tokens / "
        f"(steady_time_ms/1000) / num_gpus`, num_gpus = tp*pp*dp = {NUM_GPUS}.",
        f"- run_agg real steady run (10k-3k b=128): steady_iters="
        f"{RUNAGG_STEADY_ITERS}, steady_wall={RUNAGG_STEADY_WALL_MS:.0f} ms, "
        f"steady_output_tokens~={RUNAGG_STEADY_OUTPUT_TOKENS} -> "
        f"{RUNAGG_TOK_S_GPU:.2f} tok/s/gpu (matches validate).",
        "- Root cause of the Phase395 ~201 vs 59 gap: the diagnose "
        "`--cb-trace-out --expand-skips` harness under-counts the decode wall "
        "~6.6x (expands pure-decode skips at near-constant latency), whereas "
        "the real sim charges the KV-growth trapezoid "
        "(`_estimate_decode_skip_latency`). `--no-expand-skips` reproduces a "
        "same-order steady decode wall.",
        "- Phase395's RELATIVE merged-vs-split conclusions still hold; only its "
        "ABSOLUTE trace throughput was on the wrong basis (corrected here).",
        "",
        "## Attribution (canonical run_agg basis)",
        "",
        f"- Steady state is ~pure decode: `avg_prefill_reqs_per_iter = "
        f"{AVG_PREFILL_REQS_PER_ITER}` -> the over-count is in the DECODE path, "
        "not prefill. No real-TTFT split needed (steady throughput gap == "
        "decode wall gap).",
        "- Each decode iteration is 100% attention-bound:",
        "",
        "| Scenario | decode_bs | gen_attn ms | gen_non_attn ms | attn share |",
        "|---|---|---|---|---|",
    ]
    for name, (dbs, gen_attn, gen_non_attn) in DECODE_COMPONENTS.items():
        lines.append(
            f"| {name} | {dbs} | {gen_attn:.2f} | {gen_non_attn:.2f} | 100% |"
        )
    lines += [
        "",
        "  With `overlap_factor=0` the pure-decode total = `max(...)` = gen_attn; "
        "MoE/GEMM (gen_non_attn) is masked.",
        f"- Over-count magnitude: sim steady wall {RUNAGG_STEADY_WALL_MS:.0f} ms "
        f"vs real-implied {REAL_IMPLIED_STEADY_WALL_MS:.0f} ms "
        f"(= {RUNAGG_STEADY_OUTPUT_TOKENS} tok / ({REAL_OUT_TOK_S_GPU_10K3K} * "
        f"{NUM_GPUS})) = **{OVERCOUNT_RATIO}x** -- exactly the throughput gate.",
        "",
        "## Verdict",
        "",
        "- The standing tp16 ~1.5x over-count is **decode-side and "
        "attention-bound**: the 0.12.0 decode attention latency vs KV "
        "(`gen_attn`), accumulated over ~3000 decode steps via the KV-growth "
        "trapezoid. NOT MoE/GEMM, NOT prefill, NOT created by merged.",
        "- Opposite-sign backends (0.12.0 over-predicts, measured 0.19.0 "
        "tp4dp2ep8 under-predicts) => pure data magnitude, not a shared "
        "composition bug.",
        f"- Phase397 (`{NEXT_PHASE}`): Route beta -- measure a "
        "decode-attention-vs-KV table for Kimi tp16 on the module-boundary "
        "(0.19.0) schema and unify the two perf backends. Route gamma (freeze "
        "legacy 0.12.0 as diagnostic) is the fallback.",
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

    rows = analyze_phase396_tp16_decode_attribution()
    write_phase396_csv(args.output_csv, rows)
    write_phase396_md(args.output_md, rows)


if __name__ == "__main__":
    main()
