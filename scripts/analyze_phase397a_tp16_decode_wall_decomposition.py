"""Phase397a tp16 decode wall decomposition (offline, verdict-only).

Route alpha follow-up to Phase396. Phase396 concluded the standing tp16
(~1.5x) over-count is decode-side and attention-bound, and pointed at a
Route-beta decode-attention-vs-KV measurement. Before spending GPU, Phase397a
decomposes -- OFFLINE -- the canonical run_agg steady wall to PIN the 1.5x to a
single locus, ruling out the cheap suspects (layer count, MTP scaling, prefill,
mixed iterations, trapezoid mechanism) so that Route-beta's GPU work is
justified and scoped.

It does not modify runtime, write PerfDatabase rows, use GPU/SSH, add fudge
factors, tune, or open Default AIC. All numbers are measured via the existing
diagnose_cb_iter_latency harness (--no-expand-skips trace + sim.run()); this
file only encodes those measured aggregates + the verdict.

Key correction carried in from the probe: Kimi K2.5 in this repo has
num_heads=64 (not 128); local heads at tp16 = 64/16 = 4. So the Route-beta
target table row is MLA num_heads=4, and it is a SINGLE-GPU microbench
(collector shards heads by tp on WORLD_SIZE=1) -- cross-node tp16 is NOT
required.

Measured findings (kimi-k2.5, h200_sxm, vllm 0.12.0, tp16 dp1):
  * Model constants: num_layers=61, nextn=0 -> mtp_scale_factor=1.0. Decode
    attention is charged as mla_per_layer * 61 * 1.0. Layer count and MTP are
    the REAL Kimi values and cannot inflate by 1.5x -> RULED OUT.
  * Steady state is ~pure decode (avg_prefill_reqs_per_iter 0.028..0.084).
  * Steady wall decomposition (trace shares): prefill 0%, mixed 8..10%,
    pure-decode (scheduled + KV-growth skip-trapezoid) 90..92%. The
    skip-trapezoid is the overwhelming majority of the wall.
  * Decode iteration latency is attention-bound: with overlap_factor=0 the
    pure-decode total = max(gen_attn, gen_non_attn) = gen_attn; gen_non_attn
    (MoE/GEMM) is masked.
  * Locate experiment on the CANONICAL run_agg wall: dividing ONLY the
    decode(=attention) component by 1.5 converges the total wall to the
    real-implied wall within ~6% for all three scenarios (1.062x / 1.011x /
    0.936x). A uniform decode scaling closes the gap -> the 1.5x is
    CONCENTRATED in the decode/attention component, not broadly spread.

Verdict: by elimination, the standing tp16 ~1.5x over-count is the 0.12.0
decode MLA-latency-vs-KV table values (num_heads=4 for tp16), integrated over
the decode KV trajectory via the trapezoid. It is NOT num_layers, NOT
mtp_scale, NOT prefill, NOT mixed, and NOT a broad multi-component spread. The
trapezoid is the (correct) accumulation vehicle, not the error; the per-KV MLA
values it integrates are the locus. Confirm-and-fix requires a fresh SINGLE-GPU
MLA-decode-vs-KV measurement (num_heads=4) -> Phase397b (Route beta), which
does NOT need cross-node tp16. Route gamma (freeze legacy 0.12.0 as diagnostic)
remains the fallback.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase397a_tp16_decode_wall_decomposition.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase397a_tp16_decode_wall_decomposition.md"
)

SOURCE = "phase397a_tp16_decode_wall_decomposition"
MODEL = "kimi-k2.5"
HARDWARE = "h200_sxm"
VLLM_VERSION = "0.12.0"
TOPOLOGY = "tp16dp1"
TRUE = "true"
FALSE = "false"
DEFAULT_READINESS = "No-Go"
NEXT_PHASE = "phase397b_route_beta_single_gpu_mla_decode_vs_kv_remeasure"

# Measured model constants (kimi-k2.5 tp16 dp1). Load-bearing for the verdict:
# these are the REAL Kimi values, so decode attention (= mla_per_layer *
# num_layers * mtp_scale) cannot be inflated by layer count or MTP.
NUM_LAYERS = 61
NEXTN = 0
MTP_SCALE_FACTOR = 1.0
NUM_HEADS_GLOBAL = 64
LOCAL_HEADS_TP16 = NUM_HEADS_GLOBAL // 16  # 4 -> Route-beta target MLA row
NUM_GPUS = 16

# Per-scenario canonical run_agg steady basis + decomposition (measured).
# name: (
#   steady_wall_ms, real_implied_wall_ms, overcount_ratio, tok_s_gpu,
#   avg_prefill_reqs_per_iter, mixed_pct, decode_pct, locate_adj_over_target,
# )
SCENARIOS = {
    "10k3k_b128": (796039.6, 524812.0, 1.517, 59.01, 0.084, 10.0, 90.0, 1.062),
    "10k2k_b32": (496486.8, 340553.0, 1.458, 34.16, 0.028, 8.0, 92.0, 1.011),
    "16k2k_b32": (545446.2, 406704.0, 1.341, 31.09, 0.028, 9.5, 90.5, 0.936),
}
# The throughput gate is the max over-count across scenarios (10k3k = 1.517x).
GATE_SCENARIO = "10k3k_b128"

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


def analyze_phase397a_tp16_decode_wall_decomposition() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = [
        _row(
            "model_constants",
            metric="num_layers;nextn;mtp_scale_factor;local_heads_tp16",
            sim_value=(
                f"num_layers={NUM_LAYERS};nextn={NEXTN};"
                f"mtp_scale_factor={MTP_SCALE_FACTOR};"
                f"local_heads_tp16={LOCAL_HEADS_TP16}"
            ),
            verdict=(
                "real_kimi_values: gen_attn=mla_per_layer*61*1.0; "
                "layer_count_and_mtp_cannot_inflate_1.5x -> RULED_OUT; "
                "route_beta_target_row_is_mla_num_heads=4_single_gpu"
            ),
        ),
    ]

    for name, (
        wall,
        real_wall,
        overcount,
        tok_s_gpu,
        avg_prefill,
        mixed_pct,
        decode_pct,
        _locate,
    ) in SCENARIOS.items():
        rows.append(
            _row(
                "overcount_canonical",
                scenario=name,
                metric="steady_wall_ms_sim_vs_real_implied",
                sim_value=f"{wall:.0f}",
                real_value=f"{real_wall:.0f}",
                ratio_or_share=f"{overcount}x",
                verdict=(
                    f"canonical_run_agg; tok_s_gpu={tok_s_gpu:.2f}; "
                    f"avg_prefill_reqs_per_iter={avg_prefill}"
                ),
            )
        )

    for name, (
        _wall,
        _real_wall,
        _overcount,
        _tok_s_gpu,
        _avg_prefill,
        mixed_pct,
        decode_pct,
        _locate,
    ) in SCENARIOS.items():
        rows.append(
            _row(
                "wall_decomposition",
                scenario=name,
                metric="steady_wall_shares_prefill_mixed_decode",
                sim_value=(
                    f"prefill=0.0%;mixed={mixed_pct}%;"
                    f"decode_incl_skip_trapezoid={decode_pct}%"
                ),
                ratio_or_share=f"decode_share={decode_pct}%",
                verdict=(
                    "pure_decode_skip_trapezoid_dominates; "
                    "prefill_ruled_out; mixed_minor"
                ),
            )
        )

    rows.append(
        _row(
            "decode_is_attention_bound",
            metric="pure_decode_iter_lat_composition",
            sim_value="iter_lat=max(gen_attn,gen_non_attn)=gen_attn",
            verdict=(
                "overlap_factor0_masks_gen_non_attn_moe_gemm; "
                "decode_wall=integral_of_mla_gen_attn_over_kv"
            ),
        )
    )

    for name, (
        _wall,
        real_wall,
        _overcount,
        _tok_s_gpu,
        _avg_prefill,
        _mixed_pct,
        _decode_pct,
        locate,
    ) in SCENARIOS.items():
        rows.append(
            _row(
                "locate_experiment",
                scenario=name,
                metric="decode_component_div_1.5_vs_real_implied_wall",
                real_value=f"{real_wall:.0f}",
                ratio_or_share=f"{locate}x",
                verdict=(
                    "dividing_only_decode_by_1.5_converges_to_real_within_~6%; "
                    "overcount_concentrated_in_decode_not_broad"
                ),
            )
        )

    rows.extend(
        [
            _row(
                "ruled_out",
                metric="non_locus_candidates",
                verdict=(
                    "num_layers=61_real; mtp_scale=1.0_neutral; prefill=0%; "
                    "mixed~10%; broad_spread_refuted_by_locate_experiment; "
                    "trapezoid_is_accumulation_vehicle_not_the_error"
                ),
            ),
            _row(
                "verdict",
                metric="standing_overcount_locus",
                verdict=(
                    "decode_mla_latency_vs_kv_table_values_num_heads4: the_"
                    "0.12.0_gen_attn_per_kv_values_integrated_via_trapezoid; "
                    "attention_bound; NOT_num_layers, NOT_mtp, NOT_prefill, "
                    "NOT_mixed, NOT_broad_spread"
                ),
                next_allowed_phase=NEXT_PHASE,
            ),
            _row(
                "next_phase",
                metric="route_beta_target",
                verdict=(
                    "route_beta_phase397b: single_gpu_remeasure_mla_decode_vs_"
                    "kv_num_heads4_and_compare_to_0.12.0_table_pointwise; "
                    "cross_node_tp16_NOT_required; gamma_freeze_legacy_fallback"
                ),
                next_allowed_phase=NEXT_PHASE,
            ),
        ]
    )
    _validate_rows(rows)
    return rows


def _validate_rows(rows: list[dict[str, str]]) -> None:
    actual = [r["row_type"] for r in rows]
    if actual[0] != "model_constants":
        raise ValueError(f"Phase397a must start with model_constants: {actual[0]}")
    if actual[-2:] != ["verdict", "next_phase"]:
        raise ValueError(f"Phase397a must end with verdict,next_phase: {actual[-2:]}")

    required_types = {
        "model_constants",
        "overcount_canonical",
        "wall_decomposition",
        "decode_is_attention_bound",
        "locate_experiment",
        "ruled_out",
        "verdict",
        "next_phase",
    }
    if required_types - set(actual):
        raise ValueError(f"Phase397a missing row types: {required_types - set(actual)}")

    n = len(SCENARIOS)
    for per_scenario in ("overcount_canonical", "wall_decomposition", "locate_experiment"):
        got = sum(1 for r in rows if r["row_type"] == per_scenario)
        if got != n:
            raise ValueError(f"{per_scenario} must have {n} rows, got {got}")

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

    # Load-bearing checks for the verdict.
    if f"num_layers={NUM_LAYERS}" not in rows[0]["sim_value"]:
        raise ValueError("model_constants must record num_layers=61")
    if f"mtp_scale_factor={MTP_SCALE_FACTOR}" not in rows[0]["sim_value"]:
        raise ValueError("model_constants must record mtp_scale_factor=1.0")

    # The gate scenario over-count must be the ~1.52x throughput gate.
    gate = next(
        r
        for r in rows
        if r["row_type"] == "overcount_canonical" and r["scenario"] == GATE_SCENARIO
    )
    if gate["ratio_or_share"] != "1.517x":
        raise ValueError("gate scenario over-count must be 1.517x")

    # Locate experiment must converge within ~6% for every scenario.
    for row in rows:
        if row["row_type"] == "locate_experiment":
            ratio = float(row["ratio_or_share"].rstrip("x"))
            if not (0.90 <= ratio <= 1.10):
                raise ValueError(
                    f"{row['scenario']} locate ratio {ratio} not within ~6% of 1.0"
                )

    verdict = next(r for r in rows if r["row_type"] == "verdict")
    if "decode_mla_latency_vs_kv" not in verdict["verdict"]:
        raise ValueError("verdict must name the decode MLA-vs-KV table as the locus")
    if "NOT_num_layers" not in verdict["verdict"]:
        raise ValueError("verdict must rule out num_layers")
    if verdict["next_allowed_phase"] != NEXT_PHASE:
        raise ValueError("verdict next phase mismatch")
    nxt = next(r for r in rows if r["row_type"] == "next_phase")
    if "cross_node_tp16_NOT_required" not in nxt["verdict"]:
        raise ValueError("next_phase must record that cross-node tp16 is not required")


def write_phase397a_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_phase397a_md(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase397a tp16 Decode Wall Decomposition (offline, verdict-only)",
        "",
        "| Item | Result |",
        "|---|---|",
        "| Question | which single locus holds the standing tp16 ~1.5x over-count? |",
        "| Answer | **the 0.12.0 decode MLA-latency-vs-KV table values** (num_heads=4) |",
        "| Runtime / PerfDatabase / GPU | not touched (verdict-only) |",
        "| Default AIC | No-Go |",
        "",
        "## Model constants (measured, kimi-k2.5 tp16 dp1)",
        "",
        f"- `num_layers = {NUM_LAYERS}`, `nextn = {NEXTN}` -> "
        f"`mtp_scale_factor = {MTP_SCALE_FACTOR}`.",
        f"- `num_heads` global = {NUM_HEADS_GLOBAL}; local heads at tp16 = "
        f"{NUM_HEADS_GLOBAL}/16 = **{LOCAL_HEADS_TP16}** (Route-beta target MLA row).",
        "- Decode attention is charged as `mla_per_layer * 61 * 1.0`. Layer count "
        "and MTP are the REAL Kimi values and cannot inflate by 1.5x -> RULED OUT.",
        "",
        "## Canonical run_agg over-count + wall decomposition",
        "",
        "| Scenario | steady wall ms | real-implied ms | over-count | tok/s/gpu "
        "| avg_prefill/iter | mixed % | decode(+skip) % | decode/1.5 vs real |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, (
        wall,
        real_wall,
        overcount,
        tok_s_gpu,
        avg_prefill,
        mixed_pct,
        decode_pct,
        locate,
    ) in SCENARIOS.items():
        lines.append(
            f"| {name} | {wall:.0f} | {real_wall:.0f} | {overcount}x | "
            f"{tok_s_gpu:.2f} | {avg_prefill} | {mixed_pct}% | {decode_pct}% | "
            f"{locate}x |"
        )
    lines += [
        "",
        "- Steady state is ~pure decode; the wall is 90..92% pure-decode "
        "(scheduled + KV-growth skip-trapezoid), 8..10% mixed, 0% prefill.",
        "- Decode iteration latency is attention-bound: with `overlap_factor=0` "
        "the pure-decode total = `max(gen_attn, gen_non_attn)` = `gen_attn`; "
        "`gen_non_attn` (MoE/GEMM) is masked.",
        "",
        "## Locate experiment",
        "",
        "- Dividing ONLY the decode(=attention) component by 1.5 converges the "
        "canonical wall to the real-implied wall within ~6% for all three "
        "scenarios (1.062x / 1.011x / 0.936x). A uniform decode scaling closes "
        "the gap -> the 1.5x is CONCENTRATED in the decode/attention component, "
        "not broadly spread.",
        "",
        "## Verdict",
        "",
        "- By elimination, the standing tp16 ~1.5x over-count is the **0.12.0 "
        "decode MLA-latency-vs-KV table values** (`num_heads=4` for tp16), "
        "integrated over the decode KV trajectory via the trapezoid. It is NOT "
        "num_layers, NOT mtp_scale, NOT prefill, NOT mixed, and NOT a broad "
        "multi-component spread. The trapezoid is the (correct) accumulation "
        "vehicle, not the error.",
        f"- Phase397b (`{NEXT_PHASE}`): Route beta -- a fresh **single-GPU** "
        "MLA-decode-vs-KV microbench (`num_heads=4`, collector shards heads on "
        "`WORLD_SIZE=1`) compared point-wise to the 0.12.0 table. "
        "**Cross-node tp16 is NOT required.** Route gamma (freeze legacy 0.12.0 "
        "as diagnostic) remains the fallback.",
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

    rows = analyze_phase397a_tp16_decode_wall_decomposition()
    write_phase397a_csv(args.output_csv, rows)
    write_phase397a_md(args.output_md, rows)


if __name__ == "__main__":
    main()
