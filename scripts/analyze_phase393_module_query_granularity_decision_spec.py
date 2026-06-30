"""Phase393 module query-granularity decision spec (offline, verdict-only).

Phase392 showed that cb_sim's module binding triggers but exact-only lookup is
unreachable for mixed iterations. Phase393 pins the root cause and decides the
fix route, encoding the offline confirmation probe results. It does not modify
runtime, write PerfDatabase rows, use GPU/SSH, or open Default AIC.

Root cause (confirmed):
    cb_sim's 3-pass composition (iteration_latency._compute_3pass) queries the
    MoE / EP8-comm module table with the PREFILL CHUNK token count
    (prefill_tokens = max_num_batched_tokens - decode_bs, e.g. 8191), separately
    from the decode batch. The real vLLM fused MoE runs ONCE on the MERGED batch
    (prefill_chunk + decode_tokens = max_num_batched_tokens = 8192), as shown by
    phase124_moe_activation_rows.csv tokens_actual = 8192. The split granularity
    therefore lands on non-materialized buckets (8191//4 = 2047), while the
    merged granularity lands exactly (8192//4 = 2048).

Confirmation probe (offline, scripts/diagnose_cb_iter_latency.py driven with the
same cross-version diagnostic DB as Phase392, --isl 10000 --osl 2000
--concurrency 32 --tp 4 --dp 2 --moe-tp 1 --moe-ep 8
--max-num-batched-tokens 8192 --overlap-factor 0 --per-iteration-overhead-ms 0):
    mixed iterations exact-hit ep8+fusedmoe buckets:
      * split granularity  (prefill_tokens):              0 / 118
      * merged granularity (prefill_tokens + decode_bs): 110 / 118
    prefill iterations: 2 / 2 under both.
    merged batch token == 8192 for 112 / 118 mixed iters; the 8 misses are
    non-saturated tail iterations (merged token != 8192).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase393_module_query_granularity_decision_spec.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase393_module_query_granularity_decision_spec.md"
)

SOURCE = "phase393_module_query_granularity_decision_spec"
MODEL = "kimi-k2.5"
HARDWARE = "h200_sxm"
VLLM_VERSION = "0.19.0"
TOPOLOGY = "tp4dp2ep8"
WORKLOAD = "10k2k_b32_bt8192"
TRUE = "true"
FALSE = "false"
DEFAULT_READINESS = "No-Go"

# Confirmation probe results (see module docstring for provenance).
MIXED_N = 118
PREFILL_N = 2
SPLIT_MIXED_HIT = 0
MERGED_MIXED_HIT = 110
SATURATED_MIXED = 112
EP8_BUCKETS = "1/15/16/241/1808/2048/8192"
FUSEDMOE_BUCKETS = "1/2/15/16/30/32/241/482/1808/2048/3616/4096/8192/16384"
NEXT_PHASE = "phase394_merged_granularity_runtime_change_and_bare_error_remeasure"

FIELDNAMES = [
    "source",
    "row_type",
    "candidate",
    "workload",
    "mixed_iters",
    "prefill_iters",
    "split_mixed_exact_hit",
    "merged_mixed_exact_hit",
    "saturated_mixed_iters",
    "recommendation",
    "verdict",
    "model",
    "hardware",
    "vllm_version",
    "topology",
    "ep8_buckets",
    "fusedmoe_buckets",
    "next_allowed_phase",
    "exact_lookup_only",
    "nearest_lookup_allowed",
    "interpolation_allowed",
    "extrapolation_allowed",
    "fudge_factor_tuning_used",
    "runtime_modified",
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
    "candidate": "",
    "workload": WORKLOAD,
    "mixed_iters": "",
    "prefill_iters": "",
    "split_mixed_exact_hit": "",
    "merged_mixed_exact_hit": "",
    "saturated_mixed_iters": "",
    "recommendation": "",
    "verdict": "",
    "model": MODEL,
    "hardware": HARDWARE,
    "vllm_version": VLLM_VERSION,
    "topology": TOPOLOGY,
    "ep8_buckets": EP8_BUCKETS,
    "fusedmoe_buckets": FUSEDMOE_BUCKETS,
    "next_allowed_phase": "",
    "exact_lookup_only": TRUE,
    "nearest_lookup_allowed": FALSE,
    "interpolation_allowed": FALSE,
    "extrapolation_allowed": FALSE,
    "fudge_factor_tuning_used": FALSE,
    "runtime_modified": FALSE,
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


def analyze_phase393_module_query_granularity_decision_spec() -> list[dict[str, str]]:
    rows = [
        _row(
            "root_cause",
            verdict=(
                "3pass_queries_prefill_chunk_token_count_while_real_vllm_"
                "fused_moe_runs_on_merged_batch"
            ),
        ),
        _row(
            "evidence_phase124",
            verdict=(
                "phase124_moe_activation_tokens_actual_is_8192_merged_not_"
                "8191_split"
            ),
        ),
        _row(
            "probe_split_granularity",
            candidate="split_prefill_chunk_current",
            mixed_iters=str(MIXED_N),
            prefill_iters=str(PREFILL_N),
            split_mixed_exact_hit=f"{SPLIT_MIXED_HIT}/{MIXED_N}",
            verdict="split_granularity_misses_all_mixed_iterations",
        ),
        _row(
            "probe_merged_granularity",
            candidate="merged_batch_tokens",
            mixed_iters=str(MIXED_N),
            prefill_iters=str(PREFILL_N),
            merged_mixed_exact_hit=f"{MERGED_MIXED_HIT}/{MIXED_N}",
            saturated_mixed_iters=str(SATURATED_MIXED),
            verdict="merged_granularity_makes_93pct_mixed_iterations_exact",
        ),
        _row(
            "candidate_merged_granularity",
            candidate="merged_batch_tokens",
            recommendation="recommended_primary",
            verdict=(
                "query_moe_and_ep8_at_prefill_tokens_plus_decode_bs; attention "
                "stays_split; not_a_lookup_relaxation_not_a_scheduler_rewrite"
            ),
        ),
        _row(
            "candidate_bucketization_contract",
            candidate="round_up_to_materialized_bucket",
            recommendation="fallback_for_non_saturated_tail_iters",
            verdict=(
                "explicit_quantization_contract_relaxes_exact_only; only_if_"
                "merged_granularity_leaves_gaps"
            ),
        ),
        _row(
            "candidate_scheduler_alignment",
            candidate="rewrite_cbsim_scheduler_token_accounting",
            recommendation="last_resort",
            verdict="highest_risk_touches_scheduler_defer_unless_needed",
        ),
        _row(
            "residual_non_saturated",
            verdict=(
                f"{MIXED_N - MERGED_MIXED_HIT}_of_{MIXED_N}_mixed_iters_are_"
                "non_saturated_tail_merged_token_not_8192_handle_in_phase394"
            ),
        ),
        _row(
            "decode_residual_deferred",
            verdict=(
                "decode_2_01x_small_bucket_moe_and_ep8_alltoall_comm_deferred_"
                "to_gpu_measurement_phase_not_this_phase"
            ),
        ),
        _row(
            "verdict",
            recommendation="recommended_primary",
            verdict=(
                "adopt_merged_granularity_for_moe_ep8_module_query; "
                "0_to_93pct_mixed_reachability_without_lookup_relaxation_or_gpu"
            ),
            next_allowed_phase=NEXT_PHASE,
        ),
        _row(
            "next_phase",
            verdict=(
                "phase394_runtime_change_merged_query_then_remeasure_bare_"
                "error_vs_ground_truth"
            ),
            next_allowed_phase=NEXT_PHASE,
        ),
    ]
    _validate_rows(rows)
    return rows


def _validate_rows(rows: list[dict[str, str]]) -> None:
    expected_types = [
        "root_cause",
        "evidence_phase124",
        "probe_split_granularity",
        "probe_merged_granularity",
        "candidate_merged_granularity",
        "candidate_bucketization_contract",
        "candidate_scheduler_alignment",
        "residual_non_saturated",
        "decode_residual_deferred",
        "verdict",
        "next_phase",
    ]
    actual_types = [row.get("row_type", "") for row in rows]
    if actual_types != expected_types:
        raise ValueError(f"unexpected Phase393 row order: {actual_types}")

    for row in rows:
        label = row["row_type"]
        if set(row) != set(FIELDNAMES):
            raise ValueError(f"{label} fields do not match FIELDNAMES")
        if row["source"] != SOURCE:
            raise ValueError(f"{label} bad source")
        for guard in (
            "nearest_lookup_allowed",
            "interpolation_allowed",
            "extrapolation_allowed",
            "fudge_factor_tuning_used",
            "runtime_modified",
            "write_real_data_file",
            "gpu_allowed",
            "ssh_allowed",
            "default_aic_allowed",
            "valid_for_default",
            "perf_database",
        ):
            if row[guard] != FALSE:
                raise ValueError(f"{label} {guard} must be {FALSE}")
        for flag in ("exact_lookup_only", "diagnostic_only"):
            if row[flag] != TRUE:
                raise ValueError(f"{label} {flag} must be {TRUE}")
        if row["default_readiness"] != DEFAULT_READINESS:
            raise ValueError(f"{label} default_readiness must be {DEFAULT_READINESS}")

    by_type = {row["row_type"]: row for row in rows}
    split = by_type["probe_split_granularity"]["split_mixed_exact_hit"]
    merged = by_type["probe_merged_granularity"]["merged_mixed_exact_hit"]
    if split != f"{SPLIT_MIXED_HIT}/{MIXED_N}":
        raise ValueError("split probe value mismatch")
    if merged != f"{MERGED_MIXED_HIT}/{MIXED_N}":
        raise ValueError("merged probe value mismatch")
    # the whole verdict rests on merged >> split
    if MERGED_MIXED_HIT <= SPLIT_MIXED_HIT:
        raise ValueError("merged granularity must beat split granularity")
    if MERGED_MIXED_HIT < int(0.9 * MIXED_N):
        raise ValueError("merged granularity must reach >=90% mixed iterations")
    if by_type["candidate_merged_granularity"]["recommendation"] != "recommended_primary":
        raise ValueError("merged granularity must be the recommended primary")
    if by_type["verdict"]["next_allowed_phase"] != NEXT_PHASE:
        raise ValueError("verdict next phase mismatch")


def write_phase393_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_phase393_md(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    split_pct = round(100.0 * SPLIT_MIXED_HIT / MIXED_N, 1)
    merged_pct = round(100.0 * MERGED_MIXED_HIT / MIXED_N, 1)
    lines = [
        "# Phase393 Module Query-Granularity Decision Spec (offline, verdict-only)",
        "",
        "| Item | Result |",
        "|---|---|",
        "| Workload | 10k2k_b32 / bt8192 / tp4dp2ep8 / Kimi-K2.5 / vLLM 0.19.0 |",
        "| Root cause | 3-pass queries MoE/EP8 at the prefill chunk token "
        "count, not the merged batch |",
        "| Verdict | **adopt merged-batch query granularity for MoE/EP8** |",
        f"| Mixed reachability | split {SPLIT_MIXED_HIT}/{MIXED_N} "
        f"({split_pct}%) -> merged {MERGED_MIXED_HIT}/{MIXED_N} ({merged_pct}%) |",
        "| Default AIC | No-Go |",
        "| Runtime / PerfDatabase / GPU | not touched |",
        "",
        "## Root cause",
        "",
        "- `scheduler.py` reserves 1 token per decode request, so the mixed "
        "prefill chunk = `max_num_batched_tokens - decode_bs` (8191, 8190, ...).",
        "- `iteration_latency._compute_3pass` Pass 1 queries the MoE/EP8 module "
        "with `isl=prefill_tokens` (the split chunk). Correct for attention, "
        "wrong for the token-parallel fused MoE.",
        "- `phase124_moe_activation_rows.csv` shows the real vLLM fused MoE "
        "`tokens_actual = 8192` (merged batch), not 8191. So the materialized "
        "buckets are merged-batch token counts.",
        "",
        "## Confirmation probe (offline, exact-only, no GPU)",
        "",
        "| Phase | n | split granularity exact-hit | merged granularity "
        "exact-hit |",
        "|---|---|---|---|",
        f"| prefill | {PREFILL_N} | {PREFILL_N}/{PREFILL_N} | "
        f"{PREFILL_N}/{PREFILL_N} |",
        f"| mixed | {MIXED_N} | {SPLIT_MIXED_HIT}/{MIXED_N} ({split_pct}%) | "
        f"{MERGED_MIXED_HIT}/{MIXED_N} ({merged_pct}%) |",
        "",
        f"- Merged batch token == 8192 for {SATURATED_MIXED}/{MIXED_N} mixed "
        f"iters; the {MIXED_N - MERGED_MIXED_HIT} misses are non-saturated tail "
        "iterations (merged token != 8192).",
        "",
        "## Candidates",
        "",
        "1. **Merged-batch granularity (recommended primary)**: query MoE/EP8 "
        "at `prefill_tokens + decode_bs`; attention stays split. This is a "
        "query-granularity correction, not a lookup relaxation and not a "
        "scheduler rewrite. Lifts mixed reachability 0% -> 93%.",
        "2. **Bucketization contract (fallback)**: round non-saturated tail "
        "iterations up to the nearest materialized bucket via an explicit "
        "documented quantization. Only if merged granularity leaves gaps.",
        "3. **Scheduler alignment (last resort)**: rewrite cb_sim scheduler "
        "token accounting. Highest risk; defer unless needed.",
        "",
        "## Verdict and next step",
        "",
        "- Adopt **merged-batch query granularity** for the MoE/EP8 module "
        "lookup. It resolves the Phase392 reachability blocker for 93% of mixed "
        "iterations without relaxing exact-only or using GPU.",
        f"- Phase394: make the runtime change (`{NEXT_PHASE}`) and re-measure "
        "the bare error against `compare_10k2k_b32_dp0.csv` for prefill/mixed.",
        f"- Residual: the {MIXED_N - MERGED_MIXED_HIT} non-saturated tail iters "
        "and the decode 2.01x (small-bucket MoE + EP8 all2all comm) are handled "
        "separately; decode goes to a later GPU measurement phase.",
        "",
        "## No-Go discipline held",
        "",
        "- exact-only model contract unchanged; merged granularity is querying "
        "the correct token count, not nearest/interpolation/extrapolation.",
        "- no fudge tuning; runtime (`operations.py` / `iteration_latency.py` / "
        "`vllm_backend.py`) and PerfDatabase not modified this phase.",
        "- no GPU/SSH used; Default AIC remains No-Go.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    rows = analyze_phase393_module_query_granularity_decision_spec()
    write_phase393_csv(args.output_csv, rows)
    write_phase393_md(args.output_md, rows)


if __name__ == "__main__":
    main()
