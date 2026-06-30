"""Phase392 end-to-end attribution gate (offline, verdict-only).

This analyzer encodes the measured per-phase iteration-latency attribution for
the tp4dp2ep8 Kimi-K2.5 vLLM 0.19.0 workload (10k2k_b32, max_num_batched_tokens
8192) and emits a verdict CSV + MD. It does not run cb_sim, touch runtime, write
PerfDatabase rows, use GPU, or open Default AIC.

Measurement provenance (offline, reproducible):
    scripts/diagnose_cb_iter_latency.py collect_cb_iteration_trace was driven
    with --isl 10000 --osl 2000 --concurrency 32 --tp 4 --dp 2 --moe-tp 1
    --moe-ep 8 --max-num-batched-tokens 8192 --overlap-factor 0 under three
    configurations:
      * baseline_query_moe_0_12: PerfDatabase version 0.12.0, module binding
        OFF (legacy query_moe path).
      * module_bound_0_19_snap: a diagnostic in-memory composition that loads
        0.12.0 kernel data (gemm/attention/mla) and attaches the measured
        0.19.0 vllm_module_perf.txt module table, flipping database.version to
        0.19.0 so _query_vllm_module fires for MoE + EP8 comm. attention/GEMM
        therefore stay on 0.12.0 data (the documented "kernel reuse" assumption).
    Ground truth is docs/iter_gap_investigation/compare_10k2k_b32_dp0.csv
    (vllm_iter_lat_ms), grouped by phase.

Diagnostic-snap caveat: cb_sim's continuous-batching scheduler emits a
continuous spread of scheduled-token counts (ep8 1..2500, fusedmoe 2..5000),
while only discrete real-vLLM-derived buckets are materialized (ep8 7,
fusedmoe 14). Under exact-only lookup the run fails fast on the first mixed
iteration. To obtain full-coverage numbers the analyzer's source run used a
DIAGNOSTIC nearest-bucket snap (max relative bucket error 0.18). The snap is
diagnostic only; the model lookup contract stays exact-only. Module-bound
numbers are therefore directional, not certified.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase392_e2e_attribution_gate.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase392_e2e_attribution_gate.md"
)

SOURCE = "phase392_e2e_attribution_gate"
MODEL = "kimi-k2.5"
HARDWARE = "h200_sxm"
VLLM_VERSION = "0.19.0"
TOPOLOGY = "tp4dp2ep8"
WORKLOAD = "10k2k_b32_bt8192"
GROUND_TRUTH = "compare_10k2k_b32_dp0.csv"
TRUE = "true"
FALSE = "false"
DEFAULT_READINESS = "No-Go"

# Measured per-phase medians (ms). See module docstring for provenance.
GT = {"prefill": 1054.15, "mixed": 976.54, "pure_decode": 24.82}
BASELINE_OH0 = {"prefill": 334.92, "mixed": 333.92, "pure_decode": 21.15}
MODULE_SNAP_OH0 = {"prefill": 789.73, "mixed": 789.46, "pure_decode": 12.34}
MODULE_SNAP_OH90_DECODE = 102.34

# Reachability evidence (exact-only lookup vs cb_sim scheduler token spread).
MODULE_CALLS = 201
EXACT_REACHABLE = "ep8 19/201; fusedmoe 11/201"
TOKEN_SPREAD = "ep8 1..2500; fusedmoe 2..5000"
MATERIALIZED_BUCKETS = "ep8 7; fusedmoe 14"
SNAP_MAX_REL_ERR = "0.18"

NEXT_ROUTE = (
    "route_b_scheduler_token_shape_alignment_plus_phase393_decode_ep8_comm_"
    "and_small_bucket_moe_measurement"
)

FIELDNAMES = [
    "source",
    "row_type",
    "workload",
    "phase",
    "comparison_mode",
    "overhead_ms",
    "ground_truth_median_ms",
    "predicted_median_ms",
    "error_ratio",
    "verdict",
    "module_binding_triggered",
    "exact_reachable_iters",
    "total_module_calls",
    "cbsim_token_spread",
    "materialized_bucket_count",
    "diagnostic_snap_used",
    "snap_max_rel_err",
    "next_route",
    "model",
    "hardware",
    "vllm_version",
    "topology",
    "ground_truth_source",
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
    "workload": WORKLOAD,
    "phase": "",
    "comparison_mode": "",
    "overhead_ms": "",
    "ground_truth_median_ms": "",
    "predicted_median_ms": "",
    "error_ratio": "",
    "verdict": "",
    "module_binding_triggered": "",
    "exact_reachable_iters": "",
    "total_module_calls": "",
    "cbsim_token_spread": "",
    "materialized_bucket_count": "",
    "diagnostic_snap_used": "",
    "snap_max_rel_err": "",
    "next_route": "",
    "model": MODEL,
    "hardware": HARDWARE,
    "vllm_version": VLLM_VERSION,
    "topology": TOPOLOGY,
    "ground_truth_source": GROUND_TRUTH,
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

PHASES = ["prefill", "mixed", "pure_decode"]


def _error_ratio(ground_truth: float, predicted: float) -> float:
    if ground_truth <= 0.0 or predicted <= 0.0:
        raise ValueError("medians must be positive")
    ratio = predicted / ground_truth
    return round(max(ratio, 1.0 / ratio), 2)


def _row(row_type: str, **overrides: str) -> dict[str, str]:
    row = dict(COMMON_FIELDS)
    row["row_type"] = row_type
    row.update(overrides)
    return row


def _phase_compare_row(
    row_type: str,
    phase: str,
    comparison_mode: str,
    overhead_ms: int,
    predicted: float,
    verdict: str,
) -> dict[str, str]:
    gt = GT[phase]
    return _row(
        row_type,
        phase=phase,
        comparison_mode=comparison_mode,
        overhead_ms=str(overhead_ms),
        ground_truth_median_ms=f"{gt:.2f}",
        predicted_median_ms=f"{predicted:.2f}",
        error_ratio=f"{_error_ratio(gt, predicted):.2f}",
        verdict=verdict,
        diagnostic_snap_used=(
            TRUE if comparison_mode == "module_bound_0_19_snap" else FALSE
        ),
        snap_max_rel_err=(
            SNAP_MAX_REL_ERR
            if comparison_mode == "module_bound_0_19_snap"
            else ""
        ),
    )


def analyze_phase392_e2e_attribution_gate() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for phase in PHASES:
        rows.append(
            _phase_compare_row(
                "baseline_query_moe_0_12",
                phase,
                "baseline_query_moe_0_12",
                0,
                BASELINE_OH0[phase],
                "legacy_0_12_moe_under_predicts_prefill_mixed_decode_ok",
            )
        )
    for phase in PHASES:
        rows.append(
            _phase_compare_row(
                "module_bound_0_19_snap_oh0",
                phase,
                "module_bound_0_19_snap",
                0,
                MODULE_SNAP_OH0[phase],
                (
                    "module_table_recovers_prefill_mixed_gap"
                    if phase in {"prefill", "mixed"}
                    else "module_table_regresses_decode_vs_baseline"
                ),
            )
        )
    rows.append(
        _phase_compare_row(
            "module_bound_0_19_snap_oh90_decode",
            "pure_decode",
            "module_bound_0_19_snap",
            90,
            MODULE_SNAP_OH90_DECODE,
            "90ms_decode_overhead_overshoots_decode_4x",
        )
    )

    rows.append(
        _row(
            "reachability_exact_only",
            verdict=(
                "binding_triggers_but_exact_only_incompatible_with_cbsim_"
                "continuous_token_spread"
            ),
            module_binding_triggered=TRUE,
            exact_reachable_iters=EXACT_REACHABLE,
            total_module_calls=str(MODULE_CALLS),
            cbsim_token_spread=TOKEN_SPREAD,
            materialized_bucket_count=MATERIALIZED_BUCKETS,
            diagnostic_snap_used=TRUE,
            snap_max_rel_err=SNAP_MAX_REL_ERR,
        )
    )
    rows.append(
        _row(
            "residual_attribution",
            verdict=(
                "prefill_mixed_residual_was_moe_undermeasured_in_0_12_table_"
                "recovered_by_0_19_module; decode_residual_is_ep8_alltoall_"
                "comm_and_small_bucket_moe_not_a_missing_prefill_boundary"
            ),
        )
    )
    rows.append(
        _row(
            "verdict",
            verdict=(
                "mixed: route_a_validated_for_prefill_mixed_3x_to_1_3x; "
                "decode_regresses; gating_blocker_is_scheduler_token_shape_"
                "not_missing_gpu_buckets"
            ),
            next_route=NEXT_ROUTE,
        )
    )
    rows.append(
        _row(
            "next_phase",
            verdict=(
                "phase393_route_b_scheduler_alignment_or_lookup_contract_"
                "then_targeted_decode_ep8_comm_measurement"
            ),
            next_route=NEXT_ROUTE,
        )
    )

    _validate_rows(rows)
    return rows


def _validate_rows(rows: list[dict[str, str]]) -> None:
    expected_types = [
        "baseline_query_moe_0_12",
        "baseline_query_moe_0_12",
        "baseline_query_moe_0_12",
        "module_bound_0_19_snap_oh0",
        "module_bound_0_19_snap_oh0",
        "module_bound_0_19_snap_oh0",
        "module_bound_0_19_snap_oh90_decode",
        "reachability_exact_only",
        "residual_attribution",
        "verdict",
        "next_phase",
    ]
    actual_types = [row.get("row_type", "") for row in rows]
    if actual_types != expected_types:
        raise ValueError(f"unexpected Phase392 row order: {actual_types}")

    for row in rows:
        label = f"{row['row_type']}/{row.get('phase') or '-'}"
        if set(row) != set(FIELDNAMES):
            raise ValueError(f"{label} fields do not match FIELDNAMES")
        if row["source"] != SOURCE:
            raise ValueError(f"{label} bad source")
        # No-Go discipline: never relax lookup, never fudge, never touch runtime.
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

        # numeric compare rows must have a self-consistent error_ratio
        if row["error_ratio"]:
            gt = float(row["ground_truth_median_ms"])
            pred = float(row["predicted_median_ms"])
            recomputed = _error_ratio(gt, pred)
            if abs(recomputed - float(row["error_ratio"])) > 0.011:
                raise ValueError(
                    f"{label} error_ratio {row['error_ratio']} != recomputed "
                    f"{recomputed}"
                )

    by_phase_mode = {
        (row.get("phase"), row.get("comparison_mode")): row
        for row in rows
        if row.get("error_ratio")
    }
    # core attribution invariants (the verdict rests on these)
    prefill_base = by_phase_mode[("prefill", "baseline_query_moe_0_12")]
    prefill_mod = by_phase_mode[("prefill", "module_bound_0_19_snap")]
    if float(prefill_base["error_ratio"]) < 3.0:
        raise ValueError("prefill baseline error must be >= 3x (under-prediction)")
    if float(prefill_mod["error_ratio"]) >= float(prefill_base["error_ratio"]):
        raise ValueError("module table must improve prefill error vs baseline")

    decode_base = by_phase_mode[("pure_decode", "baseline_query_moe_0_12")]
    decode_mod = by_phase_mode[("pure_decode", "module_bound_0_19_snap")]
    if float(decode_base["error_ratio"]) > 1.3:
        raise ValueError("decode baseline error should be near-accurate (<=1.3x)")
    if float(decode_mod["error_ratio"]) <= float(decode_base["error_ratio"]):
        raise ValueError("module table is expected to regress decode in this gate")

    verdict_row = next(r for r in rows if r["row_type"] == "verdict")
    if verdict_row["next_route"] != NEXT_ROUTE:
        raise ValueError("verdict next_route mismatch")


def write_phase392_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_phase392_md(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)

    def pick(phase: str, mode: str) -> dict[str, str]:
        for row in rows:
            if row.get("phase") == phase and row.get("comparison_mode") == mode:
                return row
        raise KeyError((phase, mode))

    lines = [
        "# Phase392 End-to-End Attribution Gate (offline, verdict-only)",
        "",
        "| Item | Result |",
        "|---|---|",
        "| Workload | 10k2k_b32 / bt8192 / tp4dp2ep8 / Kimi-K2.5 / vLLM 0.19.0 |",
        f"| Ground truth | `{GROUND_TRUTH}` (vllm_iter_lat_ms, per phase) |",
        "| Module binding | triggers (MoE + EP8 comm), but exact-only "
        "unreachable on mixed/decode |",
        "| Verdict | **mixed** — Route A validated for prefill/mixed; decode "
        "regresses; gating blocker is scheduler token-shape, not GPU buckets |",
        "| Default AIC | No-Go |",
        "| Runtime / PerfDatabase | not modified |",
        "",
        "## Per-phase bare error (median ms, error_ratio = max(p/r, r/p))",
        "",
        "| Phase | Ground truth | Baseline 0.12 query_moe | err | "
        "Module-bound 0.19 (diag snap, oh=0) | err |",
        "|---|---|---|---|---|---|",
    ]
    for phase in PHASES:
        b = pick(phase, "baseline_query_moe_0_12")
        m = pick(phase, "module_bound_0_19_snap")
        lines.append(
            f"| {phase} | {b['ground_truth_median_ms']} | "
            f"{b['predicted_median_ms']} | {b['error_ratio']}x | "
            f"{m['predicted_median_ms']} | {m['error_ratio']}x |"
        )
    lines += [
        "",
        f"- `+90ms` decode overhead overshoots: pure_decode "
        f"{MODULE_SNAP_OH90_DECODE:.2f}ms vs ground truth "
        f"{GT['pure_decode']:.2f}ms ("
        f"{_error_ratio(GT['pure_decode'], MODULE_SNAP_OH90_DECODE):.2f}x). "
        "The 90ms fudge was calibrated for a different operating point.",
        "",
        "## Reachability (exact-only vs cb_sim scheduler)",
        "",
        "- Module binding fires for both `fusedmoe_runner_compute` and "
        "`ep8_comm_dispatch_combine`.",
        f"- Exact-bucket hits: {EXACT_REACHABLE} of {MODULE_CALLS} module "
        "calls — only the pure-prefill (`decode_bs=0`) iterations land on a "
        "materialized bucket (8192 -> 2048).",
        f"- cb_sim emits a **continuous** token spread ({TOKEN_SPREAD}) because "
        "mixed-phase prefill chunk = `max_num_batched_tokens - decode_bs` "
        "(8191, 8190, ...), whereas the materialized buckets "
        f"({MATERIALIZED_BUCKETS}) come from the real vLLM scheduler's discrete "
        "token counts. Under exact-only lookup cb_sim fails fast on the first "
        "mixed iteration.",
        f"- All module-bound numbers above used a DIAGNOSTIC nearest-bucket "
        f"snap (max relative bucket error {SNAP_MAX_REL_ERR}); they are "
        "directional, not certified. The model lookup contract stays "
        "exact-only.",
        "",
        "## Attribution",
        "",
        "- **prefill/mixed**: the ~3x gap was dominated by MoE compute being "
        "under-measured in the legacy 0.12 `moe_perf.txt`. The measured 0.19 "
        "module table (~2x the 0.12 MoE) recovers most of it (3.15x -> 1.33x "
        "prefill, 2.92x -> 1.24x mixed). This refutes the earlier \"residual "
        "belongs to no module\" worry for prefill.",
        "- **decode**: the 0.12 path was already accurate (1.17x). The 0.19 "
        "module table makes decode worse (2.01x) because decode-phase small "
        "buckets are under-valued and EP8 all2all comm in the decode loop is "
        "not captured — exactly what the 90ms hand-fudge was masking.",
        "",
        "## Verdict and next step",
        "",
        "- Route A (module-level perf table) is **validated for the prefill/"
        "mixed compute path**, the original big gap.",
        "- The gating blocker is **not** missing GPU buckets (measuring "
        "`8191/8190/...` would be futile artifacts). It is the cb_sim "
        "scheduler emitting token shapes that do not match vLLM's discrete "
        "buckets. Fix is **Route B scheduler token-shape alignment** (or a "
        "documented bucketization/quantization lookup contract), not naive "
        "bucket/topology expansion.",
        "- Phase393 GPU work should target the **decode path** (EP8 all2all "
        "comm + small-bucket MoE), not prefill.",
        "",
        "## No-Go discipline held",
        "",
        "- exact-only model contract unchanged; nearest/interpolation/"
        "extrapolation not enabled in runtime.",
        "- no fudge tuning used to declare success; 90ms shown as a "
        "diagnostic column only.",
        "- runtime (`operations.py` / `vllm_backend.py`) and PerfDatabase "
        "files not modified; no GPU/SSH used.",
        "- Default AIC remains No-Go.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    rows = analyze_phase392_e2e_attribution_gate()
    write_phase392_csv(args.output_csv, rows)
    write_phase392_md(args.output_md, rows)


if __name__ == "__main__":
    main()
