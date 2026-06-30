from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE382_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase382_runtime_binding_probe_sufficiency_gate.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT / "docs/iter_gap_investigation/phase383_runtime_bucket_semantics_decision_spec.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT / "docs/iter_gap_investigation/phase383_runtime_bucket_semantics_decision_spec.md"
)

SOURCE = "phase383_runtime_bucket_semantics_decision_spec"
DEFAULT_READINESS = "No-Go"
TRUE = "true"
FALSE = "false"
EP8_BUCKET_FORMULA = "max(1, raw_tokens//4)"
FUSEDMOE_BUCKET_FORMULA = "max(1, raw_tokens//4)*2"
EXISTING_BUCKETS = [1, 15, 16, 241, 1808, 2048, 8192]
FUSEDMOE_PAIRED_BUCKETS = [2, 30, 32, 482, 3616, 4096, 16384]
EXISTING_BUCKETS_TEXT = "/".join(str(bucket) for bucket in EXISTING_BUCKETS)
FUSEDMOE_PAIRED_BUCKETS_TEXT = "/".join(str(bucket) for bucket in FUSEDMOE_PAIRED_BUCKETS)
SELECTED_ROUTE = "paired_bucket_data_expansion_first"
NEXT_PHASE = "phase384_paired_bucket_expansion_spec"

FIELDNAMES = [
    "source",
    "row_type",
    "verdict",
    "phase382_prerequisite",
    "paired_lookup_candidate_count",
    "ep8_bucket_formula",
    "fusedmoe_bucket_formula",
    "runtime_bucket_semantics_verdict",
    "runtime_bucket_semantics_change_allowed",
    "runtime_change_rejection_reason",
    "selected_route",
    "existing_rows_preserved",
    "existing_module_row_count",
    "fusedmoe_paired_bucket_expansion",
    "ep8_existing_buckets",
    "bucket_128_allowed",
    "next_allowed_phase",
    "gpu_allowed",
    "ssh_allowed",
    "runtime_change",
    "new_perfdb_data",
    "default_aic_allowed",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
    "default_readiness",
    "interpolation_allowed",
    "extrapolation_allowed",
    "nearest_bucket_allowed",
]

COMMON_FIELDS = {
    "source": SOURCE,
    "verdict": "",
    "phase382_prerequisite": "",
    "paired_lookup_candidate_count": "",
    "ep8_bucket_formula": "",
    "fusedmoe_bucket_formula": "",
    "runtime_bucket_semantics_verdict": "",
    "runtime_bucket_semantics_change_allowed": FALSE,
    "runtime_change_rejection_reason": "",
    "selected_route": "",
    "existing_rows_preserved": "",
    "existing_module_row_count": "",
    "fusedmoe_paired_bucket_expansion": "",
    "ep8_existing_buckets": "",
    "bucket_128_allowed": FALSE,
    "next_allowed_phase": "",
    "gpu_allowed": FALSE,
    "ssh_allowed": FALSE,
    "runtime_change": FALSE,
    "new_perfdb_data": FALSE,
    "default_aic_allowed": FALSE,
    "diagnostic_only": TRUE,
    "valid_for_default": FALSE,
    "perf_database": FALSE,
    "default_readiness": DEFAULT_READINESS,
    "interpolation_allowed": FALSE,
    "extrapolation_allowed": FALSE,
    "nearest_bucket_allowed": FALSE,
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no data rows")
    return rows


def _require(row: dict[str, str], field: str, expected: str, label: str) -> None:
    actual = row.get(field)
    if actual != expected:
        raise ValueError(f"{label} {field} expected {expected!r}, got {actual!r}")


def _require_phase382(path: Path) -> None:
    rows = _read_csv(path)
    by_type = {row["row_type"]: row for row in rows}
    required = [
        "module_table_exact_key_inventory",
        "runtime_bucket_formula_current",
        "fusedmoe_runtime_bucket_reachability",
        "paired_lookup_candidate_gap",
        "op_level_runtime_binding_sufficient",
        "full_model_runtime_exact_lookup_blocked",
        "default_aic_blocked",
        "next_phase",
    ]
    for row_type in required:
        if row_type not in by_type:
            raise ValueError(f"Phase382 missing row {row_type}")

    inventory = by_type["module_table_exact_key_inventory"]
    formula = by_type["runtime_bucket_formula_current"]
    fused = by_type["fusedmoe_runtime_bucket_reachability"]
    gap = by_type["paired_lookup_candidate_gap"]
    op_level = by_type["op_level_runtime_binding_sufficient"]
    blocked = by_type["full_model_runtime_exact_lookup_blocked"]
    default = by_type["default_aic_blocked"]
    next_phase = by_type["next_phase"]

    _require(inventory, "module_table_exact_keys", "2_modules_x_7_buckets", "Phase382 inventory")
    _require(inventory, "table_row_count", "14", "Phase382 inventory")
    _require(inventory, "allowed_bucket_tokens", EXISTING_BUCKETS_TEXT, "Phase382 inventory")
    _require(formula, "ep8_bucket_formula", EP8_BUCKET_FORMULA, "Phase382 formula")
    _require(formula, "fusedmoe_bucket_formula", FUSEDMOE_BUCKET_FORMULA, "Phase382 formula")
    _require(fused, "fusedmoe_reachable_buckets", "16/1808/2048/8192", "Phase382 fused")
    _require(fused, "fusedmoe_unreachable_buckets", "1/15/241", "Phase382 fused")
    _require(gap, "paired_lookup_candidate_count", "0", "Phase382 paired gap")
    _require(op_level, "op_level_runtime_binding_sufficient", TRUE, "Phase382 op level")
    _require(blocked, "full_model_runtime_exact_lookup_sufficient", FALSE, "Phase382 full model")
    _require(default, "default_aic_allowed", FALSE, "Phase382 default")
    _require(next_phase, "next_allowed_phase", SOURCE, "Phase382 next")

    for row in rows:
        _require(row, "default_readiness", DEFAULT_READINESS, row["row_type"])
        _require(row, "diagnostic_only", TRUE, row["row_type"])
        _require(row, "valid_for_default", FALSE, row["row_type"])
        _require(row, "perf_database", FALSE, row["row_type"])


def _row(row_type: str, **overrides: str) -> dict[str, str]:
    row = dict(COMMON_FIELDS)
    row["row_type"] = row_type
    row.update(overrides)
    return row


def analyze_phase383_runtime_bucket_semantics_decision_spec(
    phase382_csv: Path = DEFAULT_PHASE382_CSV,
) -> list[dict[str, str]]:
    _require_phase382(phase382_csv)

    rows = [
        _row(
            "phase382_prerequisite",
            verdict="phase382_full_model_exact_lookup_blocked_confirmed",
            phase382_prerequisite="full_model_exact_lookup_blocked",
            paired_lookup_candidate_count="0",
        ),
        _row(
            "current_runtime_bucket_semantics_confirmed",
            verdict="current_runtime_bucket_semantics_are_correct",
            ep8_bucket_formula=EP8_BUCKET_FORMULA,
            fusedmoe_bucket_formula=FUSEDMOE_BUCKET_FORMULA,
            runtime_bucket_semantics_verdict="correct_current_semantics",
        ),
        _row(
            "runtime_bucket_semantics_change_rejected",
            verdict="runtime_bucket_semantics_change_rejected",
            runtime_bucket_semantics_change_allowed=FALSE,
            runtime_change_rejection_reason="would_diverge_from_real_fusedmoe_shape",
        ),
        _row(
            "paired_bucket_data_expansion_selected",
            verdict="selected_route_locked",
            selected_route=SELECTED_ROUTE,
        ),
        _row(
            "existing_module_rows_preserved",
            verdict="preserve_existing_14_rows",
            existing_rows_preserved=TRUE,
            existing_module_row_count="14",
        ),
        _row(
            "fusedmoe_paired_bucket_expansion_required",
            verdict="future_fusedmoe_paired_buckets_required",
            fusedmoe_paired_bucket_expansion=FUSEDMOE_PAIRED_BUCKETS_TEXT,
        ),
        _row(
            "ep8_bucket_set_preserved",
            verdict="existing_ep8_bucket_set_preserved",
            ep8_existing_buckets=EXISTING_BUCKETS_TEXT,
        ),
        _row(
            "bucket_128_blocked",
            verdict="bucket_128_remains_smoke_only",
            bucket_128_allowed=FALSE,
        ),
        _row(
            "perfdb_write_blocked",
            verdict="spec_only_no_perfdb_write",
            new_perfdb_data=FALSE,
            perf_database=FALSE,
        ),
        _row(
            "default_aic_blocked",
            verdict="default_aic_stays_no_go",
            default_aic_allowed=FALSE,
        ),
        _row(
            "next_phase",
            verdict="next_phase_is_paired_bucket_expansion_spec",
            next_allowed_phase=NEXT_PHASE,
        ),
    ]
    _validate_rows(rows)
    return rows


def _validate_rows(rows: list[dict[str, str]]) -> None:
    expected_row_types = [
        "phase382_prerequisite",
        "current_runtime_bucket_semantics_confirmed",
        "runtime_bucket_semantics_change_rejected",
        "paired_bucket_data_expansion_selected",
        "existing_module_rows_preserved",
        "fusedmoe_paired_bucket_expansion_required",
        "ep8_bucket_set_preserved",
        "bucket_128_blocked",
        "perfdb_write_blocked",
        "default_aic_blocked",
        "next_phase",
    ]
    actual_row_types = [row["row_type"] for row in rows]
    if actual_row_types != expected_row_types:
        raise ValueError(f"unexpected row order: {actual_row_types}")

    for row in rows:
        if set(row) != set(FIELDNAMES):
            raise ValueError(f"{row['row_type']} fields do not match FIELDNAMES")
        _require(row, "gpu_allowed", FALSE, row["row_type"])
        _require(row, "ssh_allowed", FALSE, row["row_type"])
        _require(row, "runtime_change", FALSE, row["row_type"])
        _require(row, "new_perfdb_data", FALSE, row["row_type"])
        _require(row, "default_aic_allowed", FALSE, row["row_type"])
        _require(row, "diagnostic_only", TRUE, row["row_type"])
        _require(row, "valid_for_default", FALSE, row["row_type"])
        _require(row, "perf_database", FALSE, row["row_type"])
        _require(row, "default_readiness", DEFAULT_READINESS, row["row_type"])
        _require(row, "interpolation_allowed", FALSE, row["row_type"])
        _require(row, "extrapolation_allowed", FALSE, row["row_type"])
        _require(row, "nearest_bucket_allowed", FALSE, row["row_type"])

    by_type = {row["row_type"]: row for row in rows}
    _require(
        by_type["current_runtime_bucket_semantics_confirmed"],
        "ep8_bucket_formula",
        EP8_BUCKET_FORMULA,
        "Phase383 formula",
    )
    _require(
        by_type["current_runtime_bucket_semantics_confirmed"],
        "fusedmoe_bucket_formula",
        FUSEDMOE_BUCKET_FORMULA,
        "Phase383 formula",
    )
    _require(
        by_type["runtime_bucket_semantics_change_rejected"],
        "runtime_bucket_semantics_change_allowed",
        FALSE,
        "Phase383 runtime semantics",
    )
    _require(
        by_type["paired_bucket_data_expansion_selected"],
        "selected_route",
        SELECTED_ROUTE,
        "Phase383 route",
    )
    _require(
        by_type["fusedmoe_paired_bucket_expansion_required"],
        "fusedmoe_paired_bucket_expansion",
        FUSEDMOE_PAIRED_BUCKETS_TEXT,
        "Phase383 paired buckets",
    )
    _require(
        by_type["bucket_128_blocked"],
        "bucket_128_allowed",
        FALSE,
        "Phase383 bucket 128",
    )


def write_phase383_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_phase383_md(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    paired = next(
        row for row in rows if row["row_type"] == "fusedmoe_paired_bucket_expansion_required"
    )
    ep8 = next(row for row in rows if row["row_type"] == "ep8_bucket_set_preserved")
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = f"""# Phase383 Runtime Bucket Semantics Decision Spec

Phase383 is a decision spec. It does not change runtime code, write PerfDatabase data, run GPU, or open Default AIC.

| Gate | Decision |
|---|---|
| current EP8 formula | {EP8_BUCKET_FORMULA} |
| current FusedMoE formula | {FUSEDMOE_BUCKET_FORMULA} |
| runtime semantics | Do not change runtime bucket semantics |
| selected route | {SELECTED_ROUTE} |
| preserve existing rows | 14 current vLLM module rows stay in place |
| future FusedMoE paired buckets | {paired["fusedmoe_paired_bucket_expansion"]} |
| existing EP8 buckets | {ep8["ep8_existing_buckets"]} |
| bucket 128 | blocked |
| Default AIC | No-Go |

Changing the runtime formula would make FusedMoE lookup diverge from the real module shape, so the first legal fix is paired bucket data expansion.
The next allowed phase is `{NEXT_PHASE}`.
"""
    path.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase382-csv", type=Path, default=DEFAULT_PHASE382_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    rows = analyze_phase383_runtime_bucket_semantics_decision_spec(args.phase382_csv)
    write_phase383_csv(args.output_csv, rows)
    write_phase383_md(args.output_md, rows)


if __name__ == "__main__":
    main()
