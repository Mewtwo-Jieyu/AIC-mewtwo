from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE383_CSV = (
    REPO_ROOT / "docs/iter_gap_investigation/phase383_runtime_bucket_semantics_decision_spec.csv"
)
DEFAULT_VLLM_MODULE_PERF = (
    REPO_ROOT / "src/aiconfigurator/systems/data/h200_sxm/vllm/0.19.0/vllm_module_perf.txt"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT / "docs/iter_gap_investigation/phase384_paired_bucket_expansion_spec.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT / "docs/iter_gap_investigation/phase384_paired_bucket_expansion_spec.md"
)

SOURCE = "phase384_paired_bucket_expansion_spec"
DEFAULT_READINESS = "No-Go"
TRUE = "true"
FALSE = "false"
MODEL = "kimi-k2.5"
HARDWARE = "h200_sxm"
VLLM_VERSION = "0.19.0"
TOPOLOGY = "tp4dp2ep8"
QUANT_RUNTIME = "CompressedTensorsWNA16MarlinMoEMethod"
FUSED_MODULE = "fusedmoe_runner_compute"
EP8_MODULE = "ep8_comm_dispatch_combine"
EP8_BUCKETS = [1, 15, 16, 241, 1808, 2048, 8192]
FUSEDMOE_PAIRED_BUCKETS = [2, 30, 32, 482, 3616, 4096, 16384]
EP8_BUCKETS_TEXT = "/".join(str(bucket) for bucket in EP8_BUCKETS)
FUSEDMOE_PAIRED_BUCKETS_TEXT = "/".join(str(bucket) for bucket in FUSEDMOE_PAIRED_BUCKETS)
NEXT_PHASE = "phase385_fusedmoe_paired_bucket_gpu_data_spec"

FIELDNAMES = [
    "source",
    "row_type",
    "verdict",
    "phase383_prerequisite",
    "next_allowed_phase_from_phase383",
    "runtime_bucket_semantics_change_allowed",
    "existing_rows_preserved",
    "existing_module_row_count",
    "delete_existing_rows",
    "replace_existing_rows",
    "ep8_existing_buckets",
    "ep8_bucket_change_allowed",
    "fusedmoe_existing_buckets",
    "fusedmoe_paired_bucket_expansion",
    "new_planned_row_count",
    "future_total_row_count",
    "bucket_128_allowed",
    "exact_lookup_only",
    "gpu_allowed",
    "ssh_allowed",
    "runtime_change",
    "new_perfdb_data",
    "write_real_data_file",
    "default_aic_allowed",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
    "default_readiness",
    "interpolation_allowed",
    "extrapolation_allowed",
    "nearest_bucket_allowed",
    "next_allowed_phase",
]

COMMON_FIELDS = {
    "source": SOURCE,
    "verdict": "",
    "phase383_prerequisite": "",
    "next_allowed_phase_from_phase383": "",
    "runtime_bucket_semantics_change_allowed": FALSE,
    "existing_rows_preserved": "",
    "existing_module_row_count": "",
    "delete_existing_rows": FALSE,
    "replace_existing_rows": FALSE,
    "ep8_existing_buckets": "",
    "ep8_bucket_change_allowed": FALSE,
    "fusedmoe_existing_buckets": "",
    "fusedmoe_paired_bucket_expansion": "",
    "new_planned_row_count": "",
    "future_total_row_count": "",
    "bucket_128_allowed": FALSE,
    "exact_lookup_only": TRUE,
    "gpu_allowed": FALSE,
    "ssh_allowed": FALSE,
    "runtime_change": FALSE,
    "new_perfdb_data": FALSE,
    "write_real_data_file": FALSE,
    "default_aic_allowed": FALSE,
    "diagnostic_only": TRUE,
    "valid_for_default": FALSE,
    "perf_database": FALSE,
    "default_readiness": DEFAULT_READINESS,
    "interpolation_allowed": FALSE,
    "extrapolation_allowed": FALSE,
    "nearest_bucket_allowed": FALSE,
    "next_allowed_phase": "",
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


def _require_phase383(path: Path) -> None:
    rows = _read_csv(path)
    by_type = {row["row_type"]: row for row in rows}
    required = [
        "paired_bucket_data_expansion_selected",
        "existing_module_rows_preserved",
        "fusedmoe_paired_bucket_expansion_required",
        "ep8_bucket_set_preserved",
        "bucket_128_blocked",
        "default_aic_blocked",
        "next_phase",
    ]
    for row_type in required:
        if row_type not in by_type:
            raise ValueError(f"Phase383 missing row {row_type}")

    route = by_type["paired_bucket_data_expansion_selected"]
    preserve = by_type["existing_module_rows_preserved"]
    expansion = by_type["fusedmoe_paired_bucket_expansion_required"]
    ep8 = by_type["ep8_bucket_set_preserved"]
    bucket = by_type["bucket_128_blocked"]
    default = by_type["default_aic_blocked"]
    next_phase = by_type["next_phase"]

    _require(route, "selected_route", "paired_bucket_data_expansion_first", "Phase383 route")
    _require(preserve, "existing_rows_preserved", TRUE, "Phase383 preserve")
    _require(preserve, "existing_module_row_count", "14", "Phase383 preserve")
    _require(expansion, "fusedmoe_paired_bucket_expansion", FUSEDMOE_PAIRED_BUCKETS_TEXT, "Phase383 expansion")
    _require(ep8, "ep8_existing_buckets", EP8_BUCKETS_TEXT, "Phase383 ep8")
    _require(bucket, "bucket_128_allowed", FALSE, "Phase383 bucket 128")
    _require(default, "default_aic_allowed", FALSE, "Phase383 default")
    _require(next_phase, "next_allowed_phase", SOURCE, "Phase383 next")

    for row in rows:
        _require(row, "default_readiness", DEFAULT_READINESS, row["row_type"])
        _require(row, "diagnostic_only", TRUE, row["row_type"])
        _require(row, "valid_for_default", FALSE, row["row_type"])
        _require(row, "perf_database", FALSE, row["row_type"])
        _require(row, "runtime_change", FALSE, row["row_type"])


def _load_existing_module_rows(path: Path) -> dict[str, set[int]]:
    rows = _read_csv(path)
    buckets_by_module: dict[str, set[int]] = {FUSED_MODULE: set(), EP8_MODULE: set()}
    seen: set[tuple[str, str, str, str, int, str, str]] = set()
    for row in rows:
        module = row["module_boundary"]
        bucket = int(row["bucket_tokens"])
        key = (
            row["model"],
            row["hardware"],
            row["vllm_version"],
            row["topology"],
            bucket,
            module,
            row["quant_runtime"],
        )
        if key in seen:
            raise ValueError(f"duplicate vllm module key: {key}")
        seen.add(key)
        if row["model"] != MODEL:
            raise ValueError(f"unexpected model {row['model']!r}")
        if row["hardware"] != HARDWARE:
            raise ValueError(f"unexpected hardware {row['hardware']!r}")
        if row["vllm_version"] != VLLM_VERSION:
            raise ValueError(f"unexpected vllm_version {row['vllm_version']!r}")
        if row["topology"] != TOPOLOGY:
            raise ValueError(f"unexpected topology {row['topology']!r}")
        if row["quant_runtime"] != QUANT_RUNTIME:
            raise ValueError(f"unexpected quant_runtime {row['quant_runtime']!r}")
        if module not in buckets_by_module:
            raise ValueError(f"unexpected module_boundary {module!r}")
        buckets_by_module[module].add(bucket)

    expected = set(EP8_BUCKETS)
    if buckets_by_module[FUSED_MODULE] != expected:
        raise ValueError(f"unexpected FusedMoE buckets {sorted(buckets_by_module[FUSED_MODULE])}")
    if buckets_by_module[EP8_MODULE] != expected:
        raise ValueError(f"unexpected EP8 buckets {sorted(buckets_by_module[EP8_MODULE])}")
    if len(seen) != 14:
        raise ValueError(f"expected 14 existing rows, got {len(seen)}")
    return buckets_by_module


def _format_buckets(buckets: list[int] | set[int]) -> str:
    return "/".join(str(bucket) for bucket in sorted(buckets))


def _row(row_type: str, **overrides: str) -> dict[str, str]:
    row = dict(COMMON_FIELDS)
    row["row_type"] = row_type
    row.update(overrides)
    return row


def analyze_phase384_paired_bucket_expansion_spec(
    phase383_csv: Path = DEFAULT_PHASE383_CSV,
    vllm_module_perf: Path = DEFAULT_VLLM_MODULE_PERF,
) -> list[dict[str, str]]:
    _require_phase383(phase383_csv)
    existing = _load_existing_module_rows(vllm_module_perf)

    rows = [
        _row(
            "phase383_prerequisite",
            verdict="phase383_decision_accepts_paired_bucket_expansion_first",
            phase383_prerequisite="paired_bucket_data_expansion_first",
            next_allowed_phase_from_phase383=SOURCE,
            runtime_bucket_semantics_change_allowed=FALSE,
        ),
        _row(
            "existing_vllm_module_rows_preserved",
            verdict="preserve_existing_14_rows",
            existing_rows_preserved=TRUE,
            existing_module_row_count="14",
            delete_existing_rows=FALSE,
            replace_existing_rows=FALSE,
        ),
        _row(
            "ep8_bucket_set_preserved",
            verdict="ep8_bucket_set_unchanged",
            ep8_existing_buckets=_format_buckets(existing[EP8_MODULE]),
            ep8_bucket_change_allowed=FALSE,
        ),
        _row(
            "fusedmoe_paired_bucket_expansion_planned",
            verdict="plan_seven_new_fusedmoe_paired_buckets",
            fusedmoe_existing_buckets=_format_buckets(existing[FUSED_MODULE]),
            fusedmoe_paired_bucket_expansion=FUSEDMOE_PAIRED_BUCKETS_TEXT,
            new_planned_row_count="7",
            write_real_data_file=FALSE,
        ),
        _row(
            "future_total_row_count_planned",
            verdict="future_table_would_have_21_rows_after_later_materialization",
            existing_module_row_count="14",
            new_planned_row_count="7",
            future_total_row_count="21",
            write_real_data_file=FALSE,
        ),
        _row(
            "bucket_128_blocked",
            verdict="bucket_128_remains_disallowed",
            bucket_128_allowed=FALSE,
        ),
        _row(
            "lookup_policy_exact_only",
            verdict="exact_only_no_interpolation_or_nearest_lookup",
            exact_lookup_only=TRUE,
            interpolation_allowed=FALSE,
            extrapolation_allowed=FALSE,
            nearest_bucket_allowed=FALSE,
        ),
        _row(
            "runtime_change_blocked",
            verdict="no_runtime_change_in_phase384",
            runtime_change=FALSE,
        ),
        _row(
            "gpu_data_collection_deferred",
            verdict="gpu_collection_deferred_to_later_phase",
            gpu_allowed=FALSE,
            ssh_allowed=FALSE,
        ),
        _row(
            "perfdb_write_blocked",
            verdict="phase384_does_not_write_real_perfdb_data",
            new_perfdb_data=FALSE,
            write_real_data_file=FALSE,
            perf_database=FALSE,
        ),
        _row(
            "default_aic_blocked",
            verdict="default_aic_stays_no_go",
            default_aic_allowed=FALSE,
        ),
        _row(
            "next_phase",
            verdict="next_phase_can_design_or_collect_paired_bucket_gpu_data",
            next_allowed_phase=NEXT_PHASE,
        ),
    ]
    _validate_rows(rows)
    return rows


def _validate_rows(rows: list[dict[str, str]]) -> None:
    expected_row_types = [
        "phase383_prerequisite",
        "existing_vllm_module_rows_preserved",
        "ep8_bucket_set_preserved",
        "fusedmoe_paired_bucket_expansion_planned",
        "future_total_row_count_planned",
        "bucket_128_blocked",
        "lookup_policy_exact_only",
        "runtime_change_blocked",
        "gpu_data_collection_deferred",
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
        _require(row, "write_real_data_file", FALSE, row["row_type"])
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
        by_type["existing_vllm_module_rows_preserved"],
        "existing_module_row_count",
        "14",
        "Phase384 preserve",
    )
    _require(
        by_type["ep8_bucket_set_preserved"],
        "ep8_existing_buckets",
        EP8_BUCKETS_TEXT,
        "Phase384 ep8",
    )
    _require(
        by_type["fusedmoe_paired_bucket_expansion_planned"],
        "fusedmoe_paired_bucket_expansion",
        FUSEDMOE_PAIRED_BUCKETS_TEXT,
        "Phase384 paired buckets",
    )
    _require(by_type["bucket_128_blocked"], "bucket_128_allowed", FALSE, "Phase384 bucket 128")
    _require(by_type["lookup_policy_exact_only"], "exact_lookup_only", TRUE, "Phase384 lookup")


def write_phase384_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_phase384_md(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    paired = next(row for row in rows if row["row_type"] == "fusedmoe_paired_bucket_expansion_planned")
    ep8 = next(row for row in rows if row["row_type"] == "ep8_bucket_set_preserved")
    total = next(row for row in rows if row["row_type"] == "future_total_row_count_planned")
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = f"""# Phase384 Paired Bucket Expansion Spec

Phase384 is a paired bucket expansion spec. It does not run GPU, change runtime code, write real PerfDatabase data, or open Default AIC.

| Gate | Decision |
|---|---|
| existing vLLM module rows | preserve 14 current rows |
| EP8 buckets | {ep8["ep8_existing_buckets"]} |
| future FusedMoE paired buckets | {paired["fusedmoe_paired_bucket_expansion"]} |
| planned new rows | {paired["new_planned_row_count"]} |
| future row count after later materialization | {total["future_total_row_count"]} |
| bucket 128 | blocked |
| lookup policy | exact-only; no interpolation, extrapolation, or nearest lookup |
| Default AIC | No-Go |

Do not write vllm_module_perf.txt in Phase384. The next allowed phase is `{NEXT_PHASE}`.
"""
    path.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase383-csv", type=Path, default=DEFAULT_PHASE383_CSV)
    parser.add_argument("--vllm-module-perf", type=Path, default=DEFAULT_VLLM_MODULE_PERF)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    rows = analyze_phase384_paired_bucket_expansion_spec(
        phase383_csv=args.phase383_csv,
        vllm_module_perf=args.vllm_module_perf,
    )
    write_phase384_csv(args.output_csv, rows)
    write_phase384_md(args.output_md, rows)


if __name__ == "__main__":
    main()
