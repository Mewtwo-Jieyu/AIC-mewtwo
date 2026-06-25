from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE370_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase370_route_a_perfdb_schema_readiness_gate.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase371_route_a_perfdb_schema_design_spec.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase371_route_a_perfdb_schema_design_spec.md"
)

SOURCE = "phase371_route_a_perfdb_schema_design_spec"
PHASE370_SOURCE = "phase370_route_a_perfdb_schema_readiness_gate"
DEFAULT_READINESS = "No-Go"
TRUE = "true"
FALSE = "false"
ALLOWED_BUCKETS = "1/15/16/241/1808/2048/8192"
KEY_DIMENSIONS = (
    "model;hardware;vllm_version;topology;bucket_tokens;module_boundary;quant_runtime"
)

FIELDNAMES = [
    "source",
    "row_type",
    "decision",
    "row_boundary",
    "module_boundary",
    "schema_row_allowed",
    "key_dimensions",
    "kernel_source_metadata",
    "kernel_source_lookup_key",
    "allowed_bucket_tokens",
    "excluded_bucket_tokens",
    "bucket_interpolation_allowed",
    "bucket_extrapolation_allowed",
    "end_to_end_row_allowed",
    "perfdb_write_allowed",
    "loader_query_change_allowed",
    "default_aic_allowed",
    "next_allowed_phase",
    "gpu_allowed",
    "curve_fit_allowed",
    "interpolation_allowed",
    "extrapolation_allowed",
    "default_readiness",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]

COMMON_FIELDS = {
    "source": SOURCE,
    "decision": "",
    "row_boundary": "",
    "module_boundary": "",
    "schema_row_allowed": FALSE,
    "key_dimensions": "",
    "kernel_source_metadata": "",
    "kernel_source_lookup_key": FALSE,
    "allowed_bucket_tokens": "",
    "excluded_bucket_tokens": "",
    "bucket_interpolation_allowed": FALSE,
    "bucket_extrapolation_allowed": FALSE,
    "end_to_end_row_allowed": FALSE,
    "perfdb_write_allowed": FALSE,
    "loader_query_change_allowed": FALSE,
    "default_aic_allowed": FALSE,
    "next_allowed_phase": "",
    "gpu_allowed": FALSE,
    "curve_fit_allowed": FALSE,
    "interpolation_allowed": FALSE,
    "extrapolation_allowed": FALSE,
    "default_readiness": DEFAULT_READINESS,
    "diagnostic_only": TRUE,
    "valid_for_default": FALSE,
    "perf_database": FALSE,
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


def _require_phase370_schema_readiness(path: Path) -> None:
    rows = _read_csv(path)
    if len(rows) != 8:
        raise ValueError(f"Phase370 must have exactly 8 rows, got {len(rows)}")
    expected_row_types = [
        "phase366_ep8_comm_shape_sweep_passed",
        "phase369_fusedmoe_runner_shape_sweep_passed",
        "shared_bucket_alignment_passed",
        "smoke_bucket_128_excluded",
        "kernel_source_metadata_only",
        "perfdb_schema_design_ready",
        "perfdb_write_blocked",
        "default_aic_blocked",
    ]
    row_types = [row.get("row_type") for row in rows]
    if row_types != expected_row_types:
        raise ValueError(f"Phase370 row types expected {expected_row_types}, got {row_types}")

    for row in rows:
        label = row.get("row_type") or "phase370"
        _require(row, "source", PHASE370_SOURCE, label)
        _require(row, "gpu_allowed", FALSE, label)
        _require(row, "curve_fit_allowed", FALSE, label)
        _require(row, "interpolation_allowed", FALSE, label)
        _require(row, "extrapolation_allowed", FALSE, label)
        _require(row, "default_readiness", DEFAULT_READINESS, label)
        _require(row, "diagnostic_only", TRUE, label)
        _require(row, "valid_for_default", FALSE, label)
        _require(row, "perf_database", FALSE, label)

    alignment = rows[2]
    _require(alignment, "bucket_tokens", ALLOWED_BUCKETS, "shared_bucket_alignment_passed")
    _require(
        alignment,
        "ep8_comm_bucket_tokens",
        ALLOWED_BUCKETS,
        "shared_bucket_alignment_passed",
    )
    _require(
        alignment,
        "fusedmoe_bucket_tokens",
        ALLOWED_BUCKETS,
        "shared_bucket_alignment_passed",
    )
    _require(rows[3], "excluded_bucket_tokens", "128", "smoke_bucket_128_excluded")
    _require(
        rows[4],
        "kernel_source_lookup_key",
        FALSE,
        "kernel_source_metadata_only",
    )
    _require(
        rows[5],
        "perfdb_schema_design_ready",
        TRUE,
        "perfdb_schema_design_ready",
    )
    _require(
        rows[5],
        "next_allowed_phase",
        "phase371_perfdb_schema_design_spec",
        "perfdb_schema_design_ready",
    )
    _require(rows[6], "perfdb_write_allowed", FALSE, "perfdb_write_blocked")
    _require(rows[7], "default_aic_allowed", FALSE, "default_aic_blocked")


def _schema_rows() -> list[dict[str, str]]:
    rows = [
        {
            "row_type": "schema_scope",
            "decision": "module_level_row_only",
            "row_boundary": "module_level_row_not_end_to_end_row",
            "end_to_end_row_allowed": FALSE,
        },
        {
            "row_type": "module_row_fusedmoe_runner_compute",
            "decision": "define_fusedmoe_runner_compute_module_row",
            "row_boundary": "module_level_row_not_end_to_end_row",
            "module_boundary": "fusedmoe_runner_compute",
            "schema_row_allowed": TRUE,
        },
        {
            "row_type": "module_row_ep8_comm_dispatch_combine",
            "decision": "define_ep8_comm_dispatch_combine_module_row",
            "row_boundary": "module_level_row_not_end_to_end_row",
            "module_boundary": "ep8_comm_dispatch_combine",
            "schema_row_allowed": TRUE,
        },
        {
            "row_type": "key_dimensions",
            "decision": "lock_minimal_module_perf_key_dimensions",
            "key_dimensions": KEY_DIMENSIONS,
            "kernel_source_lookup_key": FALSE,
        },
        {
            "row_type": "kernel_source_policy",
            "decision": "kernel_source_metadata_only_not_lookup_key",
            "kernel_source_metadata": "metadata_only",
            "kernel_source_lookup_key": FALSE,
        },
        {
            "row_type": "bucket_policy",
            "decision": "allow_only_phase124_real_buckets",
            "allowed_bucket_tokens": ALLOWED_BUCKETS,
            "excluded_bucket_tokens": "128",
            "bucket_interpolation_allowed": FALSE,
            "bucket_extrapolation_allowed": FALSE,
        },
        {
            "row_type": "prohibited_actions",
            "decision": "no_write_no_fit_no_default_in_phase371",
            "perfdb_write_allowed": FALSE,
            "loader_query_change_allowed": FALSE,
            "default_aic_allowed": FALSE,
        },
        {
            "row_type": "next_phase",
            "decision": "phase372_may_design_loader_query_change_plan",
            "next_allowed_phase": "phase372_loader_query_change_plan",
            "loader_query_change_allowed": FALSE,
        },
    ]
    return [{**COMMON_FIELDS, **row} for row in rows]


def analyze_phase371_route_a_perfdb_schema_design_spec(
    phase370_csv: Path = DEFAULT_PHASE370_CSV,
) -> list[dict[str, str]]:
    _require_phase370_schema_readiness(phase370_csv)
    return _schema_rows()


def _validate_output_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != 8:
        raise ValueError(f"Phase371 output must have exactly 8 rows, got {len(rows)}")
    expected_row_types = [
        "schema_scope",
        "module_row_fusedmoe_runner_compute",
        "module_row_ep8_comm_dispatch_combine",
        "key_dimensions",
        "kernel_source_policy",
        "bucket_policy",
        "prohibited_actions",
        "next_phase",
    ]
    row_types = [row.get("row_type") for row in rows]
    if row_types != expected_row_types:
        raise ValueError(f"row types expected {expected_row_types}, got {row_types}")

    for row in rows:
        label = row.get("row_type") or "phase371"
        _require(row, "source", SOURCE, label)
        _require(row, "gpu_allowed", FALSE, label)
        _require(row, "curve_fit_allowed", FALSE, label)
        _require(row, "interpolation_allowed", FALSE, label)
        _require(row, "extrapolation_allowed", FALSE, label)
        _require(row, "default_readiness", DEFAULT_READINESS, label)
        _require(row, "diagnostic_only", TRUE, label)
        _require(row, "valid_for_default", FALSE, label)
        _require(row, "perf_database", FALSE, label)
        _require(row, "perfdb_write_allowed", FALSE, label)
        _require(row, "loader_query_change_allowed", FALSE, label)
        _require(row, "default_aic_allowed", FALSE, label)

    _require(rows[0], "row_boundary", "module_level_row_not_end_to_end_row", "schema_scope")
    _require(rows[0], "end_to_end_row_allowed", FALSE, "schema_scope")
    _require(
        rows[1],
        "module_boundary",
        "fusedmoe_runner_compute",
        "module_row_fusedmoe_runner_compute",
    )
    _require(rows[1], "schema_row_allowed", TRUE, "module_row_fusedmoe_runner_compute")
    _require(
        rows[2],
        "module_boundary",
        "ep8_comm_dispatch_combine",
        "module_row_ep8_comm_dispatch_combine",
    )
    _require(rows[2], "schema_row_allowed", TRUE, "module_row_ep8_comm_dispatch_combine")
    _require(rows[3], "key_dimensions", KEY_DIMENSIONS, "key_dimensions")
    _require(rows[3], "kernel_source_lookup_key", FALSE, "key_dimensions")
    _require(rows[4], "kernel_source_metadata", "metadata_only", "kernel_source_policy")
    _require(rows[4], "kernel_source_lookup_key", FALSE, "kernel_source_policy")
    _require(rows[5], "allowed_bucket_tokens", ALLOWED_BUCKETS, "bucket_policy")
    _require(rows[5], "excluded_bucket_tokens", "128", "bucket_policy")
    _require(rows[5], "bucket_interpolation_allowed", FALSE, "bucket_policy")
    _require(rows[5], "bucket_extrapolation_allowed", FALSE, "bucket_policy")
    _require(rows[7], "next_allowed_phase", "phase372_loader_query_change_plan", "next_phase")


def write_phase371_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_output_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_phase371_md(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_output_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase371 Route A PerfDatabase Schema Design Spec",
        "",
        "| Item | Result |",
        "|---|---|",
        "| Verdict | module-level row only schema design spec |",
        "| Default AIC | No-Go |",
        "| PerfDatabase | not written |",
        "| Loader/query change | not changed |",
        "",
        "## Schema Boundary",
        "",
        "- The row boundary is module-level row only, not an end-to-end row.",
        "- Required module rows are `fusedmoe_runner_compute` and `ep8_comm_dispatch_combine`.",
        f"- Key dimensions are `{KEY_DIMENSIONS}`.",
        "- The kernel source remains metadata only and is not a lookup key.",
        f"- Allowed buckets are `{ALLOWED_BUCKETS}`; bucket `128` remains excluded.",
        "- This phase does not write PerfDatabase rows, fit curves, interpolate, extrapolate, or enable default AIC.",
        "- Phase372 may design the loader/query change plan, but Phase371 does not change loader/query behavior.",
        "",
        "## Rows",
        "",
        "| row_type | decision |",
        "|---|---|",
    ]
    for row in rows:
        lines.append("| {row_type} | {decision} |".format(**row))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase370-csv", type=Path, default=DEFAULT_PHASE370_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    rows = analyze_phase371_route_a_perfdb_schema_design_spec(args.phase370_csv)
    write_phase371_csv(args.output_csv, rows)
    write_phase371_md(args.output_md, rows)


if __name__ == "__main__":
    main()
