from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE371_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase371_route_a_perfdb_schema_design_spec.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase372_perfdb_loader_query_change_plan.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase372_perfdb_loader_query_change_plan.md"
)

SOURCE = "phase372_perfdb_loader_query_change_plan"
PHASE371_SOURCE = "phase371_route_a_perfdb_schema_design_spec"
DEFAULT_READINESS = "No-Go"
TRUE = "true"
FALSE = "false"
ALLOWED_BUCKETS = "1/15/16/241/1808/2048/8192"
MODULE_BOUNDARIES = "fusedmoe_runner_compute;ep8_comm_dispatch_combine"
KEY_DIMENSIONS = (
    "model;hardware;vllm_version;topology;bucket_tokens;module_boundary;quant_runtime"
)

FIELDNAMES = [
    "source",
    "row_type",
    "decision",
    "phase371_required",
    "change_kind",
    "loader_query_implemented",
    "runtime_changed",
    "lookup_design",
    "replace_existing_trtllm_path",
    "module_boundaries",
    "module_boundary_extension_allowed",
    "key_dimensions",
    "kernel_source_metadata",
    "kernel_source_lookup_key",
    "allowed_bucket_tokens",
    "excluded_bucket_tokens",
    "bucket_interpolation_allowed",
    "bucket_extrapolation_allowed",
    "perfdb_write_allowed",
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
    "phase371_required": FALSE,
    "change_kind": "",
    "loader_query_implemented": FALSE,
    "runtime_changed": FALSE,
    "lookup_design": "",
    "replace_existing_trtllm_path": FALSE,
    "module_boundaries": "",
    "module_boundary_extension_allowed": FALSE,
    "key_dimensions": "",
    "kernel_source_metadata": "",
    "kernel_source_lookup_key": FALSE,
    "allowed_bucket_tokens": "",
    "excluded_bucket_tokens": "",
    "bucket_interpolation_allowed": FALSE,
    "bucket_extrapolation_allowed": FALSE,
    "perfdb_write_allowed": FALSE,
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


def _require_phase371_schema_spec(path: Path) -> None:
    rows = _read_csv(path)
    if len(rows) != 8:
        raise ValueError(f"Phase371 must have exactly 8 rows, got {len(rows)}")
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
        raise ValueError(f"Phase371 row types expected {expected_row_types}, got {row_types}")

    for row in rows:
        label = row.get("row_type") or "phase371"
        _require(row, "source", PHASE371_SOURCE, label)
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
    _require(rows[1], "module_boundary", "fusedmoe_runner_compute", "module_row_fusedmoe_runner_compute")
    _require(rows[2], "module_boundary", "ep8_comm_dispatch_combine", "module_row_ep8_comm_dispatch_combine")
    _require(rows[3], "key_dimensions", KEY_DIMENSIONS, "key_dimensions")
    _require(rows[4], "kernel_source_metadata", "metadata_only", "kernel_source_policy")
    _require(rows[4], "kernel_source_lookup_key", FALSE, "kernel_source_policy")
    _require(rows[5], "allowed_bucket_tokens", ALLOWED_BUCKETS, "bucket_policy")
    _require(rows[5], "excluded_bucket_tokens", "128", "bucket_policy")
    _require(rows[7], "next_allowed_phase", "phase372_loader_query_change_plan", "next_phase")


def _plan_rows() -> list[dict[str, str]]:
    rows = [
        {
            "row_type": "phase371_schema_spec_prerequisite",
            "decision": "require_phase371_module_schema_spec",
            "phase371_required": TRUE,
        },
        {
            "row_type": "change_scope_plan_only",
            "decision": "plan_only_no_loader_query_implementation",
            "change_kind": "plan_only",
            "loader_query_implemented": FALSE,
            "runtime_changed": FALSE,
        },
        {
            "row_type": "vllm_module_lookup_boundary",
            "decision": "add_vllm_module_level_lookup_design_not_trtllm_replacement",
            "lookup_design": "add_vllm_module_level_lookup",
            "replace_existing_trtllm_path": FALSE,
        },
        {
            "row_type": "allowed_module_boundaries",
            "decision": "lock_phase371_module_boundaries",
            "module_boundaries": MODULE_BOUNDARIES,
            "module_boundary_extension_allowed": FALSE,
        },
        {
            "row_type": "key_dimensions_contract",
            "decision": "reuse_phase371_module_key_dimensions",
            "key_dimensions": KEY_DIMENSIONS,
        },
        {
            "row_type": "kernel_source_metadata_policy",
            "decision": "kernel_source_metadata_only_not_lookup_key",
            "kernel_source_metadata": "metadata_only",
            "kernel_source_lookup_key": FALSE,
        },
        {
            "row_type": "bucket_whitelist",
            "decision": "allow_only_phase124_real_buckets",
            "allowed_bucket_tokens": ALLOWED_BUCKETS,
            "excluded_bucket_tokens": "128",
            "bucket_interpolation_allowed": FALSE,
            "bucket_extrapolation_allowed": FALSE,
        },
        {
            "row_type": "prohibited_actions",
            "decision": "no_write_no_fit_no_default_in_phase372",
            "perfdb_write_allowed": FALSE,
            "default_aic_allowed": FALSE,
        },
        {
            "row_type": "next_phase",
            "decision": "phase373_may_design_loader_query_implementation_spec",
            "next_allowed_phase": "phase373_loader_query_implementation_spec",
        },
    ]
    return [{**COMMON_FIELDS, **row} for row in rows]


def analyze_phase372_perfdb_loader_query_change_plan(
    phase371_csv: Path = DEFAULT_PHASE371_CSV,
) -> list[dict[str, str]]:
    _require_phase371_schema_spec(phase371_csv)
    return _plan_rows()


def _validate_output_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != 9:
        raise ValueError(f"Phase372 output must have exactly 9 rows, got {len(rows)}")
    expected_row_types = [
        "phase371_schema_spec_prerequisite",
        "change_scope_plan_only",
        "vllm_module_lookup_boundary",
        "allowed_module_boundaries",
        "key_dimensions_contract",
        "kernel_source_metadata_policy",
        "bucket_whitelist",
        "prohibited_actions",
        "next_phase",
    ]
    row_types = [row.get("row_type") for row in rows]
    if row_types != expected_row_types:
        raise ValueError(f"row types expected {expected_row_types}, got {row_types}")

    for row in rows:
        label = row.get("row_type") or "phase372"
        _require(row, "source", SOURCE, label)
        _require(row, "loader_query_implemented", FALSE, label)
        _require(row, "runtime_changed", FALSE, label)
        _require(row, "replace_existing_trtllm_path", FALSE, label)
        _require(row, "kernel_source_lookup_key", FALSE, label)
        _require(row, "perfdb_write_allowed", FALSE, label)
        _require(row, "default_aic_allowed", FALSE, label)
        _require(row, "gpu_allowed", FALSE, label)
        _require(row, "curve_fit_allowed", FALSE, label)
        _require(row, "interpolation_allowed", FALSE, label)
        _require(row, "extrapolation_allowed", FALSE, label)
        _require(row, "default_readiness", DEFAULT_READINESS, label)
        _require(row, "diagnostic_only", TRUE, label)
        _require(row, "valid_for_default", FALSE, label)
        _require(row, "perf_database", FALSE, label)

    _require(rows[0], "phase371_required", TRUE, "phase371_schema_spec_prerequisite")
    _require(rows[1], "change_kind", "plan_only", "change_scope_plan_only")
    _require(rows[2], "lookup_design", "add_vllm_module_level_lookup", "vllm_module_lookup_boundary")
    _require(rows[2], "replace_existing_trtllm_path", FALSE, "vllm_module_lookup_boundary")
    _require(rows[3], "module_boundaries", MODULE_BOUNDARIES, "allowed_module_boundaries")
    _require(rows[3], "module_boundary_extension_allowed", FALSE, "allowed_module_boundaries")
    _require(rows[4], "key_dimensions", KEY_DIMENSIONS, "key_dimensions_contract")
    _require(rows[5], "kernel_source_metadata", "metadata_only", "kernel_source_metadata_policy")
    _require(rows[5], "kernel_source_lookup_key", FALSE, "kernel_source_metadata_policy")
    _require(rows[6], "allowed_bucket_tokens", ALLOWED_BUCKETS, "bucket_whitelist")
    _require(rows[6], "excluded_bucket_tokens", "128", "bucket_whitelist")
    _require(rows[6], "bucket_interpolation_allowed", FALSE, "bucket_whitelist")
    _require(rows[6], "bucket_extrapolation_allowed", FALSE, "bucket_whitelist")
    _require(rows[8], "next_allowed_phase", "phase373_loader_query_implementation_spec", "next_phase")


def write_phase372_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_output_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_phase372_md(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_output_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase372 PerfDatabase Loader Query Change Plan",
        "",
        "| Item | Result |",
        "|---|---|",
        "| Verdict | plan-only loader/query change design |",
        "| Default AIC | No-Go |",
        "| PerfDatabase | not written |",
        "| Loader/query implementation | not implemented |",
        "",
        "## Plan Boundary",
        "",
        "- Phase372 depends on the Phase371 module-level schema design spec.",
        "- This is plan-only; it does not implement loader/query behavior and does not change runtime behavior.",
        "- The design is to add vLLM module-level lookup design and it does not replace the existing TRT-LLM path.",
        f"- Allowed module boundaries are `{MODULE_BOUNDARIES}`.",
        f"- Key dimensions remain `{KEY_DIMENSIONS}`.",
        "- The kernel source remains metadata only and is not a lookup key.",
        f"- Allowed buckets are `{ALLOWED_BUCKETS}`; bucket `128` remains excluded.",
        "- This phase does not write PerfDatabase rows, fit curves, interpolate, extrapolate, or enable default AIC.",
        "- Phase373 may design the loader/query implementation spec or minimal implementation plan.",
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
    parser.add_argument("--phase371-csv", type=Path, default=DEFAULT_PHASE371_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    rows = analyze_phase372_perfdb_loader_query_change_plan(args.phase371_csv)
    write_phase372_csv(args.output_csv, rows)
    write_phase372_md(args.output_md, rows)


if __name__ == "__main__":
    main()
