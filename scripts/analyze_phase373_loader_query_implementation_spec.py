from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE372_CSV = (
    REPO_ROOT / "docs/iter_gap_investigation/phase372_perfdb_loader_query_change_plan.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT / "docs/iter_gap_investigation/phase373_loader_query_implementation_spec.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT / "docs/iter_gap_investigation/phase373_loader_query_implementation_spec.md"
)

SOURCE = "phase373_loader_query_implementation_spec"
PHASE372_SOURCE = "phase372_perfdb_loader_query_change_plan"
DEFAULT_READINESS = "No-Go"
TRUE = "true"
FALSE = "false"
NEW_DATA_FILE = "vllm_module_perf.txt"
COMMON_ENTRY = "src/aiconfigurator/sdk/common.py"
PERF_DATABASE_ENTRY = "src/aiconfigurator/sdk/perf_database.py"
NEW_LOADER = "load_vllm_module_data"
NEW_QUERY = "query_vllm_module"
MODULE_BOUNDARIES = "fusedmoe_runner_compute;ep8_comm_dispatch_combine"
KEY_DIMENSIONS = (
    "model;hardware;vllm_version;topology;bucket_tokens;module_boundary;quant_runtime"
)
ALLOWED_BUCKETS = "1/15/16/241/1808/2048/8192"

FIELDNAMES = [
    "source",
    "row_type",
    "decision",
    "phase372_required",
    "new_data_file",
    "existing_data_file_reused",
    "future_code_entry",
    "new_loader",
    "new_loader_implemented",
    "new_query",
    "new_query_implemented",
    "existing_query_replaced",
    "trtllm_sglang_path_changed",
    "key_dimensions",
    "module_boundaries",
    "kernel_source_metadata",
    "kernel_source_lookup_key",
    "allowed_bucket_tokens",
    "excluded_bucket_tokens",
    "bucket_interpolation_allowed",
    "bucket_extrapolation_allowed",
    "perfdb_row_write_allowed",
    "src_change_allowed",
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
    "phase372_required": FALSE,
    "new_data_file": "",
    "existing_data_file_reused": FALSE,
    "future_code_entry": "",
    "new_loader": "",
    "new_loader_implemented": FALSE,
    "new_query": "",
    "new_query_implemented": FALSE,
    "existing_query_replaced": FALSE,
    "trtllm_sglang_path_changed": FALSE,
    "key_dimensions": "",
    "module_boundaries": "",
    "kernel_source_metadata": "",
    "kernel_source_lookup_key": FALSE,
    "allowed_bucket_tokens": "",
    "excluded_bucket_tokens": "",
    "bucket_interpolation_allowed": FALSE,
    "bucket_extrapolation_allowed": FALSE,
    "perfdb_row_write_allowed": FALSE,
    "src_change_allowed": FALSE,
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


def _require_phase372_plan(path: Path) -> None:
    rows = _read_csv(path)
    if len(rows) != 9:
        raise ValueError(f"Phase372 must have exactly 9 rows, got {len(rows)}")
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
        raise ValueError(f"Phase372 row types expected {expected_row_types}, got {row_types}")

    for row in rows:
        label = row.get("row_type") or "phase372"
        _require(row, "source", PHASE372_SOURCE, label)
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

    _require(rows[1], "change_kind", "plan_only", "change_scope_plan_only")
    _require(rows[2], "lookup_design", "add_vllm_module_level_lookup", "vllm_module_lookup_boundary")
    _require(rows[3], "module_boundaries", MODULE_BOUNDARIES, "allowed_module_boundaries")
    _require(rows[4], "key_dimensions", KEY_DIMENSIONS, "key_dimensions_contract")
    _require(rows[5], "kernel_source_metadata", "metadata_only", "kernel_source_metadata_policy")
    _require(rows[6], "allowed_bucket_tokens", ALLOWED_BUCKETS, "bucket_whitelist")
    _require(rows[6], "excluded_bucket_tokens", "128", "bucket_whitelist")
    _require(rows[8], "next_allowed_phase", "phase373_loader_query_implementation_spec", "next_phase")


def _spec_rows() -> list[dict[str, str]]:
    rows = [
        {
            "row_type": "phase372_plan_prerequisite",
            "decision": "require_phase372_loader_query_change_plan",
            "phase372_required": TRUE,
        },
        {
            "row_type": "data_file_contract",
            "decision": "design_new_vllm_module_perf_file_not_moe_perf",
            "new_data_file": NEW_DATA_FILE,
            "existing_data_file_reused": FALSE,
        },
        {
            "row_type": "future_code_entry_common",
            "decision": "future_common_entry_only_no_src_change",
            "future_code_entry": COMMON_ENTRY,
            "src_change_allowed": FALSE,
        },
        {
            "row_type": "future_code_entry_perf_database",
            "decision": "future_perf_database_entry_only_no_src_change",
            "future_code_entry": PERF_DATABASE_ENTRY,
            "src_change_allowed": FALSE,
        },
        {
            "row_type": "loader_contract",
            "decision": "design_load_vllm_module_data_without_implementation",
            "new_loader": NEW_LOADER,
            "new_loader_implemented": FALSE,
        },
        {
            "row_type": "query_contract",
            "decision": "design_query_vllm_module_without_implementation",
            "new_query": NEW_QUERY,
            "new_query_implemented": FALSE,
        },
        {
            "row_type": "existing_path_guard",
            "decision": "do_not_replace_query_moe_or_change_trtllm_sglang_paths",
            "existing_query_replaced": FALSE,
            "trtllm_sglang_path_changed": FALSE,
        },
        {
            "row_type": "key_dimensions_contract",
            "decision": "reuse_phase371_phase372_key_dimensions",
            "key_dimensions": KEY_DIMENSIONS,
        },
        {
            "row_type": "module_boundary_contract",
            "decision": "allow_only_phase371_module_boundaries",
            "module_boundaries": MODULE_BOUNDARIES,
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
            "decision": "no_src_change_no_perfdb_rows_no_fit_no_default",
            "perfdb_row_write_allowed": FALSE,
            "src_change_allowed": FALSE,
            "default_aic_allowed": FALSE,
        },
        {
            "row_type": "next_phase",
            "decision": "phase374_may_do_minimal_loader_query_implementation",
            "next_allowed_phase": "phase374_minimal_loader_query_implementation",
        },
    ]
    return [{**COMMON_FIELDS, **row} for row in rows]


def analyze_phase373_loader_query_implementation_spec(
    phase372_csv: Path = DEFAULT_PHASE372_CSV,
) -> list[dict[str, str]]:
    _require_phase372_plan(phase372_csv)
    return _spec_rows()


def _validate_output_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != 13:
        raise ValueError(f"Phase373 output must have exactly 13 rows, got {len(rows)}")
    expected_row_types = [
        "phase372_plan_prerequisite",
        "data_file_contract",
        "future_code_entry_common",
        "future_code_entry_perf_database",
        "loader_contract",
        "query_contract",
        "existing_path_guard",
        "key_dimensions_contract",
        "module_boundary_contract",
        "kernel_source_metadata_policy",
        "bucket_whitelist",
        "prohibited_actions",
        "next_phase",
    ]
    row_types = [row.get("row_type") for row in rows]
    if row_types != expected_row_types:
        raise ValueError(f"row types expected {expected_row_types}, got {row_types}")

    for row in rows:
        label = row.get("row_type") or "phase373"
        _require(row, "source", SOURCE, label)
        _require(row, "new_loader_implemented", FALSE, label)
        _require(row, "new_query_implemented", FALSE, label)
        _require(row, "existing_query_replaced", FALSE, label)
        _require(row, "trtllm_sglang_path_changed", FALSE, label)
        _require(row, "kernel_source_lookup_key", FALSE, label)
        _require(row, "perfdb_row_write_allowed", FALSE, label)
        _require(row, "src_change_allowed", FALSE, label)
        _require(row, "default_aic_allowed", FALSE, label)
        _require(row, "gpu_allowed", FALSE, label)
        _require(row, "curve_fit_allowed", FALSE, label)
        _require(row, "interpolation_allowed", FALSE, label)
        _require(row, "extrapolation_allowed", FALSE, label)
        _require(row, "default_readiness", DEFAULT_READINESS, label)
        _require(row, "diagnostic_only", TRUE, label)
        _require(row, "valid_for_default", FALSE, label)
        _require(row, "perf_database", FALSE, label)

    _require(rows[0], "phase372_required", TRUE, "phase372_plan_prerequisite")
    _require(rows[1], "new_data_file", NEW_DATA_FILE, "data_file_contract")
    _require(rows[1], "existing_data_file_reused", FALSE, "data_file_contract")
    _require(rows[2], "future_code_entry", COMMON_ENTRY, "future_code_entry_common")
    _require(rows[3], "future_code_entry", PERF_DATABASE_ENTRY, "future_code_entry_perf_database")
    _require(rows[4], "new_loader", NEW_LOADER, "loader_contract")
    _require(rows[4], "new_loader_implemented", FALSE, "loader_contract")
    _require(rows[5], "new_query", NEW_QUERY, "query_contract")
    _require(rows[5], "new_query_implemented", FALSE, "query_contract")
    _require(rows[6], "existing_query_replaced", FALSE, "existing_path_guard")
    _require(rows[6], "trtllm_sglang_path_changed", FALSE, "existing_path_guard")
    _require(rows[7], "key_dimensions", KEY_DIMENSIONS, "key_dimensions_contract")
    _require(rows[8], "module_boundaries", MODULE_BOUNDARIES, "module_boundary_contract")
    _require(rows[9], "kernel_source_metadata", "metadata_only", "kernel_source_metadata_policy")
    _require(rows[9], "kernel_source_lookup_key", FALSE, "kernel_source_metadata_policy")
    _require(rows[10], "allowed_bucket_tokens", ALLOWED_BUCKETS, "bucket_whitelist")
    _require(rows[10], "excluded_bucket_tokens", "128", "bucket_whitelist")
    _require(rows[10], "bucket_interpolation_allowed", FALSE, "bucket_whitelist")
    _require(rows[10], "bucket_extrapolation_allowed", FALSE, "bucket_whitelist")
    _require(rows[12], "next_allowed_phase", "phase374_minimal_loader_query_implementation", "next_phase")


def write_phase373_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_output_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_phase373_md(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_output_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase373 Loader Query Implementation Spec",
        "",
        "| Item | Result |",
        "|---|---|",
        "| Verdict | implementation spec only |",
        "| Default AIC | No-Go |",
        "| PerfDatabase | not written |",
        "| src change | not changed |",
        "",
        "## Implementation Boundary",
        "",
        "- Phase373 depends on the Phase372 loader/query change plan.",
        f"- The new data file design is `{NEW_DATA_FILE}`; it is not placed into `moe_perf.txt`.",
        f"- Future code entry points are `{COMMON_ENTRY}` and `{PERF_DATABASE_ENTRY}`.",
        f"- The new loader name is `{NEW_LOADER}`, but Phase373 does not implement it.",
        f"- The new query name is `{NEW_QUERY}`, but Phase373 does not implement it.",
        "- The design does not replace `query_moe(...)` and does not change existing TRT-LLM/SGLang query paths.",
        f"- Key dimensions remain `{KEY_DIMENSIONS}`.",
        f"- Allowed module boundaries are `{MODULE_BOUNDARIES}`.",
        "- The kernel source remains metadata only and is not a lookup key.",
        f"- Allowed buckets are `{ALLOWED_BUCKETS}`; bucket `128` remains excluded.",
        "- This phase does not change `src/`, write PerfDatabase rows, fit curves, interpolate, extrapolate, or enable default AIC.",
        "- Phase374 may do the minimal loader/query implementation.",
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
    parser.add_argument("--phase372-csv", type=Path, default=DEFAULT_PHASE372_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    rows = analyze_phase373_loader_query_implementation_spec(args.phase372_csv)
    write_phase373_csv(args.output_csv, rows)
    write_phase373_md(args.output_md, rows)


if __name__ == "__main__":
    main()
