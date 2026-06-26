from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE376_TABLE = (
    REPO_ROOT
    / "src/aiconfigurator/systems/data/h200_sxm/vllm/0.19.0/vllm_module_perf.txt"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase377_vllm_module_runtime_binding_spec.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase377_vllm_module_runtime_binding_spec.md"
)

SOURCE = "phase377_vllm_module_runtime_binding_spec"
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
MODULE_BOUNDARIES = f"{FUSED_MODULE};{EP8_MODULE}"
REAL_BUCKETS = ["1", "15", "16", "241", "1808", "2048", "8192"]
REAL_BUCKETS_TEXT = "/".join(REAL_BUCKETS)
NEXT_PHASE = "phase378_operations_runtime_binding_minimal_implementation"

FIELDNAMES = [
    "source",
    "row_type",
    "decision",
    "phase376_table",
    "phase376_exact_key_count",
    "runtime_operation",
    "module_boundary",
    "module_boundaries",
    "hardware_source",
    "hardware_required",
    "vllm_version_source",
    "vllm_version_required",
    "model_required",
    "topology_required",
    "quant_runtime_required",
    "allowed_bucket_tokens",
    "excluded_bucket_tokens",
    "bucket_selection",
    "nearest_bucket_allowed",
    "kernel_source_metadata",
    "kernel_source_lookup_key",
    "runtime_changed",
    "operations_py_changed",
    "new_data_rows_allowed",
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
    "phase376_table": "",
    "phase376_exact_key_count": "",
    "runtime_operation": "",
    "module_boundary": "",
    "module_boundaries": "",
    "hardware_source": "",
    "hardware_required": "",
    "vllm_version_source": "",
    "vllm_version_required": "",
    "model_required": "",
    "topology_required": "",
    "quant_runtime_required": "",
    "allowed_bucket_tokens": "",
    "excluded_bucket_tokens": "",
    "bucket_selection": "",
    "nearest_bucket_allowed": FALSE,
    "kernel_source_metadata": "",
    "kernel_source_lookup_key": FALSE,
    "runtime_changed": FALSE,
    "operations_py_changed": FALSE,
    "new_data_rows_allowed": FALSE,
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


def _require_phase376_table(path: Path) -> list[dict[str, str]]:
    rows = _read_csv(path)
    if len(rows) != 14:
        raise ValueError(f"Phase376 table must have exactly 14 rows, got {len(rows)}")

    keys = set()
    seen_by_module: dict[str, list[str]] = {FUSED_MODULE: [], EP8_MODULE: []}
    for row in rows:
        label = f"phase376 row {len(keys) + 1}"
        _require(row, "model", MODEL, label)
        _require(row, "hardware", HARDWARE, label)
        _require(row, "vllm_version", VLLM_VERSION, label)
        _require(row, "topology", TOPOLOGY, label)
        _require(row, "quant_runtime", QUANT_RUNTIME, label)
        module_boundary = row.get("module_boundary", "")
        if module_boundary not in seen_by_module:
            raise ValueError(f"Phase376 module_boundary unexpected: {module_boundary!r}")
        bucket = row.get("bucket_tokens", "")
        seen_by_module[module_boundary].append(bucket)
        if bucket == "128" or bucket not in REAL_BUCKETS:
            raise ValueError(f"Phase376 buckets expected {REAL_BUCKETS}, got {bucket!r}")
        key = (
            row["model"],
            row["hardware"],
            row["vllm_version"],
            row["topology"],
            bucket,
            module_boundary,
            row["quant_runtime"],
        )
        if key in keys:
            raise ValueError(f"duplicate Phase376 exact key: {key}")
        keys.add(key)
        if float(row["latency"]) <= 0.0:
            raise ValueError(f"{label} latency must be positive")
        _require(row, "power", "0.0", label)

    for module_boundary, buckets in seen_by_module.items():
        if buckets != REAL_BUCKETS:
            raise ValueError(
                f"Phase376 buckets for {module_boundary} expected {REAL_BUCKETS}, got {buckets}"
            )
    return rows


def _row(row_type: str, **overrides: str) -> dict[str, str]:
    row = dict(COMMON_FIELDS)
    row["row_type"] = row_type
    row.update(overrides)
    return row


def _spec_rows(phase376_table: Path) -> list[dict[str, str]]:
    table = str(phase376_table)
    return [
        _row(
            "phase376_table_prerequisite",
            decision="require_phase376_14_exact_keys",
            phase376_table=table,
            phase376_exact_key_count="14",
            module_boundaries=MODULE_BOUNDARIES,
            allowed_bucket_tokens=REAL_BUCKETS_TEXT,
        ),
        _row(
            "moe_query_runtime_binding",
            decision="bind_moe_query_to_fusedmoe_runner_compute",
            runtime_operation="MoE.query(...)",
            module_boundary=FUSED_MODULE,
        ),
        _row(
            "moe_dispatch_query_runtime_binding",
            decision="bind_moe_dispatch_query_to_ep8_comm_dispatch_combine",
            runtime_operation="MoEDispatch.query(...)",
            module_boundary=EP8_MODULE,
        ),
        _row(
            "runtime_key_source_contract",
            decision="derive_exact_key_from_database_and_runtime_scope",
            hardware_source="database.system",
            hardware_required=HARDWARE,
            vllm_version_source="database.version",
            vllm_version_required=VLLM_VERSION,
            model_required=MODEL,
            topology_required=TOPOLOGY,
            quant_runtime_required=QUANT_RUNTIME,
        ),
        _row(
            "bucket_exact_only_contract",
            decision="runtime_bucket_must_match_phase376_bucket_whitelist",
            allowed_bucket_tokens=REAL_BUCKETS_TEXT,
            excluded_bucket_tokens="128",
            bucket_selection="exact_only",
            nearest_bucket_allowed=FALSE,
        ),
        _row(
            "kernel_source_metadata_policy",
            decision="keep_kernel_source_out_of_lookup_key",
            kernel_source_metadata="metadata_only",
            kernel_source_lookup_key=FALSE,
        ),
        _row(
            "prohibited_runtime_behaviors",
            decision="no_runtime_change_no_nearest_no_fit_no_default_aic",
            bucket_selection="exact_only",
            nearest_bucket_allowed=FALSE,
            default_aic_allowed=FALSE,
        ),
        _row(
            "next_phase",
            decision="phase378_may_minimally_change_operations_py",
            next_allowed_phase=NEXT_PHASE,
        ),
    ]


def _require_output_rows(rows: list[dict[str, str]]) -> None:
    expected_row_types = [
        "phase376_table_prerequisite",
        "moe_query_runtime_binding",
        "moe_dispatch_query_runtime_binding",
        "runtime_key_source_contract",
        "bucket_exact_only_contract",
        "kernel_source_metadata_policy",
        "prohibited_runtime_behaviors",
        "next_phase",
    ]
    row_types = [row.get("row_type") for row in rows]
    if row_types != expected_row_types:
        raise ValueError(f"Phase377 row types expected {expected_row_types}, got {row_types}")

    for row in rows:
        label = row.get("row_type") or "phase377"
        if set(row) != set(FIELDNAMES):
            raise ValueError(f"{label} fieldnames changed")
        _require(row, "source", SOURCE, label)
        _require(row, "runtime_changed", FALSE, label)
        _require(row, "operations_py_changed", FALSE, label)
        _require(row, "new_data_rows_allowed", FALSE, label)
        _require(row, "default_aic_allowed", FALSE, label)
        _require(row, "gpu_allowed", FALSE, label)
        _require(row, "curve_fit_allowed", FALSE, label)
        _require(row, "interpolation_allowed", FALSE, label)
        _require(row, "extrapolation_allowed", FALSE, label)
        _require(row, "default_readiness", DEFAULT_READINESS, label)
        _require(row, "diagnostic_only", TRUE, label)
        _require(row, "valid_for_default", FALSE, label)
        _require(row, "perf_database", FALSE, label)
        if row.get("excluded_bucket_tokens") == "128" and row.get("bucket_selection") != "exact_only":
            raise ValueError(f"{label} must use exact_only bucket selection")

    _require(rows[0], "phase376_exact_key_count", "14", "phase376_table_prerequisite")
    _require(rows[0], "module_boundaries", MODULE_BOUNDARIES, "phase376_table_prerequisite")
    _require(rows[1], "runtime_operation", "MoE.query(...)", "moe_query_runtime_binding")
    _require(rows[1], "module_boundary", FUSED_MODULE, "moe_query_runtime_binding")
    _require(
        rows[2],
        "runtime_operation",
        "MoEDispatch.query(...)",
        "moe_dispatch_query_runtime_binding",
    )
    _require(rows[2], "module_boundary", EP8_MODULE, "moe_dispatch_query_runtime_binding")
    _require(rows[3], "hardware_source", "database.system", "runtime_key_source_contract")
    _require(rows[3], "hardware_required", HARDWARE, "runtime_key_source_contract")
    _require(rows[3], "vllm_version_source", "database.version", "runtime_key_source_contract")
    _require(rows[3], "vllm_version_required", VLLM_VERSION, "runtime_key_source_contract")
    _require(rows[4], "allowed_bucket_tokens", REAL_BUCKETS_TEXT, "bucket_exact_only_contract")
    _require(rows[4], "excluded_bucket_tokens", "128", "bucket_exact_only_contract")
    _require(rows[4], "nearest_bucket_allowed", FALSE, "bucket_exact_only_contract")
    _require(rows[5], "kernel_source_metadata", "metadata_only", "kernel_source_metadata_policy")
    _require(rows[5], "kernel_source_lookup_key", FALSE, "kernel_source_metadata_policy")


def analyze_phase377_vllm_module_runtime_binding_spec(
    phase376_table: Path = DEFAULT_PHASE376_TABLE,
) -> list[dict[str, str]]:
    _require_phase376_table(phase376_table)
    rows = _spec_rows(phase376_table)
    _require_output_rows(rows)
    return rows


def write_phase377_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _require_output_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_phase377_md(path: Path, rows: list[dict[str, str]]) -> None:
    _require_output_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = f"""# Phase377 vLLM Module Runtime Binding Spec

Phase377 defines how `vllm_module_perf.txt` may be bound into runtime later. It does not modify `operations.py`.

| Item | Decision |
|---|---|
| Phase376 table | 14 exact keys required |
| MoE binding | `MoE.query(...)` -> `{FUSED_MODULE}` |
| EP8 comm binding | `MoEDispatch.query(...)` -> `{EP8_MODULE}` |
| Hardware source | `database.system`, must be `{HARDWARE}` |
| vLLM version source | `database.version`, must be `{VLLM_VERSION}` |
| Topology | `{TOPOLOGY}` |
| Quant runtime | `{QUANT_RUNTIME}` |
| Buckets | `{REAL_BUCKETS_TEXT}` |
| Rejected bucket | bucket `128` remains rejected |
| Bucket policy | exact only; no nearest bucket, interpolation, or extrapolation |
| Kernel source | metadata only, not a lookup key |
| operations.py | unchanged |
| Default AIC | No-Go |

Phase378 may make a minimal `operations.py` binding. That implementation must fail fast when a runtime token bucket is not one of `{REAL_BUCKETS_TEXT}`.
"""
    path.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Phase377 vLLM module runtime binding spec."
    )
    parser.add_argument("--phase376-table", type=Path, default=DEFAULT_PHASE376_TABLE)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    rows = analyze_phase377_vllm_module_runtime_binding_spec(args.phase376_table)
    write_phase377_csv(args.output_csv, rows)
    write_phase377_md(args.output_md, rows)


if __name__ == "__main__":
    main()
