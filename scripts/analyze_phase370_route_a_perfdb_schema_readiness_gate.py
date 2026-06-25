from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE366_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase366_ep8_comm_minimal_shape_sweep_result.csv"
)
DEFAULT_PHASE369_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase369_fusedmoe_runner_shape_sweep_result.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase370_route_a_perfdb_schema_readiness_gate.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase370_route_a_perfdb_schema_readiness_gate.md"
)

SOURCE = "phase370_route_a_perfdb_schema_readiness_gate"
PHASE366_SOURCE = "phase366_ep8_comm_minimal_shape_sweep_result"
PHASE369_SOURCE = "phase369_fusedmoe_runner_shape_sweep_result"
DEFAULT_READINESS = "No-Go"
TRUE = "true"
FALSE = "false"
REAL_BUCKETS = ["1", "15", "16", "241", "1808", "2048", "8192"]
REAL_BUCKETS_TEXT = "/".join(REAL_BUCKETS)
KERNEL_METADATA = "CompressedTensorsWNA16MarlinMoEMethod:Marlin"

FIELDNAMES = [
    "source",
    "row_type",
    "verdict",
    "input_evidence",
    "input_status",
    "bucket_tokens",
    "ep8_comm_bucket_tokens",
    "fusedmoe_bucket_tokens",
    "excluded_bucket_tokens",
    "kernel_source_metadata",
    "kernel_source_lookup_key",
    "perfdb_schema_design_ready",
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
    "verdict": "",
    "input_evidence": "",
    "input_status": "",
    "bucket_tokens": "",
    "ep8_comm_bucket_tokens": "",
    "fusedmoe_bucket_tokens": "",
    "excluded_bucket_tokens": "",
    "kernel_source_metadata": "",
    "kernel_source_lookup_key": "",
    "perfdb_schema_design_ready": FALSE,
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


def _require_common_result_flags(row: dict[str, str], label: str) -> None:
    _require(row, "ok", TRUE, label)
    _require(row, "worker", "worker-gn6kz", label)
    _require(row, "vllm_version", "0.19.0", label)
    _require(row, "cleanup", TRUE, label)
    _require(row, "gpu_process_residue", FALSE, label)
    _require(row, "curve_fit_allowed", FALSE, label)
    _require(row, "interpolation_allowed", FALSE, label)
    _require(row, "extrapolation_allowed", FALSE, label)
    _require(row, "default_readiness", DEFAULT_READINESS, label)
    _require(row, "diagnostic_only", TRUE, label)
    _require(row, "valid_for_default", FALSE, label)
    _require(row, "perf_database", FALSE, label)


def _require_phase366_ep8_comm_pass(path: Path) -> list[dict[str, str]]:
    rows = _read_csv(path)
    if len(rows) != 7:
        raise ValueError(f"Phase366 must have exactly 7 rows, got {len(rows)}")
    buckets = [row.get("bucket_tokens", "") for row in rows]
    if buckets != REAL_BUCKETS:
        raise ValueError(f"Phase366 buckets expected {REAL_BUCKETS}, got {buckets}")
    if "128" in buckets:
        raise ValueError("Phase366 must not include smoke bucket 128")

    for row in rows:
        label = row.get("bucket_tokens") or "phase366"
        _require(row, "source", PHASE366_SOURCE, label)
        _require_common_result_flags(row, label)
        _require(
            row,
            "measurement_boundary",
            "vllm_ep_group_dispatch_router_logits_plus_combine",
            label,
        )
        _require(row, "backend", "allgather_reducescatter", label)
        _require(row, "manager", "AgRsAll2AllManager", label)
        _require(row, "rank_error", "0", label)
        if float(row["latency_ms"]) <= 0.0:
            raise ValueError(f"{label} latency_ms must be positive")
    return rows


def _require_phase369_fusedmoe_runner_pass(path: Path) -> list[dict[str, str]]:
    rows = _read_csv(path)
    if len(rows) != 7:
        raise ValueError(f"Phase369 must have exactly 7 rows, got {len(rows)}")
    buckets = [row.get("bucket_tokens", "") for row in rows]
    if buckets != REAL_BUCKETS:
        raise ValueError(f"Phase369 buckets expected {REAL_BUCKETS}, got {buckets}")
    if "128" in buckets:
        raise ValueError("Phase369 must not include smoke bucket 128")

    for row in rows:
        label = row.get("bucket_tokens") or "phase369"
        _require(row, "source", PHASE369_SOURCE, label)
        _require_common_result_flags(row, label)
        _require(row, "measurement_boundary", "fusedmoe_forward_runner_level", label)
        _require(row, "runner_boundary_api", "FusedMoE.forward()", label)
        _require(row, "quant_runtime", "CompressedTensorsWNA16MarlinMoEMethod", label)
        _require(row, "kernel_source_metadata", KERNEL_METADATA, label)
        _require(row, "output_shape", f"{label}x7168", label)
        _require(row, "output_all_finite", TRUE, label)
        if float(row["latency_ms_median"]) <= 0.0:
            raise ValueError(f"{label} latency_ms_median must be positive")
    return rows


def _gate_rows() -> list[dict[str, str]]:
    rows = [
        {
            "row_type": "phase366_ep8_comm_shape_sweep_passed",
            "verdict": "schema_design_input_accepted",
            "input_evidence": PHASE366_SOURCE,
            "input_status": "passed_7_bucket_diagnostic",
            "bucket_tokens": REAL_BUCKETS_TEXT,
            "ep8_comm_bucket_tokens": REAL_BUCKETS_TEXT,
        },
        {
            "row_type": "phase369_fusedmoe_runner_shape_sweep_passed",
            "verdict": "schema_design_input_accepted",
            "input_evidence": PHASE369_SOURCE,
            "input_status": "passed_7_bucket_diagnostic",
            "bucket_tokens": REAL_BUCKETS_TEXT,
            "fusedmoe_bucket_tokens": REAL_BUCKETS_TEXT,
        },
        {
            "row_type": "shared_bucket_alignment_passed",
            "verdict": "shared_bucket_set_aligned",
            "bucket_tokens": REAL_BUCKETS_TEXT,
            "ep8_comm_bucket_tokens": REAL_BUCKETS_TEXT,
            "fusedmoe_bucket_tokens": REAL_BUCKETS_TEXT,
        },
        {
            "row_type": "smoke_bucket_128_excluded",
            "verdict": "smoke_bucket_excluded_from_schema_readiness",
            "excluded_bucket_tokens": "128",
        },
        {
            "row_type": "kernel_source_metadata_only",
            "verdict": "metadata_only_not_lookup_key",
            "kernel_source_metadata": KERNEL_METADATA,
            "kernel_source_lookup_key": FALSE,
        },
        {
            "row_type": "perfdb_schema_design_ready",
            "verdict": "ready_for_phase371_schema_design_spec",
            "perfdb_schema_design_ready": TRUE,
            "next_allowed_phase": "phase371_perfdb_schema_design_spec",
        },
        {
            "row_type": "perfdb_write_blocked",
            "verdict": "blocked_no_perfdb_row_or_curve",
            "perfdb_write_allowed": FALSE,
        },
        {
            "row_type": "default_aic_blocked",
            "verdict": "blocked_default_aic_no_go",
            "default_aic_allowed": FALSE,
        },
    ]
    return [{**COMMON_FIELDS, **row} for row in rows]


def analyze_phase370_route_a_perfdb_schema_readiness_gate(
    phase366_csv: Path = DEFAULT_PHASE366_CSV,
    phase369_csv: Path = DEFAULT_PHASE369_CSV,
) -> list[dict[str, str]]:
    phase366_rows = _require_phase366_ep8_comm_pass(phase366_csv)
    phase369_rows = _require_phase369_fusedmoe_runner_pass(phase369_csv)
    phase366_buckets = [row["bucket_tokens"] for row in phase366_rows]
    phase369_buckets = [row["bucket_tokens"] for row in phase369_rows]
    if phase366_buckets != phase369_buckets:
        raise ValueError(
            f"shared bucket alignment expected {phase366_buckets}, got {phase369_buckets}"
        )
    return _gate_rows()


def _validate_output_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != 8:
        raise ValueError(f"Phase370 output must have exactly 8 rows, got {len(rows)}")
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
        raise ValueError(f"row types expected {expected_row_types}, got {row_types}")

    for row in rows:
        label = row.get("row_type") or "phase370"
        _require(row, "source", SOURCE, label)
        _require(row, "gpu_allowed", FALSE, label)
        _require(row, "curve_fit_allowed", FALSE, label)
        _require(row, "interpolation_allowed", FALSE, label)
        _require(row, "extrapolation_allowed", FALSE, label)
        _require(row, "default_readiness", DEFAULT_READINESS, label)
        _require(row, "diagnostic_only", TRUE, label)
        _require(row, "valid_for_default", FALSE, label)
        _require(row, "perf_database", FALSE, label)

    _require(rows[0], "input_evidence", PHASE366_SOURCE, rows[0]["row_type"])
    _require(rows[0], "input_status", "passed_7_bucket_diagnostic", rows[0]["row_type"])
    _require(rows[1], "input_evidence", PHASE369_SOURCE, rows[1]["row_type"])
    _require(rows[1], "input_status", "passed_7_bucket_diagnostic", rows[1]["row_type"])
    _require(rows[2], "bucket_tokens", REAL_BUCKETS_TEXT, rows[2]["row_type"])
    _require(rows[2], "ep8_comm_bucket_tokens", REAL_BUCKETS_TEXT, rows[2]["row_type"])
    _require(rows[2], "fusedmoe_bucket_tokens", REAL_BUCKETS_TEXT, rows[2]["row_type"])
    _require(rows[3], "excluded_bucket_tokens", "128", rows[3]["row_type"])
    _require(rows[4], "kernel_source_metadata", KERNEL_METADATA, rows[4]["row_type"])
    _require(rows[4], "kernel_source_lookup_key", FALSE, rows[4]["row_type"])
    _require(rows[5], "perfdb_schema_design_ready", TRUE, rows[5]["row_type"])
    _require(
        rows[5],
        "next_allowed_phase",
        "phase371_perfdb_schema_design_spec",
        rows[5]["row_type"],
    )
    _require(rows[6], "perfdb_write_allowed", FALSE, rows[6]["row_type"])
    _require(rows[7], "default_aic_allowed", FALSE, rows[7]["row_type"])


def write_phase370_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_output_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_phase370_md(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_output_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase370 Route A PerfDatabase Schema Readiness Gate",
        "",
        "| Item | Result |",
        "|---|---|",
        "| Verdict | ready for Phase371 PerfDatabase schema design spec |",
        "| PerfDatabase write | not ready for PerfDatabase write |",
        "| Default AIC | No-Go |",
        "| PerfDatabase | not written |",
        "",
        "## Gate Result",
        "",
        "- Phase366 EP8 comm 7-bucket result is accepted as a schema design input.",
        "- Phase369 FusedMoE runner 7-bucket result is accepted as a schema design input.",
        f"- Shared buckets are exactly `{REAL_BUCKETS_TEXT}`.",
        "- Bucket `128` remains smoke-only and is excluded from schema readiness.",
        f"- Kernel source `{KERNEL_METADATA}` remains metadata only, not a lookup key.",
        "- Phase371 may write a PerfDatabase schema design spec.",
        "- This phase still cannot write PerfDatabase rows, fit curves, interpolate, extrapolate, or enable default AIC.",
        "",
        "## Rows",
        "",
        "| row_type | verdict |",
        "|---|---|",
    ]
    for row in rows:
        lines.append("| {row_type} | {verdict} |".format(**row))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase366-csv", type=Path, default=DEFAULT_PHASE366_CSV)
    parser.add_argument("--phase369-csv", type=Path, default=DEFAULT_PHASE369_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    rows = analyze_phase370_route_a_perfdb_schema_readiness_gate(
        args.phase366_csv,
        args.phase369_csv,
    )
    write_phase370_csv(args.output_csv, rows)
    write_phase370_md(args.output_md, rows)


if __name__ == "__main__":
    main()
