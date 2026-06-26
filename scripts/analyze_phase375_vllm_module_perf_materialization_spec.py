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
    / "docs/iter_gap_investigation/phase375_vllm_module_perf_materialization_spec.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase375_vllm_module_perf_materialization_spec.md"
)

SOURCE = "phase375_vllm_module_perf_materialization_spec"
PHASE366_SOURCE = "phase366_ep8_comm_minimal_shape_sweep_result"
PHASE369_SOURCE = "phase369_fusedmoe_runner_shape_sweep_result"
DEFAULT_READINESS = "No-Go"
TRUE = "true"
FALSE = "false"
TARGET_FILE = "vllm_module_perf.txt"
MODEL = "kimi-k2.5"
HARDWARE = "h200_sxm"
TOPOLOGY = "tp4dp2ep8"
QUANT_RUNTIME = "CompressedTensorsWNA16MarlinMoEMethod"
FUSED_MODULE = "fusedmoe_runner_compute"
EP8_MODULE = "ep8_comm_dispatch_combine"
FUSED_KERNEL_METADATA = "CompressedTensorsWNA16MarlinMoEMethod:Marlin"
EP8_KERNEL_METADATA = "allgather_reducescatter:AgRsAll2AllManager"
KEY_COLUMNS = (
    "model;hardware;vllm_version;topology;bucket_tokens;module_boundary;quant_runtime"
)
REAL_BUCKETS = ["1", "15", "16", "241", "1808", "2048", "8192"]
NEXT_PHASE = "phase376_write_vllm_module_perf_rows"

FIELDNAMES = [
    "source",
    "row_type",
    "source_phase",
    "source_latency_field",
    "materialization_target",
    "model",
    "hardware",
    "vllm_version",
    "topology",
    "bucket_tokens",
    "module_boundary",
    "quant_runtime",
    "latency_ms",
    "kernel_source_metadata",
    "kernel_source_lookup_key",
    "key_columns",
    "exact_lookup_only",
    "power_default",
    "energy_formula",
    "duplicate_key_policy",
    "missing_exact_key_policy",
    "allowed_candidate_row_count",
    "write_real_data_allowed",
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

COMMON_OUTPUT_FIELDS = {
    "source": SOURCE,
    "row_type": "candidate_materialization_row",
    "materialization_target": TARGET_FILE,
    "model": MODEL,
    "hardware": HARDWARE,
    "topology": TOPOLOGY,
    "quant_runtime": QUANT_RUNTIME,
    "kernel_source_lookup_key": FALSE,
    "key_columns": KEY_COLUMNS,
    "exact_lookup_only": TRUE,
    "power_default": "0.0",
    "energy_formula": "power*latency",
    "duplicate_key_policy": "fail_fast",
    "missing_exact_key_policy": "fail_fast",
    "allowed_candidate_row_count": "14",
    "write_real_data_allowed": FALSE,
    "next_allowed_phase": NEXT_PHASE,
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


def _require_bucket_sequence(rows: list[dict[str, str]], phase: str) -> None:
    buckets = [row.get("bucket_tokens", "") for row in rows]
    if buckets != REAL_BUCKETS:
        raise ValueError(f"{phase} buckets expected {REAL_BUCKETS}, got {buckets}")
    if "128" in buckets:
        raise ValueError(f"{phase} must not include smoke bucket 128")


def _require_phase366_ep8_comm_pass(path: Path) -> list[dict[str, str]]:
    rows = _read_csv(path)
    if len(rows) != 7:
        raise ValueError(f"Phase366 must have exactly 7 rows, got {len(rows)}")
    _require_bucket_sequence(rows, "Phase366")

    for row in rows:
        label = f"phase366 bucket {row.get('bucket_tokens', '')}"
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
    _require_bucket_sequence(rows, "Phase369")

    for row in rows:
        bucket = row.get("bucket_tokens", "")
        label = f"phase369 bucket {bucket}"
        _require(row, "source", PHASE369_SOURCE, label)
        _require_common_result_flags(row, label)
        _require(row, "measurement_boundary", "fusedmoe_forward_runner_level", label)
        _require(row, "runner_boundary_api", "FusedMoE.forward()", label)
        _require(row, "hidden_size", "7168", label)
        _require(row, "quant_runtime", QUANT_RUNTIME, label)
        _require(row, "kernel_source_metadata", FUSED_KERNEL_METADATA, label)
        _require(row, "output_shape", f"{bucket}x7168", label)
        _require(row, "output_all_finite", TRUE, label)
        if float(row["latency_ms_median"]) <= 0.0:
            raise ValueError(f"{label} latency_ms_median must be positive")
    return rows


def _candidate_row(
    *,
    source_phase: str,
    source_latency_field: str,
    source_row: dict[str, str],
    module_boundary: str,
    latency_ms: str,
    kernel_source_metadata: str,
) -> dict[str, str]:
    row = dict(COMMON_OUTPUT_FIELDS)
    row.update(
        {
            "source_phase": source_phase,
            "source_latency_field": source_latency_field,
            "vllm_version": source_row["vllm_version"],
            "bucket_tokens": source_row["bucket_tokens"],
            "module_boundary": module_boundary,
            "latency_ms": latency_ms,
            "kernel_source_metadata": kernel_source_metadata,
        }
    )
    return row


def _require_output_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != 14:
        raise ValueError(f"Phase375 must output exactly 14 rows, got {len(rows)}")
    if [row.get("bucket_tokens") for row in rows[:7]] != REAL_BUCKETS:
        raise ValueError("Phase375 fusedmoe bucket order changed")
    if [row.get("bucket_tokens") for row in rows[7:]] != REAL_BUCKETS:
        raise ValueError("Phase375 ep8 bucket order changed")

    for row in rows:
        label = f"{row.get('module_boundary')} bucket {row.get('bucket_tokens')}"
        if set(row) != set(FIELDNAMES):
            raise ValueError(f"{label} fieldnames changed")
        _require(row, "source", SOURCE, label)
        _require(row, "row_type", "candidate_materialization_row", label)
        _require(row, "materialization_target", TARGET_FILE, label)
        _require(row, "model", MODEL, label)
        _require(row, "hardware", HARDWARE, label)
        _require(row, "topology", TOPOLOGY, label)
        _require(row, "quant_runtime", QUANT_RUNTIME, label)
        _require(row, "kernel_source_lookup_key", FALSE, label)
        _require(row, "key_columns", KEY_COLUMNS, label)
        _require(row, "exact_lookup_only", TRUE, label)
        _require(row, "power_default", "0.0", label)
        _require(row, "energy_formula", "power*latency", label)
        _require(row, "duplicate_key_policy", "fail_fast", label)
        _require(row, "missing_exact_key_policy", "fail_fast", label)
        _require(row, "allowed_candidate_row_count", "14", label)
        _require(row, "write_real_data_allowed", FALSE, label)
        _require(row, "next_allowed_phase", NEXT_PHASE, label)
        _require(row, "gpu_allowed", FALSE, label)
        _require(row, "curve_fit_allowed", FALSE, label)
        _require(row, "interpolation_allowed", FALSE, label)
        _require(row, "extrapolation_allowed", FALSE, label)
        _require(row, "default_readiness", DEFAULT_READINESS, label)
        _require(row, "diagnostic_only", TRUE, label)
        _require(row, "valid_for_default", FALSE, label)
        _require(row, "perf_database", FALSE, label)
        if row["bucket_tokens"] == "128":
            raise ValueError("Phase375 must not include smoke bucket 128")
        if float(row["latency_ms"]) <= 0.0:
            raise ValueError(f"{label} latency_ms must be positive")

    fused = [row for row in rows if row["module_boundary"] == FUSED_MODULE]
    ep8 = [row for row in rows if row["module_boundary"] == EP8_MODULE]
    if len(fused) != 7 or len(ep8) != 7:
        raise ValueError("Phase375 must contain 7 rows for each module boundary")


def analyze_phase375_vllm_module_perf_materialization_spec(
    phase366_csv: Path = DEFAULT_PHASE366_CSV,
    phase369_csv: Path = DEFAULT_PHASE369_CSV,
) -> list[dict[str, str]]:
    phase366_rows = _require_phase366_ep8_comm_pass(phase366_csv)
    phase369_rows = _require_phase369_fusedmoe_runner_pass(phase369_csv)

    rows: list[dict[str, str]] = []
    for source_row in phase369_rows:
        rows.append(
            _candidate_row(
                source_phase=PHASE369_SOURCE,
                source_latency_field="latency_ms_median",
                source_row=source_row,
                module_boundary=FUSED_MODULE,
                latency_ms=source_row["latency_ms_median"],
                kernel_source_metadata=source_row["kernel_source_metadata"],
            )
        )
    for source_row in phase366_rows:
        rows.append(
            _candidate_row(
                source_phase=PHASE366_SOURCE,
                source_latency_field="latency_ms",
                source_row=source_row,
                module_boundary=EP8_MODULE,
                latency_ms=source_row["latency_ms"],
                kernel_source_metadata=EP8_KERNEL_METADATA,
            )
        )

    _require_output_rows(rows)
    return rows


def write_phase375_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _require_output_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_phase375_md(path: Path, rows: list[dict[str, str]]) -> None:
    _require_output_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = f"""# Phase375 vLLM Module Perf Materialization Spec

Phase375 defines how Phase366 EP8 comm rows and Phase369 FusedMoE runner rows may be materialized into `{TARGET_FILE}` later. It does not write the data file.

| Item | Decision |
|---|---|
| Candidate rows | 14 candidate rows |
| Module boundaries | `{FUSED_MODULE}` and `{EP8_MODULE}` |
| Buckets | `{'/'.join(REAL_BUCKETS)}` |
| Smoke bucket | bucket `128` remains excluded |
| Target file | `{TARGET_FILE}` |
| Hardware key | `{HARDWARE}` |
| Key columns | `{KEY_COLUMNS}` |
| Lookup policy | exact lookup only |
| Kernel source | kernel source remains metadata only |
| Power default | `0.0`; energy is `power*latency` |
| Duplicate key policy | fail fast |
| Missing exact key policy | fail fast |
| Real data write | blocked until Phase376 |
| Default AIC | No-Go |
| PerfDatabase | not written |

## Candidate Rows

| module_boundary | rows | source |
|---|---:|---|
| `{FUSED_MODULE}` | 7 | `{PHASE369_SOURCE}` |
| `{EP8_MODULE}` | 7 | `{PHASE366_SOURCE}` |

## Boundary

Phase375 is a local diagnostic spec. It does not use SSH or GPU, does not create `systems/.../{TARGET_FILE}`, does not interpolate, does not extrapolate, and does not create a PerfDatabase curve.
"""
    path.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Phase375 vLLM module perf materialization spec."
    )
    parser.add_argument("--phase366-csv", type=Path, default=DEFAULT_PHASE366_CSV)
    parser.add_argument("--phase369-csv", type=Path, default=DEFAULT_PHASE369_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    rows = analyze_phase375_vllm_module_perf_materialization_spec(
        args.phase366_csv, args.phase369_csv
    )
    write_phase375_csv(args.output_csv, rows)
    write_phase375_md(args.output_md, rows)


if __name__ == "__main__":
    main()
