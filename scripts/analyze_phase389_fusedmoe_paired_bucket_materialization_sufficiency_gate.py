from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE388_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase388_fusedmoe_paired_bucket_gpu_run_result.csv"
)
DEFAULT_VLLM_MODULE_PERF = (
    REPO_ROOT
    / "src/aiconfigurator/systems/data/h200_sxm/vllm/0.19.0/vllm_module_perf.txt"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/"
    "phase389_fusedmoe_paired_bucket_materialization_sufficiency_gate.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/"
    "phase389_fusedmoe_paired_bucket_materialization_sufficiency_gate.md"
)

SOURCE = "phase389_fusedmoe_paired_bucket_materialization_sufficiency_gate"
PHASE388_SOURCE = "phase388_fusedmoe_paired_bucket_gpu_run_result"
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
KERNEL_METADATA = "CompressedTensorsWNA16MarlinMoEMethod:Marlin"
EXISTING_BUCKETS = ["1", "15", "16", "241", "1808", "2048", "8192"]
PLANNED_BUCKETS = ["2", "30", "32", "482", "3616", "4096", "16384"]
EXISTING_BUCKETS_JOINED = "/".join(EXISTING_BUCKETS)
PLANNED_BUCKETS_JOINED = "/".join(PLANNED_BUCKETS)
KEY_COLUMNS = "model;hardware;vllm_version;topology;bucket_tokens;module_boundary;quant_runtime"
NEXT_PHASE = "phase390_write_fusedmoe_paired_bucket_vllm_module_perf_rows"

FIELDNAMES = [
    "source",
    "row_type",
    "verdict",
    "gate_type",
    "phase388_prerequisite",
    "materialization_target",
    "existing_row_count",
    "planned_added_row_count",
    "future_row_count",
    "existing_bucket_tokens",
    "planned_bucket_tokens",
    "forbidden_bucket_tokens",
    "existing_module_boundaries",
    "planned_module_boundary",
    "model",
    "hardware",
    "vllm_version",
    "topology",
    "quant_runtime",
    "measurement_boundary",
    "kernel_source_metadata",
    "kernel_source_lookup_key",
    "key_columns",
    "exact_lookup_only",
    "nearest_lookup_allowed",
    "interpolation_allowed",
    "extrapolation_allowed",
    "write_real_data_file",
    "gpu_allowed",
    "ssh_allowed",
    "default_aic_allowed",
    "next_allowed_phase",
    "default_readiness",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]

COMMON_FIELDS = {
    "source": SOURCE,
    "verdict": "",
    "gate_type": "materialization_sufficiency_gate",
    "phase388_prerequisite": "",
    "materialization_target": "vllm_module_perf.txt",
    "existing_row_count": "",
    "planned_added_row_count": "",
    "future_row_count": "",
    "existing_bucket_tokens": "",
    "planned_bucket_tokens": "",
    "forbidden_bucket_tokens": "128",
    "existing_module_boundaries": "",
    "planned_module_boundary": "",
    "model": MODEL,
    "hardware": HARDWARE,
    "vllm_version": VLLM_VERSION,
    "topology": TOPOLOGY,
    "quant_runtime": QUANT_RUNTIME,
    "measurement_boundary": "",
    "kernel_source_metadata": "",
    "kernel_source_lookup_key": FALSE,
    "key_columns": KEY_COLUMNS,
    "exact_lookup_only": TRUE,
    "nearest_lookup_allowed": FALSE,
    "interpolation_allowed": FALSE,
    "extrapolation_allowed": FALSE,
    "write_real_data_file": FALSE,
    "gpu_allowed": FALSE,
    "ssh_allowed": FALSE,
    "default_aic_allowed": FALSE,
    "next_allowed_phase": "",
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


def _require_phase388_result(path: Path) -> list[dict[str, str]]:
    rows = _read_csv(path)
    if len(rows) != 7:
        raise ValueError(f"Phase388 must have exactly 7 rows, got {len(rows)}")
    buckets = [row.get("bucket_tokens", "") for row in rows]
    if buckets != PLANNED_BUCKETS:
        raise ValueError(f"Phase388 buckets expected {PLANNED_BUCKETS}, got {buckets}")
    if "128" in buckets:
        raise ValueError("Phase388 must not include bucket 128")

    for row in rows:
        bucket = row["bucket_tokens"]
        label = f"phase388 bucket {bucket}"
        _require(row, "source", PHASE388_SOURCE, label)
        _require(row, "ok", TRUE, label)
        _require(row, "worker", "worker-r28f2", label)
        _require(row, "hardware", HARDWARE, label)
        _require(row, "vllm_version", VLLM_VERSION, label)
        _require(row, "topology", TOPOLOGY, label)
        _require(row, "measurement_boundary", "fusedmoe_forward_runner_level", label)
        _require(row, "runner_boundary_api", "FusedMoE.forward()", label)
        _require(row, "quant_runtime", QUANT_RUNTIME, label)
        _require(row, "kernel_source_metadata", KERNEL_METADATA, label)
        _require(row, "output_shape", f"{bucket}x7168", label)
        _require(row, "output_all_finite", TRUE, label)
        _require(row, "cleanup", TRUE, label)
        _require(row, "gpu_process_residue", FALSE, label)
        _require(row, "write_real_data_file", FALSE, label)
        _require(row, "default_readiness", DEFAULT_READINESS, label)
        _require(row, "diagnostic_only", TRUE, label)
        _require(row, "valid_for_default", FALSE, label)
        _require(row, "perf_database", FALSE, label)
        if float(row["latency_ms_median"]) <= 0.0:
            raise ValueError(f"{label} latency_ms_median must be positive")
    return rows


def _require_existing_vllm_module_perf(path: Path) -> list[dict[str, str]]:
    rows = _read_csv(path)
    if len(rows) != 14:
        raise ValueError(f"existing vllm_module_perf row count expected 14, got {len(rows)}")

    expected = [
        (bucket, FUSED_MODULE) for bucket in EXISTING_BUCKETS
    ] + [
        (bucket, EP8_MODULE) for bucket in EXISTING_BUCKETS
    ]
    actual = [(row.get("bucket_tokens", ""), row.get("module_boundary", "")) for row in rows]
    if actual != expected:
        raise ValueError(f"existing vllm_module_perf keys changed: {actual}")
    if any(row.get("bucket_tokens") == "128" for row in rows):
        raise ValueError("existing vllm_module_perf must not include bucket 128")

    for row in rows:
        label = f"{row.get('module_boundary')} bucket {row.get('bucket_tokens')}"
        _require(row, "model", MODEL, label)
        _require(row, "hardware", HARDWARE, label)
        _require(row, "vllm_version", VLLM_VERSION, label)
        _require(row, "topology", TOPOLOGY, label)
        _require(row, "quant_runtime", QUANT_RUNTIME, label)
        if float(row["latency"]) <= 0.0:
            raise ValueError(f"{label} latency must be positive")
    return rows


def _row(row_type: str, **overrides: str) -> dict[str, str]:
    row = dict(COMMON_FIELDS)
    row["row_type"] = row_type
    row.update(overrides)
    return row


def analyze_phase389_fusedmoe_paired_bucket_materialization_sufficiency_gate(
    phase388_csv: Path = DEFAULT_PHASE388_CSV,
    vllm_module_perf: Path = DEFAULT_VLLM_MODULE_PERF,
) -> list[dict[str, str]]:
    _require_phase388_result(phase388_csv)
    _require_existing_vllm_module_perf(vllm_module_perf)

    rows = [
        _row(
            "phase388_prerequisite",
            verdict="paired_fusedmoe_gpu_result_passed",
            phase388_prerequisite="fusedmoe_paired_bucket_gpu_run_passed",
            planned_bucket_tokens=PLANNED_BUCKETS_JOINED,
            planned_added_row_count="7",
            measurement_boundary="fusedmoe_forward_runner_level",
        ),
        _row(
            "existing_vllm_module_table_verified",
            verdict="existing_14_rows_preserved",
            existing_row_count="14",
            existing_bucket_tokens=EXISTING_BUCKETS_JOINED,
            existing_module_boundaries=f"{FUSED_MODULE};{EP8_MODULE}",
        ),
        _row(
            "paired_fusedmoe_rows_planned",
            verdict="seven_fusedmoe_paired_rows_can_be_materialized_next",
            planned_added_row_count="7",
            planned_bucket_tokens=PLANNED_BUCKETS_JOINED,
            planned_module_boundary=FUSED_MODULE,
            measurement_boundary="fusedmoe_forward_runner_level",
            kernel_source_metadata=KERNEL_METADATA,
        ),
        _row(
            "future_row_count_verified",
            verdict="future_table_would_have_21_rows",
            existing_row_count="14",
            planned_added_row_count="7",
            future_row_count="21",
        ),
        _row(
            "exact_lookup_policy_locked",
            verdict="exact_lookup_only_no_128_no_nearest",
            planned_bucket_tokens=PLANNED_BUCKETS_JOINED,
        ),
        _row(
            "kernel_source_metadata_only",
            verdict="kernel_source_stays_metadata_not_key",
            kernel_source_metadata=KERNEL_METADATA,
        ),
        _row(
            "phase389_no_write_no_default",
            verdict="phase389_is_gate_only",
        ),
        _row(
            "next_phase",
            verdict="phase390_can_write_seven_paired_rows",
            next_allowed_phase=NEXT_PHASE,
        ),
    ]
    _validate_rows(rows)
    return rows


def _validate_rows(rows: list[dict[str, str]]) -> None:
    expected_types = [
        "phase388_prerequisite",
        "existing_vllm_module_table_verified",
        "paired_fusedmoe_rows_planned",
        "future_row_count_verified",
        "exact_lookup_policy_locked",
        "kernel_source_metadata_only",
        "phase389_no_write_no_default",
        "next_phase",
    ]
    actual_types = [row.get("row_type", "") for row in rows]
    if actual_types != expected_types:
        raise ValueError(f"unexpected Phase389 row order: {actual_types}")

    for row in rows:
        label = row["row_type"]
        if set(row) != set(FIELDNAMES):
            raise ValueError(f"{label} fields do not match FIELDNAMES")
        _require(row, "source", SOURCE, label)
        _require(row, "gate_type", "materialization_sufficiency_gate", label)
        _require(row, "materialization_target", "vllm_module_perf.txt", label)
        _require(row, "model", MODEL, label)
        _require(row, "hardware", HARDWARE, label)
        _require(row, "vllm_version", VLLM_VERSION, label)
        _require(row, "topology", TOPOLOGY, label)
        _require(row, "quant_runtime", QUANT_RUNTIME, label)
        _require(row, "kernel_source_lookup_key", FALSE, label)
        _require(row, "key_columns", KEY_COLUMNS, label)
        _require(row, "exact_lookup_only", TRUE, label)
        _require(row, "nearest_lookup_allowed", FALSE, label)
        _require(row, "interpolation_allowed", FALSE, label)
        _require(row, "extrapolation_allowed", FALSE, label)
        _require(row, "write_real_data_file", FALSE, label)
        _require(row, "gpu_allowed", FALSE, label)
        _require(row, "ssh_allowed", FALSE, label)
        _require(row, "default_aic_allowed", FALSE, label)
        _require(row, "default_readiness", DEFAULT_READINESS, label)
        _require(row, "diagnostic_only", TRUE, label)
        _require(row, "valid_for_default", FALSE, label)
        _require(row, "perf_database", FALSE, label)
        if "128" in row.get("planned_bucket_tokens", "").split("/"):
            raise ValueError(f"{label} planned_bucket_tokens must not include 128")

    by_type = {row["row_type"]: row for row in rows}
    _require(
        by_type["phase388_prerequisite"],
        "phase388_prerequisite",
        "fusedmoe_paired_bucket_gpu_run_passed",
        "phase388_prerequisite",
    )
    _require(
        by_type["phase388_prerequisite"],
        "planned_bucket_tokens",
        PLANNED_BUCKETS_JOINED,
        "phase388_prerequisite",
    )
    _require(
        by_type["existing_vllm_module_table_verified"],
        "existing_row_count",
        "14",
        "existing rows",
    )
    _require(
        by_type["existing_vllm_module_table_verified"],
        "existing_bucket_tokens",
        EXISTING_BUCKETS_JOINED,
        "existing rows",
    )
    _require(
        by_type["paired_fusedmoe_rows_planned"],
        "planned_bucket_tokens",
        PLANNED_BUCKETS_JOINED,
        "planned rows",
    )
    _require(
        by_type["paired_fusedmoe_rows_planned"],
        "planned_added_row_count",
        "7",
        "planned rows",
    )
    _require(
        by_type["future_row_count_verified"],
        "future_row_count",
        "21",
        "future rows",
    )
    _require(by_type["next_phase"], "next_allowed_phase", NEXT_PHASE, "next phase")


def write_phase389_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_phase389_md(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase389 FusedMoE Paired Bucket Materialization Sufficiency Gate",
        "",
        "| Item | Result |",
        "|---|---|",
        "| Verdict | materialization sufficiency gate passed |",
        "| existing rows | 14 |",
        "| planned added rows | 7 |",
        "| future row count | 21 |",
        "| Default AIC | No-Go |",
        "| PerfDatabase | not written |",
        "",
        "## Decision",
        "",
        "- Phase388 has seven diagnostic FusedMoE paired bucket rows with finite output and clean cleanup.",
        "- The existing 14 rows are preserved: seven FusedMoE original rows and seven EP8 comm rows.",
        "- Phase390 may add only the seven FusedMoE paired rows for buckets `2`, `30`, `32`, `482`, `3616`, `4096`, `16384`.",
        "- Bucket `128` remains forbidden.",
        "- Lookup stays exact-only; no nearest lookup, interpolation, or extrapolation is allowed.",
        "- Kernel source remains metadata only and is not part of the lookup key.",
        "- Phase389 does not write `vllm_module_perf.txt`, write PerfDatabase rows, use GPU, SSH, or open Default AIC.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase388-csv", type=Path, default=DEFAULT_PHASE388_CSV)
    parser.add_argument("--vllm-module-perf", type=Path, default=DEFAULT_VLLM_MODULE_PERF)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    rows = analyze_phase389_fusedmoe_paired_bucket_materialization_sufficiency_gate(
        args.phase388_csv,
        args.vllm_module_perf,
    )
    write_phase389_csv(args.output_csv, rows)
    write_phase389_md(args.output_md, rows)


if __name__ == "__main__":
    main()
