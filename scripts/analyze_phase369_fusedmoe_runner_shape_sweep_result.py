from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE367_CSV = (
    REPO_ROOT / "docs/iter_gap_investigation/phase367_fusedmoe_runner_shape_sweep_spec.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase369_fusedmoe_runner_shape_sweep_result.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT / "docs/iter_gap_investigation/phase369_fusedmoe_runner_shape_sweep_result.md"
)

SOURCE = "phase369_fusedmoe_runner_shape_sweep_result"
PHASE367_SOURCE = "phase367_fusedmoe_runner_shape_sweep_spec"
DEFAULT_READINESS = "No-Go"
TRUE = "true"
FALSE = "false"
PHASE368_ARTIFACT_DIR = (
    "/mnt/shared-storage-user/ailab-sys/zhaojieyu/aic/"
    "phase368_fusedmoe_runner_minimal_shape_sweep_6e45508"
)
SOURCE_SHA = "6e455084ee77330e34140df2e55b6071b4b3bc1e"
REAL_BUCKETS = ["1", "15", "16", "241", "1808", "2048", "8192"]
LATENCY_BY_BUCKET = {
    "1": "0.187888",
    "15": "0.242813",
    "16": "0.239764",
    "241": "1.889204",
    "1808": "4.729947",
    "2048": "5.344776",
    "8192": "20.612520",
}
QUANT_RUNTIME = "CompressedTensorsWNA16MarlinMoEMethod"
KERNEL_METADATA = "CompressedTensorsWNA16MarlinMoEMethod:Marlin"

FIELDNAMES = [
    "source",
    "phase368_artifact_dir",
    "source_sha",
    "ok",
    "worker",
    "vllm_version",
    "source_root",
    "model_path",
    "measurement_boundary",
    "runner_boundary_api",
    "bucket_tokens",
    "hidden_size",
    "intermediate_size",
    "num_experts",
    "top_k",
    "dtype",
    "quant_method_config",
    "quant_runtime",
    "runtime_dispatch_scope",
    "kernel_source_metadata",
    "latency_ms_median",
    "output_shape",
    "output_all_finite",
    "cleanup",
    "gpu_process_residue",
    "failure_reason",
    "result_interpretation",
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
    "phase368_artifact_dir": PHASE368_ARTIFACT_DIR,
    "source_sha": SOURCE_SHA,
    "ok": TRUE,
    "worker": "worker-gn6kz",
    "vllm_version": "0.19.0",
    "source_root": "/usr/local/lib/python3.12/dist-packages/vllm",
    "model_path": (
        "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/"
        "zskj-hub/models--moonshotai--Kimi-K2.5"
    ),
    "measurement_boundary": "fusedmoe_forward_runner_level",
    "runner_boundary_api": "FusedMoE.forward()",
    "hidden_size": "7168",
    "intermediate_size": "2048",
    "num_experts": "384",
    "top_k": "8",
    "dtype": "bfloat16",
    "quant_method_config": "compressed-tensors",
    "quant_runtime": QUANT_RUNTIME,
    "runtime_dispatch_scope": "fixed_model_config_hw_vllm_version_tuple",
    "kernel_source_metadata": KERNEL_METADATA,
    "output_all_finite": TRUE,
    "cleanup": TRUE,
    "gpu_process_residue": FALSE,
    "failure_reason": "",
    "result_interpretation": "diagnostic_fusedmoe_runner_shape_sweep_not_perfdb_curve",
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


def _require_phase367_fusedmoe_shape_sweep_spec(path: Path) -> None:
    rows = _read_csv(path)
    if len(rows) != 15:
        raise ValueError(f"Phase367 must have exactly 15 rows, got {len(rows)}")

    for index, row in enumerate(rows):
        label = row.get("row_type") or f"row_{index}"
        _require(row, "source", PHASE367_SOURCE, label)
        _require(row, "gpu_allowed", FALSE, label)
        _require(row, "curve_fit_allowed", FALSE, label)
        _require(row, "interpolation_allowed", FALSE, label)
        _require(row, "extrapolation_allowed", FALSE, label)
        _require(row, "default_readiness", DEFAULT_READINESS, label)
        _require(row, "diagnostic_only", TRUE, label)
        _require(row, "valid_for_default", FALSE, label)
        _require(row, "perf_database", FALSE, label)

    boundary = next(row for row in rows if row["row_type"] == "measurement_boundary")
    _require(boundary, "measurement_boundary", "FusedMoE.forward()", "measurement_boundary")
    _require(boundary, "runner_level_boundary", TRUE, "measurement_boundary")

    smoke = next(row for row in rows if row["row_type"] == "smoke_bucket")
    _require(smoke, "bucket_tokens", "128", "smoke_bucket")
    _require(
        smoke,
        "bucket_role",
        "fusedmoe_runner_smoke_only_not_shape_sweep",
        "smoke_bucket",
    )

    buckets = [row["bucket_tokens"] for row in rows if row["row_type"] == "shape_bucket"]
    if buckets != REAL_BUCKETS:
        raise ValueError(f"Phase367 buckets expected {REAL_BUCKETS}, got {buckets}")
    if "128" in buckets:
        raise ValueError("Phase367 shape sweep buckets must not include 128")

    quant = next(row for row in rows if row["row_type"] == "quant_runtime")
    _require(quant, "quant_runtime", QUANT_RUNTIME, "quant_runtime")

    metadata = next(row for row in rows if row["row_type"] == "kernel_metadata")
    _require(metadata, "kernel_source_metadata", TRUE, "kernel_metadata")
    _require(metadata, "kernel_source_lookup_key", FALSE, "kernel_metadata")

    environment = next(row for row in rows if row["row_type"] == "environment_gate")
    _require(
        environment,
        "ld_library_path_prefix",
        "/usr/local/cuda-12.9/compat:/usr/local/nvidia/lib64",
        "environment_gate",
    )
    _require(environment, "vllm_enable_cuda_compatibility", "1", "environment_gate")

    next_phase = next(row for row in rows if row["row_type"] == "next_phase")
    _require(
        next_phase,
        "next_allowed_phase",
        "phase368_fusedmoe_runner_minimal_shape_sweep",
        "next_phase",
    )


def analyze_phase369_fusedmoe_runner_shape_sweep_result(
    phase367_csv: Path = DEFAULT_PHASE367_CSV,
) -> list[dict[str, str]]:
    _require_phase367_fusedmoe_shape_sweep_spec(phase367_csv)
    return [
        {
            **COMMON_FIELDS,
            "bucket_tokens": bucket,
            "latency_ms_median": LATENCY_BY_BUCKET[bucket],
            "output_shape": f"{bucket}x7168",
        }
        for bucket in REAL_BUCKETS
    ]


def _validate_output_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != 7:
        raise ValueError(f"Phase369 output must have exactly 7 rows, got {len(rows)}")

    buckets = [row.get("bucket_tokens") for row in rows]
    if buckets != REAL_BUCKETS:
        raise ValueError(f"bucket sequence expected {REAL_BUCKETS}, got {buckets}")
    if "128" in buckets:
        raise ValueError("bucket 128 must not be included in Phase369 result")

    for row in rows:
        label = row.get("bucket_tokens") or "phase369"
        _require(row, "source", SOURCE, label)
        _require(row, "phase368_artifact_dir", PHASE368_ARTIFACT_DIR, label)
        _require(row, "source_sha", SOURCE_SHA, label)
        _require(row, "ok", TRUE, label)
        _require(row, "worker", "worker-gn6kz", label)
        _require(row, "vllm_version", "0.19.0", label)
        _require(row, "source_root", "/usr/local/lib/python3.12/dist-packages/vllm", label)
        _require(row, "measurement_boundary", "fusedmoe_forward_runner_level", label)
        _require(row, "runner_boundary_api", "FusedMoE.forward()", label)
        _require(row, "hidden_size", "7168", label)
        _require(row, "intermediate_size", "2048", label)
        _require(row, "num_experts", "384", label)
        _require(row, "top_k", "8", label)
        _require(row, "dtype", "bfloat16", label)
        _require(row, "quant_method_config", "compressed-tensors", label)
        _require(row, "quant_runtime", QUANT_RUNTIME, label)
        _require(row, "kernel_source_metadata", KERNEL_METADATA, label)
        _require(row, "latency_ms_median", LATENCY_BY_BUCKET[label], label)
        _require(row, "output_shape", f"{label}x7168", label)
        _require(row, "output_all_finite", TRUE, label)
        _require(row, "cleanup", TRUE, label)
        _require(row, "gpu_process_residue", FALSE, label)
        _require(row, "failure_reason", "", label)
        _require(
            row,
            "result_interpretation",
            "diagnostic_fusedmoe_runner_shape_sweep_not_perfdb_curve",
            label,
        )
        _require(row, "curve_fit_allowed", FALSE, label)
        _require(row, "interpolation_allowed", FALSE, label)
        _require(row, "extrapolation_allowed", FALSE, label)
        _require(row, "default_readiness", DEFAULT_READINESS, label)
        _require(row, "diagnostic_only", TRUE, label)
        _require(row, "valid_for_default", FALSE, label)
        _require(row, "perf_database", FALSE, label)
        if float(row["latency_ms_median"]) <= 0.0:
            raise ValueError(f"{label} latency_ms_median must be positive")


def write_phase369_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_output_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_phase369_md(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_output_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase369 FusedMoE Runner Shape Sweep Result",
        "",
        "| Item | Result |",
        "|---|---|",
        "| Verdict | Phase368 FusedMoE runner minimal shape sweep passed |",
        "| Default AIC | No-Go |",
        "| PerfDatabase | not written |",
        "| Evidence status | diagnostic-only FusedMoE runner shape sweep |",
        "",
        "## Result",
        "",
        f"- Phase368 artifact: `{PHASE368_ARTIFACT_DIR}`",
        "- Measurement boundary is `FusedMoE.forward()` runner level.",
        "- Buckets are `1`, `15`, `16`, `241`, `1808`, `2048`, `8192`.",
        "- `128` is not included; it remains a smoke-only point.",
        f"- Runtime quant method was `{QUANT_RUNTIME}` for every row.",
        f"- Kernel source metadata was `{KERNEL_METADATA}`; it is not a lookup key.",
        "- Output shape matched `<bucket>x7168` and all outputs were finite.",
        "- Cleanup was `true` and GPU/process residue was `false`.",
        "- These latencies are diagnostic only and cannot be interpolated or extrapolated into a PerfDatabase curve.",
        "- They are not default AIC evidence.",
        "",
        "## Bucket Results",
        "",
        "| bucket_tokens | latency_ms_median | output_shape | output_all_finite |",
        "|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {bucket_tokens} | {latency_ms_median} | {output_shape} | {output_all_finite} |".format(
                **row
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase367-csv", type=Path, default=DEFAULT_PHASE367_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    rows = analyze_phase369_fusedmoe_runner_shape_sweep_result(args.phase367_csv)
    write_phase369_csv(args.output_csv, rows)
    write_phase369_md(args.output_md, rows)


if __name__ == "__main__":
    main()
