from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE366_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase366_ep8_comm_minimal_shape_sweep_result.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT / "docs/iter_gap_investigation/phase367_fusedmoe_runner_shape_sweep_spec.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT / "docs/iter_gap_investigation/phase367_fusedmoe_runner_shape_sweep_spec.md"
)

SOURCE = "phase367_fusedmoe_runner_shape_sweep_spec"
PHASE366_SOURCE = "phase366_ep8_comm_minimal_shape_sweep_result"
DEFAULT_READINESS = "No-Go"
TRUE = "true"
FALSE = "false"
REAL_BUCKETS = ["1", "15", "16", "241", "1808", "2048", "8192"]
FUTURE_GPU_SSH = (
    "ssh -CAXY "
    "ws-faaf0de74ef9a14d-worker-gn6kz.zhaojieyu+root.ailab-sys.pod"
    "@h.pjlab.org.cn"
)
CUDA_COMPAT_LD_LIBRARY_PATH = "/usr/local/cuda-12.9/compat:/usr/local/nvidia/lib64"
QUANT_RUNTIME = "CompressedTensorsWNA16MarlinMoEMethod"

FIELDNAMES = [
    "source",
    "row_type",
    "bucket_tokens",
    "bucket_role",
    "decision",
    "phase366_prerequisite",
    "measurement_boundary",
    "runner_level_boundary",
    "quant_runtime",
    "kernel_source_metadata",
    "kernel_source_lookup_key",
    "ld_library_path_prefix",
    "vllm_enable_cuda_compatibility",
    "stop_rule",
    "next_allowed_phase",
    "future_gpu_ssh",
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
    "bucket_tokens": "",
    "bucket_role": "",
    "decision": "",
    "phase366_prerequisite": "",
    "measurement_boundary": "",
    "runner_level_boundary": "",
    "quant_runtime": "",
    "kernel_source_metadata": "",
    "kernel_source_lookup_key": "",
    "ld_library_path_prefix": "",
    "vllm_enable_cuda_compatibility": "",
    "stop_rule": "",
    "next_allowed_phase": "phase368_fusedmoe_runner_minimal_shape_sweep",
    "future_gpu_ssh": "",
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


def _require_phase366_ep8_comm_shape_sweep_pass(path: Path) -> None:
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
        _require(row, "ok", TRUE, label)
        _require(row, "worker", "worker-gn6kz", label)
        _require(row, "vllm_version", "0.19.0", label)
        _require(
            row,
            "measurement_boundary",
            "vllm_ep_group_dispatch_router_logits_plus_combine",
            label,
        )
        _require(row, "backend", "allgather_reducescatter", label)
        _require(row, "manager", "AgRsAll2AllManager", label)
        _require(row, "rank_error", "0", label)
        _require(row, "cleanup", TRUE, label)
        _require(row, "gpu_process_residue", FALSE, label)
        _require(row, "curve_fit_allowed", FALSE, label)
        _require(row, "interpolation_allowed", FALSE, label)
        _require(row, "extrapolation_allowed", FALSE, label)
        _require(row, "default_readiness", DEFAULT_READINESS, label)
        _require(row, "diagnostic_only", TRUE, label)
        _require(row, "valid_for_default", FALSE, label)
        _require(row, "perf_database", FALSE, label)


def _spec_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = [
        {
            "row_type": "prerequisite",
            "decision": "phase366_ep8_comm_shape_sweep_pass_required",
            "phase366_prerequisite": "phase366_ep8_comm_minimal_shape_sweep_passed",
        },
        {
            "row_type": "measurement_boundary",
            "decision": "measure_fusedmoe_forward_runner_level",
            "measurement_boundary": "FusedMoE.forward()",
            "runner_level_boundary": TRUE,
        },
        {
            "row_type": "smoke_bucket",
            "bucket_tokens": "128",
            "bucket_role": "fusedmoe_runner_smoke_only_not_shape_sweep",
            "decision": "exclude_from_fusedmoe_shape_sweep",
        },
    ]

    rows.extend(
        {
            "row_type": "shape_bucket",
            "bucket_tokens": bucket,
            "bucket_role": "phase124_real_bucket",
            "decision": "required_for_fusedmoe_runner_minimal_shape_sweep",
            "measurement_boundary": "FusedMoE.forward()",
            "runner_level_boundary": TRUE,
        }
        for bucket in REAL_BUCKETS
    )

    rows.extend(
        [
            {
                "row_type": "quant_runtime",
                "decision": "runtime_quant_method_must_be_captured",
                "quant_runtime": QUANT_RUNTIME,
            },
            {
                "row_type": "kernel_metadata",
                "decision": "kernel_source_metadata_only_not_perfdb_lookup_key",
                "kernel_source_metadata": TRUE,
                "kernel_source_lookup_key": FALSE,
            },
            {
                "row_type": "environment_gate",
                "decision": "preload_cuda_compat_before_direct_smoke",
                "ld_library_path_prefix": CUDA_COMPAT_LD_LIBRARY_PATH,
                "vllm_enable_cuda_compatibility": "1",
            },
            {
                "row_type": "stop_rules",
                "decision": "fail_fast_no_retry_no_bare_kernel_fallback",
                "stop_rule": (
                    "ptx_failure;forward_context_failure;quant_runtime_drift;"
                    "gpu_or_process_residue"
                ),
            },
            {
                "row_type": "next_phase",
                "decision": "phase368_fusedmoe_runner_minimal_shape_sweep_only",
                "future_gpu_ssh": FUTURE_GPU_SSH,
            },
        ]
    )
    return [{**COMMON_FIELDS, **row} for row in rows]


def analyze_phase367_fusedmoe_runner_shape_sweep_spec(
    phase366_csv: Path = DEFAULT_PHASE366_CSV,
) -> list[dict[str, str]]:
    _require_phase366_ep8_comm_shape_sweep_pass(phase366_csv)
    return _spec_rows()


def _validate_output_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != 15:
        raise ValueError(f"Phase367 output must have exactly 15 rows, got {len(rows)}")

    row_types = [row.get("row_type") for row in rows]
    expected_types = [
        "prerequisite",
        "measurement_boundary",
        "smoke_bucket",
        "shape_bucket",
        "shape_bucket",
        "shape_bucket",
        "shape_bucket",
        "shape_bucket",
        "shape_bucket",
        "shape_bucket",
        "quant_runtime",
        "kernel_metadata",
        "environment_gate",
        "stop_rules",
        "next_phase",
    ]
    if row_types != expected_types:
        raise ValueError(f"Phase367 row_type sequence expected {expected_types}, got {row_types}")

    for index, row in enumerate(rows):
        label = row.get("row_type") or f"row_{index}"
        _require(row, "source", SOURCE, label)
        _require(row, "gpu_allowed", FALSE, label)
        _require(row, "curve_fit_allowed", FALSE, label)
        _require(row, "interpolation_allowed", FALSE, label)
        _require(row, "extrapolation_allowed", FALSE, label)
        _require(row, "default_readiness", DEFAULT_READINESS, label)
        _require(row, "diagnostic_only", TRUE, label)
        _require(row, "valid_for_default", FALSE, label)
        _require(row, "perf_database", FALSE, label)

    boundary = rows[1]
    _require(boundary, "measurement_boundary", "FusedMoE.forward()", "measurement_boundary")
    _require(boundary, "runner_level_boundary", TRUE, "measurement_boundary")

    smoke = rows[2]
    _require(smoke, "bucket_tokens", "128", "smoke_bucket")
    _require(
        smoke,
        "bucket_role",
        "fusedmoe_runner_smoke_only_not_shape_sweep",
        "smoke_bucket",
    )

    buckets = [row["bucket_tokens"] for row in rows if row["row_type"] == "shape_bucket"]
    if buckets != REAL_BUCKETS:
        raise ValueError(f"shape buckets expected {REAL_BUCKETS}, got {buckets}")
    if "128" in buckets:
        raise ValueError("shape sweep buckets must not include 128")

    quant = rows[10]
    _require(quant, "quant_runtime", QUANT_RUNTIME, "quant_runtime")

    metadata = rows[11]
    _require(metadata, "kernel_source_metadata", TRUE, "kernel_metadata")
    _require(metadata, "kernel_source_lookup_key", FALSE, "kernel_metadata")

    environment = rows[12]
    _require(
        environment,
        "ld_library_path_prefix",
        CUDA_COMPAT_LD_LIBRARY_PATH,
        "environment_gate",
    )
    _require(environment, "vllm_enable_cuda_compatibility", "1", "environment_gate")

    stop_rule = rows[13].get("stop_rule", "")
    for token in [
        "ptx_failure",
        "forward_context_failure",
        "quant_runtime_drift",
        "gpu_or_process_residue",
    ]:
        if token not in stop_rule:
            raise ValueError(f"stop_rules missing {token}")

    _require(rows[14], "future_gpu_ssh", FUTURE_GPU_SSH, "next_phase")
    _require(
        rows[14],
        "next_allowed_phase",
        "phase368_fusedmoe_runner_minimal_shape_sweep",
        "next_phase",
    )


def write_phase367_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_output_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_phase367_md(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_output_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Phase367 FusedMoE Runner Shape Sweep Spec",
                "",
                "| Item | Result |",
                "|---|---|",
                "| Verdict | FusedMoE runner shape sweep spec only |",
                "| Default AIC | No-Go |",
                "| PerfDatabase | not written |",
                "| GPU | not run in Phase367 |",
                "",
                "## Spec",
                "",
                "- Measurement boundary is `FusedMoE.forward() runner level`.",
                "- Shape sweep buckets are fixed to `1`, `15`, `16`, `241`, `1808`, `2048`, `8192`.",
                "- `128` remains smoke-only and is excluded from the shape sweep.",
                f"- Runtime quant method must be `{QUANT_RUNTIME}`.",
                "- The kernel source is metadata only and is not a PerfDatabase lookup key.",
                "- Direct smoke requires `LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:/usr/local/nvidia/lib64` and `VLLM_ENABLE_CUDA_COMPATIBILITY=1` before process start.",
                "- Stop on PTX failure, forward-context failure, quant runtime drift, or GPU/process residue.",
                "- Phase368 may run the minimal FusedMoE runner shape sweep on the recorded worker only.",
                "",
                "## Future GPU Entry",
                "",
                f"- Recorded only: `{FUTURE_GPU_SSH}`",
                "- Phase367 does not permit SSH or GPU execution.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase366-csv", type=Path, default=DEFAULT_PHASE366_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    rows = analyze_phase367_fusedmoe_runner_shape_sweep_spec(args.phase366_csv)
    write_phase367_csv(args.output_csv, rows)
    write_phase367_md(args.output_md, rows)


if __name__ == "__main__":
    main()
