from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE356_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase356_route_a_fusedmoe_runner_boundary_decision.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase359b_fusedmoe_cuda_compat_smoke_result.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase359b_fusedmoe_cuda_compat_smoke_result.md"
)

SOURCE = "phase359b_fusedmoe_cuda_compat_smoke_result"
PHASE356_SOURCE = "phase356_route_a_fusedmoe_runner_boundary_decision"
DEFAULT_READINESS = "No-Go"
TRUE = "true"
FALSE = "false"

FIELDNAMES = [
    "source",
    "event",
    "decision",
    "worker",
    "vllm_version",
    "source_root",
    "measurement_boundary",
    "runner_boundary_api",
    "num_tokens",
    "hidden_size",
    "intermediate_size",
    "num_experts",
    "top_k",
    "dtype",
    "quant_method_config",
    "quant_method_runtime",
    "kernel_source_metadata",
    "kernel_source_lookup_key",
    "perf_database_row",
    "ld_library_path_prefix",
    "vllm_enable_cuda_compatibility",
    "forward_context",
    "static_all_moe_layers",
    "ok",
    "latency_ms_median",
    "output_shape",
    "output_all_finite",
    "interpretation",
    "next_required_action",
    "default_readiness",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]

COMMON_FIELDS = {
    "source": SOURCE,
    "worker": "worker-892rz",
    "vllm_version": "0.19.0",
    "source_root": "/usr/local/lib/python3.12/dist-packages/vllm",
    "measurement_boundary": "fusedmoe_forward_runner_level",
    "runner_boundary_api": "FusedMoE.forward()",
    "num_tokens": "128",
    "hidden_size": "7168",
    "intermediate_size": "2048",
    "num_experts": "384",
    "top_k": "8",
    "dtype": "bfloat16",
    "quant_method_config": "compressed-tensors",
    "quant_method_runtime": "CompressedTensorsWNA16MarlinMoEMethod",
    "kernel_source_metadata": "CompressedTensorsWNA16MarlinMoEMethod:Marlin",
    "kernel_source_lookup_key": FALSE,
    "perf_database_row": FALSE,
    "ld_library_path_prefix": "",
    "vllm_enable_cuda_compatibility": "",
    "forward_context": "",
    "static_all_moe_layers": "",
    "ok": FALSE,
    "latency_ms_median": "",
    "output_shape": "",
    "output_all_finite": "",
    "default_readiness": DEFAULT_READINESS,
    "diagnostic_only": TRUE,
    "valid_for_default": FALSE,
    "perf_database": FALSE,
}

SPEC_ROWS = [
    {
        "event": "initial_direct_python_smoke_failed",
        "decision": "ptx_failed_without_process_start_cuda_compat_preload",
        "interpretation": (
            "cudaErrorUnsupportedPtxVersion_before_compat_libcuda_preload"
        ),
        "next_required_action": (
            "start_direct_smoke_process_with_cuda_12_9_compat_libcuda_first"
        ),
    },
    {
        "event": "process_start_cuda_compat_preload",
        "decision": "ptx_cleared_by_process_start_cuda_compat_preload",
        "ld_library_path_prefix": (
            "/usr/local/cuda-12.9/compat:/usr/local/nvidia/lib64"
        ),
        "vllm_enable_cuda_compatibility": "1",
        "interpretation": (
            "direct_smoke_process_start_requires_preloaded_cuda_compat_path"
        ),
        "next_required_action": "add_forward_context_for_direct_runner_smoke",
    },
    {
        "event": "per_forward_context_required",
        "decision": "requires_per_forward_set_forward_context",
        "ld_library_path_prefix": (
            "/usr/local/cuda-12.9/compat:/usr/local/nvidia/lib64"
        ),
        "vllm_enable_cuda_compatibility": "1",
        "forward_context": "per_forward_set_forward_context_none_num_tokens_128",
        "static_all_moe_layers": "phase358_cuda_compat_per_forward_context_retry",
        "interpretation": (
            "direct_smoke_must_recreate_vllm_serve_model_runner_lifecycle"
        ),
        "next_required_action": "run_single_runner_boundary_smoke_with_context",
    },
    {
        "event": "fusedmoe_forward_runner_smoke_passed",
        "decision": "runner_boundary_smoke_pass",
        "ld_library_path_prefix": (
            "/usr/local/cuda-12.9/compat:/usr/local/nvidia/lib64"
        ),
        "vllm_enable_cuda_compatibility": "1",
        "forward_context": "per_forward_set_forward_context_none_num_tokens_128",
        "static_all_moe_layers": "phase358_cuda_compat_per_forward_context_retry",
        "ok": TRUE,
        "latency_ms_median": "1.039261",
        "output_shape": "128x7168",
        "output_all_finite": TRUE,
        "interpretation": (
            "runner_boundary_smoke_pass_not_perfdb_or_default_aic_evidence"
        ),
        "next_required_action": "phase360_ep8_alltoall_single_point_spec",
    },
]


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


def _require_phase356_runner_boundary(path: Path) -> None:
    rows = _read_csv(path)
    if len(rows) != 1:
        raise ValueError(f"Phase356 must have exactly 1 row, got {len(rows)}")

    row = rows[0]
    label = "phase356"
    _require(row, "source", PHASE356_SOURCE, label)
    _require(row, "measurement_boundary", "fusedmoe_forward_runner_level", label)
    _require(row, "runner_boundary_api", "FusedMoE.forward()", label)
    _require(row, "locked_kernel_path_guard_required", FALSE, label)
    _require(row, "kernel_source_lookup_key", FALSE, label)
    _require(row, "kernel_source_metadata", TRUE, label)
    _require(row, "gpu_allowed", FALSE, label)
    _require(row, "default_readiness", DEFAULT_READINESS, label)
    _require(row, "diagnostic_only", TRUE, label)
    _require(row, "valid_for_default", FALSE, label)
    _require(row, "perf_database", FALSE, label)


def analyze_phase359b_fusedmoe_cuda_compat_smoke_result(
    phase356_csv: Path = DEFAULT_PHASE356_CSV,
) -> list[dict[str, str]]:
    _require_phase356_runner_boundary(phase356_csv)
    return [{**COMMON_FIELDS, **row} for row in SPEC_ROWS]


def _validate_output_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != 4:
        raise ValueError(f"Phase359b output must have exactly 4 rows, got {len(rows)}")

    events = [row.get("event") for row in rows]
    expected_events = [row["event"] for row in SPEC_ROWS]
    if events != expected_events:
        raise ValueError(f"Phase359b event sequence expected {expected_events}, got {events}")

    for index, row in enumerate(rows):
        label = row.get("event") or f"row_{index}"
        _require(row, "source", SOURCE, label)
        _require(row, "worker", "worker-892rz", label)
        _require(row, "vllm_version", "0.19.0", label)
        _require(row, "source_root", "/usr/local/lib/python3.12/dist-packages/vllm", label)
        _require(row, "measurement_boundary", "fusedmoe_forward_runner_level", label)
        _require(row, "runner_boundary_api", "FusedMoE.forward()", label)
        _require(row, "num_tokens", "128", label)
        _require(row, "hidden_size", "7168", label)
        _require(row, "intermediate_size", "2048", label)
        _require(row, "num_experts", "384", label)
        _require(row, "top_k", "8", label)
        _require(row, "kernel_source_metadata", "CompressedTensorsWNA16MarlinMoEMethod:Marlin", label)
        _require(row, "kernel_source_lookup_key", FALSE, label)
        _require(row, "perf_database_row", FALSE, label)
        _require(row, "default_readiness", DEFAULT_READINESS, label)
        _require(row, "diagnostic_only", TRUE, label)
        _require(row, "valid_for_default", FALSE, label)
        _require(row, "perf_database", FALSE, label)

    preload = rows[1]
    _require(
        preload,
        "ld_library_path_prefix",
        "/usr/local/cuda-12.9/compat:/usr/local/nvidia/lib64",
        "process_start_cuda_compat_preload",
    )
    _require(preload, "vllm_enable_cuda_compatibility", "1", "process_start_cuda_compat_preload")

    context = rows[2]
    _require(
        context,
        "forward_context",
        "per_forward_set_forward_context_none_num_tokens_128",
        "per_forward_context_required",
    )
    _require(
        context,
        "static_all_moe_layers",
        "phase358_cuda_compat_per_forward_context_retry",
        "per_forward_context_required",
    )

    final = rows[3]
    _require(final, "ok", TRUE, "fusedmoe_forward_runner_smoke_passed")
    _require(final, "latency_ms_median", "1.039261", "fusedmoe_forward_runner_smoke_passed")
    _require(final, "output_shape", "128x7168", "fusedmoe_forward_runner_smoke_passed")
    _require(final, "output_all_finite", TRUE, "fusedmoe_forward_runner_smoke_passed")


def write_phase359b_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_output_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_phase359b_md(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_output_rows(rows)
    final = rows[-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Phase359b FusedMoE CUDA Compat Smoke Result",
                "",
                "| Item | Result |",
                "|---|---|",
                "| Verdict | runner-boundary smoke pass |",
                "| Default AIC | No-Go |",
                "| PerfDatabase | not written |",
                "| Evidence status | diagnostic-only runner smoke |",
                "",
                "## Result",
                "",
                "- VLLM_ENABLE_CUDA_COMPATIBILITY=1 is effective when the direct smoke process starts with `/usr/local/cuda-12.9/compat` before `/usr/local/nvidia/lib64` in `LD_LIBRARY_PATH`.",
                "- The direct smoke is not equivalent to vLLM serve worker subprocess environment: `vllm serve` can propagate the compat setting before worker startup, while direct Python already has libcuda binding pressure at process start.",
                "- Each direct `FusedMoE.forward()` call must run inside `set_forward_context`, with `static_all_moe_layers` registered for the same runner layer.",
                "- `kernel_source=CompressedTensorsWNA16MarlinMoEMethod:Marlin` is measurement metadata only; kernel_source stays measurement metadata and is not a PerfDatabase lookup key.",
                "- The latency value proves this runner boundary can execute, but it is not a PerfDatabase row and is not default AIC evidence.",
                "",
                "## Final Smoke Row",
                "",
                f"- ok: `{final['ok']}`",
                f"- latency_ms_median: `{final['latency_ms_median']}`",
                f"- output_shape: `{final['output_shape']}`",
                f"- output_all_finite: `{final['output_all_finite']}`",
                f"- diagnostic_only: `{final['diagnostic_only']}`",
                f"- valid_for_default: `{final['valid_for_default']}`",
                f"- perf_database: `{final['perf_database']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase356-csv", type=Path, default=DEFAULT_PHASE356_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    rows = analyze_phase359b_fusedmoe_cuda_compat_smoke_result(args.phase356_csv)
    write_phase359b_csv(args.output_csv, rows)
    write_phase359b_md(args.output_md, rows)


if __name__ == "__main__":
    main()
