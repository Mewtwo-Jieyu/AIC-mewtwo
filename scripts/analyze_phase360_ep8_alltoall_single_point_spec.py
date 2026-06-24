from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE359B_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase359b_fusedmoe_cuda_compat_smoke_result.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase360_ep8_alltoall_single_point_spec.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase360_ep8_alltoall_single_point_spec.md"
)

SOURCE = "phase360_ep8_alltoall_single_point_spec"
PHASE359B_SOURCE = "phase359b_fusedmoe_cuda_compat_smoke_result"
DEFAULT_READINESS = "No-Go"
TRUE = "true"
FALSE = "false"

FIELDNAMES = [
    "source",
    "spec_row",
    "decision",
    "required_phase359b_event",
    "required_phase359b_ok",
    "required_cuda_compat_preload",
    "required_vllm_enable_cuda_compatibility",
    "measurement_boundary",
    "measurement_contract",
    "schema_reuse",
    "fusedmoe_total_latency",
    "route_label",
    "runtime_backend_required",
    "generic_backend_allowed",
    "artifact_contract",
    "stop_rule",
    "retry_policy",
    "next_allowed_phase",
    "gpu_allowed",
    "default_readiness",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]

COMMON_FIELDS = {
    "source": SOURCE,
    "required_phase359b_event": "",
    "required_phase359b_ok": "",
    "required_cuda_compat_preload": "",
    "required_vllm_enable_cuda_compatibility": "",
    "measurement_boundary": "",
    "measurement_contract": "",
    "schema_reuse": "",
    "fusedmoe_total_latency": "",
    "route_label": "",
    "runtime_backend_required": "",
    "generic_backend_allowed": "",
    "artifact_contract": "",
    "stop_rule": "",
    "retry_policy": "",
    "next_allowed_phase": "",
    "gpu_allowed": FALSE,
    "default_readiness": DEFAULT_READINESS,
    "diagnostic_only": TRUE,
    "valid_for_default": FALSE,
    "perf_database": FALSE,
}

SPEC_ROWS = [
    {
        "spec_row": "phase359b_runner_smoke_prerequisite",
        "decision": "requires_phase359b_pass_and_cuda_compat_preload",
        "required_phase359b_event": "fusedmoe_forward_runner_smoke_passed",
        "required_phase359b_ok": TRUE,
        "required_cuda_compat_preload": (
            "LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:/usr/local/nvidia/lib64"
        ),
        "required_vllm_enable_cuda_compatibility": "1",
        "measurement_contract": (
            "phase359b_runner_boundary_pass_is_prerequisite_not_comm_evidence"
        ),
        "next_allowed_phase": "phase361_ep8_comm_single_point_gpu_run",
    },
    {
        "spec_row": "ep8_comm_boundary",
        "decision": "measure_vllm_0190_runner_level_ep_comm_path",
        "measurement_boundary": "vllm_0190_runner_level_ep_comm_path",
        "measurement_contract": (
            "measure_actual_vllm_ep_comm_path_separate_from_wideep_and_total_moe"
        ),
        "schema_reuse": "reject_wideep_schema",
        "fusedmoe_total_latency": "not_comm_boundary",
        "next_allowed_phase": "phase361_ep8_comm_single_point_gpu_run",
    },
    {
        "spec_row": "runtime_backend_capture",
        "decision": "capture_actual_runtime_backend_not_route_label",
        "measurement_boundary": "vllm_0190_runner_level_ep_comm_path",
        "measurement_contract": (
            "runtime_backend_capture_required_not_alltoall_string_only"
        ),
        "route_label": "ep8_alltoall_single_point_spec",
        "runtime_backend_required": (
            "allgather_reducescatter;deepep_*;flashinfer_*;other_explicit_backend"
        ),
        "generic_backend_allowed": FALSE,
        "next_allowed_phase": "phase361_ep8_comm_single_point_gpu_run",
    },
    {
        "spec_row": "single_point_artifact_contract",
        "decision": "emit_one_diagnostic_csv_row_only",
        "measurement_boundary": "vllm_0190_runner_level_ep_comm_path",
        "artifact_contract": (
            "one_row_csv:worker,vllm_version,backend,shape,latency_ms,ok,cleanup"
        ),
        "measurement_contract": (
            "single_point_artifact_is_diagnostic_contract_not_perfdb_row"
        ),
        "next_allowed_phase": "phase361_ep8_comm_single_point_gpu_run",
    },
    {
        "spec_row": "stop_rules",
        "decision": "fail_fast_no_retry_no_bare_kernel_fallback",
        "stop_rule": (
            "ptx_failure;context_failure;unknown_backend;gpu_or_process_residue"
        ),
        "retry_policy": "no_retry_no_bare_kernel_fallback",
        "measurement_contract": (
            "stop_on_ptx_context_backend_or_cleanup_failure_without_fallback"
        ),
        "next_allowed_phase": "phase361_only_if_spec_inputs_are_available",
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


def _require_phase359b_runner_smoke_pass(path: Path) -> None:
    rows = _read_csv(path)
    if len(rows) != 4:
        raise ValueError(f"Phase359b must have exactly 4 rows, got {len(rows)}")

    final = rows[-1]
    label = "phase359b"
    _require(final, "source", PHASE359B_SOURCE, label)
    _require(final, "event", "fusedmoe_forward_runner_smoke_passed", label)
    _require(final, "decision", "runner_boundary_smoke_pass", label)
    _require(final, "ok", TRUE, label)
    _require(
        final,
        "ld_library_path_prefix",
        "/usr/local/cuda-12.9/compat:/usr/local/nvidia/lib64",
        label,
    )
    _require(final, "vllm_enable_cuda_compatibility", "1", label)
    _require(final, "next_required_action", "phase360_ep8_alltoall_single_point_spec", label)
    _require(final, "default_readiness", DEFAULT_READINESS, label)
    _require(final, "diagnostic_only", TRUE, label)
    _require(final, "valid_for_default", FALSE, label)
    _require(final, "perf_database", FALSE, label)


def analyze_phase360_ep8_alltoall_single_point_spec(
    phase359b_csv: Path = DEFAULT_PHASE359B_CSV,
) -> list[dict[str, str]]:
    _require_phase359b_runner_smoke_pass(phase359b_csv)
    return [{**COMMON_FIELDS, **row} for row in SPEC_ROWS]


def _validate_output_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != 5:
        raise ValueError(f"Phase360 output must have exactly 5 rows, got {len(rows)}")

    spec_rows = [row.get("spec_row") for row in rows]
    expected_rows = [row["spec_row"] for row in SPEC_ROWS]
    if spec_rows != expected_rows:
        raise ValueError(f"Phase360 row sequence expected {expected_rows}, got {spec_rows}")

    for index, row in enumerate(rows):
        label = row.get("spec_row") or f"row_{index}"
        _require(row, "source", SOURCE, label)
        _require(row, "gpu_allowed", FALSE, label)
        _require(row, "default_readiness", DEFAULT_READINESS, label)
        _require(row, "diagnostic_only", TRUE, label)
        _require(row, "valid_for_default", FALSE, label)
        _require(row, "perf_database", FALSE, label)

    prerequisite = rows[0]
    _require(
        prerequisite,
        "required_phase359b_event",
        "fusedmoe_forward_runner_smoke_passed",
        "phase359b_runner_smoke_prerequisite",
    )
    _require(
        prerequisite,
        "required_phase359b_ok",
        TRUE,
        "phase359b_runner_smoke_prerequisite",
    )
    _require(
        prerequisite,
        "required_cuda_compat_preload",
        "LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:/usr/local/nvidia/lib64",
        "phase359b_runner_smoke_prerequisite",
    )

    boundary = rows[1]
    _require(
        boundary,
        "measurement_boundary",
        "vllm_0190_runner_level_ep_comm_path",
        "ep8_comm_boundary",
    )
    _require(boundary, "schema_reuse", "reject_wideep_schema", "ep8_comm_boundary")
    _require(boundary, "fusedmoe_total_latency", "not_comm_boundary", "ep8_comm_boundary")

    backend = rows[2]
    _require(backend, "route_label", "ep8_alltoall_single_point_spec", "runtime_backend_capture")
    _require(
        backend,
        "runtime_backend_required",
        "allgather_reducescatter;deepep_*;flashinfer_*;other_explicit_backend",
        "runtime_backend_capture",
    )
    _require(backend, "generic_backend_allowed", FALSE, "runtime_backend_capture")

    artifact = rows[3]
    _require(
        artifact,
        "artifact_contract",
        "one_row_csv:worker,vllm_version,backend,shape,latency_ms,ok,cleanup",
        "single_point_artifact_contract",
    )

    stop_rules = rows[4]
    _require(
        stop_rules,
        "stop_rule",
        "ptx_failure;context_failure;unknown_backend;gpu_or_process_residue",
        "stop_rules",
    )
    _require(stop_rules, "retry_policy", "no_retry_no_bare_kernel_fallback", "stop_rules")


def write_phase360_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_output_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_phase360_md(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_output_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Phase360 EP8 Alltoall Single Point Spec",
                "",
                "| Item | Result |",
                "|---|---|",
                "| Verdict | run spec only |",
                "| Default AIC | No-Go |",
                "| PerfDatabase | not written |",
                "| GPU | not run in Phase360 |",
                "",
                "## Scope",
                "",
                "- Phase360 is a run spec, not performance evidence.",
                "- Phase361 is the first allowed GPU single-point run, and only if the exact runtime inputs are available.",
                "- WideEP schema is not reused; the run must measure the vLLM 0.19.0 runner-level EP comm path.",
                "- FusedMoE total latency is not the EP8 comm boundary.",
                "- The route name may say alltoall, but the actual runtime backend must be captured.",
                "- The single-point artifact may produce one diagnostic CSV row only; it must not become a PerfDatabase row or default AIC evidence.",
                "",
                "## Stop Rules",
                "",
                "- Stop on PTX failure.",
                "- Stop on forward/context setup failure.",
                "- Stop when the backend cannot be identified.",
                "- Stop when GPU or process residue remains after cleanup.",
                "- Do not retry, and do not fall back to a bare kernel measurement.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase359b-csv", type=Path, default=DEFAULT_PHASE359B_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    rows = analyze_phase360_ep8_alltoall_single_point_spec(args.phase359b_csv)
    write_phase360_csv(args.output_csv, rows)
    write_phase360_md(args.output_md, rows)


if __name__ == "__main__":
    main()
