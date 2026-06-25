from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE363_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase363_ep8_comm_single_point_sufficiency_gate.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT / "docs/iter_gap_investigation/phase364_ep8_comm_shape_expansion_spec.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT / "docs/iter_gap_investigation/phase364_ep8_comm_shape_expansion_spec.md"
)

SOURCE = "phase364_ep8_comm_shape_expansion_spec"
PHASE363_SOURCE = "phase363_ep8_comm_single_point_sufficiency_gate"
DEFAULT_READINESS = "No-Go"
TRUE = "true"
FALSE = "false"
FUTURE_GPU_SSH = (
    "ssh -CAXY "
    "ws-faaf0de74ef9a14d-worker-gn6kz.zhaojieyu+root.ailab-sys.pod"
    "@h.pjlab.org.cn"
)
CUDA_COMPAT_LD_LIBRARY_PATH = "/usr/local/cuda-12.9/compat:/usr/local/nvidia/lib64"
REAL_BUCKETS = ["1", "15", "16", "241", "1808", "2048", "8192"]

FIELDNAMES = [
    "source",
    "row_type",
    "bucket_tokens",
    "bucket_role",
    "decision",
    "phase363_prerequisite",
    "backend_required",
    "manager_required",
    "ld_library_path_prefix",
    "vllm_enable_cuda_compatibility",
    "stop_rule",
    "curve_fit_allowed",
    "interpolation_allowed",
    "extrapolation_allowed",
    "perfdb_curve_allowed",
    "next_allowed_phase",
    "future_gpu_ssh",
    "gpu_allowed",
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
    "phase363_prerequisite": "",
    "backend_required": "",
    "manager_required": "",
    "ld_library_path_prefix": "",
    "vllm_enable_cuda_compatibility": "",
    "stop_rule": "",
    "curve_fit_allowed": FALSE,
    "interpolation_allowed": FALSE,
    "extrapolation_allowed": FALSE,
    "perfdb_curve_allowed": FALSE,
    "next_allowed_phase": "phase365_minimal_shape_sweep",
    "future_gpu_ssh": "",
    "gpu_allowed": FALSE,
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


def _require_phase363_single_point_insufficient(path: Path) -> None:
    rows = _read_csv(path)
    if len(rows) != 5:
        raise ValueError(f"Phase363 must have exactly 5 rows, got {len(rows)}")

    by_gate = {row.get("gate", ""): row for row in rows}
    expected_gates = [
        "phase362_prerequisite",
        "single_point_sufficiency",
        "default_aic_gate",
        "shape_expansion_required",
        "future_gpu_entry",
    ]
    if list(by_gate) != expected_gates:
        raise ValueError(f"Phase363 gate sequence expected {expected_gates}, got {list(by_gate)}")

    for row in rows:
        label = row.get("gate") or "phase363"
        _require(row, "source", PHASE363_SOURCE, label)
        _require(row, "curve_fit_allowed", FALSE, label)
        _require(row, "interpolation_allowed", FALSE, label)
        _require(row, "extrapolation_allowed", FALSE, label)
        _require(row, "perfdb_curve_allowed", FALSE, label)
        _require(row, "gpu_allowed", FALSE, label)
        _require(row, "default_readiness", DEFAULT_READINESS, label)
        _require(row, "diagnostic_only", TRUE, label)
        _require(row, "valid_for_default", FALSE, label)
        _require(row, "perf_database", FALSE, label)

    single_point = by_gate["single_point_sufficiency"]
    _require(
        single_point,
        "verdict",
        "insufficient_for_perfdb_curve",
        "single_point_sufficiency",
    )
    _require(
        by_gate["shape_expansion_required"],
        "verdict",
        "shape_expansion_spec_required",
        "shape_expansion_required",
    )
    _require(
        by_gate["future_gpu_entry"],
        "future_gpu_ssh",
        FUTURE_GPU_SSH,
        "future_gpu_entry",
    )


def _spec_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = [
        {
            "row_type": "prerequisite",
            "decision": "phase363_single_point_insufficient_curve_fit_disallowed",
            "phase363_prerequisite": "insufficient_for_perfdb_curve",
        },
        {
            "row_type": "smoke_bucket",
            "bucket_tokens": "128",
            "bucket_role": "runner_comm_smoke_only_not_real_coverage",
            "decision": "exclude_from_perfdb_curve_fit",
        },
    ]

    rows.extend(
        {
            "row_type": "real_bucket",
            "bucket_tokens": bucket,
            "bucket_role": "phase124_real_bucket",
            "decision": "required_for_minimal_shape_sweep",
        }
        for bucket in REAL_BUCKETS
    )

    rows.extend(
        [
            {
                "row_type": "backend_gate",
                "decision": "runtime_capture_backend_and_manager_per_bucket",
                "backend_required": "allgather_reducescatter",
                "manager_required": "AgRsAll2AllManager",
            },
            {
                "row_type": "environment_gate",
                "decision": "preload_cuda_compat_before_direct_smoke",
                "ld_library_path_prefix": CUDA_COMPAT_LD_LIBRARY_PATH,
                "vllm_enable_cuda_compatibility": "1",
            },
            {
                "row_type": "stop_rules",
                "decision": "fail_fast_no_retry_no_fallback",
                "stop_rule": (
                    "ptx_failure;context_failure;unknown_backend;"
                    "gpu_or_process_residue;backend_drift_from_allgather_reducescatter"
                ),
            },
            {
                "row_type": "next_phase",
                "decision": "phase365_minimal_shape_sweep_only_not_full_gpu_matrix",
                "future_gpu_ssh": FUTURE_GPU_SSH,
            },
        ]
    )

    return [{**COMMON_FIELDS, **row} for row in rows]


def analyze_phase364_ep8_comm_shape_expansion_spec(
    phase363_csv: Path = DEFAULT_PHASE363_CSV,
) -> list[dict[str, str]]:
    _require_phase363_single_point_insufficient(phase363_csv)
    return _spec_rows()


def _validate_output_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != 13:
        raise ValueError(f"Phase364 output must have exactly 13 rows, got {len(rows)}")

    row_types = [row.get("row_type") for row in rows]
    expected_types = [
        "prerequisite",
        "smoke_bucket",
        "real_bucket",
        "real_bucket",
        "real_bucket",
        "real_bucket",
        "real_bucket",
        "real_bucket",
        "real_bucket",
        "backend_gate",
        "environment_gate",
        "stop_rules",
        "next_phase",
    ]
    if row_types != expected_types:
        raise ValueError(f"Phase364 row_type sequence expected {expected_types}, got {row_types}")

    for index, row in enumerate(rows):
        label = row.get("row_type") or f"row_{index}"
        _require(row, "source", SOURCE, label)
        _require(row, "curve_fit_allowed", FALSE, label)
        _require(row, "interpolation_allowed", FALSE, label)
        _require(row, "extrapolation_allowed", FALSE, label)
        _require(row, "perfdb_curve_allowed", FALSE, label)
        _require(row, "gpu_allowed", FALSE, label)
        _require(row, "default_readiness", DEFAULT_READINESS, label)
        _require(row, "diagnostic_only", TRUE, label)
        _require(row, "valid_for_default", FALSE, label)
        _require(row, "perf_database", FALSE, label)

    _require(
        rows[0],
        "decision",
        "phase363_single_point_insufficient_curve_fit_disallowed",
        "prerequisite",
    )
    _require(rows[1], "bucket_tokens", "128", "smoke_bucket")
    _require(
        rows[1],
        "bucket_role",
        "runner_comm_smoke_only_not_real_coverage",
        "smoke_bucket",
    )

    real_buckets = [row["bucket_tokens"] for row in rows if row["row_type"] == "real_bucket"]
    if real_buckets != REAL_BUCKETS:
        raise ValueError(f"real buckets expected {REAL_BUCKETS}, got {real_buckets}")
    if "128" in real_buckets:
        raise ValueError("smoke bucket 128 must not be real coverage")

    backend = rows[9]
    _require(backend, "backend_required", "allgather_reducescatter", "backend_gate")
    _require(backend, "manager_required", "AgRsAll2AllManager", "backend_gate")

    environment = rows[10]
    _require(
        environment,
        "ld_library_path_prefix",
        CUDA_COMPAT_LD_LIBRARY_PATH,
        "environment_gate",
    )
    _require(environment, "vllm_enable_cuda_compatibility", "1", "environment_gate")

    stop_rule = rows[11].get("stop_rule", "")
    for token in [
        "ptx_failure",
        "context_failure",
        "unknown_backend",
        "gpu_or_process_residue",
        "backend_drift_from_allgather_reducescatter",
    ]:
        if token not in stop_rule:
            raise ValueError(f"stop_rules missing {token}")

    _require(rows[12], "future_gpu_ssh", FUTURE_GPU_SSH, "next_phase")
    _require(
        rows[12],
        "decision",
        "phase365_minimal_shape_sweep_only_not_full_gpu_matrix",
        "next_phase",
    )


def write_phase364_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_output_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_phase364_md(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_output_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Phase364 EP8 Comm Shape Expansion Spec",
                "",
                "| Item | Result |",
                "|---|---|",
                "| Verdict | shape expansion spec only |",
                "| Default AIC | No-Go |",
                "| PerfDatabase | not written |",
                "| GPU | not run in Phase364 |",
                "",
                "## Gate Result",
                "",
                "- Phase363 proved the EP8 single point is insufficient for a PerfDatabase curve.",
                "- `128` remains a runner/comm smoke point and is excluded from real coverage.",
                "- Real bucket coverage is fixed to Phase124 buckets: `1`, `15`, `16`, `241`, `1808`, `2048`, `8192`.",
                "- Every future GPU bucket must runtime capture `backend` and `manager`.",
                "- Direct smoke requires `LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:/usr/local/nvidia/lib64` and `VLLM_ENABLE_CUDA_COMPATIBILITY=1` before process start.",
                "- Stop on PTX failure, forward-context failure, unknown backend, GPU/process residue, or backend drift away from `allgather_reducescatter`.",
                "- The next allowed phase is Phase365 minimal shape sweep, not a full GPU matrix.",
                "",
                "## Future GPU Entry",
                "",
                f"- Recorded only: `{FUTURE_GPU_SSH}`",
                "- Phase364 does not permit SSH or GPU execution.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase363-csv", type=Path, default=DEFAULT_PHASE363_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    rows = analyze_phase364_ep8_comm_shape_expansion_spec(args.phase363_csv)
    write_phase364_csv(args.output_csv, rows)
    write_phase364_md(args.output_md, rows)


if __name__ == "__main__":
    main()
