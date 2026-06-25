from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE360_CSV = (
    REPO_ROOT / "docs/iter_gap_investigation/phase360_ep8_alltoall_single_point_spec.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT / "docs/iter_gap_investigation/phase362_ep8_comm_single_point_result.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT / "docs/iter_gap_investigation/phase362_ep8_comm_single_point_result.md"
)

SOURCE = "phase362_ep8_comm_single_point_result"
PHASE360_SOURCE = "phase360_ep8_alltoall_single_point_spec"
DEFAULT_READINESS = "No-Go"
TRUE = "true"
FALSE = "false"

FIELDNAMES = [
    "source",
    "phase361_artifact_dir",
    "ok",
    "worker",
    "vllm_version",
    "source_root",
    "measurement_boundary",
    "route_label",
    "backend",
    "manager",
    "latency_ms",
    "shape",
    "cleanup",
    "gpu_process_residue",
    "rank_error",
    "result_interpretation",
    "next_required_action",
    "default_readiness",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]

RESULT_ROW = {
    "source": SOURCE,
    "phase361_artifact_dir": (
        "/mnt/shared-storage-user/ailab-sys/zhaojieyu/aic/"
        "phase361_ep8_comm_single_point_c68cfa1"
    ),
    "ok": TRUE,
    "worker": "worker-892rz",
    "vllm_version": "0.19.0",
    "source_root": "/usr/local/lib/python3.12/dist-packages/vllm",
    "measurement_boundary": "vllm_ep_group_dispatch_router_logits_plus_combine",
    "route_label": "ep8_alltoall_single_point_result",
    "backend": "allgather_reducescatter",
    "manager": "AgRsAll2AllManager",
    "latency_ms": "0.590688",
    "shape": (
        "num_tokens=128,local_tokens=32,hidden_size=7168,num_experts=384,"
        "top_k=8,dtype=bfloat16"
    ),
    "cleanup": TRUE,
    "gpu_process_residue": FALSE,
    "rank_error": "0",
    "result_interpretation": (
        "runner_level_ep_comm_single_point_pass_not_perfdb_or_default_aic_evidence"
    ),
    "next_required_action": "phase363_shape_expansion_or_sufficiency_gate",
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


def _require_phase360_runtime_backend_contract(path: Path) -> None:
    rows = _read_csv(path)
    if len(rows) != 5:
        raise ValueError(f"Phase360 must have exactly 5 rows, got {len(rows)}")

    for index, row in enumerate(rows):
        label = row.get("spec_row") or f"row_{index}"
        _require(row, "source", PHASE360_SOURCE, label)
        _require(row, "default_readiness", DEFAULT_READINESS, label)
        _require(row, "diagnostic_only", TRUE, label)
        _require(row, "valid_for_default", FALSE, label)
        _require(row, "perf_database", FALSE, label)

    backend = rows[2]
    _require(backend, "spec_row", "runtime_backend_capture", "phase360")
    _require(
        backend,
        "runtime_backend_required",
        "allgather_reducescatter;deepep_*;flashinfer_*;other_explicit_backend",
        "phase360",
    )
    _require(backend, "generic_backend_allowed", FALSE, "phase360")

    artifact = rows[3]
    _require(artifact, "spec_row", "single_point_artifact_contract", "phase360")
    _require(
        artifact,
        "artifact_contract",
        "one_row_csv:worker,vllm_version,backend,shape,latency_ms,ok,cleanup",
        "phase360",
    )


def analyze_phase362_ep8_comm_single_point_result(
    phase360_csv: Path = DEFAULT_PHASE360_CSV,
) -> list[dict[str, str]]:
    _require_phase360_runtime_backend_contract(phase360_csv)
    return [dict(RESULT_ROW)]


def _validate_output_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != 1:
        raise ValueError(f"Phase362 output must have exactly 1 row, got {len(rows)}")

    row = rows[0]
    _require(row, "source", SOURCE, "phase362")
    _require(row, "ok", TRUE, "phase362")
    _require(row, "worker", "worker-892rz", "phase362")
    _require(row, "vllm_version", "0.19.0", "phase362")
    _require(row, "source_root", "/usr/local/lib/python3.12/dist-packages/vllm", "phase362")
    _require(
        row,
        "measurement_boundary",
        "vllm_ep_group_dispatch_router_logits_plus_combine",
        "phase362",
    )
    _require(row, "backend", "allgather_reducescatter", "phase362")
    _require(row, "manager", "AgRsAll2AllManager", "phase362")
    _require(row, "latency_ms", "0.590688", "phase362")
    _require(row, "cleanup", TRUE, "phase362")
    _require(row, "gpu_process_residue", FALSE, "phase362")
    _require(row, "rank_error", "0", "phase362")
    _require(row, "default_readiness", DEFAULT_READINESS, "phase362")
    _require(row, "diagnostic_only", TRUE, "phase362")
    _require(row, "valid_for_default", FALSE, "phase362")
    _require(row, "perf_database", FALSE, "phase362")
    if row.get("backend") == "alltoall":
        raise ValueError("phase362 backend must not be generic alltoall")


def write_phase362_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_output_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_phase362_md(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_output_rows(rows)
    row = rows[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Phase362 EP8 Comm Single Point Result",
                "",
                "| Item | Result |",
                "|---|---|",
                "| Verdict | EP8 comm single point pass |",
                "| Default AIC | No-Go |",
                "| PerfDatabase | not written |",
                "| Evidence status | diagnostic-only single point |",
                "",
                "## Result",
                "",
                f"- Phase361 artifact: `{row['phase361_artifact_dir']}`",
                "- Phase361 pass only proves the runner-level EP comm single point is measurable.",
                "- `alltoall` is only the route label; the actual backend is `allgather_reducescatter`.",
                "- Runtime manager was captured as `AgRsAll2AllManager`.",
                "- `0.590688` ms is not a PerfDatabase row and is not default AIC evidence.",
                "- The next decision should be shape expansion or a sufficiency gate, not direct e2e/default AIC promotion.",
                "",
                "## Captured Row",
                "",
                f"- ok: `{row['ok']}`",
                f"- worker: `{row['worker']}`",
                f"- vllm_version: `{row['vllm_version']}`",
                f"- source_root: `{row['source_root']}`",
                f"- measurement_boundary: `{row['measurement_boundary']}`",
                f"- backend: `{row['backend']}`",
                f"- manager: `{row['manager']}`",
                f"- latency_ms: `{row['latency_ms']}`",
                f"- cleanup: `{row['cleanup']}`",
                f"- diagnostic_only: `{row['diagnostic_only']}`",
                f"- valid_for_default: `{row['valid_for_default']}`",
                f"- perf_database: `{row['perf_database']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase360-csv", type=Path, default=DEFAULT_PHASE360_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    rows = analyze_phase362_ep8_comm_single_point_result(args.phase360_csv)
    write_phase362_csv(args.output_csv, rows)
    write_phase362_md(args.output_md, rows)


if __name__ == "__main__":
    main()
