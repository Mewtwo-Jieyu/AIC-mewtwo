from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE364_CSV = (
    REPO_ROOT / "docs/iter_gap_investigation/phase364_ep8_comm_shape_expansion_spec.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase366_ep8_comm_minimal_shape_sweep_result.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase366_ep8_comm_minimal_shape_sweep_result.md"
)

SOURCE = "phase366_ep8_comm_minimal_shape_sweep_result"
PHASE364_SOURCE = "phase364_ep8_comm_shape_expansion_spec"
DEFAULT_READINESS = "No-Go"
TRUE = "true"
FALSE = "false"
PHASE365_ARTIFACT_DIR = (
    "/mnt/shared-storage-user/ailab-sys/zhaojieyu/aic/"
    "phase365_ep8_comm_minimal_shape_sweep_d422088"
)
REAL_BUCKETS = ["1", "15", "16", "241", "1808", "2048", "8192"]
LATENCY_BY_BUCKET = {
    "1": "0.157088",
    "15": "0.109920",
    "16": "0.083776",
    "241": "0.288960",
    "1808": "1.192704",
    "2048": "1.237760",
    "8192": "2.237024",
}

FIELDNAMES = [
    "source",
    "phase365_artifact_dir",
    "ok",
    "worker",
    "vllm_version",
    "source_root",
    "measurement_boundary",
    "bucket_tokens",
    "backend",
    "manager",
    "configured_backend",
    "shape",
    "latency_ms",
    "rank_error",
    "cleanup",
    "gpu_process_residue",
    "failure_reason",
    "runtime_dispatch_scope",
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
    "phase365_artifact_dir": PHASE365_ARTIFACT_DIR,
    "ok": TRUE,
    "worker": "worker-gn6kz",
    "vllm_version": "0.19.0",
    "source_root": "/usr/local/lib/python3.12/dist-packages/vllm",
    "measurement_boundary": "vllm_ep_group_dispatch_router_logits_plus_combine",
    "backend": "allgather_reducescatter",
    "manager": "AgRsAll2AllManager",
    "configured_backend": "allgather_reducescatter",
    "rank_error": "0",
    "cleanup": TRUE,
    "gpu_process_residue": FALSE,
    "failure_reason": "",
    "runtime_dispatch_scope": "fixed_model_config_hw_vllm_version_tuple",
    "result_interpretation": "diagnostic_shape_sweep_not_perfdb_curve",
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


def _require_phase364_shape_expansion_spec(path: Path) -> None:
    rows = _read_csv(path)
    if len(rows) != 13:
        raise ValueError(f"Phase364 must have exactly 13 rows, got {len(rows)}")

    for index, row in enumerate(rows):
        label = row.get("row_type") or f"row_{index}"
        _require(row, "source", PHASE364_SOURCE, label)
        _require(row, "curve_fit_allowed", FALSE, label)
        _require(row, "interpolation_allowed", FALSE, label)
        _require(row, "extrapolation_allowed", FALSE, label)
        _require(row, "perfdb_curve_allowed", FALSE, label)
        _require(row, "gpu_allowed", FALSE, label)
        _require(row, "default_readiness", DEFAULT_READINESS, label)
        _require(row, "diagnostic_only", TRUE, label)
        _require(row, "valid_for_default", FALSE, label)
        _require(row, "perf_database", FALSE, label)

    real_buckets = [row["bucket_tokens"] for row in rows if row["row_type"] == "real_bucket"]
    if real_buckets != REAL_BUCKETS:
        raise ValueError(f"real buckets expected {REAL_BUCKETS}, got {real_buckets}")
    if "128" in real_buckets:
        raise ValueError("real buckets must not include smoke bucket 128")

    smoke = next(row for row in rows if row["row_type"] == "smoke_bucket")
    _require(smoke, "bucket_tokens", "128", "smoke_bucket")
    _require(
        smoke,
        "bucket_role",
        "runner_comm_smoke_only_not_real_coverage",
        "smoke_bucket",
    )

    backend = next(row for row in rows if row["row_type"] == "backend_gate")
    _require(backend, "backend_required", "allgather_reducescatter", "backend_gate")
    _require(backend, "manager_required", "AgRsAll2AllManager", "backend_gate")

    next_phase = next(row for row in rows if row["row_type"] == "next_phase")
    _require(
        next_phase,
        "next_allowed_phase",
        "phase365_minimal_shape_sweep",
        "next_phase",
    )


def _shape_for_bucket(bucket: str) -> str:
    local_tokens = {
        "1": "1",
        "15": "4",
        "16": "4",
        "241": "61",
        "1808": "452",
        "2048": "512",
        "8192": "2048",
    }[bucket]
    return (
        f"num_tokens={bucket},local_tokens={local_tokens},hidden_size=7168,"
        "num_experts=384,top_k=8,dtype=bfloat16"
    )


def analyze_phase366_ep8_comm_minimal_shape_sweep_result(
    phase364_csv: Path = DEFAULT_PHASE364_CSV,
) -> list[dict[str, str]]:
    _require_phase364_shape_expansion_spec(phase364_csv)
    return [
        {
            **COMMON_FIELDS,
            "bucket_tokens": bucket,
            "shape": _shape_for_bucket(bucket),
            "latency_ms": LATENCY_BY_BUCKET[bucket],
        }
        for bucket in REAL_BUCKETS
    ]


def _validate_output_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != 7:
        raise ValueError(f"Phase366 output must have exactly 7 rows, got {len(rows)}")

    buckets = [row.get("bucket_tokens") for row in rows]
    if buckets != REAL_BUCKETS:
        raise ValueError(f"bucket sequence expected {REAL_BUCKETS}, got {buckets}")
    if "128" in buckets:
        raise ValueError("bucket 128 must not be included in Phase366 result")

    for row in rows:
        label = row.get("bucket_tokens") or "phase366"
        _require(row, "source", SOURCE, label)
        _require(row, "phase365_artifact_dir", PHASE365_ARTIFACT_DIR, label)
        _require(row, "ok", TRUE, label)
        _require(row, "worker", "worker-gn6kz", label)
        _require(row, "vllm_version", "0.19.0", label)
        _require(row, "source_root", "/usr/local/lib/python3.12/dist-packages/vllm", label)
        _require(
            row,
            "measurement_boundary",
            "vllm_ep_group_dispatch_router_logits_plus_combine",
            label,
        )
        _require(row, "backend", "allgather_reducescatter", label)
        _require(row, "manager", "AgRsAll2AllManager", label)
        _require(row, "configured_backend", "allgather_reducescatter", label)
        _require(row, "latency_ms", LATENCY_BY_BUCKET[label], label)
        _require(row, "rank_error", "0", label)
        _require(row, "cleanup", TRUE, label)
        _require(row, "gpu_process_residue", FALSE, label)
        _require(row, "failure_reason", "", label)
        _require(row, "curve_fit_allowed", FALSE, label)
        _require(row, "interpolation_allowed", FALSE, label)
        _require(row, "extrapolation_allowed", FALSE, label)
        _require(row, "default_readiness", DEFAULT_READINESS, label)
        _require(row, "diagnostic_only", TRUE, label)
        _require(row, "valid_for_default", FALSE, label)
        _require(row, "perf_database", FALSE, label)
        if float(row["latency_ms"]) <= 0.0:
            raise ValueError(f"{label} latency_ms must be positive")


def write_phase366_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_output_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_phase366_md(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_output_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase366 EP8 Comm Minimal Shape Sweep Result",
        "",
        "| Item | Result |",
        "|---|---|",
        "| Verdict | Phase365 EP8 comm minimal shape sweep passed |",
        "| Default AIC | No-Go |",
        "| PerfDatabase | not written |",
        "| Evidence status | diagnostic-only shape sweep |",
        "",
        "## Result",
        "",
        f"- Phase365 artifact: `{PHASE365_ARTIFACT_DIR}`",
        "- `128` is not included; it remains a smoke-only point from Phase361/364.",
        "- All 7 Phase124 real buckets passed with backend `allgather_reducescatter`.",
        "- Runtime manager was captured as `AgRsAll2AllManager` for every bucket.",
        "- Rank errors are `0`, cleanup is `true`, and GPU/process residue is `false` for every row.",
        "- These latencies are diagnostic only and cannot be interpolated or extrapolated into a PerfDatabase curve.",
        "- They are not default AIC evidence.",
        "",
        "## Bucket Results",
        "",
        "| bucket_tokens | latency_ms | backend | manager | rank_error |",
        "|---:|---:|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            "| {bucket_tokens} | {latency_ms} | {backend} | {manager} | {rank_error} |".format(
                **row
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase364-csv", type=Path, default=DEFAULT_PHASE364_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    rows = analyze_phase366_ep8_comm_minimal_shape_sweep_result(args.phase364_csv)
    write_phase366_csv(args.output_csv, rows)
    write_phase366_md(args.output_md, rows)


if __name__ == "__main__":
    main()
