from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE386_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase386_fusedmoe_paired_bucket_gpu_run_decision_spec.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase388_fusedmoe_paired_bucket_gpu_run_result.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase388_fusedmoe_paired_bucket_gpu_run_result.md"
)

SOURCE = "phase388_fusedmoe_paired_bucket_gpu_run_result"
PHASE386_SOURCE = "phase386_fusedmoe_paired_bucket_gpu_run_decision_spec"
PHASE387_ARTIFACT_DIR = (
    "/mnt/shared-storage-user/ailab-sys/zhaojieyu/aic/"
    "phase387_fusedmoe_paired_bucket_gpu_run_2ee80b7"
)
SOURCE_SHA = "2ee80b7c8e3a390227b7d19942f5ba6514e20ea1"
DEFAULT_READINESS = "No-Go"
TRUE = "true"
FALSE = "false"
TARGET_BUCKETS = ["2", "30", "32", "482", "3616", "4096", "16384"]
TARGET_BUCKETS_JOINED = "/".join(TARGET_BUCKETS)
LATENCY_BY_BUCKET = {
    "2": "0.188482",
    "30": "0.355168",
    "32": "0.353423",
    "482": "2.040215",
    "3616": "9.283537",
    "4096": "10.359648",
    "16384": "41.146568",
}
MEASUREMENT_BOUNDARY = "fusedmoe_forward_runner_level"
RUNNER_BOUNDARY_API = "FusedMoE.forward()"
HARDWARE = "h200_sxm"
VLLM_VERSION = "0.19.0"
SOURCE_ROOT = "/usr/local/lib/python3.12/dist-packages/vllm"
TOPOLOGY = "tp4dp2ep8"
QUANT_RUNTIME = "CompressedTensorsWNA16MarlinMoEMethod"
KERNEL_METADATA = "CompressedTensorsWNA16MarlinMoEMethod:Marlin"
STOP_RULES = (
    "ptx_or_compat_error;"
    "forward_context_error;"
    "quant_runtime_mismatch;"
    "shape_or_output_non_finite;"
    "gpu_or_process_residue;"
    "missing_bucket"
)

FIELDNAMES = [
    "source",
    "phase387_artifact_dir",
    "source_sha",
    "ok",
    "worker",
    "hardware",
    "vllm_version",
    "source_root",
    "topology",
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
    "write_real_data_file",
    "default_readiness",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]

COMMON_FIELDS = {
    "source": SOURCE,
    "phase387_artifact_dir": PHASE387_ARTIFACT_DIR,
    "source_sha": SOURCE_SHA,
    "ok": TRUE,
    "worker": "worker-r28f2",
    "hardware": HARDWARE,
    "vllm_version": VLLM_VERSION,
    "source_root": SOURCE_ROOT,
    "topology": TOPOLOGY,
    "model_path": (
        "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/"
        "zskj-hub/models--moonshotai--Kimi-K2.5"
    ),
    "measurement_boundary": MEASUREMENT_BOUNDARY,
    "runner_boundary_api": RUNNER_BOUNDARY_API,
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
    "result_interpretation": "diagnostic_fusedmoe_paired_bucket_gpu_run_not_perfdb_row",
    "write_real_data_file": FALSE,
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


def _require_phase386_gpu_run_decision_spec(path: Path) -> None:
    rows = _read_csv(path)
    by_type = {row["row_type"]: row for row in rows}
    required = [
        "target_bucket_set_verified",
        "measurement_boundary_verified",
        "environment_key_verified",
        "quant_runtime_verified",
        "stop_rules_locked",
        "gpu_run_allowed_next_phase_only",
        "next_phase",
    ]
    for row_type in required:
        if row_type not in by_type:
            raise ValueError(f"Phase386 missing row {row_type}")

    for row in rows:
        label = row.get("row_type") or "phase386"
        _require(row, "source", PHASE386_SOURCE, label)
        _require(row, "gpu_result", FALSE, label)
        _require(row, "diagnostic_only", TRUE, label)
        _require(row, "valid_for_default", FALSE, label)
        _require(row, "perf_database", FALSE, label)
        _require(row, "default_readiness", DEFAULT_READINESS, label)

    buckets = by_type["target_bucket_set_verified"]
    boundary = by_type["measurement_boundary_verified"]
    env = by_type["environment_key_verified"]
    quant = by_type["quant_runtime_verified"]
    stop = by_type["stop_rules_locked"]
    run_gate = by_type["gpu_run_allowed_next_phase_only"]
    next_phase = by_type["next_phase"]

    _require(buckets, "target_bucket_tokens", TARGET_BUCKETS_JOINED, "Phase386 buckets")
    _require(buckets, "target_bucket_count", "7", "Phase386 buckets")
    _require(boundary, "measurement_boundary", MEASUREMENT_BOUNDARY, "Phase386 boundary")
    _require(boundary, "runner_boundary_api", RUNNER_BOUNDARY_API, "Phase386 boundary")
    _require(env, "hardware", HARDWARE, "Phase386 env")
    _require(env, "vllm_version", VLLM_VERSION, "Phase386 env")
    _require(env, "topology", TOPOLOGY, "Phase386 env")
    _require(quant, "quant_runtime", QUANT_RUNTIME, "Phase386 quant")
    _require(stop, "stop_rules", STOP_RULES, "Phase386 stop rules")
    _require(run_gate, "future_gpu_run_allowed", TRUE, "Phase386 run gate")
    _require(next_phase, "next_allowed_phase", "phase387_fusedmoe_paired_bucket_gpu_run", "Phase386 next")


def analyze_phase388_fusedmoe_paired_bucket_gpu_run_result(
    phase386_csv: Path = DEFAULT_PHASE386_CSV,
) -> list[dict[str, str]]:
    _require_phase386_gpu_run_decision_spec(phase386_csv)
    rows = [
        {
            **COMMON_FIELDS,
            "bucket_tokens": bucket,
            "latency_ms_median": LATENCY_BY_BUCKET[bucket],
            "output_shape": f"{bucket}x7168",
        }
        for bucket in TARGET_BUCKETS
    ]
    _validate_output_rows(rows)
    return rows


def _validate_output_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != 7:
        raise ValueError(f"Phase388 output must have exactly 7 rows, got {len(rows)}")

    buckets = [row.get("bucket_tokens") for row in rows]
    if buckets != TARGET_BUCKETS:
        raise ValueError(f"bucket sequence expected {TARGET_BUCKETS}, got {buckets}")
    if "128" in buckets:
        raise ValueError("bucket 128 must not be included in Phase388 result")

    for row in rows:
        label = row.get("bucket_tokens") or "phase388"
        if set(row) != set(FIELDNAMES):
            raise ValueError(f"{label} fields do not match FIELDNAMES")
        _require(row, "source", SOURCE, label)
        _require(row, "phase387_artifact_dir", PHASE387_ARTIFACT_DIR, label)
        _require(row, "source_sha", SOURCE_SHA, label)
        _require(row, "ok", TRUE, label)
        _require(row, "worker", "worker-r28f2", label)
        _require(row, "hardware", HARDWARE, label)
        _require(row, "vllm_version", VLLM_VERSION, label)
        _require(row, "source_root", SOURCE_ROOT, label)
        _require(row, "topology", TOPOLOGY, label)
        _require(row, "measurement_boundary", MEASUREMENT_BOUNDARY, label)
        _require(row, "runner_boundary_api", RUNNER_BOUNDARY_API, label)
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
            "diagnostic_fusedmoe_paired_bucket_gpu_run_not_perfdb_row",
            label,
        )
        _require(row, "write_real_data_file", FALSE, label)
        _require(row, "default_readiness", DEFAULT_READINESS, label)
        _require(row, "diagnostic_only", TRUE, label)
        _require(row, "valid_for_default", FALSE, label)
        _require(row, "perf_database", FALSE, label)
        if float(row["latency_ms_median"]) <= 0.0:
            raise ValueError(f"{label} latency_ms_median must be positive")


def write_phase388_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_output_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_phase388_md(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_output_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase388 FusedMoE Paired Bucket GPU Run Result",
        "",
        "| Item | Result |",
        "|---|---|",
        "| Verdict | Phase387 FusedMoE paired bucket GPU run passed |",
        "| Default AIC | No-Go |",
        "| PerfDatabase | not written |",
        "| vllm_module_perf.txt | not written |",
        "| Evidence status | diagnostic-only FusedMoE.forward paired bucket GPU run |",
        "",
        "## Result",
        "",
        f"- Phase387 artifact: `{PHASE387_ARTIFACT_DIR}`",
        "- Measurement boundary is `FusedMoE.forward()` runner level.",
        "- Buckets are `2`, `30`, `32`, `482`, `3616`, `4096`, `16384`.",
        "- `128` is not included.",
        f"- Runtime quant method was `{QUANT_RUNTIME}` for every row.",
        f"- Kernel source metadata was `{KERNEL_METADATA}`; it is not a lookup key.",
        "- Output shape matched `<bucket>x7168` and all outputs were finite.",
        "- Cleanup was `true` and GPU/process residue was `false`.",
        "- Phase388 does not write `vllm_module_perf.txt`, write PerfDatabase rows, or open Default AIC.",
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
    parser.add_argument("--phase386-csv", type=Path, default=DEFAULT_PHASE386_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    rows = analyze_phase388_fusedmoe_paired_bucket_gpu_run_result(args.phase386_csv)
    write_phase388_csv(args.output_csv, rows)
    write_phase388_md(args.output_md, rows)


if __name__ == "__main__":
    main()
