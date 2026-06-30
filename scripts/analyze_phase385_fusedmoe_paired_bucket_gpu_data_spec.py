from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE384_CSV = (
    REPO_ROOT / "docs/iter_gap_investigation/phase384_paired_bucket_expansion_spec.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT / "docs/iter_gap_investigation/phase385_fusedmoe_paired_bucket_gpu_data_spec.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT / "docs/iter_gap_investigation/phase385_fusedmoe_paired_bucket_gpu_data_spec.md"
)

SOURCE = "phase385_fusedmoe_paired_bucket_gpu_data_spec"
DEFAULT_READINESS = "No-Go"
TRUE = "true"
FALSE = "false"
TARGET_BUCKETS = [2, 30, 32, 482, 3616, 4096, 16384]
TARGET_BUCKETS_TEXT = "/".join(str(bucket) for bucket in TARGET_BUCKETS)
MEASUREMENT_BOUNDARY = "fusedmoe_forward_runner_level"
RUNNER_BOUNDARY_API = "FusedMoE.forward()"
HARDWARE = "h200_sxm"
VLLM_VERSION = "0.19.0"
TOPOLOGY = "tp4dp2ep8"
QUANT_RUNTIME = "CompressedTensorsWNA16MarlinMoEMethod"
NEXT_PHASE = "phase386_fusedmoe_paired_bucket_gpu_run_decision"
PHASE384_NEXT = SOURCE

FIELDNAMES = [
    "source",
    "row_type",
    "verdict",
    "phase384_prerequisite",
    "phase384_next_allowed_phase",
    "target_bucket_tokens",
    "target_bucket_count",
    "measurement_boundary",
    "runner_boundary_api",
    "hardware",
    "vllm_version",
    "topology",
    "quant_runtime",
    "artifact_contract",
    "ep8_recollection_allowed",
    "existing_rows_preserved",
    "existing_module_row_count",
    "delete_existing_rows",
    "write_real_data_file",
    "bucket_128_allowed",
    "gpu_allowed",
    "ssh_allowed",
    "runtime_change",
    "new_perfdb_data",
    "default_aic_allowed",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
    "default_readiness",
    "interpolation_allowed",
    "extrapolation_allowed",
    "nearest_bucket_allowed",
    "next_allowed_phase",
]

COMMON_FIELDS = {
    "source": SOURCE,
    "verdict": "",
    "phase384_prerequisite": "",
    "phase384_next_allowed_phase": "",
    "target_bucket_tokens": "",
    "target_bucket_count": "",
    "measurement_boundary": "",
    "runner_boundary_api": "",
    "hardware": "",
    "vllm_version": "",
    "topology": "",
    "quant_runtime": "",
    "artifact_contract": "",
    "ep8_recollection_allowed": FALSE,
    "existing_rows_preserved": "",
    "existing_module_row_count": "",
    "delete_existing_rows": FALSE,
    "write_real_data_file": FALSE,
    "bucket_128_allowed": FALSE,
    "gpu_allowed": FALSE,
    "ssh_allowed": FALSE,
    "runtime_change": FALSE,
    "new_perfdb_data": FALSE,
    "default_aic_allowed": FALSE,
    "diagnostic_only": TRUE,
    "valid_for_default": FALSE,
    "perf_database": FALSE,
    "default_readiness": DEFAULT_READINESS,
    "interpolation_allowed": FALSE,
    "extrapolation_allowed": FALSE,
    "nearest_bucket_allowed": FALSE,
    "next_allowed_phase": "",
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


def _require_phase384(path: Path) -> None:
    rows = _read_csv(path)
    by_type = {row["row_type"]: row for row in rows}
    required = [
        "phase383_prerequisite",
        "existing_vllm_module_rows_preserved",
        "fusedmoe_paired_bucket_expansion_planned",
        "ep8_bucket_set_preserved",
        "lookup_policy_exact_only",
        "bucket_128_blocked",
        "perfdb_write_blocked",
        "default_aic_blocked",
        "next_phase",
    ]
    for row_type in required:
        if row_type not in by_type:
            raise ValueError(f"Phase384 missing row {row_type}")

    prerequisite = by_type["phase383_prerequisite"]
    preserve = by_type["existing_vllm_module_rows_preserved"]
    expansion = by_type["fusedmoe_paired_bucket_expansion_planned"]
    ep8 = by_type["ep8_bucket_set_preserved"]
    lookup = by_type["lookup_policy_exact_only"]
    bucket = by_type["bucket_128_blocked"]
    perfdb = by_type["perfdb_write_blocked"]
    default = by_type["default_aic_blocked"]
    next_phase = by_type["next_phase"]

    _require(
        prerequisite,
        "phase383_prerequisite",
        "paired_bucket_data_expansion_first",
        "Phase384 prerequisite",
    )
    _require(preserve, "existing_rows_preserved", TRUE, "Phase384 preserve")
    _require(preserve, "existing_module_row_count", "14", "Phase384 preserve")
    _require(expansion, "fusedmoe_paired_bucket_expansion", TARGET_BUCKETS_TEXT, "Phase384 expansion")
    _require(expansion, "write_real_data_file", FALSE, "Phase384 expansion")
    _require(ep8, "ep8_bucket_change_allowed", FALSE, "Phase384 EP8")
    _require(lookup, "exact_lookup_only", TRUE, "Phase384 lookup")
    _require(bucket, "bucket_128_allowed", FALSE, "Phase384 bucket 128")
    _require(perfdb, "write_real_data_file", FALSE, "Phase384 perfdb")
    _require(default, "default_aic_allowed", FALSE, "Phase384 default")
    _require(next_phase, "next_allowed_phase", PHASE384_NEXT, "Phase384 next")

    for row in rows:
        _require(row, "gpu_allowed", FALSE, row["row_type"])
        _require(row, "ssh_allowed", FALSE, row["row_type"])
        _require(row, "runtime_change", FALSE, row["row_type"])
        _require(row, "new_perfdb_data", FALSE, row["row_type"])
        _require(row, "write_real_data_file", FALSE, row["row_type"])
        _require(row, "default_readiness", DEFAULT_READINESS, row["row_type"])
        _require(row, "diagnostic_only", TRUE, row["row_type"])
        _require(row, "valid_for_default", FALSE, row["row_type"])
        _require(row, "perf_database", FALSE, row["row_type"])


def _row(row_type: str, **overrides: str) -> dict[str, str]:
    row = dict(COMMON_FIELDS)
    row["row_type"] = row_type
    row.update(overrides)
    return row


def analyze_phase385_fusedmoe_paired_bucket_gpu_data_spec(
    phase384_csv: Path = DEFAULT_PHASE384_CSV,
) -> list[dict[str, str]]:
    _require_phase384(phase384_csv)

    rows = [
        _row(
            "phase384_prerequisite",
            verdict="phase384_paired_bucket_expansion_spec_passed",
            phase384_prerequisite="paired_bucket_expansion_spec_passed",
            phase384_next_allowed_phase=PHASE384_NEXT,
            existing_rows_preserved=TRUE,
            existing_module_row_count="14",
        ),
        _row(
            "target_bucket_set_locked",
            verdict="seven_fusedmoe_paired_buckets_locked",
            target_bucket_tokens=TARGET_BUCKETS_TEXT,
            target_bucket_count="7",
            bucket_128_allowed=FALSE,
        ),
        _row(
            "measurement_boundary_locked",
            verdict="fusedmoe_forward_runner_level_locked",
            measurement_boundary=MEASUREMENT_BOUNDARY,
            runner_boundary_api=RUNNER_BOUNDARY_API,
        ),
        _row(
            "environment_key_locked",
            verdict="h200_sxm_vllm019_tp4dp2ep8_locked",
            hardware=HARDWARE,
            vllm_version=VLLM_VERSION,
            topology=TOPOLOGY,
        ),
        _row(
            "quant_runtime_locked",
            verdict="compressed_tensors_marlin_quant_runtime_locked",
            quant_runtime=QUANT_RUNTIME,
        ),
        _row(
            "collection_artifact_contract",
            verdict="future_gpu_run_artifact_contract_defined",
            artifact_contract=(
                "one_row_per_target_bucket_with_latency_output_shape_finite_quant_cleanup"
            ),
        ),
        _row(
            "no_ep8_recollection",
            verdict="ep8_not_recollected_in_paired_bucket_data_plan",
            ep8_recollection_allowed=FALSE,
        ),
        _row(
            "preserve_existing_data_rows",
            verdict="preserve_existing_14_rows_and_do_not_write_table",
            existing_rows_preserved=TRUE,
            existing_module_row_count="14",
            delete_existing_rows=FALSE,
            write_real_data_file=FALSE,
        ),
        _row(
            "lookup_policy_exact_only",
            verdict="exact_only_no_interpolation_extrapolation_or_nearest",
            interpolation_allowed=FALSE,
            extrapolation_allowed=FALSE,
            nearest_bucket_allowed=FALSE,
        ),
        _row(
            "bucket_128_blocked",
            verdict="bucket_128_remains_disallowed",
            bucket_128_allowed=FALSE,
        ),
        _row(
            "gpu_run_deferred",
            verdict="phase385_does_not_use_gpu_or_ssh",
            gpu_allowed=FALSE,
            ssh_allowed=FALSE,
        ),
        _row(
            "perfdb_write_blocked",
            verdict="phase385_does_not_write_vllm_module_perf",
            new_perfdb_data=FALSE,
            write_real_data_file=FALSE,
            perf_database=FALSE,
        ),
        _row(
            "default_aic_blocked",
            verdict="default_aic_stays_no_go",
            default_aic_allowed=FALSE,
        ),
        _row(
            "next_phase",
            verdict="next_phase_decides_whether_to_run_gpu",
            next_allowed_phase=NEXT_PHASE,
        ),
    ]
    _validate_rows(rows)
    return rows


def _validate_rows(rows: list[dict[str, str]]) -> None:
    expected_row_types = [
        "phase384_prerequisite",
        "target_bucket_set_locked",
        "measurement_boundary_locked",
        "environment_key_locked",
        "quant_runtime_locked",
        "collection_artifact_contract",
        "no_ep8_recollection",
        "preserve_existing_data_rows",
        "lookup_policy_exact_only",
        "bucket_128_blocked",
        "gpu_run_deferred",
        "perfdb_write_blocked",
        "default_aic_blocked",
        "next_phase",
    ]
    actual_row_types = [row["row_type"] for row in rows]
    if actual_row_types != expected_row_types:
        raise ValueError(f"unexpected row order: {actual_row_types}")

    for row in rows:
        if set(row) != set(FIELDNAMES):
            raise ValueError(f"{row['row_type']} fields do not match FIELDNAMES")
        _require(row, "gpu_allowed", FALSE, row["row_type"])
        _require(row, "ssh_allowed", FALSE, row["row_type"])
        _require(row, "runtime_change", FALSE, row["row_type"])
        _require(row, "new_perfdb_data", FALSE, row["row_type"])
        _require(row, "write_real_data_file", FALSE, row["row_type"])
        _require(row, "default_aic_allowed", FALSE, row["row_type"])
        _require(row, "diagnostic_only", TRUE, row["row_type"])
        _require(row, "valid_for_default", FALSE, row["row_type"])
        _require(row, "perf_database", FALSE, row["row_type"])
        _require(row, "default_readiness", DEFAULT_READINESS, row["row_type"])
        _require(row, "interpolation_allowed", FALSE, row["row_type"])
        _require(row, "extrapolation_allowed", FALSE, row["row_type"])
        _require(row, "nearest_bucket_allowed", FALSE, row["row_type"])

    by_type = {row["row_type"]: row for row in rows}
    _require(by_type["target_bucket_set_locked"], "target_bucket_tokens", TARGET_BUCKETS_TEXT, "Phase385 target")
    _require(by_type["target_bucket_set_locked"], "target_bucket_count", "7", "Phase385 target")
    _require(
        by_type["measurement_boundary_locked"],
        "measurement_boundary",
        MEASUREMENT_BOUNDARY,
        "Phase385 boundary",
    )
    _require(
        by_type["measurement_boundary_locked"],
        "runner_boundary_api",
        RUNNER_BOUNDARY_API,
        "Phase385 boundary",
    )
    _require(by_type["environment_key_locked"], "hardware", HARDWARE, "Phase385 env")
    _require(by_type["environment_key_locked"], "vllm_version", VLLM_VERSION, "Phase385 env")
    _require(by_type["environment_key_locked"], "topology", TOPOLOGY, "Phase385 env")
    _require(by_type["quant_runtime_locked"], "quant_runtime", QUANT_RUNTIME, "Phase385 quant")
    _require(by_type["no_ep8_recollection"], "ep8_recollection_allowed", FALSE, "Phase385 EP8")
    _require(by_type["bucket_128_blocked"], "bucket_128_allowed", FALSE, "Phase385 bucket 128")


def write_phase385_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_phase385_md(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    target = next(row for row in rows if row["row_type"] == "target_bucket_set_locked")
    env = next(row for row in rows if row["row_type"] == "environment_key_locked")
    quant = next(row for row in rows if row["row_type"] == "quant_runtime_locked")
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = f"""# Phase385 FusedMoE Paired Bucket GPU Data Spec

Phase385 is a spec-only phase. It defines future FusedMoE paired bucket GPU data collection, but does not SSH, run GPU, write `vllm_module_perf.txt`, or open Default AIC.

| Gate | Decision |
|---|---|
| target buckets | {target["target_bucket_tokens"]} |
| measurement boundary | FusedMoE.forward() runner level |
| hardware | {env["hardware"]} |
| vLLM version | {env["vllm_version"]} |
| topology | {env["topology"]} |
| quant runtime | {quant["quant_runtime"]} |
| EP8 | Do not collect EP8 again |
| existing rows | preserve current 14 rows |
| bucket 128 | blocked |
| lookup policy | exact-only; no interpolation, extrapolation, or nearest lookup |
| Default AIC | No-Go |

Do not run GPU in Phase385. The next allowed phase is `{NEXT_PHASE}`.
"""
    path.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase384-csv", type=Path, default=DEFAULT_PHASE384_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    rows = analyze_phase385_fusedmoe_paired_bucket_gpu_data_spec(args.phase384_csv)
    write_phase385_csv(args.output_csv, rows)
    write_phase385_md(args.output_md, rows)


if __name__ == "__main__":
    main()
