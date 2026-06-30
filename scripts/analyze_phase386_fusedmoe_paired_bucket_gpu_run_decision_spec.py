from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE385_CSV = (
    REPO_ROOT / "docs/iter_gap_investigation/phase385_fusedmoe_paired_bucket_gpu_data_spec.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase386_fusedmoe_paired_bucket_gpu_run_decision_spec.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase386_fusedmoe_paired_bucket_gpu_run_decision_spec.md"
)

SOURCE = "phase386_fusedmoe_paired_bucket_gpu_run_decision_spec"
DEFAULT_READINESS = "No-Go"
TRUE = "true"
FALSE = "false"
TARGET_BUCKETS = "2/30/32/482/3616/4096/16384"
TARGET_BUCKET_COUNT = "7"
MEASUREMENT_BOUNDARY = "fusedmoe_forward_runner_level"
RUNNER_BOUNDARY_API = "FusedMoE.forward()"
HARDWARE = "h200_sxm"
VLLM_VERSION = "0.19.0"
TOPOLOGY = "tp4dp2ep8"
QUANT_RUNTIME = "CompressedTensorsWNA16MarlinMoEMethod"
PHASE385_NEXT = "phase386_fusedmoe_paired_bucket_gpu_run_decision"
NEXT_PHASE = "phase387_fusedmoe_paired_bucket_gpu_run"
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
    "row_type",
    "verdict",
    "phase385_prerequisite",
    "phase385_next_allowed_phase",
    "decision_type",
    "gpu_result",
    "target_bucket_tokens",
    "target_bucket_count",
    "measurement_boundary",
    "runner_boundary_api",
    "hardware",
    "vllm_version",
    "topology",
    "quant_runtime",
    "stop_rules",
    "artifact_contract",
    "future_gpu_run_allowed",
    "gpu_allowed",
    "ssh_allowed",
    "runtime_change",
    "new_perfdb_data",
    "write_real_data_file",
    "default_aic_allowed",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
    "default_readiness",
    "next_allowed_phase",
]

COMMON_FIELDS = {
    "source": SOURCE,
    "verdict": "",
    "phase385_prerequisite": "",
    "phase385_next_allowed_phase": "",
    "decision_type": "gpu_run_decision_spec",
    "gpu_result": FALSE,
    "target_bucket_tokens": "",
    "target_bucket_count": "",
    "measurement_boundary": "",
    "runner_boundary_api": "",
    "hardware": "",
    "vllm_version": "",
    "topology": "",
    "quant_runtime": "",
    "stop_rules": "",
    "artifact_contract": "",
    "future_gpu_run_allowed": FALSE,
    "gpu_allowed": FALSE,
    "ssh_allowed": FALSE,
    "runtime_change": FALSE,
    "new_perfdb_data": FALSE,
    "write_real_data_file": FALSE,
    "default_aic_allowed": FALSE,
    "diagnostic_only": TRUE,
    "valid_for_default": FALSE,
    "perf_database": FALSE,
    "default_readiness": DEFAULT_READINESS,
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


def _require_phase385(path: Path) -> None:
    rows = _read_csv(path)
    by_type = {row["row_type"]: row for row in rows}
    required = [
        "phase384_prerequisite",
        "target_bucket_set_locked",
        "measurement_boundary_locked",
        "environment_key_locked",
        "quant_runtime_locked",
        "collection_artifact_contract",
        "next_phase",
    ]
    for row_type in required:
        if row_type not in by_type:
            raise ValueError(f"Phase385 missing row {row_type}")

    prereq = by_type["phase384_prerequisite"]
    buckets = by_type["target_bucket_set_locked"]
    boundary = by_type["measurement_boundary_locked"]
    env = by_type["environment_key_locked"]
    quant = by_type["quant_runtime_locked"]
    artifact = by_type["collection_artifact_contract"]
    next_phase = by_type["next_phase"]

    _require(prereq, "phase384_prerequisite", "paired_bucket_expansion_spec_passed", "Phase385 prereq")
    _require(buckets, "target_bucket_tokens", TARGET_BUCKETS, "Phase385 buckets")
    _require(buckets, "target_bucket_count", TARGET_BUCKET_COUNT, "Phase385 buckets")
    _require(boundary, "measurement_boundary", MEASUREMENT_BOUNDARY, "Phase385 boundary")
    _require(boundary, "runner_boundary_api", RUNNER_BOUNDARY_API, "Phase385 boundary")
    _require(env, "hardware", HARDWARE, "Phase385 env")
    _require(env, "vllm_version", VLLM_VERSION, "Phase385 env")
    _require(env, "topology", TOPOLOGY, "Phase385 env")
    _require(quant, "quant_runtime", QUANT_RUNTIME, "Phase385 quant")
    if not artifact["artifact_contract"]:
        raise ValueError("Phase385 artifact_contract is empty")
    _require(next_phase, "next_allowed_phase", PHASE385_NEXT, "Phase385 next")

    for row in rows:
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


def _row(row_type: str, **overrides: str) -> dict[str, str]:
    row = dict(COMMON_FIELDS)
    row["row_type"] = row_type
    row.update(overrides)
    return row


def analyze_phase386_fusedmoe_paired_bucket_gpu_run_decision_spec(
    phase385_csv: Path = DEFAULT_PHASE385_CSV,
) -> list[dict[str, str]]:
    _require_phase385(phase385_csv)

    rows = [
        _row(
            "phase385_prerequisite",
            verdict="phase385_spec_prerequisite_passed",
            phase385_prerequisite="fusedmoe_paired_bucket_gpu_data_spec_passed",
            phase385_next_allowed_phase=PHASE385_NEXT,
        ),
        _row(
            "decision_type_locked",
            verdict="gpu_run_decision_spec_not_result",
            decision_type="gpu_run_decision_spec",
            gpu_result=FALSE,
        ),
        _row(
            "target_bucket_set_verified",
            verdict="seven_target_buckets_verified",
            target_bucket_tokens=TARGET_BUCKETS,
            target_bucket_count=TARGET_BUCKET_COUNT,
        ),
        _row(
            "measurement_boundary_verified",
            verdict="fusedmoe_forward_runner_level_verified",
            measurement_boundary=MEASUREMENT_BOUNDARY,
            runner_boundary_api=RUNNER_BOUNDARY_API,
        ),
        _row(
            "environment_key_verified",
            verdict="h200_sxm_vllm019_tp4dp2ep8_verified",
            hardware=HARDWARE,
            vllm_version=VLLM_VERSION,
            topology=TOPOLOGY,
        ),
        _row(
            "quant_runtime_verified",
            verdict="compressed_tensors_marlin_quant_runtime_verified",
            quant_runtime=QUANT_RUNTIME,
        ),
        _row(
            "stop_rules_locked",
            verdict="future_gpu_run_stop_rules_locked",
            stop_rules=STOP_RULES,
        ),
        _row(
            "artifact_contract_locked",
            verdict="future_gpu_artifact_contract_locked",
            artifact_contract="one_row_per_bucket_plus_error_if_stopped",
        ),
        _row(
            "gpu_run_allowed_next_phase_only",
            verdict="gpu_run_can_only_happen_in_phase387",
            future_gpu_run_allowed=TRUE,
        ),
        _row(
            "phase386_no_gpu_no_ssh",
            verdict="phase386_itself_is_local_only",
            gpu_allowed=FALSE,
            ssh_allowed=FALSE,
        ),
        _row(
            "perfdb_write_blocked",
            verdict="no_real_table_write_in_phase386",
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
            verdict="next_phase_is_real_gpu_run",
            next_allowed_phase=NEXT_PHASE,
        ),
    ]
    _validate_rows(rows)
    return rows


def _validate_rows(rows: list[dict[str, str]]) -> None:
    expected_row_types = [
        "phase385_prerequisite",
        "decision_type_locked",
        "target_bucket_set_verified",
        "measurement_boundary_verified",
        "environment_key_verified",
        "quant_runtime_verified",
        "stop_rules_locked",
        "artifact_contract_locked",
        "gpu_run_allowed_next_phase_only",
        "phase386_no_gpu_no_ssh",
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
        _require(row, "gpu_result", FALSE, row["row_type"])
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

    by_type = {row["row_type"]: row for row in rows}
    _require(by_type["decision_type_locked"], "decision_type", "gpu_run_decision_spec", "Phase386 decision")
    _require(by_type["target_bucket_set_verified"], "target_bucket_tokens", TARGET_BUCKETS, "Phase386 buckets")
    _require(by_type["target_bucket_set_verified"], "target_bucket_count", TARGET_BUCKET_COUNT, "Phase386 buckets")
    _require(
        by_type["measurement_boundary_verified"],
        "measurement_boundary",
        MEASUREMENT_BOUNDARY,
        "Phase386 boundary",
    )
    _require(
        by_type["measurement_boundary_verified"],
        "runner_boundary_api",
        RUNNER_BOUNDARY_API,
        "Phase386 boundary",
    )
    _require(by_type["environment_key_verified"], "hardware", HARDWARE, "Phase386 env")
    _require(by_type["environment_key_verified"], "vllm_version", VLLM_VERSION, "Phase386 env")
    _require(by_type["environment_key_verified"], "topology", TOPOLOGY, "Phase386 env")
    _require(by_type["quant_runtime_verified"], "quant_runtime", QUANT_RUNTIME, "Phase386 quant")
    _require(by_type["stop_rules_locked"], "stop_rules", STOP_RULES, "Phase386 stop rules")
    _require(by_type["next_phase"], "next_allowed_phase", NEXT_PHASE, "Phase386 next")


def write_phase386_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_phase386_md(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    buckets = next(row for row in rows if row["row_type"] == "target_bucket_set_verified")
    env = next(row for row in rows if row["row_type"] == "environment_key_verified")
    stop = next(row for row in rows if row["row_type"] == "stop_rules_locked")
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = f"""# Phase386 FusedMoE Paired Bucket GPU Run Decision Spec

Phase386 is a GPU run decision spec. This is not GPU result evidence. It does not SSH, run GPU, write real data, or open Default AIC.

| Gate | Decision |
|---|---|
| decision type | GPU run decision spec |
| target buckets | {buckets["target_bucket_tokens"]} |
| measurement boundary | FusedMoE.forward() runner level |
| hardware | {env["hardware"]} |
| vLLM version | {env["vllm_version"]} |
| topology | {env["topology"]} |
| quant runtime | {QUANT_RUNTIME} |
| stop rules | {stop["stop_rules"]} |
| Phase386 GPU/SSH | blocked |
| Default AIC | No-Go |

The next allowed phase is `{NEXT_PHASE}`.
"""
    path.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase385-csv", type=Path, default=DEFAULT_PHASE385_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    rows = analyze_phase386_fusedmoe_paired_bucket_gpu_run_decision_spec(args.phase385_csv)
    write_phase386_csv(args.output_csv, rows)
    write_phase386_md(args.output_md, rows)


if __name__ == "__main__":
    main()
