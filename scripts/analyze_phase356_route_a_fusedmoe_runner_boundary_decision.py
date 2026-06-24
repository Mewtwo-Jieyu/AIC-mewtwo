from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE349_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase349_route_a_moe_measurement_api_contract.csv"
)
DEFAULT_PHASE351_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase351_route_a_kernel_source_key_contract.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase356_route_a_fusedmoe_runner_boundary_decision.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase356_route_a_fusedmoe_runner_boundary_decision.md"
)

SOURCE = "phase356_route_a_fusedmoe_runner_boundary_decision"
PHASE349_SOURCE = "phase349_route_a_moe_measurement_api_contract"
PHASE351_SOURCE = "phase351_route_a_kernel_source_key_contract"
DEFAULT_READINESS = "No-Go"
TRUE = "true"
FALSE = "false"

FIELDNAMES = [
    "source",
    "worker",
    "vllm_version",
    "source_root",
    "measurement_boundary",
    "runner_boundary_api",
    "locked_kernel_path_guard_required",
    "locked_kernel_path_guard_decision",
    "kernel_source_lookup_key",
    "kernel_source_metadata",
    "runtime_dispatch_deterministic",
    "runtime_dispatch_scope",
    "source_check_evidence",
    "source_check_interpretation",
    "blocking_reasons",
    "candidate_kernel_paths",
    "gpu_smoke_readiness",
    "next_allowed_phase",
    "gpu_allowed",
    "default_readiness",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]

EXPECTED_PHASE349 = [
    ("primary_measurement_boundary", "candidate_fusedmoe_default_runner_boundary"),
    ("bare_fused_experts_probe", "auxiliary_kernel_probe_not_perfdb_row"),
    (
        "required_input_tensors",
        "blocked_until_hidden_states_router_weights_expert_weights_defined",
    ),
    (
        "required_quant_fields",
        "blocked_until_dtype_quant_method_kernel_source_defined",
    ),
    ("kernel_source_capture", "required_for_any_measurement_row"),
    ("measurement_output", "latency_ms_only_no_default_prediction"),
    ("gpu_smoke_readiness", "blocked_until_api_contract_and_kernel_source_clear"),
]

EXPECTED_PHASE351 = [
    ("kernel_source_identity", "required_key_dimension"),
    ("logical_moe_key_without_kernel_source", "rejected_ambiguous_identity"),
    ("lock_single_kernel_path_option", "allowed_only_with_source_check_guard"),
    (
        "perfdb_schema_update_option",
        "candidate_requires_loader_query_contract_change",
    ),
    ("measurement_row_eligibility", "blocked_until_kernel_source_identity_defined"),
    ("gpu_smoke_readiness", "blocked_until_kernel_source_contract_clears"),
]

COMMON_FLAGS = {
    "gpu_allowed": FALSE,
    "default_readiness": DEFAULT_READINESS,
    "diagnostic_only": TRUE,
    "valid_for_default": FALSE,
    "perf_database": FALSE,
}

SPEC_ROWS = [
    {
        "source": SOURCE,
        "worker": "worker-892rz",
        "vllm_version": "0.19.0",
        "source_root": "/usr/local/lib/python3.12/dist-packages/vllm",
        "measurement_boundary": "fusedmoe_forward_runner_level",
        "runner_boundary_api": "FusedMoE.forward()",
        "locked_kernel_path_guard_required": FALSE,
        "locked_kernel_path_guard_decision": "superseded_not_applicable",
        "kernel_source_lookup_key": FALSE,
        "kernel_source_metadata": TRUE,
        "runtime_dispatch_deterministic": TRUE,
        "runtime_dispatch_scope": "fixed_model_config_hw_vllm_version_tuple",
        "source_check_evidence": (
            "KimiMoE;FusedMoE;DefaultMoERunner;torch.ops.vllm.moe_forward"
        ),
        "source_check_interpretation": (
            "single_kernel_guard_not_applicable_runner_measurement_retained"
        ),
        "blocking_reasons": (
            "quant_method_runtime_selection;unquantized_backend_runtime_selection;"
            "shape_specific_fallback;multiple_expert_kernel_classes"
        ),
        "candidate_kernel_paths": "Triton;Cutlass;DeepGemm;FlashInfer;Marlin;TRTLLM",
        "gpu_smoke_readiness": (
            "blocked_until_fusedmoe_forward_tensor_shapes_defined"
        ),
        "next_allowed_phase": "phase358_moe_tensor_shape_smoke_spec",
        **COMMON_FLAGS,
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


def _require_flags(row: dict[str, str], label: str) -> None:
    _require(row, "gpu_allowed", FALSE, label)
    _require(row, "default_readiness", DEFAULT_READINESS, label)
    _require(row, "diagnostic_only", TRUE, label)
    _require(row, "valid_for_default", FALSE, label)
    _require(row, "perf_database", FALSE, label)


def _require_phase349_moe_contract(path: Path) -> None:
    rows = _read_csv(path)
    if len(rows) != len(EXPECTED_PHASE349):
        raise ValueError(f"Phase349 must have exactly 7 rows, got {len(rows)}")

    for row, (contract_row, decision) in zip(rows, EXPECTED_PHASE349):
        label = contract_row
        _require(row, "source", PHASE349_SOURCE, label)
        _require(row, "contract_row", contract_row, label)
        _require(row, "decision", decision, label)
        _require_flags(row, label)

    contracts = {row["contract_row"]: row["required_contract"] for row in rows}
    required_fragments = {
        "primary_measurement_boundary": "default_vllm_fusedmoe_runner_boundary",
        "kernel_source_capture": "kernel_source_must_be_part",
        "gpu_smoke_readiness": "kernel_source_contract",
    }
    for contract_row, fragment in required_fragments.items():
        if fragment not in contracts.get(contract_row, ""):
            raise ValueError(
                f"{contract_row} missing required contract fragment {fragment!r}"
            )


def _require_phase351_kernel_source_contract(path: Path) -> None:
    rows = _read_csv(path)
    if len(rows) != len(EXPECTED_PHASE351):
        raise ValueError(f"Phase351 must have exactly 6 rows, got {len(rows)}")

    for row, (contract_row, decision) in zip(rows, EXPECTED_PHASE351):
        label = contract_row
        _require(row, "source", PHASE351_SOURCE, label)
        _require(row, "contract_row", contract_row, label)
        _require(row, "decision", decision, label)
        _require_flags(row, label)

    contracts = {row["contract_row"]: row["key_contract"] for row in rows}
    required_fragments = {
        "kernel_source_identity": "include_kernel_source",
        "perfdb_schema_update_option": "add_kernel_source",
        "gpu_smoke_readiness": "kernel_source_contract",
    }
    for contract_row, fragment in required_fragments.items():
        if fragment not in contracts.get(contract_row, ""):
            raise ValueError(
                f"{contract_row} missing key contract fragment {fragment!r}"
            )


def analyze_route_a_fusedmoe_runner_boundary_decision(
    phase349_csv: Path = DEFAULT_PHASE349_CSV,
    phase351_csv: Path = DEFAULT_PHASE351_CSV,
) -> list[dict[str, str]]:
    _require_phase349_moe_contract(phase349_csv)
    _require_phase351_kernel_source_contract(phase351_csv)
    return [dict(row) for row in SPEC_ROWS]


def _validate_output_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != 1:
        raise ValueError(f"Phase356 output must have exactly 1 row, got {len(rows)}")

    row = rows[0]
    _require(row, "source", SOURCE, "phase356")
    _require(row, "worker", "worker-892rz", "phase356")
    _require(row, "vllm_version", "0.19.0", "phase356")
    _require(
        row,
        "source_root",
        "/usr/local/lib/python3.12/dist-packages/vllm",
        "phase356",
    )
    _require(
        row,
        "measurement_boundary",
        "fusedmoe_forward_runner_level",
        "phase356",
    )
    _require(row, "runner_boundary_api", "FusedMoE.forward()", "phase356")
    _require(row, "locked_kernel_path_guard_required", FALSE, "phase356")
    _require(row, "kernel_source_lookup_key", FALSE, "phase356")
    _require(row, "kernel_source_metadata", TRUE, "phase356")
    _require(row, "runtime_dispatch_deterministic", TRUE, "phase356")
    _require(
        row,
        "runtime_dispatch_scope",
        "fixed_model_config_hw_vllm_version_tuple",
        "phase356",
    )
    _require(
        row,
        "gpu_smoke_readiness",
        "blocked_until_fusedmoe_forward_tensor_shapes_defined",
        "phase356",
    )
    _require(
        row,
        "next_allowed_phase",
        "phase358_moe_tensor_shape_smoke_spec",
        "phase356",
    )
    _require_flags(row, "phase356")


def write_route_a_fusedmoe_runner_boundary_decision_csv(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    _validate_output_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_route_a_fusedmoe_runner_boundary_decision_doc(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    _validate_output_rows(rows)
    row = rows[0]
    lines = [
        "# Phase356 Route A FusedMoE Runner Boundary Decision",
        "",
        "| Item | Decision |",
        "|---|---|",
        "| Default AIC | No-Go |",
        "| GPU allowed | false |",
        "| Diagnostic only | true |",
        "| Valid for default | false |",
        "| PerfDatabase | false |",
        "",
        "Phase356 changes the Route A MoE measurement boundary to the "
        "FusedMoE.forward() runner level. It does not SSH, does not run GPU, "
        "does not change runtime, and does not write PerfDatabase rows.",
        "",
        "kernel_source stays measurement metadata. It is not a PerfDatabase "
        "lookup key for this runner-level measurement decision.",
        "",
        "Runtime dispatch is deterministic only inside a fixed model config, "
        "hardware, and vLLM version tuple. The next GPU step remains blocked "
        "until FusedMoE.forward() tensor shapes are defined.",
        "",
        "## Decision Evidence (Phase355 Source Check Evidence)",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| worker | {row['worker']} |",
        f"| vllm_version | {row['vllm_version']} |",
        f"| source_root | {row['source_root']} |",
        f"| blocking_reasons | {row['blocking_reasons']} |",
        f"| candidate_kernel_paths | {row['candidate_kernel_paths']} |",
        "",
        "These facts show that a source-level single kernel guard is not "
        "applicable. They do not block runner-level measurement because the "
        "measurement boundary is FusedMoE.forward() runner level.",
        "",
        "The locked kernel path guard is not applicable for this boundary and "
        "is superseded by the runner-level contract.",
        "",
        "| Field | Value |",
        "|---|---|",
    ]
    for field in FIELDNAMES:
        lines.append(f"| {field} | {row[field]} |")
    lines.extend(
        [
            "",
            "Conclusion: Route A proceeds only as a diagnostic FusedMoE.forward() "
            "runner-level measurement spec. GPU smoke is still blocked until "
            "tensor shapes are defined.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write the Phase356 Route A FusedMoE runner boundary decision."
    )
    parser.add_argument("--phase349-csv", type=Path, default=DEFAULT_PHASE349_CSV)
    parser.add_argument("--phase351-csv", type=Path, default=DEFAULT_PHASE351_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = analyze_route_a_fusedmoe_runner_boundary_decision(
        args.phase349_csv,
        args.phase351_csv,
    )
    write_route_a_fusedmoe_runner_boundary_decision_csv(args.output_csv, rows)
    write_route_a_fusedmoe_runner_boundary_decision_doc(args.output_md, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
