from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE347_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase347_route_a_schema_source_reconciliation.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase349_route_a_moe_measurement_api_contract.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase349_route_a_moe_measurement_api_contract.md"
)

SOURCE = "phase349_route_a_moe_measurement_api_contract"
PHASE347_SOURCE = "phase347_route_a_schema_source_reconciliation"
DEFAULT_READINESS = "No-Go"
TRUE = "true"
FALSE = "false"

FIELDNAMES = [
    "source",
    "contract_row",
    "measurement_boundary",
    "decision",
    "required_contract",
    "next_required_action",
    "gpu_allowed",
    "default_readiness",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]

EXPECTED_PHASE347 = [
    (
        "remote_vllm_version_source_root",
        "cleared_baseline_only",
    ),
    (
        "moe_measurement_api_contract",
        "blocked_define_fusedmoe_or_fused_experts_api",
    ),
    (
        "kernel_source_key_contract",
        "blocked_add_or_lock_kernel_source",
    ),
    (
        "vllm_ep8_comm_schema_contract",
        "blocked_define_vllm_ep8_comm_scope",
    ),
    (
        "attention_gemm_reuse_policy",
        "blocked_assumption_pending_0190_validation",
    ),
    (
        "token_shape_fidelity_input",
        "retained_prerun_only_no_trace_feedback",
    ),
    (
        "gpu_smoke_design",
        "blocked_until_contracts_clear",
    ),
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
        "contract_row": "primary_measurement_boundary",
        "measurement_boundary": "fusedmoe_default_runner",
        "decision": "candidate_fusedmoe_default_runner_boundary",
        "required_contract": "measure_the_default_vllm_fusedmoe_runner_boundary_before_perfdb_row",
        "next_required_action": "define_fusedmoe_default_runner_call_contract",
        **COMMON_FLAGS,
    },
    {
        "source": SOURCE,
        "contract_row": "bare_fused_experts_probe",
        "measurement_boundary": "bare_fused_experts",
        "decision": "auxiliary_kernel_probe_not_perfdb_row",
        "required_contract": "only_use_bare_fused_experts_to_explain_kernel_behavior_not_default_model",
        "next_required_action": "keep_bare_probe_out_of_perfdb_until_boundary_matches_route_a",
        **COMMON_FLAGS,
    },
    {
        "source": SOURCE,
        "contract_row": "required_input_tensors",
        "measurement_boundary": "fusedmoe_default_runner",
        "decision": "blocked_until_hidden_states_router_weights_expert_weights_defined",
        "required_contract": "hidden_states_router_weights_expert_weights_and_token_shapes",
        "next_required_action": "define_tensor_shape_dtype_and_ownership_for_measurement",
        **COMMON_FLAGS,
    },
    {
        "source": SOURCE,
        "contract_row": "required_quant_fields",
        "measurement_boundary": "fusedmoe_default_runner",
        "decision": "blocked_until_dtype_quant_method_kernel_source_defined",
        "required_contract": "dtype_quant_method_kernel_source_and_distribution",
        "next_required_action": "define_quant_and_kernel_source_fields_before_smoke",
        **COMMON_FLAGS,
    },
    {
        "source": SOURCE,
        "contract_row": "kernel_source_capture",
        "measurement_boundary": "all_moe_measurement_rows",
        "decision": "required_for_any_measurement_row",
        "required_contract": "kernel_source_must_be_part_of_any_route_a_moe_measurement_identity",
        "next_required_action": "write_kernel_source_key_contract_next",
        **COMMON_FLAGS,
    },
    {
        "source": SOURCE,
        "contract_row": "measurement_output",
        "measurement_boundary": "route_a_moe_measurement",
        "decision": "latency_ms_only_no_default_prediction",
        "required_contract": "output_is_latency_ms_not_throughput_prediction_or_default_correction",
        "next_required_action": "keep_measurement_output_out_of_default_aic",
        **COMMON_FLAGS,
    },
    {
        "source": SOURCE,
        "contract_row": "gpu_smoke_readiness",
        "measurement_boundary": "route_a_moe_smoke",
        "decision": "blocked_until_api_contract_and_kernel_source_clear",
        "required_contract": "gpu_smoke_requires_moe_api_contract_and_kernel_source_contract_first",
        "next_required_action": "clear_moe_api_contract_and_kernel_source_before_gpu_smoke",
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


def _require_phase347_reconciliation_gates(path: Path) -> None:
    rows = _read_csv(path)
    if len(rows) != len(EXPECTED_PHASE347):
        raise ValueError(f"Phase347 must have exactly 7 rows, got {len(rows)}")

    for row, (gate, decision) in zip(rows, EXPECTED_PHASE347):
        label = gate
        _require(row, "source", PHASE347_SOURCE, label)
        _require(row, "gate", gate, label)
        _require(row, "decision", decision, label)
        _require(row, "gpu_allowed", FALSE, label)
        _require(row, "default_readiness", DEFAULT_READINESS, label)
        _require(row, "diagnostic_only", TRUE, label)
        _require(row, "valid_for_default", FALSE, label)
        _require(row, "perf_database", FALSE, label)

    required_actions = {
        row["gate"]: row["next_required_action"]
        for row in rows
        if row["gate"]
        in {
            "moe_measurement_api_contract",
            "kernel_source_key_contract",
            "gpu_smoke_design",
        }
    }
    required_fragments = {
        "moe_measurement_api_contract": "define_measurement_api",
        "kernel_source_key_contract": "kernel_source",
        "gpu_smoke_design": "before_any_gpu_smoke_spec",
    }
    for gate, fragment in required_fragments.items():
        if fragment not in required_actions.get(gate, ""):
            raise ValueError(f"{gate} missing next action fragment {fragment!r}")


def analyze_route_a_moe_measurement_api_contract(
    phase347_csv: Path = DEFAULT_PHASE347_CSV,
) -> list[dict[str, str]]:
    _require_phase347_reconciliation_gates(phase347_csv)
    return [dict(row) for row in SPEC_ROWS]


def _validate_output_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != 7:
        raise ValueError(f"Phase349 output must have exactly 7 rows, got {len(rows)}")

    expected_rows = [row["contract_row"] for row in SPEC_ROWS]
    contract_rows = [row.get("contract_row") for row in rows]
    if contract_rows != expected_rows:
        raise ValueError(f"Phase349 contract row order mismatch: {contract_rows!r}")

    expected_decisions = [row["decision"] for row in SPEC_ROWS]
    decisions = [row.get("decision") for row in rows]
    if decisions != expected_decisions:
        raise ValueError(f"Phase349 decision order mismatch: {decisions!r}")

    for row in rows:
        label = row["contract_row"]
        _require(row, "source", SOURCE, label)
        _require(row, "gpu_allowed", FALSE, label)
        _require(row, "default_readiness", DEFAULT_READINESS, label)
        _require(row, "diagnostic_only", TRUE, label)
        _require(row, "valid_for_default", FALSE, label)
        _require(row, "perf_database", FALSE, label)


def write_route_a_moe_measurement_api_contract_csv(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    _validate_output_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_route_a_moe_measurement_api_contract_doc(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    _validate_output_rows(rows)
    lines = [
        "# Phase349 Route A MoE Measurement API Contract",
        "",
        "| Item | Decision |",
        "|---|---|",
        "| Default AIC | No-Go |",
        "| GPU allowed | false |",
        "| Diagnostic only | true |",
        "| Valid for default | false |",
        "| PerfDatabase | false |",
        "",
        "Phase349 only defines the Route A MoE measurement API contract. It "
        "does not run GPU, does not change runtime behavior, and does not write "
        "PerfDatabase rows.",
        "",
        "The primary candidate boundary is the vLLM FusedMoE default runner "
        "boundary. A bare fused_experts is only an auxiliary kernel probe, not "
        "a PerfDatabase row, because it does not prove the default Route A "
        "module boundary used by vLLM.",
        "",
        "The measurement output is latency_ms only. It is not a throughput "
        "prediction, not a global correction, and not default evidence.",
        "",
        "GPU smoke remains blocked until tensor inputs, quant fields, and "
        "kernel_source identity are defined first.",
        "",
        "| Contract row | Boundary | Decision | Required contract |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {contract_row} | {measurement_boundary} | {decision} | {required_contract} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Conclusion: Route A MoE measurement is still diagnostic-only. The "
            "next step is the kernel_source key contract; EP8 comm schema is "
            "out of scope for this phase.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write the Phase349 Route A MoE measurement API contract."
    )
    parser.add_argument("--phase347-csv", type=Path, default=DEFAULT_PHASE347_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = analyze_route_a_moe_measurement_api_contract(args.phase347_csv)
    write_route_a_moe_measurement_api_contract_csv(args.output_csv, rows)
    write_route_a_moe_measurement_api_contract_doc(args.output_md, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
