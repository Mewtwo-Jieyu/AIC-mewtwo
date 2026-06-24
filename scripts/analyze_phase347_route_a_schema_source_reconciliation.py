from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE344_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase344_route_a_source_schema_audit.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase347_route_a_schema_source_reconciliation.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase347_route_a_schema_source_reconciliation.md"
)

SOURCE = "phase347_route_a_schema_source_reconciliation"
PHASE344_SOURCE = "phase344_route_a_source_schema_audit"
DEFAULT_READINESS = "No-Go"
TRUE = "true"
FALSE = "false"

FIELDNAMES = [
    "source",
    "gate",
    "phase346_fact",
    "decision",
    "next_required_action",
    "gpu_allowed",
    "default_readiness",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]

EXPECTED_PHASE344 = [
    (
        "vllm_0190_moe_kernel_api",
        "blocked_pending_vllm_0190_source_check",
    ),
    (
        "perfdb_moe_kernel_source_schema",
        "blocked_perfdb_schema_missing_kernel_source_key",
    ),
    (
        "vllm_ep8_comm_scope",
        "blocked_ep8_comm_schema_pending_source_check",
    ),
    (
        "legacy_attention_gemm_reuse_assumption",
        "assumption_pending_validation",
    ),
    (
        "cb_sim_token_shape_fidelity_input",
        "retained_as_prerun_input_dependency",
    ),
    (
        "full_trace_feedback_guard",
        "rejected_data_leakage",
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
        "gate": "remote_vllm_version_source_root",
        "phase346_fact": "remote_vllm_version_0_19_0_source_root_cleared",
        "decision": "cleared_baseline_only",
        "next_required_action": "use_as_source_check_baseline_only",
        **COMMON_FLAGS,
    },
    {
        "source": SOURCE,
        "gate": "moe_measurement_api_contract",
        "phase346_fact": "kimi_moe_calls_fusedmoe_default_runner_and_torch_ops_moe_forward",
        "decision": "blocked_define_fusedmoe_or_fused_experts_api",
        "next_required_action": "define_measurement_api_tensor_shapes_weights_router_quant",
        **COMMON_FLAGS,
    },
    {
        "source": SOURCE,
        "gate": "kernel_source_key_contract",
        "phase346_fact": "multiple_moe_kernel_paths_possible_under_same_logical_query",
        "decision": "blocked_add_or_lock_kernel_source",
        "next_required_action": "add_kernel_source_to_perfdb_key_or_lock_single_kernel_path",
        **COMMON_FLAGS,
    },
    {
        "source": SOURCE,
        "gate": "vllm_ep8_comm_schema_contract",
        "phase346_fact": "vllm_ep8_comm_has_multiple_all2all_backends_not_wideep_schema",
        "decision": "blocked_define_vllm_ep8_comm_scope",
        "next_required_action": "define_vllm_ep8_comm_backend_and_measured_schema",
        **COMMON_FLAGS,
    },
    {
        "source": SOURCE,
        "gate": "attention_gemm_reuse_policy",
        "phase346_fact": "local_h200_vllm_perf_tables_are_0_12_0_only",
        "decision": "blocked_assumption_pending_0190_validation",
        "next_required_action": "validate_0190_attention_gemm_kernel_drift_before_reuse",
        **COMMON_FLAGS,
    },
    {
        "source": SOURCE,
        "gate": "token_shape_fidelity_input",
        "phase346_fact": "token_shape_remains_prerun_input_only",
        "decision": "retained_prerun_only_no_trace_feedback",
        "next_required_action": "keep_cb_sim_token_shape_as_gate_without_trace_backfill",
        **COMMON_FLAGS,
    },
    {
        "source": SOURCE,
        "gate": "gpu_smoke_design",
        "phase346_fact": "single_point_gpu_smoke_not_ready",
        "decision": "blocked_until_contracts_clear",
        "next_required_action": "write_schema_source_reconciliation_before_any_gpu_smoke_spec",
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


def _require_phase344_source_schema_gates(path: Path) -> None:
    rows = _read_csv(path)
    if len(rows) != len(EXPECTED_PHASE344):
        raise ValueError(f"Phase344 must have exactly 6 rows, got {len(rows)}")

    for row, (gate, decision) in zip(rows, EXPECTED_PHASE344):
        label = gate
        _require(row, "source", PHASE344_SOURCE, label)
        _require(row, "gate", gate, label)
        _require(row, "decision", decision, label)
        _require(row, "gpu_allowed", FALSE, label)
        _require(row, "default_readiness", DEFAULT_READINESS, label)
        _require(row, "diagnostic_only", TRUE, label)
        _require(row, "valid_for_default", FALSE, label)
        _require(row, "perf_database", FALSE, label)

    blockers = {
        row["gate"]: row["blocking_condition"]
        for row in rows
        if row["gate"] in {
            "vllm_0190_moe_kernel_api",
            "perfdb_moe_kernel_source_schema",
            "vllm_ep8_comm_scope",
            "legacy_attention_gemm_reuse_assumption",
        }
    }
    required_blocker_fragments = {
        "vllm_0190_moe_kernel_api": "not_locally_verified",
        "perfdb_moe_kernel_source_schema": "does_not_key_kernel_source",
        "vllm_ep8_comm_scope": "not_vllm_ep8_alltoall_schema",
        "legacy_attention_gemm_reuse_assumption": "0120_only",
    }
    for gate, fragment in required_blocker_fragments.items():
        if fragment not in blockers.get(gate, ""):
            raise ValueError(f"{gate} missing blocker fragment {fragment!r}")


def analyze_route_a_schema_source_reconciliation(
    phase344_csv: Path = DEFAULT_PHASE344_CSV,
) -> list[dict[str, str]]:
    _require_phase344_source_schema_gates(phase344_csv)
    return [dict(row) for row in SPEC_ROWS]


def _validate_output_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != 7:
        raise ValueError(f"Phase347 output must have exactly 7 rows, got {len(rows)}")

    expected_gates = [row["gate"] for row in SPEC_ROWS]
    gates = [row.get("gate") for row in rows]
    if gates != expected_gates:
        raise ValueError(f"Phase347 gate order mismatch: {gates!r}")

    expected_decisions = [row["decision"] for row in SPEC_ROWS]
    decisions = [row.get("decision") for row in rows]
    if decisions != expected_decisions:
        raise ValueError(f"Phase347 decision order mismatch: {decisions!r}")

    for row in rows:
        label = row["gate"]
        _require(row, "source", SOURCE, label)
        _require(row, "gpu_allowed", FALSE, label)
        _require(row, "default_readiness", DEFAULT_READINESS, label)
        _require(row, "diagnostic_only", TRUE, label)
        _require(row, "valid_for_default", FALSE, label)
        _require(row, "perf_database", FALSE, label)


def write_route_a_schema_source_reconciliation_csv(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    _validate_output_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_route_a_schema_source_reconciliation_doc(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    _validate_output_rows(rows)
    lines = [
        "# Phase347 Route A Schema Source Reconciliation",
        "",
        "| Item | Decision |",
        "|---|---|",
        "| Default AIC | No-Go |",
        "| GPU allowed | false |",
        "| Diagnostic only | true |",
        "| Valid for default | false |",
        "| PerfDatabase | false |",
        "",
        "Phase346 cleared only the remote vLLM baseline: H worker vLLM is "
        "0.19.0 and the source root was locatable. That is not enough for a "
        "single-point GPU smoke.",
        "",
        "This is not a GPU smoke spec. The next step is contract/spec work: "
        "MoE measurement API contract, kernel_source schema contract, and "
        "vLLM EP8 comm schema.",
        "",
        "MoE measurement API contract must decide whether Route A measures "
        "FusedMoE or bare fused_experts, and must define tensor shapes, "
        "weights, router logits, dtype, quant, and kernel path before any GPU.",
        "",
        "kernel_source schema contract is still blocked because one logical "
        "MoE query may map to multiple vLLM kernel paths. PerfDatabase cannot "
        "accept that ambiguity as a default model key.",
        "",
        "vLLM EP8 comm schema is still blocked because vLLM exposes multiple "
        "all2all backends; WideEP schema must not be reused.",
        "",
        "| Gate | Phase346 fact | Decision | Next required action |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {gate} | {phase346_fact} | {decision} | {next_required_action} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Conclusion: GPU smoke design remains blocked until the MoE "
            "measurement API, kernel_source key contract, and vLLM EP8 comm "
            "schema clear first.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write the Phase347 Route A schema/source reconciliation audit."
    )
    parser.add_argument("--phase344-csv", type=Path, default=DEFAULT_PHASE344_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = analyze_route_a_schema_source_reconciliation(args.phase344_csv)
    write_route_a_schema_source_reconciliation_csv(args.output_csv, rows)
    write_route_a_schema_source_reconciliation_doc(args.output_md, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
