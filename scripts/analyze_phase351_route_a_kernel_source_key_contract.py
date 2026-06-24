from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE349_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase349_route_a_moe_measurement_api_contract.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase351_route_a_kernel_source_key_contract.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase351_route_a_kernel_source_key_contract.md"
)

SOURCE = "phase351_route_a_kernel_source_key_contract"
PHASE349_SOURCE = "phase349_route_a_moe_measurement_api_contract"
DEFAULT_READINESS = "No-Go"
TRUE = "true"
FALSE = "false"

FIELDNAMES = [
    "source",
    "contract_row",
    "key_contract",
    "decision",
    "allowed_route",
    "blocking_condition",
    "next_required_action",
    "gpu_allowed",
    "default_readiness",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]

EXPECTED_PHASE349 = [
    (
        "primary_measurement_boundary",
        "candidate_fusedmoe_default_runner_boundary",
    ),
    (
        "bare_fused_experts_probe",
        "auxiliary_kernel_probe_not_perfdb_row",
    ),
    (
        "required_input_tensors",
        "blocked_until_hidden_states_router_weights_expert_weights_defined",
    ),
    (
        "required_quant_fields",
        "blocked_until_dtype_quant_method_kernel_source_defined",
    ),
    (
        "kernel_source_capture",
        "required_for_any_measurement_row",
    ),
    (
        "measurement_output",
        "latency_ms_only_no_default_prediction",
    ),
    (
        "gpu_smoke_readiness",
        "blocked_until_api_contract_and_kernel_source_clear",
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
        "contract_row": "kernel_source_identity",
        "key_contract": "include_kernel_source_in_route_a_moe_measurement_identity",
        "decision": "required_key_dimension",
        "allowed_route": "schema_update_or_locked_source_guard",
        "blocking_condition": "kernel_source_not_yet_defined_as_key_dimension",
        "next_required_action": "define_kernel_source_value_set_and_capture_point",
        **COMMON_FLAGS,
    },
    {
        "source": SOURCE,
        "contract_row": "logical_moe_key_without_kernel_source",
        "key_contract": "reject_quant_distribution_topk_experts_hidden_inter_moe_tp_moe_ep_num_tokens_without_kernel_source",
        "decision": "rejected_ambiguous_identity",
        "allowed_route": "not_allowed",
        "blocking_condition": "same_logical_moe_key_can_map_to_multiple_kernel_paths",
        "next_required_action": "do_not_reuse_old_vllm_moe_key_without_kernel_source",
        **COMMON_FLAGS,
    },
    {
        "source": SOURCE,
        "contract_row": "lock_single_kernel_path_option",
        "key_contract": "lock_exactly_one_kernel_source_with_source_check_guard",
        "decision": "allowed_only_with_source_check_guard",
        "allowed_route": "single_kernel_path_lock",
        "blocking_condition": "requires_source_check_to_prove_one_kernel_path",
        "next_required_action": "write_source_check_guard_for_locked_kernel_path",
        **COMMON_FLAGS,
    },
    {
        "source": SOURCE,
        "contract_row": "perfdb_schema_update_option",
        "key_contract": "add_kernel_source_to_loader_query_and_measurement_row_identity",
        "decision": "candidate_requires_loader_query_contract_change",
        "allowed_route": "schema_update_candidate",
        "blocking_condition": "perfdb_loader_query_contract_not_changed_in_phase351",
        "next_required_action": "open_separate_phase_for_perfdb_loader_query_schema",
        **COMMON_FLAGS,
    },
    {
        "source": SOURCE,
        "contract_row": "measurement_row_eligibility",
        "key_contract": "measurement_row_requires_kernel_source_identity_before_perfdb",
        "decision": "blocked_until_kernel_source_identity_defined",
        "allowed_route": "blocked",
        "blocking_condition": "measurement_row_identity_incomplete",
        "next_required_action": "choose_kernel_source_key_or_locked_path_before_row_is_eligible",
        **COMMON_FLAGS,
    },
    {
        "source": SOURCE,
        "contract_row": "gpu_smoke_readiness",
        "key_contract": "gpu_smoke_requires_kernel_source_contract_before_run",
        "decision": "blocked_until_kernel_source_contract_clears",
        "allowed_route": "blocked",
        "blocking_condition": "kernel_source_contract_not_cleared",
        "next_required_action": "choose_schema_update_or_locked_kernel_path_before_gpu_smoke",
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


def _require_phase349_moe_contract(path: Path) -> None:
    rows = _read_csv(path)
    if len(rows) != len(EXPECTED_PHASE349):
        raise ValueError(f"Phase349 must have exactly 7 rows, got {len(rows)}")

    for row, (contract_row, decision) in zip(rows, EXPECTED_PHASE349):
        label = contract_row
        _require(row, "source", PHASE349_SOURCE, label)
        _require(row, "contract_row", contract_row, label)
        _require(row, "decision", decision, label)
        _require(row, "gpu_allowed", FALSE, label)
        _require(row, "default_readiness", DEFAULT_READINESS, label)
        _require(row, "diagnostic_only", TRUE, label)
        _require(row, "valid_for_default", FALSE, label)
        _require(row, "perf_database", FALSE, label)

    required_contracts = {
        row["contract_row"]: row["required_contract"]
        for row in rows
        if row["contract_row"]
        in {
            "kernel_source_capture",
            "required_quant_fields",
            "gpu_smoke_readiness",
        }
    }
    required_fragments = {
        "kernel_source_capture": "kernel_source_must_be_part",
        "required_quant_fields": "kernel_source",
        "gpu_smoke_readiness": "kernel_source_contract",
    }
    for contract_row, fragment in required_fragments.items():
        if fragment not in required_contracts.get(contract_row, ""):
            raise ValueError(
                f"{contract_row} missing required contract fragment {fragment!r}"
            )


def analyze_route_a_kernel_source_key_contract(
    phase349_csv: Path = DEFAULT_PHASE349_CSV,
) -> list[dict[str, str]]:
    _require_phase349_moe_contract(phase349_csv)
    return [dict(row) for row in SPEC_ROWS]


def _validate_output_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != 6:
        raise ValueError(f"Phase351 output must have exactly 6 rows, got {len(rows)}")

    expected_rows = [row["contract_row"] for row in SPEC_ROWS]
    contract_rows = [row.get("contract_row") for row in rows]
    if contract_rows != expected_rows:
        raise ValueError(f"Phase351 contract row order mismatch: {contract_rows!r}")

    expected_decisions = [row["decision"] for row in SPEC_ROWS]
    decisions = [row.get("decision") for row in rows]
    if decisions != expected_decisions:
        raise ValueError(f"Phase351 decision order mismatch: {decisions!r}")

    for row in rows:
        label = row["contract_row"]
        _require(row, "source", SOURCE, label)
        _require(row, "gpu_allowed", FALSE, label)
        _require(row, "default_readiness", DEFAULT_READINESS, label)
        _require(row, "diagnostic_only", TRUE, label)
        _require(row, "valid_for_default", FALSE, label)
        _require(row, "perf_database", FALSE, label)


def write_route_a_kernel_source_key_contract_csv(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    _validate_output_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_route_a_kernel_source_key_contract_doc(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    _validate_output_rows(rows)
    lines = [
        "# Phase351 Route A kernel_source Key Contract",
        "",
        "| Item | Decision |",
        "|---|---|",
        "| Default AIC | No-Go |",
        "| GPU allowed | false |",
        "| Diagnostic only | true |",
        "| Valid for default | false |",
        "| PerfDatabase | false |",
        "",
        "Phase351 only defines the Route A kernel_source key contract. It "
        "does not update PerfDatabase loader or query code, and it does not "
        "run GPU.",
        "",
        "There are two legal routes before any measurement row can become "
        "eligible: include kernel_source in the key, or lock exactly one "
        "kernel path with a source-check guard.",
        "",
        "The ambiguous old vLLM MoE key is rejected. A logical key made only "
        "from quant, distribution, topk, experts, hidden, inter, moe_tp, "
        "moe_ep, and num_tokens is not enough when multiple kernel paths can "
        "produce different latency.",
        "",
        "The schema update route is only a candidate. A separate phase must "
        "change loader/query tests before any PerfDatabase row can be written.",
        "",
        "| Contract row | Key contract | Decision | Next required action |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {contract_row} | {key_contract} | {decision} | {next_required_action} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Conclusion: GPU smoke and PerfDatabase writes remain blocked "
            "until the kernel_source identity is either added to the key "
            "contract or locked to a single source-checked path.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write the Phase351 Route A kernel_source key contract."
    )
    parser.add_argument("--phase349-csv", type=Path, default=DEFAULT_PHASE349_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = analyze_route_a_kernel_source_key_contract(args.phase349_csv)
    write_route_a_kernel_source_key_contract_csv(args.output_csv, rows)
    write_route_a_kernel_source_key_contract_doc(args.output_md, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
