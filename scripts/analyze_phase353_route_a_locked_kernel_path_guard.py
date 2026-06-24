from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE351_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase351_route_a_kernel_source_key_contract.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase353_route_a_locked_kernel_path_guard.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase353_route_a_locked_kernel_path_guard.md"
)

SOURCE = "phase353_route_a_locked_kernel_path_guard"
PHASE351_SOURCE = "phase351_route_a_kernel_source_key_contract"
DEFAULT_READINESS = "No-Go"
TRUE = "true"
FALSE = "false"

FIELDNAMES = [
    "source",
    "guard_row",
    "guard_contract",
    "decision",
    "blocking_condition",
    "next_required_action",
    "gpu_allowed",
    "default_readiness",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]

EXPECTED_PHASE351 = [
    (
        "kernel_source_identity",
        "required_key_dimension",
    ),
    (
        "logical_moe_key_without_kernel_source",
        "rejected_ambiguous_identity",
    ),
    (
        "lock_single_kernel_path_option",
        "allowed_only_with_source_check_guard",
    ),
    (
        "perfdb_schema_update_option",
        "candidate_requires_loader_query_contract_change",
    ),
    (
        "measurement_row_eligibility",
        "blocked_until_kernel_source_identity_defined",
    ),
    (
        "gpu_smoke_readiness",
        "blocked_until_kernel_source_contract_clears",
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
        "guard_row": "route_choice",
        "guard_contract": "choose_locked_kernel_path_guard_before_schema_update",
        "decision": "locked_kernel_path_guard_first",
        "blocking_condition": "kernel_path_not_yet_source_checked",
        "next_required_action": "write_h_source_check_for_single_kernel_path",
        **COMMON_FLAGS,
    },
    {
        "source": SOURCE,
        "guard_row": "perfdb_schema_update",
        "guard_contract": "do_not_change_perfdb_schema_in_phase353",
        "decision": "deferred_until_measurement_value_set_exists",
        "blocking_condition": "measurement_kernel_source_value_set_not_defined",
        "next_required_action": "defer_schema_update_until_source_checked_value_exists",
        **COMMON_FLAGS,
    },
    {
        "source": SOURCE,
        "guard_row": "kernel_source_value_set",
        "guard_contract": "source_check_must_name_exact_kernel_path",
        "decision": "blocked_pending_source_checked_kernel_name",
        "blocking_condition": "kernel_source_value_set_unknown_before_h_source_check",
        "next_required_action": "collect_source_checked_kernel_name_without_gpu",
        **COMMON_FLAGS,
    },
    {
        "source": SOURCE,
        "guard_row": "source_check_guard",
        "guard_contract": "prove_exactly_one_kernel_path_before_any_gpu_smoke",
        "decision": "required_before_any_gpu_smoke",
        "blocking_condition": "no_locked_path_guard_yet",
        "next_required_action": "fail_if_multiple_kernel_paths_or_missing_anchor",
        **COMMON_FLAGS,
    },
    {
        "source": SOURCE,
        "guard_row": "guard_failure_policy",
        "guard_contract": "fail_fast_without_gpu_when_guard_fails",
        "decision": "fail_fast_no_gpu_run",
        "blocking_condition": "source_check_guard_failure",
        "next_required_action": "stop_and_reconcile_schema_before_gpu",
        **COMMON_FLAGS,
    },
    {
        "source": SOURCE,
        "guard_row": "measurement_row_eligibility",
        "guard_contract": "locked_path_smoke_is_diagnostic_not_perfdb_row",
        "decision": "diagnostic_smoke_only_not_perfdb",
        "blocking_condition": "perfdb_schema_not_updated_and_value_set_not_validated",
        "next_required_action": "keep_measurement_rows_out_of_perfdb",
        **COMMON_FLAGS,
    },
    {
        "source": SOURCE,
        "guard_row": "gpu_smoke_readiness",
        "guard_contract": "gpu_smoke_requires_locked_guard_to_clear_first",
        "decision": "blocked_until_locked_guard_clears",
        "blocking_condition": "locked_kernel_path_guard_not_cleared",
        "next_required_action": "run_h_source_check_only_after_guard_spec_is_committed",
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


def _require_phase351_kernel_source_contract(path: Path) -> None:
    rows = _read_csv(path)
    if len(rows) != len(EXPECTED_PHASE351):
        raise ValueError(f"Phase351 must have exactly 6 rows, got {len(rows)}")

    for row, (contract_row, decision) in zip(rows, EXPECTED_PHASE351):
        label = contract_row
        _require(row, "source", PHASE351_SOURCE, label)
        _require(row, "contract_row", contract_row, label)
        _require(row, "decision", decision, label)
        _require(row, "gpu_allowed", FALSE, label)
        _require(row, "default_readiness", DEFAULT_READINESS, label)
        _require(row, "diagnostic_only", TRUE, label)
        _require(row, "valid_for_default", FALSE, label)
        _require(row, "perf_database", FALSE, label)

    required_contracts = {
        row["contract_row"]: row["key_contract"]
        for row in rows
        if row["contract_row"]
        in {
            "kernel_source_identity",
            "lock_single_kernel_path_option",
            "perfdb_schema_update_option",
            "gpu_smoke_readiness",
        }
    }
    required_fragments = {
        "kernel_source_identity": "include_kernel_source",
        "lock_single_kernel_path_option": "lock_exactly_one_kernel_source",
        "perfdb_schema_update_option": "add_kernel_source_to_loader_query",
        "gpu_smoke_readiness": "kernel_source_contract",
    }
    for contract_row, fragment in required_fragments.items():
        if fragment not in required_contracts.get(contract_row, ""):
            raise ValueError(
                f"{contract_row} missing key contract fragment {fragment!r}"
            )


def analyze_route_a_locked_kernel_path_guard(
    phase351_csv: Path = DEFAULT_PHASE351_CSV,
) -> list[dict[str, str]]:
    _require_phase351_kernel_source_contract(phase351_csv)
    return [dict(row) for row in SPEC_ROWS]


def _validate_output_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != 7:
        raise ValueError(f"Phase353 output must have exactly 7 rows, got {len(rows)}")

    expected_rows = [row["guard_row"] for row in SPEC_ROWS]
    guard_rows = [row.get("guard_row") for row in rows]
    if guard_rows != expected_rows:
        raise ValueError(f"Phase353 guard row order mismatch: {guard_rows!r}")

    expected_decisions = [row["decision"] for row in SPEC_ROWS]
    decisions = [row.get("decision") for row in rows]
    if decisions != expected_decisions:
        raise ValueError(f"Phase353 decision order mismatch: {decisions!r}")

    for row in rows:
        label = row["guard_row"]
        _require(row, "source", SOURCE, label)
        _require(row, "gpu_allowed", FALSE, label)
        _require(row, "default_readiness", DEFAULT_READINESS, label)
        _require(row, "diagnostic_only", TRUE, label)
        _require(row, "valid_for_default", FALSE, label)
        _require(row, "perf_database", FALSE, label)


def write_route_a_locked_kernel_path_guard_csv(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    _validate_output_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_route_a_locked_kernel_path_guard_doc(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    _validate_output_rows(rows)
    lines = [
        "# Phase353 Route A Locked Kernel Path Guard",
        "",
        "| Item | Decision |",
        "|---|---|",
        "| Default AIC | No-Go |",
        "| GPU allowed | false |",
        "| Diagnostic only | true |",
        "| Valid for default | false |",
        "| PerfDatabase | false |",
        "",
        "Phase353 selects locked kernel path guard first. It does not SSH, "
        "does not run GPU, does not change runtime, and does not update "
        "PerfDatabase.",
        "",
        "PerfDatabase schema update is deferred until the measurement value "
        "set exists. Without a source-checked kernel name, adding schema fields "
        "would only move ambiguity into the table.",
        "",
        "The required next step is a source-check guard that proves exactly "
        "one vLLM MoE kernel path. If the guard cannot prove that, fail fast "
        "and do not run GPU.",
        "",
        "Any later smoke is diagnostic-only until a separate phase updates "
        "PerfDatabase loader/query contracts and tests.",
        "",
        "| Guard row | Guard contract | Decision | Next required action |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {guard_row} | {guard_contract} | {decision} | {next_required_action} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Conclusion: GPU smoke remains blocked until the locked kernel "
            "path source-check guard clears.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write the Phase353 Route A locked kernel path guard spec."
    )
    parser.add_argument("--phase351-csv", type=Path, default=DEFAULT_PHASE351_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = analyze_route_a_locked_kernel_path_guard(args.phase351_csv)
    write_route_a_locked_kernel_path_guard_csv(args.output_csv, rows)
    write_route_a_locked_kernel_path_guard_doc(args.output_md, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
