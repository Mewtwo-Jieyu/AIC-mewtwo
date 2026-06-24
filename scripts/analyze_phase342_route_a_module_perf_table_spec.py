from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE339_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase339_prerun_model_feasibility.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase342_route_a_module_perf_table_spec.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase342_route_a_module_perf_table_spec.md"
)

SOURCE = "phase342_route_a_module_perf_table_spec"
PHASE339_SOURCE = "phase339_prerun_model_feasibility"
DEFAULT_READINESS = "No-Go"
TRUE = "true"
FALSE = "false"

FIELDNAMES = [
    "source",
    "candidate",
    "role",
    "decision",
    "blocking_condition",
    "next_phase",
    "gpu_allowed",
    "default_readiness",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]

EXPECTED_PHASE339 = [
    (
        "scope_key_only_candidate",
        "rejected_scope_only_no_throughput_prediction",
    ),
    (
        "postrun_trace_feature_candidate",
        "rejected_postrun_trace_leakage",
    ),
    (
        "topology_specific_cadence_boundary_candidate",
        "diagnostic_only_postrun_explanation",
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
        "candidate": "route_a_module_perf_table_candidate",
        "role": "primary_model_candidate",
        "decision": "retained_as_primary_candidate",
        "blocking_condition": "module_perf_table_assumptions_pending_source_check",
        "next_phase": "phase343_vllm_0190_moe_source_api_check",
        **COMMON_FLAGS,
    },
    {
        "source": SOURCE,
        "candidate": "route_b_runtime_shape_input_layer",
        "role": "token_shape_source",
        "decision": "absorbed_as_token_shape_source",
        "blocking_condition": "cb_sim_token_shape_fidelity_must_be_sufficient",
        "next_phase": "phase343_source_check_before_any_gpu",
        **COMMON_FLAGS,
    },
    {
        "source": SOURCE,
        "candidate": "legacy_attention_gemm_reuse",
        "role": "legacy_assumption",
        "decision": "assumption_pending_validation",
        "blocking_condition": "vllm_0190_kernel_source_must_match_legacy_table_scope",
        "next_phase": "phase343_kernel_source_schema_check",
        **COMMON_FLAGS,
    },
    {
        "source": SOURCE,
        "candidate": "vllm_ep8_comm_modeling",
        "role": "comm_schema_candidate",
        "decision": "schema_pending_source_check",
        "blocking_condition": "vllm_ep8_comm_scope_must_be_defined_before_gpu",
        "next_phase": "phase343_ep8_comm_source_schema_check",
        **COMMON_FLAGS,
    },
    {
        "source": SOURCE,
        "candidate": "full_trace_feedback_to_perf_table",
        "role": "forbidden_feedback_path",
        "decision": "rejected_data_leakage",
        "blocking_condition": "data_leakage_from_full_trace_feedback",
        "next_phase": "not_allowed",
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


def _require_phase339_closeout(path: Path) -> None:
    rows = _read_csv(path)
    if len(rows) != len(EXPECTED_PHASE339):
        raise ValueError(f"Phase339 must have exactly 3 rows, got {len(rows)}")

    for row, (candidate, verdict) in zip(rows, EXPECTED_PHASE339):
        label = candidate
        _require(row, "source", PHASE339_SOURCE, label)
        _require(row, "candidate", candidate, label)
        _require(row, "verdict", verdict, label)
        _require(row, "gpu_run_allowed", FALSE, label)
        _require(row, "default_readiness", DEFAULT_READINESS, label)
        _require(row, "diagnostic_only", TRUE, label)
        _require(row, "valid_for_default", FALSE, label)
        _require(row, "perf_database", FALSE, label)

    leakage_reason = rows[1].get("reason", "")
    if "leakage_risk" not in leakage_reason:
        raise ValueError("postrun_trace_feature_candidate must keep leakage_risk")

    default_model_reason = rows[2].get("reason", "")
    if "not_a_default_model" not in default_model_reason:
        raise ValueError(
            "topology_specific_cadence_boundary_candidate must stay not_a_default_model"
        )


def analyze_route_a_module_perf_table_spec(
    phase339_csv: Path = DEFAULT_PHASE339_CSV,
) -> list[dict[str, str]]:
    _require_phase339_closeout(phase339_csv)
    return [dict(row) for row in SPEC_ROWS]


def _validate_output_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != 5:
        raise ValueError(f"Phase342 output must have exactly 5 rows, got {len(rows)}")

    expected_candidates = [row["candidate"] for row in SPEC_ROWS]
    candidates = [row.get("candidate") for row in rows]
    if candidates != expected_candidates:
        raise ValueError(f"Phase342 candidate order mismatch: {candidates!r}")

    for row in rows:
        label = row["candidate"]
        _require(row, "source", SOURCE, label)
        _require(row, "gpu_allowed", FALSE, label)
        _require(row, "default_readiness", DEFAULT_READINESS, label)
        _require(row, "diagnostic_only", TRUE, label)
        _require(row, "valid_for_default", FALSE, label)
        _require(row, "perf_database", FALSE, label)


def write_route_a_module_perf_table_spec_csv(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    _validate_output_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_route_a_module_perf_table_spec_doc(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    _validate_output_rows(rows)
    lines = [
        "# Phase342 Route A Module Perf Table Spec",
        "",
        "| Item | Decision |",
        "|---|---|",
        "| Route A | vLLM 0.19.0 module-level perf table |",
        "| Route B | absorbed as token-shape input layer |",
        "| Full trace feedback | rejected_data_leakage |",
        "| Default AIC | No-Go |",
        "| GPU allowed | false |",
        "| PerfDatabase | false |",
        "",
        "Route A becomes the primary diagnostic model candidate: "
        "_compute_3pass -> run_static -> PerfDatabase query -> compose. "
        "This phase only specifies the route and gates; it is not a default model.",
        "",
        "Pre-run inputs: model_arch_dims, hardware, topology, scheduled_tokens, "
        "phase, quant, distribution, moe_tp, moe_ep. Output: iteration_latency_ms.",
        "",
        "MoE key: num_tokens, hidden, inter, topk, experts, moe_tp, moe_ep, "
        "quant, distribution, is_context.",
        "",
        "Initial validation gate: mean error <=20%, max error <=30%, "
        "direction must be correct. Parameters must be fixed on tp8; tp4dp2 "
        "is holdout.",
        "",
        "No-Go gates: unstable vLLM 0.19.0 kernel source schema, missing "
        "standalone MoE microbench, unclear EP8 comm scope, insufficient "
        "cb_sim token-shape fidelity, or any need to backfill module latency "
        "from full trace.",
        "",
        "| Candidate | Role | Decision | Blocking condition | Next phase |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {candidate} | {role} | {decision} | {blocking_condition} | {next_phase} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "| Flag | Value |",
            "|---|---|",
            "| diagnostic_only | true |",
            "| valid_for_default | false |",
            "| perf_database | false |",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write the Phase342 Route A module perf table spec audit."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_PHASE339_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = analyze_route_a_module_perf_table_spec(args.input_csv)
    write_route_a_module_perf_table_spec_csv(args.output_csv, rows)
    write_route_a_module_perf_table_spec_doc(args.output_md, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
