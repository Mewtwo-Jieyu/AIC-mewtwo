from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE342_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase342_route_a_module_perf_table_spec.csv"
)
DEFAULT_PERF_DATABASE_PY = REPO_ROOT / "src/aiconfigurator/sdk/perf_database.py"
DEFAULT_OPERATIONS_PY = REPO_ROOT / "src/aiconfigurator/sdk/operations.py"
DEFAULT_H200_VLLM_DIR = (
    REPO_ROOT / "src/aiconfigurator/systems/data/h200_sxm/vllm"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase344_route_a_source_schema_audit.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase344_route_a_source_schema_audit.md"
)

SOURCE = "phase344_route_a_source_schema_audit"
PHASE342_SOURCE = "phase342_route_a_module_perf_table_spec"
DEFAULT_READINESS = "No-Go"
TRUE = "true"
FALSE = "false"

FIELDNAMES = [
    "source",
    "gate",
    "decision",
    "blocking_condition",
    "next_allowed_phase",
    "gpu_allowed",
    "default_readiness",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]

EXPECTED_PHASE342 = [
    (
        "route_a_module_perf_table_candidate",
        "retained_as_primary_candidate",
    ),
    (
        "route_b_runtime_shape_input_layer",
        "absorbed_as_token_shape_source",
    ),
    (
        "legacy_attention_gemm_reuse",
        "assumption_pending_validation",
    ),
    (
        "vllm_ep8_comm_modeling",
        "schema_pending_source_check",
    ),
    (
        "full_trace_feedback_to_perf_table",
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
        "gate": "vllm_0190_moe_kernel_api",
        "decision": "blocked_pending_vllm_0190_source_check",
        "blocking_condition": "vllm_0190_moe_kernel_api_not_locally_verified",
        "next_allowed_phase": "phase345_blocked_until_source_schema_clears",
        **COMMON_FLAGS,
    },
    {
        "source": SOURCE,
        "gate": "perfdb_moe_kernel_source_schema",
        "decision": "blocked_perfdb_schema_missing_kernel_source_key",
        "blocking_condition": "perfdb_vllm_moe_query_does_not_key_kernel_source",
        "next_allowed_phase": "phase345_blocked_until_perfdb_schema_update_or_source_check",
        **COMMON_FLAGS,
    },
    {
        "source": SOURCE,
        "gate": "vllm_ep8_comm_scope",
        "decision": "blocked_ep8_comm_schema_pending_source_check",
        "blocking_condition": "current_comm_model_is_custom_allreduce_nccl_not_vllm_ep8_alltoall_schema",
        "next_allowed_phase": "phase345_blocked_until_ep8_comm_schema_clears",
        **COMMON_FLAGS,
    },
    {
        "source": SOURCE,
        "gate": "legacy_attention_gemm_reuse_assumption",
        "decision": "assumption_pending_validation",
        "blocking_condition": "h200_vllm_local_perf_table_is_0120_only",
        "next_allowed_phase": "phase345_blocked_until_0190_attention_gemm_source_check",
        **COMMON_FLAGS,
    },
    {
        "source": SOURCE,
        "gate": "cb_sim_token_shape_fidelity_input",
        "decision": "retained_as_prerun_input_dependency",
        "blocking_condition": "token_shape_fidelity_must_be_verified_before_full_validation",
        "next_allowed_phase": "phase345_source_check_only_before_gpu_smoke",
        **COMMON_FLAGS,
    },
    {
        "source": SOURCE,
        "gate": "full_trace_feedback_guard",
        "decision": "rejected_data_leakage",
        "blocking_condition": "full_trace_must_not_generate_perfdb_rows",
        "next_allowed_phase": "not_allowed",
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


def _require_contains(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise ValueError(f"{label} missing required source anchor: {needle}")


def _extract_vllm_query_branch(text: str) -> str:
    marker = "elif self.backend == common.BackendName.vllm.value:"
    start = text.find(marker)
    if start < 0:
        raise ValueError("query_moe vLLM branch missing")

    end_marker = "else:\n                    raise NotImplementedError"
    end = text.find(end_marker, start)
    if end < 0:
        raise ValueError("query_moe vLLM branch end marker missing")
    return text[start:end]


def _extract_operations_vllm_moe_dispatch_branch(text: str) -> str:
    class_marker = "class MoEDispatch(Operation):"
    class_start = text.find(class_marker)
    if class_start < 0:
        raise ValueError("MoEDispatch class missing in operations.py")

    start_marker = "elif database.backend == common.BackendName.vllm.value:"
    start = text.find(start_marker, class_start)
    if start < 0:
        raise ValueError("MoEDispatch vLLM branch missing in operations.py")

    end_marker = "elif database.backend == common.BackendName.sglang.value:"
    end = text.find(end_marker, start)
    if end < 0:
        raise ValueError("MoEDispatch vLLM branch end marker missing in operations.py")
    return text[start:end]


def _require_phase342_route_a_spec(path: Path) -> None:
    rows = _read_csv(path)
    if len(rows) != len(EXPECTED_PHASE342):
        raise ValueError(f"Phase342 must have exactly 5 rows, got {len(rows)}")

    for row, (candidate, decision) in zip(rows, EXPECTED_PHASE342):
        label = candidate
        _require(row, "source", PHASE342_SOURCE, label)
        _require(row, "candidate", candidate, label)
        _require(row, "decision", decision, label)
        _require(row, "gpu_allowed", FALSE, label)
        _require(row, "default_readiness", DEFAULT_READINESS, label)
        _require(row, "diagnostic_only", TRUE, label)
        _require(row, "valid_for_default", FALSE, label)
        _require(row, "perf_database", FALSE, label)


def _require_perf_database_schema(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")

    _require_contains(text, "def load_moe_data(moe_file):", "PerfDatabase")
    _require_contains(text, 'kernel_source = row["kernel_source"]', "PerfDatabase")
    _require_contains(
        text,
        'moe_data = moe_low_latency_data if kernel_source == "moe_torch_flow_min_latency" else moe_default_data',
        "PerfDatabase",
    )

    branch = _extract_vllm_query_branch(text)
    if "kernel_source" in branch:
        raise ValueError("query_moe vLLM branch must not use kernel_source yet")
    _require_contains(
        branch,
        "moe_dict = self._moe_data[quant_mode][used_workload_distribution]",
        "query_moe vLLM branch",
    )
    for needle in [
        "[topk][num_experts][hidden_size]",
        "inter_size",
        "][moe_tp_size][moe_ep_size]",
        "self._nearest_1d_point_helper(",
        "num_tokens",
    ]:
        _require_contains(branch, needle, "query_moe vLLM branch")

    for needle in [
        "def query_custom_allreduce(",
        "def query_nccl(",
        'operation == "all_gather" or operation == "alltoall" or operation == "reduce_scatter"',
    ]:
        _require_contains(text, needle, "PerfDatabase comm schema")


def _require_operations_vllm_comm_schema(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    branch = _extract_operations_vllm_moe_dispatch_branch(text)

    for forbidden in ["query_wideep", "wideep_alltoall"]:
        if forbidden in branch:
            raise ValueError(
                f"MoEDispatch vLLM branch must not use forbidden anchor: {forbidden}"
            )

    for needle in [
        "database.query_custom_allreduce(",
        "database.query_nccl(",
        '"all_gather" if self._pre_dispatch else "reduce_scatter"',
    ]:
        _require_contains(branch, needle, "MoEDispatch vLLM branch")


def _require_h200_vllm_local_table_scope(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    versions = sorted(child.name for child in path.iterdir() if child.is_dir())
    if versions != ["0.12.0"]:
        raise ValueError(f"H200 vLLM local perf table must be 0.12.0 only, got {versions!r}")

    version_dir = path / "0.12.0"
    for filename in [
        "moe_perf.txt",
        "custom_allreduce_perf.txt",
        "context_attention_perf.txt",
        "generation_attention_perf.txt",
        "gemm_perf.txt",
    ]:
        if not (version_dir / filename).exists():
            raise FileNotFoundError(version_dir / filename)

    header = (version_dir / "moe_perf.txt").read_text(encoding="utf-8").splitlines()[0]
    required_columns = {
        "framework",
        "version",
        "op_name",
        "kernel_source",
        "num_tokens",
        "hidden_size",
        "inter_size",
        "topk",
        "num_experts",
        "moe_tp_size",
        "moe_ep_size",
        "distribution",
        "latency",
    }
    columns = set(header.split(","))
    if not required_columns.issubset(columns):
        missing = sorted(required_columns - columns)
        raise ValueError(f"H200 vLLM 0.12.0 MoE table missing columns: {missing!r}")


def analyze_route_a_source_schema_audit(
    phase342_csv: Path = DEFAULT_PHASE342_CSV,
    perf_database_py: Path = DEFAULT_PERF_DATABASE_PY,
    operations_py: Path = DEFAULT_OPERATIONS_PY,
    h200_vllm_dir: Path = DEFAULT_H200_VLLM_DIR,
) -> list[dict[str, str]]:
    _require_phase342_route_a_spec(phase342_csv)
    _require_perf_database_schema(perf_database_py)
    _require_operations_vllm_comm_schema(operations_py)
    _require_h200_vllm_local_table_scope(h200_vllm_dir)
    return [dict(row) for row in SPEC_ROWS]


def _validate_output_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != 6:
        raise ValueError(f"Phase344 output must have exactly 6 rows, got {len(rows)}")

    expected_gates = [row["gate"] for row in SPEC_ROWS]
    gates = [row.get("gate") for row in rows]
    if gates != expected_gates:
        raise ValueError(f"Phase344 gate order mismatch: {gates!r}")

    expected_decisions = [row["decision"] for row in SPEC_ROWS]
    decisions = [row.get("decision") for row in rows]
    if decisions != expected_decisions:
        raise ValueError(f"Phase344 decision order mismatch: {decisions!r}")

    for row in rows:
        label = row["gate"]
        _require(row, "source", SOURCE, label)
        _require(row, "gpu_allowed", FALSE, label)
        _require(row, "default_readiness", DEFAULT_READINESS, label)
        _require(row, "diagnostic_only", TRUE, label)
        _require(row, "valid_for_default", FALSE, label)
        _require(row, "perf_database", FALSE, label)


def write_route_a_source_schema_audit_csv(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    _validate_output_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_route_a_source_schema_audit_doc(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    _validate_output_rows(rows)
    lines = [
        "# Phase344 Route A Source Schema Audit",
        "",
        "| Item | Decision |",
        "|---|---|",
        "| Default AIC | No-Go |",
        "| GPU allowed | false |",
        "| Diagnostic only | true |",
        "| Valid for default | false |",
        "| PerfDatabase | false |",
        "",
        "vLLM 0.19.0 source/schema is not cleared. Phase344 only audits the "
        "local source and schema boundary for Route A; it does not run GPU, "
        "benchmark, SSH, default AIC, or PerfDatabase writes.",
        "",
        "PerfDatabase vLLM MoE query does not use kernel_source as a key. "
        "The local loader reads kernel_source, but ordinary MoE data is still "
        "split only into default versus low-latency buckets; the vLLM query "
        "path keys by quant, distribution, topk, experts, hidden, inter, "
        "moe_tp, moe_ep, and num_tokens.",
        "",
        "Phase344 simultaneously anchors the PerfDatabase MoE query schema "
        "and the operations.py vLLM MoEDispatch comm branch.",
        "",
        "H200 vLLM local perf table version | 0.12.0 only. The existing local "
        "attention, GEMM, MoE, and custom allreduce tables therefore remain a "
        "legacy assumption for any vLLM 0.19.0 Route A path.",
        "",
        "EP8 communication remains schema-pending: current local comm logic is "
        "custom_allreduce and NCCL all_gather/alltoall/reduce_scatter modeling, "
        "not a cleared vLLM EP8 alltoall measured schema.",
        "",
        "full trace must stay validation-only. It must not feed module latency "
        "back into PerfDatabase rows.",
        "",
        "| Gate | Decision | Blocking condition | Next allowed phase |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {gate} | {decision} | {blocking_condition} | {next_allowed_phase} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Conclusion: Route A is retained as the next modeling route, but "
            "Phase345 GPU smoke is blocked until the vLLM 0.19.0 MoE kernel "
            "API, EP8 comm schema, and PerfDatabase query key contract are "
            "source-cleared.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write the Phase344 Route A source/schema diagnostic audit."
    )
    parser.add_argument("--phase342-csv", type=Path, default=DEFAULT_PHASE342_CSV)
    parser.add_argument(
        "--perf-database-py", type=Path, default=DEFAULT_PERF_DATABASE_PY
    )
    parser.add_argument("--operations-py", type=Path, default=DEFAULT_OPERATIONS_PY)
    parser.add_argument("--h200-vllm-dir", type=Path, default=DEFAULT_H200_VLLM_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = analyze_route_a_source_schema_audit(
        phase342_csv=args.phase342_csv,
        perf_database_py=args.perf_database_py,
        operations_py=args.operations_py,
        h200_vllm_dir=args.h200_vllm_dir,
    )
    write_route_a_source_schema_audit_csv(args.output_csv, rows)
    write_route_a_source_schema_audit_doc(args.output_md, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
