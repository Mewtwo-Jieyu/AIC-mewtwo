from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE377_CSV = (
    REPO_ROOT / "docs/iter_gap_investigation/phase377_vllm_module_runtime_binding_spec.csv"
)
DEFAULT_OPERATIONS_PY = REPO_ROOT / "src/aiconfigurator/sdk/operations.py"
DEFAULT_BASE_BACKEND_PY = REPO_ROOT / "src/aiconfigurator/sdk/backends/base_backend.py"
DEFAULT_PHASE378_TEST = REPO_ROOT / "tests/unit/sdk/backends/test_cb_simulator.py"
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase380_vllm_module_runtime_binding_sufficiency_gate.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase380_vllm_module_runtime_binding_sufficiency_gate.md"
)

SOURCE = "phase380_vllm_module_runtime_binding_sufficiency_gate"
DEFAULT_READINESS = "No-Go"
TRUE = "true"
FALSE = "false"
RUNTIME_MODEL = "moonshotai/Kimi-K2.5"
PERFDB_MODEL = "kimi-k2.5"
HARDWARE = "h200_sxm"
VLLM_VERSION = "0.19.0"
TOPOLOGY = "tp4dp2ep8"
QUANT_RUNTIME = "CompressedTensorsWNA16MarlinMoEMethod"
FUSED_MODULE = "fusedmoe_runner_compute"
EP8_MODULE = "ep8_comm_dispatch_combine"
MODULE_BOUNDARIES = f"{FUSED_MODULE};{EP8_MODULE}"
REAL_BUCKETS = ["1", "15", "16", "241", "1808", "2048", "8192"]
REAL_BUCKETS_TEXT = "/".join(REAL_BUCKETS)
EXACT_GUARD = "model;hardware;vllm_version;topology;bucket_tokens;module_boundary;quant_runtime"
VALIDATOR_RESULT = "1.50x/1.47x/1.79x"
NEXT_PHASE = "phase381_local_end_to_end_exact_bucket_probe"

FIELDNAMES = [
    "source",
    "row_type",
    "verdict",
    "phase377_contract",
    "runtime_binding",
    "exact_lookup_guard",
    "model_required",
    "perfdb_model_key",
    "hardware_required",
    "vllm_version_required",
    "topology_required",
    "allowed_bucket_tokens",
    "excluded_bucket_tokens",
    "module_boundaries",
    "quant_runtime_required",
    "integration_level",
    "unit_coverage",
    "validator_result",
    "next_allowed_phase",
    "topology_guard",
    "topology_source",
    "post_dispatch_bucket_guard",
    "post_dispatch_combined_row_policy",
    "perfdb_data_changed",
    "new_data_rows_allowed",
    "schema_changed",
    "gpu_allowed",
    "ssh_allowed",
    "default_aic_allowed",
    "nearest_bucket_allowed",
    "curve_fit_allowed",
    "interpolation_allowed",
    "extrapolation_allowed",
    "default_readiness",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]

COMMON_FIELDS = {
    "source": SOURCE,
    "verdict": "",
    "phase377_contract": "",
    "runtime_binding": "",
    "exact_lookup_guard": "",
    "model_required": "",
    "perfdb_model_key": "",
    "hardware_required": "",
    "vllm_version_required": "",
    "topology_required": "",
    "allowed_bucket_tokens": "",
    "excluded_bucket_tokens": "",
    "module_boundaries": "",
    "quant_runtime_required": "",
    "integration_level": "unit_plus_validator_evidence_only",
    "unit_coverage": "",
    "validator_result": "",
    "next_allowed_phase": "",
    "topology_guard": "",
    "topology_source": "",
    "post_dispatch_bucket_guard": "",
    "post_dispatch_combined_row_policy": "",
    "perfdb_data_changed": FALSE,
    "new_data_rows_allowed": FALSE,
    "schema_changed": FALSE,
    "gpu_allowed": FALSE,
    "ssh_allowed": FALSE,
    "default_aic_allowed": FALSE,
    "nearest_bucket_allowed": FALSE,
    "curve_fit_allowed": FALSE,
    "interpolation_allowed": FALSE,
    "extrapolation_allowed": FALSE,
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


def _require_text(path: Path, snippets: dict[str, str]) -> str:
    text = path.read_text(encoding="utf-8")
    for label, snippet in snippets.items():
        if snippet not in text:
            raise ValueError(f"{path} missing {label}: {snippet}")
    return text


def _require_phase377_contract(path: Path) -> None:
    rows = _read_csv(path)
    key = next((row for row in rows if row.get("row_type") == "runtime_key_source_contract"), None)
    if key is None:
        raise ValueError("Phase377 runtime_key_source_contract row missing")
    _require(key, "hardware_required", HARDWARE, "phase377 runtime key")
    _require(key, "vllm_version_required", VLLM_VERSION, "phase377 runtime key")
    _require(key, "model_required", PERFDB_MODEL, "phase377 runtime key")
    _require(key, "topology_required", TOPOLOGY, "phase377 runtime key")
    _require(key, "quant_runtime_required", QUANT_RUNTIME, "phase377 runtime key")

    bucket = next((row for row in rows if row.get("row_type") == "bucket_exact_only_contract"), None)
    if bucket is None:
        raise ValueError("Phase377 bucket_exact_only_contract row missing")
    _require(bucket, "allowed_bucket_tokens", REAL_BUCKETS_TEXT, "phase377 bucket")
    _require(bucket, "excluded_bucket_tokens", "128", "phase377 bucket")
    _require(bucket, "nearest_bucket_allowed", FALSE, "phase377 bucket")

    for row in rows:
        _require(row, "default_readiness", DEFAULT_READINESS, row["row_type"])
        _require(row, "diagnostic_only", TRUE, row["row_type"])
        _require(row, "valid_for_default", FALSE, row["row_type"])
        _require(row, "perf_database", FALSE, row["row_type"])


def _require_runtime_binding_code(operations_py: Path, base_backend_py: Path, phase378_test: Path) -> None:
    _require_text(
        operations_py,
        {
            "runtime model guard": f'_VLLM_MODULE_RUNTIME_MODEL = "{RUNTIME_MODEL}"',
            "perfdb model key": f'_VLLM_MODULE_PERFDB_MODEL = "{PERFDB_MODEL}"',
            "hardware guard": f'_VLLM_MODULE_HARDWARE = "{HARDWARE}"',
            "version guard": f'_VLLM_MODULE_VERSION = "{VLLM_VERSION}"',
            "topology guard": f'_VLLM_MODULE_TOPOLOGY = "{TOPOLOGY}"',
            "topology predicate": "and topology == _VLLM_MODULE_TOPOLOGY",
            "bucket helper": "def _validate_vllm_module_bucket(bucket_tokens: int) -> None:",
            "post dispatch bucket guard": "_validate_vllm_module_bucket(scaled_num_tokens)",
            "post dispatch zero policy": "return PerformanceResult(0.0, energy=0.0)",
            "fusedmoe module binding": 'module_boundary="fusedmoe_runner_compute"',
            "ep8 module binding": 'module_boundary="ep8_comm_dispatch_combine"',
        },
    )
    _require_text(
        base_backend_py,
        {
            "topology derivation": "vllm_module_topology = (",
            "topology exact key": (
                'f"tp{model.config.tp_size}dp{model.config.attention_dp_size}'
                'ep{model.config.moe_ep_size}"'
            ),
            "topology query argument": "vllm_module_topology=vllm_module_topology",
            "model path fallback": 'getattr(model, "model_name", getattr(model, "model_path", ""))',
        },
    )
    _require_text(
        phase378_test,
        {
            "kimi positive": "test_vllm_moe_uses_module_perf_exact_bucket",
            "non kimi guard": "test_vllm_moe_non_kimi_scope_keeps_existing_query_moe_path",
            "topology guard": "test_vllm_moe_non_topology_scope_keeps_existing_query_moe_path",
            "post dispatch bucket guard": "test_vllm_dispatch_post_scope_rejects_non_whitelisted_bucket",
            "base backend topology propagation": (
                "test_run_static_passes_model_path_and_topology_when_model_name_is_absent"
            ),
        },
    )


def _row(row_type: str, **overrides: str) -> dict[str, str]:
    row = dict(COMMON_FIELDS)
    row["row_type"] = row_type
    row.update(overrides)
    return row


def _gate_rows() -> list[dict[str, str]]:
    return [
        _row(
            "phase377_contract_prerequisite",
            verdict="contract_locked",
            phase377_contract=EXACT_GUARD,
            model_required=RUNTIME_MODEL,
            perfdb_model_key=PERFDB_MODEL,
            hardware_required=HARDWARE,
            vllm_version_required=VLLM_VERSION,
            topology_required=TOPOLOGY,
            allowed_bucket_tokens=REAL_BUCKETS_TEXT,
            excluded_bucket_tokens="128",
            module_boundaries=MODULE_BOUNDARIES,
            quant_runtime_required=QUANT_RUNTIME,
        ),
        _row(
            "phase378_runtime_binding_implemented",
            verdict="implemented_but_not_default_ready",
            runtime_binding="implemented",
            exact_lookup_guard=EXACT_GUARD,
            model_required=RUNTIME_MODEL,
            perfdb_model_key=PERFDB_MODEL,
            module_boundaries=MODULE_BOUNDARIES,
        ),
        _row(
            "phase379_topology_guard_implemented",
            verdict="implemented_exact_topology_guard",
            topology_guard="implemented",
            topology_source="vllm_module_topology",
            topology_required=TOPOLOGY,
        ),
        _row(
            "phase379_post_dispatch_bucket_guard_implemented",
            verdict="implemented_fail_fast_before_zero_row",
            post_dispatch_bucket_guard="implemented",
            post_dispatch_combined_row_policy="zero_after_bucket_check",
            allowed_bucket_tokens=REAL_BUCKETS_TEXT,
            excluded_bucket_tokens="128",
        ),
        _row(
            "exact_lookup_guard_complete",
            verdict="exact_lookup_only",
            exact_lookup_guard=EXACT_GUARD,
            model_required=RUNTIME_MODEL,
            perfdb_model_key=PERFDB_MODEL,
            hardware_required=HARDWARE,
            vllm_version_required=VLLM_VERSION,
            topology_required=TOPOLOGY,
            allowed_bucket_tokens=REAL_BUCKETS_TEXT,
            excluded_bucket_tokens="128",
            module_boundaries=MODULE_BOUNDARIES,
            quant_runtime_required=QUANT_RUNTIME,
            nearest_bucket_allowed=FALSE,
        ),
        _row(
            "unit_coverage_present",
            verdict="unit_coverage_for_runtime_binding_guards_present",
            unit_coverage=(
                "kimi_scope;non_kimi_scope;topology_guard;post_dispatch_bucket_guard;"
                "bucket_128_failfast;base_backend_topology"
            ),
        ),
        _row(
            "validator_stability_preserved",
            verdict="validator_baseline_preserved",
            validator_result=VALIDATOR_RESULT,
        ),
        _row(
            "perfdb_data_unchanged",
            verdict="no_new_data_no_schema_change",
            perfdb_data_changed=FALSE,
            new_data_rows_allowed=FALSE,
            schema_changed=FALSE,
        ),
        _row(
            "default_aic_blocked",
            verdict="blocked_default_aic_no_go",
            default_aic_allowed=FALSE,
        ),
        _row(
            "next_phase",
            verdict="phase381_local_end_to_end_exact_bucket_probe_only",
            next_allowed_phase=NEXT_PHASE,
        ),
    ]


def analyze_phase380_vllm_module_runtime_binding_sufficiency_gate(
    phase377_csv: Path = DEFAULT_PHASE377_CSV,
    operations_py: Path = DEFAULT_OPERATIONS_PY,
    base_backend_py: Path = DEFAULT_BASE_BACKEND_PY,
    phase378_test: Path = DEFAULT_PHASE378_TEST,
) -> list[dict[str, str]]:
    _require_phase377_contract(phase377_csv)
    _require_runtime_binding_code(operations_py, base_backend_py, phase378_test)
    return _gate_rows()


def _validate_rows(rows: list[dict[str, str]]) -> None:
    if [row.get("row_type") for row in rows] != [row["row_type"] for row in _gate_rows()]:
        raise ValueError("Phase380 row order or row_type set changed")
    for index, row in enumerate(rows, start=1):
        for field in FIELDNAMES:
            if field not in row:
                raise ValueError(f"row {index} missing field {field}")
        for field in (
            "gpu_allowed",
            "ssh_allowed",
            "default_aic_allowed",
            "new_data_rows_allowed",
            "schema_changed",
            "curve_fit_allowed",
            "interpolation_allowed",
            "extrapolation_allowed",
            "valid_for_default",
            "perf_database",
        ):
            if row[field] != FALSE:
                raise ValueError(f"row {index} {field} must remain false")
        _require(row, "default_readiness", DEFAULT_READINESS, f"row {index}")
        _require(row, "diagnostic_only", TRUE, f"row {index}")


def write_phase380_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_phase380_md(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = f"""# Phase380 vLLM Module Runtime Binding Sufficiency Gate

Phase380 records that runtime binding implemented, but it is not default AIC evidence.

| Gate | Verdict |
|---|---|
| runtime binding | implemented |
| exact lookup guard | {EXACT_GUARD} |
| runtime model | {RUNTIME_MODEL} |
| PerfDB model key | {PERFDB_MODEL} |
| topology | {TOPOLOGY} |
| buckets | {REAL_BUCKETS_TEXT}; bucket `128` remains rejected |
| integration level | unit + validator evidence only |
| validator | {VALIDATOR_RESULT} |
| PerfDatabase | no new rows, no schema change |
| GPU / SSH | not allowed |
| Default AIC | No-Go |

Phase381 is the next allowed phase: local end-to-end exact-bucket probe only.
It must not open default AIC, fit curves, interpolate, extrapolate, or write new PerfDatabase data.
"""
    path.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase377-csv", type=Path, default=DEFAULT_PHASE377_CSV)
    parser.add_argument("--operations-py", type=Path, default=DEFAULT_OPERATIONS_PY)
    parser.add_argument("--base-backend-py", type=Path, default=DEFAULT_BASE_BACKEND_PY)
    parser.add_argument("--phase378-test", type=Path, default=DEFAULT_PHASE378_TEST)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    rows = analyze_phase380_vllm_module_runtime_binding_sufficiency_gate(
        phase377_csv=args.phase377_csv,
        operations_py=args.operations_py,
        base_backend_py=args.base_backend_py,
        phase378_test=args.phase378_test,
    )
    write_phase380_csv(args.output_csv, rows)
    write_phase380_md(args.output_md, rows)


if __name__ == "__main__":
    main()
