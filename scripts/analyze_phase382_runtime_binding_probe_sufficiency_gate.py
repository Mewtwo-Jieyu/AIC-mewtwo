from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE381_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase381_local_end_to_end_exact_bucket_probe.csv"
)
DEFAULT_VLLM_MODULE_PERF = (
    REPO_ROOT
    / "src/aiconfigurator/systems/data/h200_sxm/vllm/0.19.0/vllm_module_perf.txt"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase382_runtime_binding_probe_sufficiency_gate.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase382_runtime_binding_probe_sufficiency_gate.md"
)

SOURCE = "phase382_runtime_binding_probe_sufficiency_gate"
DEFAULT_READINESS = "No-Go"
TRUE = "true"
FALSE = "false"
MODEL = "kimi-k2.5"
HARDWARE = "h200_sxm"
VLLM_VERSION = "0.19.0"
TOPOLOGY = "tp4dp2ep8"
QUANT_RUNTIME = "CompressedTensorsWNA16MarlinMoEMethod"
FUSED_MODULE = "fusedmoe_runner_compute"
EP8_MODULE = "ep8_comm_dispatch_combine"
MODULE_BOUNDARIES = f"{FUSED_MODULE};{EP8_MODULE}"
BUCKETS = [1, 15, 16, 241, 1808, 2048, 8192]
BUCKETS_TEXT = "/".join(str(bucket) for bucket in BUCKETS)
NEXT_PHASE = "phase383_runtime_bucket_semantics_decision_spec"
PHASE381_NEXT = "phase382_runtime_binding_probe_sufficiency_gate"
EP8_BUCKET_FORMULA = "max(1, raw_tokens//4)"
FUSEDMOE_BUCKET_FORMULA = "max(1, raw_tokens//4)*2"

FIELDNAMES = [
    "source",
    "row_type",
    "verdict",
    "phase381_prerequisite",
    "module_table_exact_keys",
    "table_row_count",
    "module_count",
    "bucket_count",
    "allowed_bucket_tokens",
    "module_boundaries",
    "ep8_bucket_formula",
    "fusedmoe_bucket_formula",
    "ep8_reachable_buckets",
    "fusedmoe_reachable_buckets",
    "fusedmoe_unreachable_buckets",
    "paired_lookup_required",
    "paired_lookup_candidates",
    "paired_lookup_candidate_count",
    "combined_full_model_status",
    "op_level_runtime_binding_sufficient",
    "full_model_runtime_exact_lookup_sufficient",
    "next_allowed_phase",
    "new_perfdb_data",
    "gpu_allowed",
    "ssh_allowed",
    "default_aic_allowed",
    "interpolation_allowed",
    "extrapolation_allowed",
    "nearest_bucket_allowed",
    "default_readiness",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]

COMMON_FIELDS = {
    "source": SOURCE,
    "verdict": "",
    "phase381_prerequisite": "",
    "module_table_exact_keys": "",
    "table_row_count": "",
    "module_count": "",
    "bucket_count": "",
    "allowed_bucket_tokens": BUCKETS_TEXT,
    "module_boundaries": MODULE_BOUNDARIES,
    "ep8_bucket_formula": "",
    "fusedmoe_bucket_formula": "",
    "ep8_reachable_buckets": "",
    "fusedmoe_reachable_buckets": "",
    "fusedmoe_unreachable_buckets": "",
    "paired_lookup_required": "",
    "paired_lookup_candidates": "",
    "paired_lookup_candidate_count": "",
    "combined_full_model_status": "",
    "op_level_runtime_binding_sufficient": FALSE,
    "full_model_runtime_exact_lookup_sufficient": FALSE,
    "next_allowed_phase": "",
    "new_perfdb_data": FALSE,
    "gpu_allowed": FALSE,
    "ssh_allowed": FALSE,
    "default_aic_allowed": FALSE,
    "interpolation_allowed": FALSE,
    "extrapolation_allowed": FALSE,
    "nearest_bucket_allowed": FALSE,
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


def _require_phase381(path: Path) -> None:
    rows = _read_csv(path)
    by_type = {row["row_type"]: row for row in rows}
    required = [
        "fusedmoe_runner_compute_run_static_exact_bucket_probe",
        "ep8_comm_dispatch_combine_run_static_pre_post_probe",
        "combined_full_model_exact_bucket_probe",
        "default_aic_blocked",
        "next_phase",
    ]
    for row_type in required:
        if row_type not in by_type:
            raise ValueError(f"Phase381 missing row {row_type}")

    fused = by_type["fusedmoe_runner_compute_run_static_exact_bucket_probe"]
    ep8 = by_type["ep8_comm_dispatch_combine_run_static_pre_post_probe"]
    combined = by_type["combined_full_model_exact_bucket_probe"]
    default = by_type["default_aic_blocked"]
    next_phase = by_type["next_phase"]

    _require(fused, "raw_tokens", "32", "Phase381 fused probe")
    _require(fused, "bucket_tokens", "16", "Phase381 fused probe")
    _require(fused, "module_boundary", FUSED_MODULE, "Phase381 fused probe")
    _require(ep8, "raw_tokens", "64", "Phase381 ep8 probe")
    _require(ep8, "bucket_tokens", "16", "Phase381 ep8 probe")
    _require(ep8, "module_boundary", EP8_MODULE, "Phase381 ep8 probe")
    _require(combined, "combined_full_model_status", "blocked_by_bucket_mapping", "Phase381 combined")
    _require(default, "default_aic_allowed", FALSE, "Phase381 default")
    _require(next_phase, "next_allowed_phase", PHASE381_NEXT, "Phase381 next")

    for row in rows:
        _require(row, "default_readiness", DEFAULT_READINESS, row["row_type"])
        _require(row, "diagnostic_only", TRUE, row["row_type"])
        _require(row, "valid_for_default", FALSE, row["row_type"])
        _require(row, "perf_database", FALSE, row["row_type"])


def _load_module_table(path: Path) -> dict[str, set[int]]:
    rows = _read_csv(path)
    seen: set[tuple[str, str, str, str, int, str, str]] = set()
    buckets_by_module: dict[str, set[int]] = {FUSED_MODULE: set(), EP8_MODULE: set()}
    for row in rows:
        module = row["module_boundary"]
        bucket = int(row["bucket_tokens"])
        key = (
            row["model"],
            row["hardware"],
            row["vllm_version"],
            row["topology"],
            bucket,
            module,
            row["quant_runtime"],
        )
        if key in seen:
            raise ValueError(f"duplicate vllm module exact key: {key}")
        seen.add(key)
        if row["model"] != MODEL:
            raise ValueError(f"unexpected model {row['model']!r}")
        if row["hardware"] != HARDWARE:
            raise ValueError(f"unexpected hardware {row['hardware']!r}")
        if row["vllm_version"] != VLLM_VERSION:
            raise ValueError(f"unexpected vllm_version {row['vllm_version']!r}")
        if row["topology"] != TOPOLOGY:
            raise ValueError(f"unexpected topology {row['topology']!r}")
        if row["quant_runtime"] != QUANT_RUNTIME:
            raise ValueError(f"unexpected quant_runtime {row['quant_runtime']!r}")
        if module not in buckets_by_module:
            raise ValueError(f"unexpected module_boundary {module!r}")
        buckets_by_module[module].add(bucket)

    expected = set(BUCKETS)
    for module, buckets in buckets_by_module.items():
        if buckets != expected:
            raise ValueError(f"{module} buckets expected {sorted(expected)}, got {sorted(buckets)}")
    if len(seen) != 14:
        raise ValueError(f"expected 14 exact keys, got {len(seen)}")
    return buckets_by_module


def _format_buckets(buckets: list[int] | set[int]) -> str:
    return "/".join(str(bucket) for bucket in sorted(buckets))


def _paired_candidates(buckets: set[int]) -> list[str]:
    return [f"{bucket}->{bucket * 2}" for bucket in sorted(buckets) if bucket * 2 in buckets]


def _row(row_type: str, **overrides: str) -> dict[str, str]:
    row = dict(COMMON_FIELDS)
    row["row_type"] = row_type
    row.update(overrides)
    return row


def analyze_phase382_runtime_binding_probe_sufficiency_gate(
    phase381_csv: Path = DEFAULT_PHASE381_CSV,
    vllm_module_perf: Path = DEFAULT_VLLM_MODULE_PERF,
) -> list[dict[str, str]]:
    _require_phase381(phase381_csv)
    buckets_by_module = _load_module_table(vllm_module_perf)
    all_buckets = buckets_by_module[FUSED_MODULE]
    fused_reachable = {bucket for bucket in all_buckets if bucket % 2 == 0}
    fused_unreachable = all_buckets - fused_reachable
    ep8_reachable = buckets_by_module[EP8_MODULE]
    pairs = _paired_candidates(all_buckets)

    rows = [
        _row(
            "phase381_prerequisite",
            verdict="phase381_local_exact_lookup_op_level_passed",
            phase381_prerequisite="local_exact_lookup_passed",
            combined_full_model_status="blocked_by_bucket_mapping",
            op_level_runtime_binding_sufficient=TRUE,
            full_model_runtime_exact_lookup_sufficient=FALSE,
        ),
        _row(
            "module_table_exact_key_inventory",
            verdict="module_table_inventory_locked",
            module_table_exact_keys="2_modules_x_7_buckets",
            table_row_count="14",
            module_count="2",
            bucket_count="7",
        ),
        _row(
            "runtime_bucket_formula_current",
            verdict="current_formula_captured",
            ep8_bucket_formula=EP8_BUCKET_FORMULA,
            fusedmoe_bucket_formula=FUSEDMOE_BUCKET_FORMULA,
        ),
        _row(
            "ep8_runtime_bucket_reachability",
            verdict="all_registered_buckets_reachable",
            ep8_reachable_buckets=_format_buckets(ep8_reachable),
            ep8_bucket_formula=EP8_BUCKET_FORMULA,
        ),
        _row(
            "fusedmoe_runtime_bucket_reachability",
            verdict="partial_registered_bucket_reachability",
            fusedmoe_reachable_buckets=_format_buckets(fused_reachable),
            fusedmoe_unreachable_buckets=_format_buckets(fused_unreachable),
            fusedmoe_bucket_formula=FUSEDMOE_BUCKET_FORMULA,
        ),
        _row(
            "full_model_paired_lookup_requirement",
            verdict="requires_registered_b_and_2b_pair",
            paired_lookup_required="bucket_b_and_2b_both_registered",
            ep8_bucket_formula=EP8_BUCKET_FORMULA,
            fusedmoe_bucket_formula=FUSEDMOE_BUCKET_FORMULA,
        ),
        _row(
            "paired_lookup_candidate_gap",
            verdict="no_common_raw_token_pair",
            paired_lookup_required="bucket_b_and_2b_both_registered",
            paired_lookup_candidates=";".join(pairs),
            paired_lookup_candidate_count=str(len(pairs)),
            full_model_runtime_exact_lookup_sufficient=FALSE,
        ),
        _row(
            "op_level_runtime_binding_sufficient",
            verdict="op_level_binding_sufficient_for_next_gate",
            op_level_runtime_binding_sufficient=TRUE,
            full_model_runtime_exact_lookup_sufficient=FALSE,
        ),
        _row(
            "full_model_runtime_exact_lookup_blocked",
            verdict="full_model_blocked_by_bucket_semantics",
            combined_full_model_status="blocked_by_bucket_mapping",
            full_model_runtime_exact_lookup_sufficient=FALSE,
            next_allowed_phase=NEXT_PHASE,
        ),
        _row(
            "default_aic_blocked",
            verdict="blocked_default_aic_no_go",
            default_aic_allowed=FALSE,
        ),
        _row(
            "next_phase",
            verdict="phase383_runtime_bucket_semantics_decision_spec",
            next_allowed_phase=NEXT_PHASE,
        ),
    ]
    _validate_rows(rows)
    return rows


def _validate_rows(rows: list[dict[str, str]]) -> None:
    expected = [
        "phase381_prerequisite",
        "module_table_exact_key_inventory",
        "runtime_bucket_formula_current",
        "ep8_runtime_bucket_reachability",
        "fusedmoe_runtime_bucket_reachability",
        "full_model_paired_lookup_requirement",
        "paired_lookup_candidate_gap",
        "op_level_runtime_binding_sufficient",
        "full_model_runtime_exact_lookup_blocked",
        "default_aic_blocked",
        "next_phase",
    ]
    if [row.get("row_type") for row in rows] != expected:
        raise ValueError("Phase382 row order or row_type set changed")
    for index, row in enumerate(rows, start=1):
        for field in FIELDNAMES:
            if field not in row:
                raise ValueError(f"row {index} missing field {field}")
        for field in (
            "new_perfdb_data",
            "gpu_allowed",
            "ssh_allowed",
            "default_aic_allowed",
            "interpolation_allowed",
            "extrapolation_allowed",
            "nearest_bucket_allowed",
            "valid_for_default",
            "perf_database",
        ):
            if row[field] != FALSE:
                raise ValueError(f"row {index} {field} must remain false")
        _require(row, "default_readiness", DEFAULT_READINESS, f"row {index}")
        _require(row, "diagnostic_only", TRUE, f"row {index}")


def write_phase382_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_phase382_md(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    fused = next(row for row in rows if row["row_type"] == "fusedmoe_runtime_bucket_reachability")
    pair_gap = next(row for row in rows if row["row_type"] == "paired_lookup_candidate_gap")
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = f"""# Phase382 Runtime Binding Probe Sufficiency Gate

Phase382 separates two questions: op-level binding is sufficient, but full-model exact lookup remains blocked.
It does not SSH, run GPU, write new PerfDatabase data, or open Default AIC.

| Gate | Verdict |
|---|---|
| module table | 14 exact keys: 2 modules x 7 buckets |
| current EP8 formula | {EP8_BUCKET_FORMULA} |
| current FusedMoE formula | {FUSEDMOE_BUCKET_FORMULA} |
| FusedMoE reachable buckets | {fused["fusedmoe_reachable_buckets"]} |
| FusedMoE unreachable buckets | {fused["fusedmoe_unreachable_buckets"]} |
| paired full-model candidates | {pair_gap["paired_lookup_candidate_count"]} |
| op-level runtime binding | sufficient |
| full-model exact lookup | blocked |
| Default AIC | No-Go |

max(1, ...) 不改变当前 paired gap，只修正公式表达。

The next allowed phase is `{NEXT_PHASE}`.
It must decide whether to change runtime bucket semantics or design paired-bucket data expansion.
"""
    path.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase381-csv", type=Path, default=DEFAULT_PHASE381_CSV)
    parser.add_argument("--vllm-module-perf", type=Path, default=DEFAULT_VLLM_MODULE_PERF)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    rows = analyze_phase382_runtime_binding_probe_sufficiency_gate(
        phase381_csv=args.phase381_csv,
        vllm_module_perf=args.vllm_module_perf,
    )
    write_phase382_csv(args.output_csv, rows)
    write_phase382_md(args.output_md, rows)


if __name__ == "__main__":
    main()
