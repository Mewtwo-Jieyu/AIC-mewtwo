#!/usr/bin/env python3
"""Enumerate vLLM module buckets required by 0.19 tp4dp2ep8 validation rows."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aiconfigurator.sdk import common, operations  # noqa: E402
from aiconfigurator.sdk.backends.vllm_backend import VLLMBackend  # noqa: E402
from aiconfigurator.sdk.config import RuntimeConfig  # noqa: E402
from aiconfigurator.sdk.performance_result import PerformanceResult  # noqa: E402
from scripts import validate_cb_simulator as validate_cb  # noqa: E402


DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase397w_required_buckets.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase397w_required_buckets.md"
VLLM_MODULE_TABLE = (
    REPO_ROOT
    / "src/aiconfigurator/systems/data/h200_sxm/vllm/0.19.0/vllm_module_perf.txt"
)
MEASURED_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase397k_measured_0190"

MODULE_BOUNDARIES = ("ep8_comm_dispatch_combine", "fusedmoe_runner_compute")
CSV_FIELDS = [
    "source",
    "scenario",
    "isl",
    "osl",
    "batch_size",
    "max_num_batched_tokens",
    "topology",
    "module_boundary",
    "requested_buckets",
    "requested_count",
    "materialized_buckets",
    "materialized_count",
    "missing_buckets",
    "missing_count",
    "bucket_128_required_by_runtime",
    "real_measurement_status",
    "acceptance_gate_eligible",
    "record_mode",
    "exact_lookup_only",
    "nearest_lookup_allowed",
    "interpolation_allowed",
    "extrapolation_allowed",
    "gpu_collection_started",
    "write_real_data_file",
    "default_readiness",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
    "verdict",
]


class _RecordingDatabase:
    def __init__(self, database) -> None:
        self._database = database
        self.calls: list[tuple[str, int]] = []

    def __getattr__(self, name: str):
        return getattr(self._database, name)

    def query_vllm_module(
        self,
        model: str,
        hardware: str,
        vllm_version: str,
        topology: str,
        bucket_tokens: int,
        module_boundary: str,
        quant_runtime: str,
    ) -> PerformanceResult:
        self.calls.append((module_boundary, int(bucket_tokens)))
        return PerformanceResult(0.0, energy=0.0)


def _slash(values: Iterable[int]) -> str:
    return "/".join(str(value) for value in sorted(values))


def _measured_scenarios(repo_root: Path) -> set[str]:
    measured_root = repo_root / MEASURED_ROOT.relative_to(REPO_ROOT)
    return {path.parent.name for path in measured_root.glob("*/meta.json")}


def _tp4dp2_scenarios() -> list:
    scenarios = [
        pt
        for pt in validate_cb.MULTI_CONFIG_DATA
        if (pt.tp, pt.dp, pt.moe_ep) == (4, 2, 8)
    ]
    names = {pt.name for pt in scenarios}
    if "K2.5-tp4ep8dp2-32k3k-bt65536" not in names:
        base = next(pt for pt in scenarios if pt.name == "K2.5-tp4ep8dp2-32k3k")
        scenarios.append(
            replace(
                base,
                name="K2.5-tp4ep8dp2-32k3k-bt65536",
                max_num_batched_tokens=65536,
            )
        )
    return scenarios


def _materialized_buckets(repo_root: Path) -> dict[str, set[int]]:
    table = repo_root / VLLM_MODULE_TABLE.relative_to(REPO_ROOT)
    by_boundary = {boundary: set() for boundary in MODULE_BOUNDARIES}
    with table.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            boundary = row["module_boundary"]
            if boundary in by_boundary:
                by_boundary[boundary].add(int(row["bucket_tokens"]))
    return by_boundary


def _record_requested_buckets(point) -> dict[str, set[int]]:
    model, database, _ = validate_cb._load_model_and_db(
        tp=point.tp,
        dp=point.dp,
        moe_tp=point.moe_tp,
        moe_ep=point.moe_ep,
    )
    recorder = _RecordingDatabase(database)
    backend = VLLMBackend()
    cb_config = validate_cb._make_cb_config(
        point.isl,
        point.batch_size,
        overlap_factor=0.0,
        per_iteration_overhead_ms=0.0,
        max_num_batched_tokens=point.max_num_batched_tokens,
    )

    original_validator = operations._validate_vllm_module_bucket
    operations._validate_vllm_module_bucket = lambda bucket_tokens: None
    try:
        backend.run_agg(
            model,
            recorder,
            RuntimeConfig(
                batch_size=point.batch_size,
                isl=point.isl,
                osl=point.osl,
            ),
            ctx_tokens=point.isl,
            database_mode=common.DatabaseMode.HYBRID,
            method="cb_sim",
            cb_config=cb_config,
        )
    finally:
        operations._validate_vllm_module_bucket = original_validator

    requested = {boundary: set() for boundary in MODULE_BOUNDARIES}
    for boundary, bucket in recorder.calls:
        if boundary in requested:
            requested[boundary].add(bucket)
    return requested


def _row(
    *,
    point,
    boundary: str,
    requested: set[int],
    materialized: set[int],
    measured: set[str],
) -> dict[str, str]:
    missing = requested - materialized
    measured_status = (
        "phase397k_measured_0190"
        if point.name in measured
        else "planned_missing_real_measurement"
    )
    gate_eligible = measured_status == "phase397k_measured_0190"
    if missing:
        verdict = "gpu_collection_required_before_exact_lookup_gate"
    else:
        verdict = "covered_by_current_vllm_module_table"
    if not gate_eligible:
        verdict = "not_acceptance_gate_row_until_real_measurement_exists"

    return {
        "source": "phase397w_required_buckets",
        "scenario": point.name,
        "isl": str(point.isl),
        "osl": str(point.osl),
        "batch_size": str(point.batch_size),
        "max_num_batched_tokens": str(point.max_num_batched_tokens),
        "topology": "tp4dp2ep8",
        "module_boundary": boundary,
        "requested_buckets": _slash(requested),
        "requested_count": str(len(requested)),
        "materialized_buckets": _slash(materialized),
        "materialized_count": str(len(materialized)),
        "missing_buckets": _slash(missing),
        "missing_count": str(len(missing)),
        "bucket_128_required_by_runtime": str(128 in requested).lower(),
        "real_measurement_status": measured_status,
        "acceptance_gate_eligible": str(gate_eligible).lower(),
        "record_mode": "operations_validator_bypassed_db_query_recorded",
        "exact_lookup_only": "true",
        "nearest_lookup_allowed": "false",
        "interpolation_allowed": "false",
        "extrapolation_allowed": "false",
        "gpu_collection_started": "false",
        "write_real_data_file": "false",
        "default_readiness": "No-Go",
        "diagnostic_only": "true",
        "valid_for_default": "false",
        "perf_database": "false",
        "verdict": verdict,
    }


def analyze_phase397w_required_buckets(repo_root: Path = REPO_ROOT) -> list[dict[str, str]]:
    measured = _measured_scenarios(repo_root)
    materialized = _materialized_buckets(repo_root)
    rows: list[dict[str, str]] = []
    for point in _tp4dp2_scenarios():
        requested = _record_requested_buckets(point)
        for boundary in MODULE_BOUNDARIES:
            rows.append(
                _row(
                    point=point,
                    boundary=boundary,
                    requested=requested[boundary],
                    materialized=materialized[boundary],
                    measured=measured,
                )
            )
    return rows


def _validate_rows(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("phase397w rows must not be empty")
    for row in rows:
        if row["diagnostic_only"] != "true":
            raise ValueError("diagnostic_only must remain true")
        if row["valid_for_default"] != "false":
            raise ValueError("valid_for_default must remain false")
        if row["perf_database"] != "false":
            raise ValueError("perf_database must remain false")
        if row["write_real_data_file"] != "false":
            raise ValueError("write_real_data_file must remain false")
        if row["default_readiness"] != "No-Go":
            raise ValueError("default_readiness must remain No-Go")
        if row["nearest_lookup_allowed"] != "false":
            raise ValueError("nearest_lookup_allowed must remain false")
        if row["interpolation_allowed"] != "false":
            raise ValueError("interpolation_allowed must remain false")
        if row["extrapolation_allowed"] != "false":
            raise ValueError("extrapolation_allowed must remain false")


def write_phase397w_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_phase397w_md(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    missing_rows = [row for row in rows if row["missing_count"] != "0"]
    required_128 = any(row["bucket_128_required_by_runtime"] == "true" for row in rows)
    planned_missing = [
        row["scenario"]
        for row in rows
        if row["real_measurement_status"] == "planned_missing_real_measurement"
    ]

    lines = [
        "# Phase397w required buckets",
        "",
        "| Item | Result |",
        "|---|---|",
        f"| Rows analyzed | {len(rows)} |",
        f"| Rows with missing buckets | {len(missing_rows)} |",
        f"| bucket 128 required by current runtime | {str(required_128).lower()} |",
        f"| planned scenario without 0.19-real measurement | {', '.join(sorted(set(planned_missing))) or 'none'} |",
        "| GPU collection | GPU collection is not started in this phase397w enumeration step |",
        "| Default AIC | No-Go |",
        "",
        "## Missing buckets",
        "",
        "| Scenario | Boundary | Missing count | Missing buckets |",
        "|---|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {row['module_boundary']} | "
            f"{row['missing_count']} | {row['missing_buckets'] or 'none'} |"
        )

    lines.extend(
        [
            "",
            "## Verdict",
            "",
            "The current tp4dp2ep8 simulator emits a broad bucket family, not only bucket 2000. "
            "A naive GPU bucket-widen pass would have to materialize many scheduler-shaped buckets, "
            "including bucket 128, before exact lookup can pass.",
            "",
            "This is diagnostic evidence only. It does not write `vllm_module_perf.txt`, "
            "does not relax exact lookup, and does not enable Default AIC.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args(argv)

    rows = analyze_phase397w_required_buckets(args.repo_root)
    write_phase397w_csv(args.out_csv, rows)
    write_phase397w_md(args.out_md, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
