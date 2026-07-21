#!/usr/bin/env python3
"""Audit Phase467 performance sources for the six Kimi validation scenarios."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import validate_cb_simulator as validate

from aiconfigurator.sdk import common
from aiconfigurator.sdk.backends.vllm_backend import VLLMBackend
from aiconfigurator.sdk.config import RuntimeConfig
from aiconfigurator.sdk.perf_source import PerfSourceRecord


SCHEMA = "phase467_perf_source_audit_v2"
BASELINE_COMMIT = "1a24a99edd6fb21d8918bc07c2307dc46266653b"
DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "iter_gap_investigation" / "phase467_perf_sources"
NUMERIC_TOLERANCE = 1e-9
EXPECTED_SIM_OUTPUT_TOK_S_GPU = {
    "K2.5-tp8ep8-8k2k": 165.98449298431973,
    "K2.5-tp8ep8-32k3k": 49.50268539489951,
    "K2.5-tp4ep8dp2-8k2k": 163.07297901767987,
    "K2.5-tp4ep8dp2-32k3k": 64.04920249565279,
    "K2.5-tp8ep8-8k2k-bt65536": 155.86985187282406,
    "K2.5-tp4ep8dp2-8k2k-bt65536": 127.4640555948869,
}


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _enum_name(value):
    return getattr(value, "name", value)


def _actual_calibration_scope(point, *, model=None, database=None) -> dict:
    config = getattr(model, "config", None)
    scope = {
        "system": getattr(database, "system", validate.SYSTEM),
        "backend": getattr(database, "backend", validate.BACKEND),
        "version": getattr(database, "version", validate.VALIDATION_DB_VERSION),
        "model": getattr(model, "model_path", validate.MODEL_PATH),
        "moe_tp_size": getattr(config, "moe_tp_size", point.moe_tp),
        "moe_ep_size": getattr(config, "moe_ep_size", point.moe_ep),
    }
    if model is not None:
        scope.update(
            {
                "hidden_size": getattr(model, "_hidden_size", None),
                "inter_size": getattr(model, "_moe_inter_size", None),
                "topk": getattr(model, "_topk", None),
                "num_experts": getattr(model, "_num_experts", None),
                "quant_mode": _enum_name(getattr(config, "moe_quant_mode", None)),
            }
        )
    return scope


def _unresolved_source_row(point, phase, operation, source_index, source_type, source_id) -> dict:
    return {
        "scenario": point.name,
        "phase": phase,
        "operation": operation,
        "source_index": source_index,
        "source_type": source_type,
        "source_id": source_id,
        "data_file_sha256": "",
        "query_key": "{}",
        "row_key": "{}",
        "domain": "{}",
        "support_keys": "[]",
        "formula_id": "",
        "formula_inputs": "{}",
        "original_source": "",
        "anchor_id": "",
        "scale": "",
        "scope": "{}",
        "approved": False,
        "transforms": "[]",
    }


def audit_summary_sources(point, summary, *, model=None, database=None) -> list[dict]:
    rows: list[dict] = []
    actual_scope = _actual_calibration_scope(point, model=model, database=database)
    phase_maps = {
        "context": summary.get_context_source_map(),
        "generation": summary.get_generation_source_map(),
    }
    for phase, source_map in phase_maps.items():
        for operation, sources in sorted(source_map.items()):
            if not sources:
                rows.append(
                    _unresolved_source_row(
                        point,
                        phase,
                        operation,
                        0,
                        "missing",
                        "",
                    )
                )
                continue
            for source_index, source in enumerate(sources):
                if not isinstance(source, PerfSourceRecord):
                    rows.append(
                        _unresolved_source_row(
                            point,
                            phase,
                            operation,
                            source_index,
                            "unknown",
                            type(source).__name__,
                        )
                    )
                    continue
                if source.source_type == "calibrated":
                    source.validate_scope(actual_scope)
                payload = source.to_dict()
                rows.append(
                    {
                        "scenario": point.name,
                        "phase": phase,
                        "operation": operation,
                        "source_index": source_index,
                        "source_type": source.source_type,
                        "source_id": source.source_id,
                        "data_file_sha256": source.data_file_sha256 or "",
                        "query_key": _json(payload["query_key"]),
                        "row_key": _json(payload["row_key"]),
                        "domain": _json(payload["domain"]),
                        "support_keys": _json(payload["support_keys"]),
                        "formula_id": source.formula_id or "",
                        "formula_inputs": _json(payload["formula_inputs"]),
                        "original_source": source.original_source or "",
                        "anchor_id": source.anchor_id or "",
                        "scale": "" if source.scale is None else repr(source.scale),
                        "scope": _json(payload["scope"]),
                        "approved": source.approved,
                        "transforms": _json(payload["transforms"]),
                    }
                )
    if not rows:
        rows.append(
            _unresolved_source_row(
                point,
                "",
                "unregistered_iteration_cost",
                0,
                "missing",
                "",
            )
        )
    return rows


def audit_charge_ledger(point, summary, *, model=None, database=None) -> list[dict]:
    ledger = summary.get_iteration_charge_ledger()
    if not ledger:
        raise ValueError(f"scenario_has_no_iteration_charge_ledger:{point.name}")
    actual_scope = _actual_calibration_scope(point, model=model, database=database)
    rows: list[dict] = []
    for entry_index, entry in enumerate(ledger):
        missing_details = [
            {
                "phase": issue.phase,
                "operation": issue.operation,
                "latency_ms": issue.latency_ms,
            }
            for issue in entry.missing_sources
        ]
        unknown_details: list[dict] = []
        unapproved_details: list[dict] = []
        registered_charge_ms = 0.0
        valid_charge_ms = 0.0
        missing_source_charge_ms = 0.0
        unknown_source_charge_ms = 0.0
        unapproved_source_charge_ms = 0.0
        for charge in entry.charges:
            charge_ms = float(charge.latency_ms)
            if charge_ms == 0.0:
                continue
            registered_charge_ms += charge_ms
            charge_missing = not charge.sources or (
                charge.charge_id == "modeled_iteration" and bool(missing_details)
            )
            charge_unknown = False
            charge_unapproved = False
            if not charge.sources:
                missing_details.append(
                    {
                        "phase": charge.phase,
                        "operation": charge.charge_id,
                        "latency_ms": charge_ms,
                    }
                )
            for source_index, source in enumerate(charge.sources):
                if not isinstance(source, PerfSourceRecord):
                    charge_unknown = True
                    unknown_details.append(
                        {
                            "phase": charge.phase,
                            "charge_id": charge.charge_id,
                            "source_index": source_index,
                            "record_type": type(source).__name__,
                        }
                    )
                    continue
                if source.source_type == "structural" and not source.approved:
                    charge_unapproved = True
                    unapproved_details.append(
                        {
                            "phase": charge.phase,
                            "charge_id": charge.charge_id,
                            "source_index": source_index,
                            "formula_id": source.formula_id,
                        }
                    )
                if source.source_type == "calibrated":
                    source.validate_scope(actual_scope)
            if charge_missing:
                missing_source_charge_ms += charge_ms
            if charge_unknown:
                unknown_source_charge_ms += charge_ms
            if charge_unapproved:
                unapproved_source_charge_ms += charge_ms
            if not (charge_missing or charge_unknown or charge_unapproved):
                valid_charge_ms += charge_ms

        reconciliation_delta_ms = float(entry.reconciliation_delta_ms)
        if not entry.reconciled:
            status = "RECONCILIATION_FAILURE"
        elif missing_details:
            status = "MISSING_SOURCE"
        elif unknown_details:
            status = "UNKNOWN_SOURCE"
        elif unapproved_details:
            status = "UNAPPROVED_SOURCE"
        else:
            status = "PASS"
        rows.append(
            {
                "scenario": point.name,
                "entry_index": entry_index,
                "workload": _json(dict(entry.workload)),
                "total_ms": float(entry.total_ms),
                "registered_charge_ms": registered_charge_ms,
                "valid_charge_ms": valid_charge_ms,
                "missing_source_charge_ms": missing_source_charge_ms,
                "unknown_source_charge_ms": unknown_source_charge_ms,
                "unapproved_source_charge_ms": unapproved_source_charge_ms,
                "reconciliation_delta_ms": reconciliation_delta_ms,
                "missing_details": _json(missing_details),
                "unknown_details": _json(unknown_details),
                "unapproved_details": _json(unapproved_details),
                "status": status,
            }
        )
    return rows


def collect_audit() -> tuple[list[dict], list[dict], list[dict]]:
    backend = VLLMBackend()
    loaded: dict[tuple[int, int, int, int], tuple] = {}
    numeric_rows: list[dict] = []
    source_rows: list[dict] = []
    charge_rows: list[dict] = []
    for point in validate.MULTI_CONFIG_DATA:
        key = (point.tp, point.dp, point.moe_tp, point.moe_ep)
        if key not in loaded:
            loaded[key] = validate._load_model_and_db(
                tp=point.tp,
                dp=point.dp,
                moe_tp=point.moe_tp,
                moe_ep=point.moe_ep,
            )[:2]
        model, database = loaded[key]
        cb_config = validate._make_official_validation_cb_config(
            point,
            overlap_factor=0.0,
            ep8_per_iteration_overhead_ms=validate.DEFAULT_EP8_PER_ITERATION_OVERHEAD_MS,
        )
        with validate._official_validation_metric_scope(point):
            summary = backend.run_agg(
                model,
                database,
                RuntimeConfig(batch_size=point.batch_size, isl=point.isl, osl=point.osl),
                ctx_tokens=point.max_num_batched_tokens,
                database_mode=common.DatabaseMode.HYBRID,
                method="cb_sim",
                cb_config=cb_config,
            )
        result = summary.get_result_dict()
        actual = float(result["tokens/s/gpu"])
        expected = EXPECTED_SIM_OUTPUT_TOK_S_GPU[point.name]
        absolute_delta = abs(actual - expected)
        if absolute_delta > NUMERIC_TOLERANCE:
            raise ValueError(
                "numeric_invariance_failed:"
                f"{point.name}:expected={expected!r}:actual={actual!r}:delta={absolute_delta!r}"
            )
        numeric_rows.append(
            {
                "scenario": point.name,
                "before_sim_output_tok_s_gpu": expected,
                "after_sim_output_tok_s_gpu": actual,
                "absolute_delta": absolute_delta,
                "tolerance": NUMERIC_TOLERANCE,
                "status": "PASS",
            }
        )
        source_rows.extend(
            audit_summary_sources(point, summary, model=model, database=database)
        )
        charge_rows.extend(
            audit_charge_ledger(point, summary, model=model, database=database)
        )
    return numeric_rows, source_rows, charge_rows


def build_report(numeric_rows: list[dict], source_rows: list[dict], charge_rows: list[dict]) -> dict:
    source_types = Counter(row["source_type"] for row in source_rows)
    operations = {(row["scenario"], row["phase"], row["operation"]) for row in source_rows}
    unresolved_operations = {
        (row["scenario"], row["phase"], row["operation"])
        for row in source_rows
        if row["source_type"] in {"missing", "unknown"}
    }
    operations_with_source = operations - unresolved_operations
    coverage_denominator_ms = sum(
        max(abs(float(row["total_ms"])), abs(float(row["registered_charge_ms"])))
        for row in charge_rows
    )
    valid_charge_ms = sum(abs(float(row["valid_charge_ms"])) for row in charge_rows)
    source_coverage = (
        valid_charge_ms / coverage_denominator_ms
        if coverage_denominator_ms != 0.0
        else 1.0
    )
    missing_sources = [row for row in charge_rows if float(row["missing_source_charge_ms"]) != 0.0]
    unknown_sources = [row for row in charge_rows if float(row["unknown_source_charge_ms"]) != 0.0]
    unapproved_sources = [row for row in charge_rows if float(row["unapproved_source_charge_ms"]) != 0.0]
    reconciliation_failures = [
        row
        for row in charge_rows
        if not math.isclose(float(row["reconciliation_delta_ms"]), 0.0, rel_tol=0.0, abs_tol=1e-9)
    ]
    calibration_debt = {
        (row["source_id"], row["anchor_id"], row["scale"], row["scope"])
        for row in source_rows
        if row["source_type"] == "calibrated"
    }
    return {
        "schema": SCHEMA,
        "numeric_baseline_commit": BASELINE_COMMIT,
        "status": (
            "FAIL"
            if missing_sources or unknown_sources or unapproved_sources or reconciliation_failures
            else "PASS"
        ),
        "scenario_count": len(numeric_rows),
        "charged_operator_count": len(operations),
        "charged_operator_with_source_count": len(operations_with_source),
        "charge_ledger_entry_count": len(charge_rows),
        "source_coverage": source_coverage,
        "coverage_denominator_ms": coverage_denominator_ms,
        "valid_charge_ms": valid_charge_ms,
        "source_record_count": len(source_rows),
        "source_type_counts": dict(sorted(source_types.items())),
        "missing_source_count": len(missing_sources),
        "missing_sources": missing_sources,
        "unknown_source_count": len(unknown_sources),
        "unknown_sources": unknown_sources,
        "unapproved_source_count": len(unapproved_sources),
        "unapproved_sources": unapproved_sources,
        "reconciliation_failure_count": len(reconciliation_failures),
        "reconciliation_failures": reconciliation_failures,
        "calibration_debt": [
            {
                "source_id": source_id,
                "anchor_id": anchor_id,
                "scale": float(scale),
                "scope": json.loads(scope),
                "status": "technical_debt_existing_phase397v",
            }
            for source_id, anchor_id, scale, scope in sorted(calibration_debt)
        ],
        "numeric_invariance": numeric_rows,
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
        "default_aic": "No-Go",
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# Phase467 Performance Source Audit",
        "",
        "| Check | Result |",
        "|---|---:|",
        f"| Six-scenario numeric invariance | {report['scenario_count']}/6 PASS |",
        f"| Charged operators total | {report['charged_operator_count']} |",
        f"| Charged operators with sources | {report['charged_operator_with_source_count']} |",
        f"| Source coverage | {report['source_coverage']:.1%} |",
        f"| Source records | {report['source_record_count']} |",
        f"| Charge ledger entries | {report['charge_ledger_entry_count']} |",
        f"| Missing sources | {report['missing_source_count']} |",
        f"| Unknown sources | {report['unknown_source_count']} |",
        f"| Unapproved sources | {report['unapproved_source_count']} |",
        f"| Reconciliation failures | {report['reconciliation_failure_count']} |",
        f"| Numeric baseline commit | `{report['numeric_baseline_commit']}` |",
        "| Default AIC | No-Go |",
        "",
        "## Source Types",
        "",
        "| Type | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| {name} | {count} |" for name, count in report["source_type_counts"].items())
    lines.extend(["", "## Calibration Debt", ""])
    if report["calibration_debt"]:
        lines.extend(["| Source | Anchor | Scale | Status |", "|---|---|---:|---|"])
        lines.extend(
            f"| {row['source_id']} | {row['anchor_id']} | {row['scale']:.12f} | {row['status']} |"
            for row in report["calibration_debt"]
        )
    else:
        lines.append("No calibrated sources were charged.")
    issue_rows = {}
    for issue_type in (
        "missing_sources",
        "unknown_sources",
        "unapproved_sources",
        "reconciliation_failures",
    ):
        for row in report[issue_type]:
            issue_rows[(row["scenario"], row["entry_index"], row["status"])] = row
    lines.extend(["", "## Audit Failures", ""])
    if issue_rows:
        lines.extend(
            [
                "| Scenario | Entry | Status | Missing | Unknown | Unapproved | Reconciliation delta (ms) |",
                "|---|---:|---|---:|---:|---:|---:|",
            ]
        )
        lines.extend(
            "| {scenario} | {entry_index} | {status} | {missing_source_charge_ms:.6f} | "
            "{unknown_source_charge_ms:.6f} | {unapproved_source_charge_ms:.6f} | "
            "{reconciliation_delta_ms:.6f} |".format(**row)
            for row in issue_rows.values()
        )
    else:
        lines.append("No missing, unknown, unapproved, or unreconciled charges.")
    lines.extend(
        [
            "",
            "`diagnostic_only=true`, `valid_for_default=false`, `perf_database=false`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(out_dir: Path = DEFAULT_OUT_DIR) -> dict:
    numeric_rows, source_rows, charge_rows = collect_audit()
    report = build_report(numeric_rows, source_rows, charge_rows)
    _write_csv(out_dir / "phase467_numeric_invariance.csv", numeric_rows)
    _write_csv(out_dir / "phase467_perf_sources.csv", source_rows)
    _write_csv(out_dir / "phase467_charge_ledger.csv", charge_rows)
    (out_dir / "phase467_source_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(out_dir / "phase467_source_audit.md", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    report = run(args.out_dir)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
