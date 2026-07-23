#!/usr/bin/env python3
"""Validate and summarize a Phase469B no-model runtime preflight."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_phase469b_fpm_runtime_preflight as runner


DEFAULT_OUTPUT_DIR = (
    SCRIPT_DIR.parent
    / "docs"
    / "iter_gap_investigation"
    / "phase469b_fpm_runtime_preflight"
)


class AnalysisContractError(ValueError):
    """The Phase469B result cannot support the claimed state."""


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json_object(path: Path, *, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisContractError(f"{label}_json_invalid") from exc
    if not isinstance(value, dict):
        raise AnalysisContractError(f"{label}_object_required")
    return value


def _markdown(result: dict) -> str:
    reasons = result.get("blocking_reasons") or []
    reason_lines = [f"- `{reason}`" for reason in reasons] or ["- None"]
    not_evaluated = result.get("not_evaluated") or []
    not_evaluated_lines = [
        f"- `{item}`" for item in not_evaluated
    ] or ["- None"]
    attestation = result.get("attestation") or {}
    model = attestation.get("model_identity") or {}
    flat = model.get("flat_model_identity") or {}
    plan = attestation.get("plan") or {}
    return "\n".join(
        [
            "# Phase469B FPM Runtime Preflight",
            "",
            f"- Status: `{result['status']}`",
            f"- Grid coverage: `{result['grid_coverage']}`",
            f"- Execution manifest: `{result['execution_manifest_sha256']}`",
            f"- Source bundle manifest: `{result['source_bundle_manifest_sha256']}`",
            f"- Model identity: `{result['model_identity_status']}`",
            f"- AIC model config: `{model.get('aic_model_config_sha256', '')}`",
            f"- Checkpoint config: `{model.get('checkpoint_config_sha256', '')}`",
            f"- Flat model fingerprint: `{flat.get('fingerprint_sha256', '')}`",
            f"- Planner: `{plan.get('status', '')}`",
            "- Model loaded: `false`",
            "- Benchmark executed: `false`",
            "- diagnostic_only: `true`",
            "- valid_for_default: `false`",
            "- perf_database: `false`",
            "- Default AIC: `No-Go`",
            "",
            "## Blocking Reasons",
            "",
            *reason_lines,
            "",
            "## Not Evaluated",
            "",
            *not_evaluated_lines,
            "",
        ]
    )


def analyze_preflight_result(
    result_path: Path,
    *,
    output_dir: Path,
    execution_manifest_path: Path,
    source_bundle_manifest_path: Path,
) -> str:
    execution_manifest = _load_json_object(
        execution_manifest_path,
        label="execution_manifest",
    )
    source_bundle_manifest = _load_json_object(
        source_bundle_manifest_path,
        label="source_bundle_manifest",
    )
    try:
        runner.validate_coordinator_manifest(
            SCRIPT_DIR.parent,
            execution_manifest,
        )
        runner.validate_source_bundle_manifest_document(source_bundle_manifest)
        runner.validate_manifest_chain(
            execution_manifest,
            source_bundle_manifest,
        )
    except runner.PreflightContractError as exc:
        raise AnalysisContractError(f"manifest_contract:{exc}") from exc
    expected_execution = execution_manifest["sha256"]
    expected_source = source_bundle_manifest["sha256"]
    result = _load_json_object(result_path, label="result")
    if result.get("schema") == runner.SUPERSEDED_RESULT_SCHEMA:
        raise AnalysisContractError("SUPERSEDED_INVALID_CONFIG_COMPARISON")
    if not isinstance(result, dict) or result.get("schema") != runner.RESULT_SCHEMA:
        raise AnalysisContractError("result_schema_invalid")
    if result.get("status") not in {
        "READY_FOR_TP8_CANARY",
        "BLOCKED_RUNTIME_CONTRACT",
    }:
        raise AnalysisContractError("result_status_invalid")
    if result.get("execution_manifest_sha256") != expected_execution:
        raise AnalysisContractError("execution_manifest_mismatch")
    if result.get("source_bundle_manifest_sha256") != expected_source:
        raise AnalysisContractError("source_bundle_manifest_mismatch")
    reasons = result.get("blocking_reasons")
    if not isinstance(reasons, list) or any(not isinstance(item, str) for item in reasons):
        raise AnalysisContractError("blocking_reasons_invalid")
    if result["status"] == "READY_FOR_TP8_CANARY" and reasons:
        raise AnalysisContractError("ready_result_has_blocking_reasons")
    if result["status"] == "BLOCKED_RUNTIME_CONTRACT" and not reasons:
        raise AnalysisContractError("blocked_result_has_no_reason")
    derived = runner.evaluate_worker_attestation(
        result.get("attestation") or {},
        execution_manifest=execution_manifest,
        source_bundle_manifest=source_bundle_manifest,
    )
    if (
        derived["status"] != result["status"]
        or derived["blocking_reasons"] != reasons
        or derived["model_identity_status"] != result.get("model_identity_status")
        or derived["not_evaluated"] != result.get("not_evaluated")
    ):
        raise AnalysisContractError("result_attestation_reconciliation")
    if result.get("grid_coverage") != "NOT_EVALUATED_RUNTIME_DETERMINED":
        raise AnalysisContractError("grid_coverage_contract_invalid")
    if (
        result.get("model_loaded") is not False
        or result.get("benchmark_executed") is not False
    ):
        raise AnalysisContractError("execution_boundary_invalid")
    if (
        result.get("diagnostic_only") is not True
        or result.get("valid_for_default") is not False
        or result.get("perf_database") is not False
        or result.get("default_aic") != "No-Go"
    ):
        raise AnalysisContractError("result_boundary_invalid")
    _write_json(output_dir / "phase469b_runtime_preflight.json", result)
    (output_dir / "phase469b_runtime_preflight.md").write_text(
        _markdown(result), encoding="utf-8"
    )
    return result["status"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--source-bundle-manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        status = analyze_preflight_result(
            args.result,
            output_dir=args.output_dir,
            execution_manifest_path=args.execution_manifest,
            source_bundle_manifest_path=args.source_bundle_manifest,
        )
    except AnalysisContractError as exc:
        print(f"BLOCKED_RUNTIME_CONTRACT:{exc}", file=sys.stderr)
        return 1
    print(status)
    return 0 if status == "READY_FOR_TP8_CANARY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
