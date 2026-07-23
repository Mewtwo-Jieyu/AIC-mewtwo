"""Tests for the Phase469B runtime preflight analyzer."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER_PATH = REPO_ROOT / "scripts" / "run_phase469b_fpm_runtime_preflight.py"
ANALYZER_PATH = REPO_ROOT / "scripts" / "analyze_phase469b_fpm_runtime_preflight.py"
ARTIFACT_DIR = (
    REPO_ROOT
    / "docs"
    / "iter_gap_investigation"
    / "phase469b_fpm_runtime_preflight"
)


def _load(path: Path, name: str):
    assert path.is_file(), f"missing script:{path}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _source_manifest(runner) -> dict:
    return runner._with_digest(
        {
            "schema": runner.SOURCE_BUNDLE_SCHEMA,
            "aic_commit": runner.AIC_COMMIT,
            "dynamo_commit": runner.DYNAMO_COMMIT,
            "source_attestation_only": True,
            "runtime_installation": False,
            "files": [
                {"path": path, "sha256": digest}
                for path, digest in sorted(runner.PINNED_SOURCE_FILES.items())
            ],
        }
    )


def _execution_manifest(runner, source_manifest: dict) -> dict:
    return runner.build_coordinator_manifest(
        REPO_ROOT,
        base_commit="1" * 40,
        source_bundle_manifest_sha256=source_manifest["sha256"],
    )


def _runtime_absent_attestation(runner, execution: dict, source: dict) -> dict:
    semantic = execution["target"]["aic_model_semantic_projection"]
    return {
        "schema": runner.WORKER_ATTESTATION_SCHEMA,
        "execution_manifest_sha256": execution["sha256"],
        "host": {
            "ssh_host": runner.DEFAULT_SSH_HOST,
            "hostkey_fingerprint": runner.EXPECTED_HOSTKEY_FINGERPRINT,
            "hostname": "worker",
        },
        "gpu": {
            "rows": [
                f"{index}, GPU-{index:02d}, NVIDIA H200, 143771, 570.133.20"
                for index in range(8)
            ],
            "residue_before": "",
            "residue_after": "",
        },
        "process_residue": {"before": "", "after": ""},
        "software": {
            "vllm_version": "0.19.0",
            "dynamo_module": "",
            "dynamo_source_sha256": "",
            "dynamo_methods": [],
            "dynamo_artifact_fields": [],
            "dynamo_fpm_fields": [],
            "aiconfigurator_runtime_module": "",
            "aiconfigurator_runtime_files": {},
        },
        "source_bundle": {
            "manifest_sha256": source["sha256"],
            "all_files_match": True,
            "source_attestation_only": True,
            "runtime_installation": False,
        },
        "model_identity": {
            "model_id": runner.EXPECTED_MODEL_ID,
            "model_load_path": runner.DEFAULT_MODEL_PATH,
            "aic_model_config_sha256": runner.EXPECTED_AIC_MODEL_CONFIG_SHA256,
            "checkpoint_config_sha256": (
                runner.EXPECTED_CHECKPOINT_CONFIG_SHA256
            ),
            "flat_model_identity": {
                "schema": "phase466_flat_model_fingerprint_v1",
                "identity_scope": "same_flat_mirror_instance",
                "fingerprint_sha256": runner.EXPECTED_FLAT_MODEL_FINGERPRINT,
                "shards": [{} for _ in range(64)],
            },
            "model_semantic_identity": {
                "status": "PASS",
                "aic_projection": semantic,
                "checkpoint_projection": semantic,
            },
        },
        "quant": {
            "expected_quant_runtime": runner.EXPECTED_QUANT_RUNTIME,
            "actual_quant_runtime": runner.DEFERRED_QUANT_RUNTIME,
        },
        "plan": {
            "planner_commit": runner.AIC_COMMIT,
            "status": "NOT_EVALUATED_RUNTIME_UNAVAILABLE",
            "cells": [],
            "points": "runtime-determined",
        },
        "grid_coverage": "NOT_EVALUATED_RUNTIME_DETERMINED",
        "model_loaded": False,
        "benchmark_executed": False,
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
        "default_aic": "No-Go",
    }


def _write_contract(tmp_path: Path, runner) -> tuple[Path, Path, Path, dict]:
    source = _source_manifest(runner)
    execution = _execution_manifest(runner, source)
    result = runner.evaluate_worker_attestation(
        _runtime_absent_attestation(runner, execution, source),
        execution_manifest=execution,
        source_bundle_manifest=source,
    )
    result_path = tmp_path / "result.json"
    execution_path = tmp_path / "execution.json"
    source_path = tmp_path / "source.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    execution_path.write_text(json.dumps(execution), encoding="utf-8")
    source_path.write_text(json.dumps(source), encoding="utf-8")
    return result_path, execution_path, source_path, result


def test_blocked_result_writes_artifacts_and_returns_nonzero(tmp_path: Path) -> None:
    runner = _load(RUNNER_PATH, "phase469b_runner_for_analyzer")
    analyzer = _load(ANALYZER_PATH, "phase469b_analyzer_blocked")
    result_path, execution_path, source_path, _ = _write_contract(
        tmp_path, runner
    )

    status = analyzer.analyze_preflight_result(
        result_path,
        output_dir=tmp_path / "out",
        execution_manifest_path=execution_path,
        source_bundle_manifest_path=source_path,
    )

    assert status == "BLOCKED_RUNTIME_CONTRACT"
    assert (tmp_path / "out" / "phase469b_runtime_preflight.json").is_file()
    assert (tmp_path / "out" / "phase469b_runtime_preflight.md").is_file()


def test_analyzer_rejects_false_ready_and_boundary_drift(tmp_path: Path) -> None:
    runner = _load(RUNNER_PATH, "phase469b_runner_false_ready")
    analyzer = _load(ANALYZER_PATH, "phase469b_analyzer_false_ready")
    result_path, execution_path, source_path, result = _write_contract(
        tmp_path, runner
    )
    result["status"] = "READY_FOR_TP8_CANARY"
    result["diagnostic_only"] = False
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(analyzer.AnalysisContractError):
        analyzer.analyze_preflight_result(
            result_path,
            output_dir=tmp_path / "out",
            execution_manifest_path=execution_path,
            source_bundle_manifest_path=source_path,
        )


def test_analyzer_rejects_manifest_mismatch(tmp_path: Path) -> None:
    runner = _load(RUNNER_PATH, "phase469b_runner_manifest_mismatch")
    analyzer = _load(ANALYZER_PATH, "phase469b_analyzer_manifest_mismatch")
    result_path, execution_path, source_path, result = _write_contract(
        tmp_path, runner
    )
    result["execution_manifest_sha256"] = "f" * 64
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(analyzer.AnalysisContractError, match="manifest"):
        analyzer.analyze_preflight_result(
            result_path,
            output_dir=tmp_path / "out",
            execution_manifest_path=execution_path,
            source_bundle_manifest_path=source_path,
        )


def test_analyzer_rejects_not_evaluated_drift(tmp_path: Path) -> None:
    runner = _load(RUNNER_PATH, "phase469b_runner_not_evaluated")
    analyzer = _load(ANALYZER_PATH, "phase469b_analyzer_not_evaluated")
    result_path, execution_path, source_path, result = _write_contract(
        tmp_path, runner
    )
    result["not_evaluated"] = []
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(
        analyzer.AnalysisContractError,
        match="attestation_reconciliation",
    ):
        analyzer.analyze_preflight_result(
            result_path,
            output_dir=tmp_path / "out",
            execution_manifest_path=execution_path,
            source_bundle_manifest_path=source_path,
        )


def test_v1_result_is_explicitly_superseded(tmp_path: Path) -> None:
    runner = _load(RUNNER_PATH, "phase469b_runner_v1_superseded")
    analyzer = _load(ANALYZER_PATH, "phase469b_analyzer_v1_superseded")
    result_path, execution_path, source_path, result = _write_contract(
        tmp_path, runner
    )
    result["schema"] = runner.SUPERSEDED_RESULT_SCHEMA
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(
        analyzer.AnalysisContractError,
        match="SUPERSEDED_INVALID_CONFIG_COMPARISON",
    ):
        analyzer.analyze_preflight_result(
            result_path,
            output_dir=tmp_path / "out",
            execution_manifest_path=execution_path,
            source_bundle_manifest_path=source_path,
        )


def test_v2_result_remains_bound_historical_external_block_evidence() -> None:
    history = json.loads(
        (ARTIFACT_DIR / "phase469b_runtime_preflight_v2_history.json").read_text(
            encoding="utf-8"
        )
    )

    assert history["status"] == "VALID_HISTORICAL_EXTERNAL_RUNTIME_BLOCK"
    assert history["result_schema"] == "phase469b_fpm_runtime_preflight_v2"
    for file_key, digest_key in (
        ("result_file", "result_file_sha256"),
        ("result_markdown_file", "result_markdown_file_sha256"),
        ("execution_manifest_file", "execution_manifest_file_sha256"),
        ("source_bundle_manifest_file", "source_bundle_manifest_file_sha256"),
    ):
        path = ARTIFACT_DIR / history[file_key]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == history[digest_key]


def test_analyzer_reads_manifest_files_and_rejects_tampered_content(
    tmp_path: Path,
) -> None:
    runner = _load(RUNNER_PATH, "phase469b_runner_manifest_content")
    analyzer = _load(ANALYZER_PATH, "phase469b_analyzer_manifest_content")
    result_path, execution_path, source_path, _ = _write_contract(
        tmp_path, runner
    )
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    execution["target"]["ssh_host"] = "tampered-worker.example"
    execution_path.write_text(json.dumps(execution), encoding="utf-8")

    with pytest.raises(analyzer.AnalysisContractError, match="manifest"):
        analyzer.analyze_preflight_result(
            result_path,
            output_dir=tmp_path / "out",
            execution_manifest_path=execution_path,
            source_bundle_manifest_path=source_path,
        )


def test_analyzer_rejects_unbound_source_manifest(tmp_path: Path) -> None:
    runner = _load(RUNNER_PATH, "phase469b_runner_manifest_chain")
    analyzer = _load(ANALYZER_PATH, "phase469b_analyzer_manifest_chain")
    result_path, execution_path, source_path, _ = _write_contract(
        tmp_path, runner
    )
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    execution["source_bundle_manifest_sha256"] = "c" * 64
    execution.pop("sha256")
    execution = runner._with_digest(execution)
    execution_path.write_text(json.dumps(execution), encoding="utf-8")

    with pytest.raises(analyzer.AnalysisContractError, match="manifest"):
        analyzer.analyze_preflight_result(
            result_path,
            output_dir=tmp_path / "out",
            execution_manifest_path=execution_path,
            source_bundle_manifest_path=source_path,
        )
