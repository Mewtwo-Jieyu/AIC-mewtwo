"""Tests for the Phase469B no-model runtime preflight runner."""
from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "run_phase469b_fpm_runtime_preflight.py"


def _load():
    assert SCRIPT.is_file(), "Phase469B runner is missing"
    spec = importlib.util.spec_from_file_location(
        "run_phase469b_fpm_runtime_preflight_test",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _source_manifest(module) -> dict:
    return {
        "sha256": "b" * 64,
        "source_attestation_only": True,
        "runtime_installation": False,
    }


def _execution_manifest(module) -> dict:
    aic, _ = _semantic_configs()
    return {
        "sha256": "a" * 64,
        "source_bundle_manifest_sha256": "b" * 64,
        "target": {
            "ssh_host": module.DEFAULT_SSH_HOST,
            "hostkey_fingerprint": module.EXPECTED_HOSTKEY_FINGERPRINT,
            "model_path": module.DEFAULT_MODEL_PATH,
            "aic_model_config_sha256": module.EXPECTED_AIC_MODEL_CONFIG_SHA256,
            "checkpoint_config_sha256": module.EXPECTED_CHECKPOINT_CONFIG_SHA256,
            "flat_model_fingerprint": module.EXPECTED_FLAT_MODEL_FINGERPRINT,
            "flat_model_identity_scope": "same_flat_mirror_instance",
            "aic_model_semantic_projection": module._aic_model_semantic_projection(aic),
        },
    }


def _ready_attestation(module) -> dict:
    semantic = _execution_manifest(module)["target"]["aic_model_semantic_projection"]
    return {
        "schema": module.WORKER_ATTESTATION_SCHEMA,
        "execution_manifest_sha256": "a" * 64,
        "host": {
            "ssh_host": module.DEFAULT_SSH_HOST,
            "hostkey_fingerprint": module.EXPECTED_HOSTKEY_FINGERPRINT,
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
            "dynamo_module": "dynamo.vllm.instrumented_scheduler",
            "dynamo_source_sha256": module.PINNED_DYNAMO_SCHEDULER_SHA256,
            "dynamo_methods": list(module.REQUIRED_DYNAMO_METHODS),
            "dynamo_artifact_fields": list(module.REQUIRED_DYNAMO_ARTIFACT_FIELDS),
            "dynamo_fpm_fields": list(module.REQUIRED_FPM_FIELDS),
            "aiconfigurator_runtime_module": (
                "collector.fpm_forward.planner"
            ),
            "aiconfigurator_runtime_files": dict(
                module.PINNED_AIC_RUNTIME_FILES
            ),
        },
        "source_bundle": {
            "manifest_sha256": "b" * 64,
            "all_files_match": True,
            "source_attestation_only": True,
            "runtime_installation": False,
        },
        "model_identity": {
            "model_id": module.EXPECTED_MODEL_ID,
            "model_load_path": module.DEFAULT_MODEL_PATH,
            "aic_model_config_sha256": module.EXPECTED_AIC_MODEL_CONFIG_SHA256,
            "checkpoint_config_sha256": module.EXPECTED_CHECKPOINT_CONFIG_SHA256,
            "flat_model_identity": {
                "schema": "phase466_flat_model_fingerprint_v1",
                "identity_scope": "same_flat_mirror_instance",
                "fingerprint_sha256": module.EXPECTED_FLAT_MODEL_FINGERPRINT,
                "shards": [{} for _ in range(64)],
            },
            "model_semantic_identity": {
                "status": "PASS",
                "aic_projection": semantic,
                "checkpoint_projection": semantic,
            },
        },
        "quant": {
            "expected_quant_runtime": module.EXPECTED_QUANT_RUNTIME,
            "actual_quant_runtime": module.DEFERRED_QUANT_RUNTIME,
        },
        "plan": {
            "planner_commit": module.AIC_COMMIT,
            "status": "EVALUATED",
            "cells": deepcopy(module.REQUIRED_PLAN_CELLS),
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


def test_ready_attestation_requires_every_runtime_contract() -> None:
    module = _load()

    result = module.evaluate_worker_attestation(
        _ready_attestation(module),
        execution_manifest=_execution_manifest(module),
        source_bundle_manifest=_source_manifest(module),
    )

    assert result["status"] == "READY_FOR_TP8_CANARY"
    assert result["grid_coverage"] == "NOT_EVALUATED_RUNTIME_DETERMINED"


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda row: row["software"].update(vllm_version="0.18.0"), "vllm_version"),
        (lambda row: row["gpu"]["rows"].pop(), "gpu_identity"),
        (
            lambda row: row["software"].update(dynamo_source_sha256="d" * 64),
            "dynamo_runtime_incompatible",
        ),
        (
            lambda row: row["source_bundle"].update(all_files_match=False),
            "source_bundle",
        ),
        (
            lambda row: row["software"].update(dynamo_fpm_fields=[]),
            "dynamo_runtime_incompatible",
        ),
        (
            lambda row: row["software"]["aiconfigurator_runtime_files"].update(
                {"collector/fpm_forward/planner.py": "f" * 64}
            ),
            "aiconfigurator_runtime_incompatible",
        ),
        (
            lambda row: row["model_identity"]["flat_model_identity"].update(
                fingerprint_sha256="c" * 64
            ),
            "flat_model_identity",
        ),
        (
            lambda row: row["quant"].update(actual_quant_runtime="FilledTooEarly"),
            "actual_quant_deferred",
        ),
        (lambda row: row["gpu"].update(residue_after="pid=7"), "gpu_residue"),
        (lambda row: row["process_residue"].update(before="vllm"), "process_residue"),
        (lambda row: row["plan"]["cells"].pop(), "planner_contract"),
    ],
)
def test_runtime_contract_failures_block_canary(mutation, reason: str) -> None:
    module = _load()
    attestation = _ready_attestation(module)
    mutation(attestation)

    result = module.evaluate_worker_attestation(
        attestation,
        execution_manifest=_execution_manifest(module),
        source_bundle_manifest=_source_manifest(module),
    )

    assert result["status"] == "BLOCKED_RUNTIME_CONTRACT"
    assert reason in result["blocking_reasons"]


def test_execution_manifest_mismatch_blocks_canary() -> None:
    module = _load()
    attestation = _ready_attestation(module)

    result = module.evaluate_worker_attestation(
        attestation,
        execution_manifest={
            **_execution_manifest(module),
            "sha256": "f" * 64,
        },
        source_bundle_manifest=_source_manifest(module),
    )

    assert result["status"] == "BLOCKED_RUNTIME_CONTRACT"
    assert "execution_manifest" in result["blocking_reasons"]


def test_execution_target_mismatch_blocks_canary() -> None:
    module = _load()
    attestation = _ready_attestation(module)
    attestation["host"]["ssh_host"] = "wrong-worker.example"

    result = module.evaluate_worker_attestation(
        attestation,
        execution_manifest=_execution_manifest(module),
        source_bundle_manifest=_source_manifest(module),
    )

    assert result["status"] == "BLOCKED_RUNTIME_CONTRACT"
    assert "execution_target" in result["blocking_reasons"]


def test_execution_manifest_must_bind_loaded_source_manifest() -> None:
    module = _load()
    execution = _execution_manifest(module)
    execution["source_bundle_manifest_sha256"] = "c" * 64

    with pytest.raises(
        module.PreflightContractError,
        match="source_bundle_manifest_chain",
    ):
        module.validate_manifest_chain(execution, _source_manifest(module))


def test_source_bundle_manifest_rejects_missing_or_changed_file(tmp_path: Path) -> None:
    module = _load()
    bundle = tmp_path / "bundle"
    source = bundle / "pinned" / "example.py"
    source.parent.mkdir(parents=True)
    source.write_text("value = 1\n", encoding="utf-8")
    manifest = module.build_source_bundle_manifest(
        bundle,
        expected_files={"pinned/example.py": module.sha256_file(source)},
    )
    assert module.validate_source_bundle_manifest(
        bundle,
        manifest,
        expected_files={"pinned/example.py": module.sha256_file(source)},
    )

    source.write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(module.PreflightContractError, match="source_bundle_hash"):
        module.validate_source_bundle_manifest(
            bundle,
            manifest,
            expected_files={"pinned/example.py": manifest["files"][0]["sha256"]},
        )
    source.unlink()
    with pytest.raises(module.PreflightContractError, match="source_bundle_missing"):
        module.validate_source_bundle_manifest(
            bundle,
            manifest,
            expected_files={"pinned/example.py": manifest["files"][0]["sha256"]},
        )


def test_source_bundle_manifest_cannot_replace_pinned_file_set(tmp_path: Path) -> None:
    module = _load()
    source = tmp_path / "replacement.py"
    source.write_text("value = 1\n", encoding="utf-8")
    manifest = module.build_source_bundle_manifest(
        tmp_path,
        expected_files={"replacement.py": module.sha256_file(source)},
    )

    with pytest.raises(module.PreflightContractError, match="source_bundle_file_set"):
        module.validate_source_bundle_manifest(tmp_path, manifest)


def test_coordinator_manifest_binds_tool_bytes_and_base_commit(tmp_path: Path) -> None:
    module = _load()
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    for relative in module.COORDINATOR_TOOL_PATHS:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative + "\n", encoding="utf-8")
    config_path = repo / module.AIC_MODEL_CONFIG_RELATIVE_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_bytes(
        (REPO_ROOT / module.AIC_MODEL_CONFIG_RELATIVE_PATH).read_bytes()
    )

    manifest = module.build_coordinator_manifest(
        repo,
        base_commit="1" * 40,
        source_bundle_manifest_sha256="2" * 64,
    )

    assert manifest["base_commit"] == "1" * 40
    assert set(manifest["tools"]) == set(module.COORDINATOR_TOOL_PATHS)
    (scripts / "run_phase469b_fpm_runtime_preflight.py").write_text(
        "changed\n", encoding="utf-8"
    )
    with pytest.raises(module.PreflightContractError, match="tool_hash"):
        module.validate_coordinator_manifest(repo, manifest)


def test_coordinator_manifest_accepts_custom_execution_target(
    tmp_path: Path,
) -> None:
    module = _load()
    repo = tmp_path / "repo"
    for relative in module.COORDINATOR_TOOL_PATHS:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative + "\n", encoding="utf-8")
    config_path = repo / module.AIC_MODEL_CONFIG_RELATIVE_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_bytes(
        (REPO_ROOT / module.AIC_MODEL_CONFIG_RELATIVE_PATH).read_bytes()
    )
    manifest = module.build_coordinator_manifest(
        repo,
        base_commit="1" * 40,
        source_bundle_manifest_sha256="2" * 64,
        ssh_host="ws-custom-worker.other-user+root.ailab-sys.pod@h.pjlab.org.cn",
        hostkey_fingerprint="SHA256:" + "A" * 43,
        model_path="/mnt/models/Kimi-K2.5",
    )

    assert module.validate_coordinator_manifest(repo, manifest)


def test_worker_failure_still_writes_blocked_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load()
    output = tmp_path / "worker.json"
    monkeypatch.setattr(
        module,
        "collect_worker_attestation",
        lambda **kwargs: {
            **_ready_attestation(module),
            "software": {
                **_ready_attestation(module)["software"],
                "dynamo_module": "",
            },
        },
    )
    monkeypatch.setattr(module, "validate_coordinator_manifest", lambda *args: True)
    monkeypatch.setattr(module, "validate_source_bundle_manifest", lambda *args: True)

    exit_code = module.run_worker_preflight(
        output_path=output,
        execution_manifest=_execution_manifest(module),
        source_bundle_root=tmp_path,
        source_bundle_manifest=_source_manifest(module),
        model_path=Path(module.DEFAULT_MODEL_PATH),
        ssh_host=module.DEFAULT_SSH_HOST,
        hostkey_fingerprint=module.EXPECTED_HOSTKEY_FINGERPRINT,
    )

    assert exit_code == 1
    assert json.loads(output.read_text())["status"] == "BLOCKED_RUNTIME_CONTRACT"


def test_unknown_worker_exception_propagates_without_formal_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load()
    output = tmp_path / "worker.json"
    monkeypatch.setattr(module, "validate_coordinator_manifest", lambda *args: True)
    monkeypatch.setattr(module, "validate_source_bundle_manifest", lambda *args: True)
    monkeypatch.setattr(
        module,
        "collect_worker_attestation",
        lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("unexpected worker failure")
        ),
    )

    with pytest.raises(RuntimeError, match="unexpected worker failure"):
        module.run_worker_preflight(
            output_path=output,
            execution_manifest=_execution_manifest(module),
            source_bundle_root=tmp_path,
            source_bundle_manifest=_source_manifest(module),
            model_path=Path(module.DEFAULT_MODEL_PATH),
            ssh_host=module.DEFAULT_SSH_HOST,
            hostkey_fingerprint=module.EXPECTED_HOSTKEY_FINGERPRINT,
        )

    assert not output.exists()


def test_worker_target_tamper_fails_without_formal_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load()
    output = tmp_path / "worker.json"
    monkeypatch.setattr(module, "validate_coordinator_manifest", lambda *args: True)
    monkeypatch.setattr(module, "validate_source_bundle_manifest", lambda *args: True)

    with pytest.raises(module.PreflightContractError, match="execution_target_mismatch"):
        module.run_worker_preflight(
            output_path=output,
            execution_manifest=_execution_manifest(module),
            source_bundle_root=tmp_path,
            source_bundle_manifest=_source_manifest(module),
            model_path=Path(module.DEFAULT_MODEL_PATH),
            ssh_host="tampered-worker.example+root.ailab-sys.pod@h.pjlab.org.cn",
            hostkey_fingerprint=module.EXPECTED_HOSTKEY_FINGERPRINT,
        )

    assert not output.exists()


def test_phase469b_v3_schema_and_historical_boundaries() -> None:
    module = _load()

    assert module.SOURCE_BUNDLE_SCHEMA.endswith("_v3")
    assert module.EXECUTION_MANIFEST_SCHEMA.endswith("_v3")
    assert module.WORKER_ATTESTATION_SCHEMA.endswith("_v3")
    assert module.RESULT_SCHEMA.endswith("_v3")
    assert module.SUPERSEDED_RESULT_SCHEMA.endswith("_v1")
    assert module.HISTORICAL_RESULT_SCHEMA.endswith("_v2")


def test_software_identity_treats_only_explicit_absence_as_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load()
    responses = iter(
        [
            "0.19.0",
            json.dumps({"status": "ABSENT"}),
            json.dumps({"status": "ABSENT"}),
        ]
    )
    monkeypatch.setattr(module, "_capture", lambda *args, **kwargs: next(responses))

    identity = module._software_identity({})

    assert identity["dynamo_module"] == ""
    assert identity["aiconfigurator_runtime_module"] == ""


def test_software_identity_rejects_malformed_probe_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load()
    responses = iter(["0.19.0", "not-json"])
    monkeypatch.setattr(module, "_capture", lambda *args, **kwargs: next(responses))

    with pytest.raises(
        module.PreflightContractError,
        match="dynamo_probe_json_invalid",
    ):
        module._software_identity({})


def test_collection_uses_planner_cells_when_runtime_is_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load()
    monkeypatch.setattr(module, "validate_source_bundle_manifest", lambda *args: True)
    monkeypatch.setattr(module, "_runtime_env", lambda: {})
    monkeypatch.setattr(module, "_gpu_residue", lambda env: "")
    monkeypatch.setattr(module, "_process_residue", lambda env: "")
    monkeypatch.setattr(
        module,
        "_gpu_rows",
        lambda env: [f"{i}, GPU-{i}, NVIDIA H200, 143771, 570" for i in range(8)],
    )
    monkeypatch.setattr(
        module,
        "_software_identity",
        lambda env: _ready_attestation(module)["software"],
    )
    monkeypatch.setattr(
        module,
        "collect_model_identity",
        lambda **kwargs: _ready_attestation(module)["model_identity"],
    )
    monkeypatch.setattr(
        module,
        "_collect_plan_cells",
        lambda **kwargs: deepcopy(module.REQUIRED_PLAN_CELLS),
    )
    monkeypatch.setattr(module, "_capture", lambda *args, **kwargs: "worker")

    attestation = module.collect_worker_attestation(
        execution_root=tmp_path,
        execution_manifest=_execution_manifest(module),
        source_bundle_root=tmp_path,
        source_bundle_manifest=_source_manifest(module),
    )

    assert attestation["plan"]["cells"] == module.REQUIRED_PLAN_CELLS
    assert "error" not in attestation["plan"]


def test_collection_propagates_planner_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load()
    monkeypatch.setattr(module, "validate_source_bundle_manifest", lambda *args: True)
    monkeypatch.setattr(module, "_runtime_env", lambda: {})
    monkeypatch.setattr(module, "_gpu_residue", lambda env: "")
    monkeypatch.setattr(module, "_process_residue", lambda env: "")
    monkeypatch.setattr(
        module,
        "_gpu_rows",
        lambda env: [f"{i}, GPU-{i}, NVIDIA H200, 143771, 570" for i in range(8)],
    )
    monkeypatch.setattr(
        module,
        "_software_identity",
        lambda env: _ready_attestation(module)["software"],
    )
    monkeypatch.setattr(
        module,
        "collect_model_identity",
        lambda **kwargs: _ready_attestation(module)["model_identity"],
    )
    monkeypatch.setattr(
        module,
        "_collect_plan_cells",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("planner bug")),
    )

    with pytest.raises(RuntimeError, match="planner bug"):
        module.collect_worker_attestation(
            execution_root=tmp_path,
            execution_manifest=_execution_manifest(module),
            source_bundle_root=tmp_path,
            source_bundle_manifest=_source_manifest(module),
        )


def _semantic_configs() -> tuple[dict, dict]:
    aic = {
        "architectures": ["KimiK25ForConditionalGeneration"],
        "model_type": "kimi_k25",
        "quant_algo": "w4a16",
        "torch_dtype": "bfloat16",
        "first_k_dense_replace": 1,
        "hidden_size": 7168,
        "intermediate_size": 18432,
        "kv_lora_rank": 512,
        "max_position_embeddings": 262144,
        "moe_intermediate_size": 2048,
        "moe_layer_freq": 1,
        "n_routed_experts": 384,
        "n_shared_experts": 1,
        "num_attention_heads": 64,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 61,
        "num_key_value_heads": 64,
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "routed_scaling_factor": 2.827,
        "tie_word_embeddings": False,
        "use_cache": True,
        "v_head_dim": 128,
        "vocab_size": 163840,
    }
    text = {
        key: value
        for key, value in aic.items()
        if key not in {"architectures", "model_type", "quant_algo", "torch_dtype"}
    }
    text.update(
        {
            "architectures": ["DeepseekV3ForCausalLM"],
            "model_type": "kimi_k2",
            "dtype": "bfloat16",
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "config_groups": {
                    "group_0": {
                        "input_activations": None,
                        "output_activations": None,
                        "weights": {"num_bits": 4},
                    }
                },
            },
        }
    )
    checkpoint = {
        "architectures": ["KimiK25ForConditionalGeneration"],
        "model_type": "kimi_k25",
        "dtype": "bfloat16",
        "text_config": text,
        "wrapper_only": "makes the full config hash different",
    }
    return aic, checkpoint


def test_different_config_hashes_can_match_semantic_projection() -> None:
    module = _load()
    aic, checkpoint = _semantic_configs()

    identity = module.build_model_semantic_identity(aic, checkpoint)

    assert module.canonical_sha256(aic) != module.canonical_sha256(checkpoint)
    assert identity["status"] == "PASS"
    assert identity["aic_projection"] == identity["checkpoint_projection"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda checkpoint: checkpoint["text_config"].update(hidden_size=4096),
        lambda checkpoint: checkpoint.update(dtype="float16"),
        lambda checkpoint: checkpoint["text_config"]["quantization_config"].update(
            quant_method="gptq"
        ),
        lambda checkpoint: checkpoint["text_config"]["quantization_config"][
            "config_groups"
        ]["group_0"]["weights"].update(num_bits=8),
        lambda checkpoint: checkpoint["text_config"]["quantization_config"][
            "config_groups"
        ]["group_0"].update(input_activations={"num_bits": 8}),
    ],
)
def test_model_semantic_identity_rejects_core_dtype_and_quant_drift(
    mutation,
) -> None:
    module = _load()
    aic, checkpoint = _semantic_configs()
    mutation(checkpoint)

    with pytest.raises(module.PreflightContractError, match="model_semantic_identity"):
        module.build_model_semantic_identity(aic, checkpoint)


def test_runtime_absence_has_only_two_root_blockers() -> None:
    module = _load()
    attestation = _ready_attestation(module)
    attestation["software"].update(
        dynamo_module="",
        dynamo_source_sha256="",
        dynamo_methods=[],
        dynamo_artifact_fields=[],
        dynamo_fpm_fields=[],
        aiconfigurator_runtime_module="",
        aiconfigurator_runtime_files={},
    )
    attestation["plan"] = {
        "planner_commit": module.AIC_COMMIT,
        "status": "NOT_EVALUATED_RUNTIME_UNAVAILABLE",
        "cells": [],
        "points": "runtime-determined",
    }

    result = module.evaluate_worker_attestation(
        attestation,
        execution_manifest=_execution_manifest(module),
        source_bundle_manifest=_source_manifest(module),
    )

    assert result["blocking_reasons"] == [
        "dynamo_runtime_absent",
        "aiconfigurator_runtime_absent",
    ]
    assert result["model_identity_status"] == "PASS"
    assert result["not_evaluated"] == [
        "dynamo_source_methods_artifacts_fpm_fields",
        "planner_cells",
        "runtime_grid_coverage",
    ]
    assert result["grid_coverage"] == "NOT_EVALUATED_RUNTIME_DETERMINED"
