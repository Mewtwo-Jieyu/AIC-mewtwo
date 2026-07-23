#!/usr/bin/env python3
"""Run the Phase469B no-model FPM runtime preflight."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any


SOURCE_BUNDLE_SCHEMA = "phase469b_fpm_source_bundle_v3"
EXECUTION_MANIFEST_SCHEMA = "phase469b_fpm_execution_manifest_v3"
WORKER_ATTESTATION_SCHEMA = "phase469b_fpm_worker_attestation_v3"
RESULT_SCHEMA = "phase469b_fpm_runtime_preflight_v3"
SUPERSEDED_RESULT_SCHEMA = "phase469b_fpm_runtime_preflight_v1"
HISTORICAL_RESULT_SCHEMA = "phase469b_fpm_runtime_preflight_v2"
AIC_COMMIT = "cf1b3cbbc00e0a822891abd29310d68a259e79ed"
DYNAMO_COMMIT = "41882ae9b07232eed4850fb1daf8c958abb2556a"
EXPECTED_MODEL_ID = "moonshotai/Kimi-K2.5"
DEFAULT_SSH_HOST = (
    "ws-faaf0de74ef9a14d-worker-b7mnp.zhaojieyu+root.ailab-sys.pod@h.pjlab.org.cn"
)
DEFAULT_MODEL_PATH = (
    "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/"
    "zskj-hub/models--moonshotai--Kimi-K2.5"
)
EXPECTED_AIC_MODEL_CONFIG_SHA256 = (
    "58944cf9cf456619768616f38acd75b3d7ccc26d8b6a64d9e312aba4c8f0b927"
)
EXPECTED_CHECKPOINT_CONFIG_SHA256 = (
    "8364a2eb7f427fd2117cc3d6729818dfd9957d33e22773d8e681a9a0ca97ddd3"
)
EXPECTED_FLAT_MODEL_FINGERPRINT = (
    "8847e7eafbad3f71d0006de6c305ea090bb1b9ee191c192fdb5cfc6c8742548b"
)
EXPECTED_QUANT_RUNTIME = "CompressedTensorsWNA16MarlinMoEMethod"
DEFERRED_QUANT_RUNTIME = "DEFERRED_TO_MODEL_CANARY"
EXPECTED_HOSTKEY_FINGERPRINT = (
    "SHA256:IVjkRgeMAWHa+gA/SVMZBy2yZBg+m04S1mBpzb2Ufrk"
)
PINNED_DYNAMO_SCHEDULER_SHA256 = (
    "443255313134f32331c6b40cfc90fe72064f207e089c8d95867f8963c22e20a4"
)
REQUIRED_DYNAMO_METHODS = (
    "schedule",
    "_bench_init",
    "_bench_build_grid",
    "_bench_step",
    "_bench_write_results",
)
REQUIRED_DYNAMO_ARTIFACT_FIELDS = (
    "run_id",
    "grid_digest",
    "iteration_groups",
)
REQUIRED_FPM_FIELDS = (
    "num_prefill_requests",
    "sum_prefill_tokens",
    "sum_prefill_kv_tokens",
    "num_decode_requests",
    "sum_decode_kv_tokens",
)
TP8_TOPOLOGY = "tp8pp1dp1moetp1ep8cp1"
DP2_TOPOLOGY = "tp4pp1dp2moetp1ep8cp1"
REQUIRED_PLAN_CELLS = [
    {"topology": topology, "workload_kind": workload_kind}
    for topology in (TP8_TOPOLOGY, DP2_TOPOLOGY)
    for workload_kind in ("prefill", "decode")
]
COORDINATOR_TOOL_PATHS = (
    "scripts/run_phase469b_fpm_runtime_preflight.py",
    "scripts/analyze_phase469b_fpm_runtime_preflight.py",
    "scripts/run_phase466_low_overhead_probe.py",
)
AIC_MODEL_CONFIG_RELATIVE_PATH = (
    "src/aiconfigurator/model_configs/moonshotai--Kimi-K2.5_config.json"
)
COORDINATOR_IDENTITY_PATHS = (AIC_MODEL_CONFIG_RELATIVE_PATH,)
SEMANTIC_WRAPPER_FIELDS = ("architectures", "model_type")
SEMANTIC_TEXT_FIELDS = (
    "first_k_dense_replace",
    "hidden_size",
    "intermediate_size",
    "kv_lora_rank",
    "max_position_embeddings",
    "moe_intermediate_size",
    "moe_layer_freq",
    "n_routed_experts",
    "n_shared_experts",
    "num_attention_heads",
    "num_experts_per_tok",
    "num_hidden_layers",
    "num_key_value_heads",
    "q_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "routed_scaling_factor",
    "tie_word_embeddings",
    "use_cache",
    "v_head_dim",
    "vocab_size",
)
PROCESS_PATTERN = (
    "vllm.entrypoints|ray::|raylet|gcs_server|VLLM::APIServer|"
    "VLLM::EngineCore|dynamo.frontend|dynamo.worker"
)
RUNTIME_PATH = (
    "/opt/py3/bin:/usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/local/sbin:"
    "/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)
RUNTIME_LD_LIBRARY_PATH = (
    "/nccl/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:"
    "/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu:"
    "/usr/local/cuda-12.9/compat:"
    "/usr/local/lib/python3.12/dist-packages/torch/lib"
)

PINNED_SOURCE_FILES = {
    "aic/collector/__init__.py": "9454980082f7d5aa78ed9eee35500c45cabbc0725736d89d14075455fb3426d0",
    "aic/collector/fpm_forward/__init__.py": "d84ab403a3c367f1f86ab476afe7a3b006f3fb53ee3159edfcad3f887b73a92f",
    "aic/collector/fpm_forward/capabilities.py": "83ce8d8036408b5f06a5132e86ac3eb9f38b74e4ee4c2ef6509b82d3e905383b",
    "aic/collector/fpm_forward/config.py": "5244c98b56a5496e5c7f5e5f863e0b1a45356ed2d4d70a5361473e890f80c38a",
    "aic/collector/fpm_forward/memory_admission.py": "f319e5bd6957fcf8e6b82387508ef7fde16d63bc2b4b15cf877069d42466641c",
    "aic/collector/fpm_forward/model_capability.py": "bb4f67bc94568ac32db158d6afe35e9e2c526a7b31f67640c6e2fd38c8209337",
    "aic/collector/fpm_forward/native_artifact.py": "13d52d344c76b2287d2a9404d353faf044e2885d9ca0cd830093f06980cb0113",
    "aic/collector/fpm_forward/planner.py": "6fbab3c49429f92ee11c3873ea18bada7926812f36485cb69a03b89d5f4f0ddf",
    "aic/collector/fpm_forward/topology.py": "99d6584e1d55c5d58f3c09d3041d980b8b14c802c37d2b23b3400e0ee0281f07",
    "aic/collector/fpm_forward/types.py": "d32e8b55794f447957f5c287e3e07b33d3616f03afcd3c06e34e5c862087e91f",
    "aic/collector/model_cases.py": "45c2b38a9b494629ae81cdaaf015911374a9961625e203bb96063bd8d9120f64",
    "aic/collector/cases/models/KimiK25ForConditionalGeneration_cases.yaml": "5e6619e0845a371e52aff408b7bb9302372a1458ebe0595055529a36661f3e34",
    "dynamo/components/src/dynamo/common/forward_pass_metrics.py": "523dfc4e7ccbc5a37dbcf03b331a41b628227ad8a5d9c64fda8abc96dd31b188",
    "dynamo/components/src/dynamo/vllm/instrumented_scheduler.py": PINNED_DYNAMO_SCHEDULER_SHA256,
}
PINNED_AIC_RUNTIME_FILES = {
    path.removeprefix("aic/"): digest
    for path, digest in PINNED_SOURCE_FILES.items()
    if path.startswith("aic/")
}


class PreflightContractError(RuntimeError):
    """Phase469B identity or runtime contract is invalid."""


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _with_digest(payload: dict) -> dict:
    result = dict(payload)
    result["sha256"] = canonical_sha256(payload)
    return result


def _validate_embedded_digest(payload: dict, *, label: str) -> str:
    expected = payload.get("sha256")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise PreflightContractError(f"{label}_sha256_invalid")
    content = dict(payload)
    content.pop("sha256")
    if canonical_sha256(content) != expected:
        raise PreflightContractError(f"{label}_sha256_mismatch")
    return expected


def build_source_bundle_manifest(
    bundle_root: Path,
    *,
    expected_files: dict[str, str] = PINNED_SOURCE_FILES,
) -> dict:
    files = []
    for relative, expected_hash in sorted(expected_files.items()):
        path = bundle_root / relative
        if not path.is_file():
            raise PreflightContractError(f"source_bundle_missing:{relative}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise PreflightContractError(f"source_bundle_hash:{relative}")
        files.append({"path": relative, "sha256": actual_hash})
    return _with_digest(
        {
            "schema": SOURCE_BUNDLE_SCHEMA,
            "aic_commit": AIC_COMMIT,
            "dynamo_commit": DYNAMO_COMMIT,
            "source_attestation_only": True,
            "runtime_installation": False,
            "files": files,
        }
    )


def validate_source_bundle_manifest_document(
    manifest: dict,
    *,
    expected_files: dict[str, str] = PINNED_SOURCE_FILES,
) -> bool:
    _validate_embedded_digest(manifest, label="source_bundle_manifest")
    if (
        manifest.get("schema") != SOURCE_BUNDLE_SCHEMA
        or manifest.get("aic_commit") != AIC_COMMIT
        or manifest.get("dynamo_commit") != DYNAMO_COMMIT
        or manifest.get("source_attestation_only") is not True
        or manifest.get("runtime_installation") is not False
    ):
        raise PreflightContractError("source_bundle_contract_mismatch")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise PreflightContractError("source_bundle_files_invalid")
    manifest_files = {
        item.get("path"): item.get("sha256")
        for item in files
        if isinstance(item, dict)
    }
    if manifest_files != expected_files:
        raise PreflightContractError("source_bundle_file_set_mismatch")
    seen = set()
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise PreflightContractError("source_bundle_file_entry_invalid")
        relative = item["path"]
        if relative in seen or not isinstance(relative, str):
            raise PreflightContractError("source_bundle_file_duplicate")
        seen.add(relative)
    return True


def validate_source_bundle_manifest(
    bundle_root: Path,
    manifest: dict,
    *,
    expected_files: dict[str, str] = PINNED_SOURCE_FILES,
) -> bool:
    validate_source_bundle_manifest_document(
        manifest,
        expected_files=expected_files,
    )
    for item in manifest["files"]:
        relative = item["path"]
        path = bundle_root / relative
        if not path.is_file():
            raise PreflightContractError(f"source_bundle_missing:{relative}")
        if sha256_file(path) != item["sha256"]:
            raise PreflightContractError(f"source_bundle_hash:{relative}")
    return True


def materialize_source_bundle(
    *, aic_source_root: Path, dynamo_source_root: Path, output_root: Path
) -> dict:
    if output_root.exists():
        raise PreflightContractError(f"source_bundle_exists:{output_root}")
    for relative in PINNED_SOURCE_FILES:
        repository, source_relative = relative.split("/", 1)
        source_root = aic_source_root if repository == "aic" else dynamo_source_root
        source = source_root / source_relative
        target = output_root / relative
        if not source.is_file():
            raise PreflightContractError(f"source_bundle_input_missing:{relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return build_source_bundle_manifest(output_root)


def build_coordinator_manifest(
    repo_root: Path,
    *,
    base_commit: str,
    source_bundle_manifest_sha256: str,
    ssh_host: str = DEFAULT_SSH_HOST,
    hostkey_fingerprint: str = EXPECTED_HOSTKEY_FINGERPRINT,
    model_path: str = DEFAULT_MODEL_PATH,
) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", base_commit):
        raise PreflightContractError("base_commit_invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", source_bundle_manifest_sha256):
        raise PreflightContractError("source_bundle_manifest_sha256_invalid")
    tools = {}
    for relative in COORDINATOR_TOOL_PATHS:
        path = repo_root / relative
        if not path.is_file():
            raise PreflightContractError(f"tool_missing:{relative}")
        tools[relative] = sha256_file(path)
    identity_files = {}
    for relative in COORDINATOR_IDENTITY_PATHS:
        path = repo_root / relative
        if not path.is_file():
            raise PreflightContractError(f"identity_file_missing:{relative}")
        identity_files[relative] = sha256_file(path)
    aic_config = _strict_json_object(
        (repo_root / AIC_MODEL_CONFIG_RELATIVE_PATH).read_bytes(),
        label="aic_model_config",
    )
    aic_config_sha256 = canonical_sha256(aic_config)
    if aic_config_sha256 != EXPECTED_AIC_MODEL_CONFIG_SHA256:
        raise PreflightContractError("aic_model_config_sha256_mismatch")
    aic_projection = _aic_model_semantic_projection(aic_config)
    payload = {
        "schema": EXECUTION_MANIFEST_SCHEMA,
        "base_commit": base_commit,
        "source_bundle_manifest_sha256": source_bundle_manifest_sha256,
        "tools": tools,
        "identity_files": identity_files,
        "target": {
            "ssh_host": ssh_host,
            "hostkey_fingerprint": hostkey_fingerprint,
            "model_path": model_path,
            "aic_model_config_sha256": aic_config_sha256,
            "checkpoint_config_sha256": EXPECTED_CHECKPOINT_CONFIG_SHA256,
            "flat_model_fingerprint": EXPECTED_FLAT_MODEL_FINGERPRINT,
            "flat_model_identity_scope": "same_flat_mirror_instance",
            "aic_model_semantic_projection": aic_projection,
        },
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
        "default_aic": "No-Go",
    }
    _validate_execution_target(payload["target"], aic_projection=aic_projection)
    return _with_digest(payload)


def _validate_execution_target(
    target: dict,
    *,
    aic_projection: dict,
) -> None:
    expected_keys = {
        "ssh_host",
        "hostkey_fingerprint",
        "model_path",
        "aic_model_config_sha256",
        "checkpoint_config_sha256",
        "flat_model_fingerprint",
        "flat_model_identity_scope",
        "aic_model_semantic_projection",
    }
    if set(target) != expected_keys:
        raise PreflightContractError("execution_manifest_target_fields")
    ssh_host = target["ssh_host"]
    if not isinstance(ssh_host, str) or not re.fullmatch(
        r"[A-Za-z0-9._+-]+@h\.pjlab\.org\.cn",
        ssh_host,
    ):
        raise PreflightContractError("execution_manifest_ssh_host")
    hostkey = target["hostkey_fingerprint"]
    if not isinstance(hostkey, str) or not re.fullmatch(
        r"SHA256:[A-Za-z0-9+/]{43}=?",
        hostkey,
    ):
        raise PreflightContractError("execution_manifest_hostkey")
    model_path = target["model_path"]
    if not isinstance(model_path, str):
        raise PreflightContractError("execution_manifest_model_path")
    normalized_model_path = PurePosixPath(model_path)
    if (
        not normalized_model_path.is_absolute()
        or str(normalized_model_path) != model_path
        or model_path == "/"
        or ".." in normalized_model_path.parts
    ):
        raise PreflightContractError("execution_manifest_model_path")
    expected_identity = {
        "aic_model_config_sha256": EXPECTED_AIC_MODEL_CONFIG_SHA256,
        "checkpoint_config_sha256": EXPECTED_CHECKPOINT_CONFIG_SHA256,
        "flat_model_fingerprint": EXPECTED_FLAT_MODEL_FINGERPRINT,
        "flat_model_identity_scope": "same_flat_mirror_instance",
        "aic_model_semantic_projection": aic_projection,
    }
    if any(target.get(key) != value for key, value in expected_identity.items()):
        raise PreflightContractError("execution_manifest_model_identity")


def validate_coordinator_manifest(repo_root: Path, manifest: dict) -> bool:
    _validate_embedded_digest(manifest, label="execution_manifest")
    if (
        manifest.get("schema") != EXECUTION_MANIFEST_SCHEMA
        or not re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("base_commit", "")))
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            str(manifest.get("source_bundle_manifest_sha256", "")),
        )
    ):
        raise PreflightContractError("execution_manifest_schema")
    tools = manifest.get("tools")
    if not isinstance(tools, dict) or set(tools) != set(COORDINATOR_TOOL_PATHS):
        raise PreflightContractError("execution_manifest_tools")
    for relative, expected in tools.items():
        path = repo_root / relative
        if not path.is_file():
            raise PreflightContractError(f"tool_missing:{relative}")
        if sha256_file(path) != expected:
            raise PreflightContractError(f"tool_hash:{relative}")
    identity_files = manifest.get("identity_files")
    if not isinstance(identity_files, dict) or set(identity_files) != set(
        COORDINATOR_IDENTITY_PATHS
    ):
        raise PreflightContractError("execution_manifest_identity_files")
    for relative, expected in identity_files.items():
        path = repo_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise PreflightContractError(f"identity_file_hash:{relative}")
    target = manifest.get("target")
    if not isinstance(target, dict):
        raise PreflightContractError("execution_manifest_target")
    _validate_execution_target(
        target,
        aic_projection=_aic_model_semantic_projection(
            _strict_json_object(
                (repo_root / AIC_MODEL_CONFIG_RELATIVE_PATH).read_bytes(),
                label="aic_model_config",
            )
        ),
    )
    if (
        manifest.get("diagnostic_only") is not True
        or manifest.get("valid_for_default") is not False
        or manifest.get("perf_database") is not False
        or manifest.get("default_aic") != "No-Go"
    ):
        raise PreflightContractError("execution_manifest_boundaries")
    return True


def validate_manifest_chain(
    execution_manifest: dict,
    source_bundle_manifest: dict,
) -> bool:
    if execution_manifest.get(
        "source_bundle_manifest_sha256"
    ) != source_bundle_manifest.get("sha256"):
        raise PreflightContractError("source_bundle_manifest_chain")
    return True


def _strict_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise PreflightContractError(f"json_duplicate_key:{label}:{key}")
            result[key] = value
        return result

    try:
        parsed = json.loads(raw, object_pairs_hook=reject_duplicates)
    except PreflightContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreflightContractError(f"json_invalid:{label}") from exc
    if not isinstance(parsed, dict):
        raise PreflightContractError(f"json_object_required:{label}")
    return parsed


def fingerprint_flat_model(model_path: Path) -> dict:
    phase466_path = Path(__file__).resolve().parent / "run_phase466_low_overhead_probe.py"
    if not phase466_path.is_file():
        raise PreflightContractError("phase466_fingerprint_tool_missing")
    spec = importlib.util.spec_from_file_location(
        "phase466_flat_model_fingerprint",
        phase466_path,
    )
    if spec is None or spec.loader is None:
        raise PreflightContractError("phase466_fingerprint_tool_invalid")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        identity = module.fingerprint_flat_model(model_path)
    except RuntimeError as exc:
        raise PreflightContractError(str(exc)) from exc
    if (
        identity.get("schema") != "phase466_flat_model_fingerprint_v1"
        or identity.get("identity_scope") != "same_flat_mirror_instance"
    ):
        raise PreflightContractError("flat_model_fingerprint_contract")
    return identity


def _quant_projection(config: dict, *, source: str) -> dict:
    if source == "aic":
        if config.get("quant_algo") != "w4a16":
            raise PreflightContractError("model_semantic_identity:quant_algo")
        return {
            "quant_method": "compressed-tensors",
            "weight_bits": 4,
            "activation_quantized": False,
        }
    quant = config.get("quantization_config")
    if not isinstance(quant, dict):
        raise PreflightContractError("model_semantic_identity:quant_method")
    groups = quant.get("config_groups")
    if quant.get("quant_method") != "compressed-tensors" or not isinstance(
        groups, dict
    ) or not groups:
        raise PreflightContractError("model_semantic_identity:quant_method")
    for group in groups.values():
        if not isinstance(group, dict):
            raise PreflightContractError("model_semantic_identity:quant_group")
        weights = group.get("weights")
        if (
            group.get("input_activations") is not None
            or group.get("output_activations") is not None
            or not isinstance(weights, dict)
            or weights.get("num_bits") != 4
        ):
            raise PreflightContractError("model_semantic_identity:quant_scheme")
    return {
        "quant_method": "compressed-tensors",
        "weight_bits": 4,
        "activation_quantized": False,
    }


def _aic_model_semantic_projection(config: dict) -> dict:
    projection = {
        field: config.get(field)
        for field in (*SEMANTIC_WRAPPER_FIELDS, *SEMANTIC_TEXT_FIELDS)
    }
    projection["dtype"] = config.get("torch_dtype")
    projection["quantization"] = _quant_projection(config, source="aic")
    if any(value is None for value in projection.values()):
        raise PreflightContractError("model_semantic_identity:aic_field_missing")
    return projection


def _checkpoint_model_semantic_projection(config: dict) -> dict:
    text = config.get("text_config")
    if not isinstance(text, dict):
        raise PreflightContractError("model_semantic_identity:text_config")
    projection = {
        field: config.get(field)
        for field in SEMANTIC_WRAPPER_FIELDS
    }
    projection.update({field: text.get(field) for field in SEMANTIC_TEXT_FIELDS})
    dtype = config.get("dtype")
    if dtype is None or text.get("dtype") != dtype:
        raise PreflightContractError("model_semantic_identity:dtype")
    projection["dtype"] = dtype
    projection["quantization"] = _quant_projection(text, source="checkpoint")
    if any(value is None for value in projection.values()):
        raise PreflightContractError(
            "model_semantic_identity:checkpoint_field_missing"
        )
    return projection


def build_model_semantic_identity(aic_config: dict, checkpoint_config: dict) -> dict:
    aic_projection = _aic_model_semantic_projection(aic_config)
    checkpoint_projection = _checkpoint_model_semantic_projection(checkpoint_config)
    if aic_projection != checkpoint_projection:
        mismatches = [
            field
            for field in sorted(aic_projection)
            if aic_projection.get(field) != checkpoint_projection.get(field)
        ]
        raise PreflightContractError(
            f"model_semantic_identity:{','.join(mismatches)}"
        )
    return {
        "status": "PASS",
        "aic_projection": aic_projection,
        "checkpoint_projection": checkpoint_projection,
    }


def collect_model_identity(
    *,
    model_path: Path,
    aic_config_path: Path,
) -> dict:
    root = model_path.resolve()
    checkpoint_config_path = root / "config.json"
    if not checkpoint_config_path.is_file() or not aic_config_path.is_file():
        raise PreflightContractError("model_identity_file_missing")
    aic_config = _strict_json_object(
        aic_config_path.read_bytes(),
        label="aic_model_config",
    )
    checkpoint_config = _strict_json_object(
        checkpoint_config_path.read_bytes(),
        label="checkpoint_config",
    )
    flat_identity = fingerprint_flat_model(root)
    return {
        "model_id": EXPECTED_MODEL_ID,
        "model_load_path": str(root),
        "aic_model_config_sha256": canonical_sha256(aic_config),
        "checkpoint_config_sha256": canonical_sha256(checkpoint_config),
        "flat_model_identity": flat_identity,
        "model_semantic_identity": build_model_semantic_identity(
            aic_config,
            checkpoint_config,
        ),
    }


def _runtime_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = RUNTIME_PATH
    env["LD_LIBRARY_PATH"] = RUNTIME_LD_LIBRARY_PATH
    return env


def _capture(
    argv: list[str],
    *,
    env: dict[str, str],
    accepted_returncodes: set[int] | None = None,
) -> str:
    accepted = accepted_returncodes or {0}
    completed = subprocess.run(
        argv,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode not in accepted:
        raise PreflightContractError(
            f"command_failed:{argv[0]}:{completed.returncode}:"
            f"{completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _gpu_rows(env: dict[str, str]) -> list[str]:
    raw = _capture(
        [
            "/usr/local/nvidia/bin/nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        env=env,
    )
    return raw.splitlines()


def _gpu_residue(env: dict[str, str]) -> str:
    return _capture(
        [
            "/usr/local/nvidia/bin/nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        env=env,
    )


def _process_residue(env: dict[str, str]) -> str:
    return _capture(
        ["pgrep", "-af", PROCESS_PATTERN],
        env=env,
        accepted_returncodes={0, 1},
    )


def _software_identity(env: dict[str, str]) -> dict:
    vllm_version = _capture(
        ["python3", "-c", "import vllm; print(vllm.__version__)"], env=env
    )
    dynamo_probe = "\n".join(
        [
            "import dataclasses",
            "import hashlib",
            "import importlib.util",
            "import inspect",
            "import json",
            "import pathlib",
            "try:",
            "    spec = importlib.util.find_spec('dynamo.vllm.instrumented_scheduler')",
            "except ModuleNotFoundError:",
            "    spec = None",
            "if spec is None:",
            "    print(json.dumps({'status': 'ABSENT'}))",
            "else:",
            "    from dynamo.vllm import instrumented_scheduler as scheduler",
            "    from dynamo.common.forward_pass_metrics import ForwardPassMetrics",
            "    cls = scheduler.InstrumentedScheduler",
            "    source_path = inspect.getsourcefile(scheduler)",
            "    source_bytes = pathlib.Path(source_path).read_bytes()",
            "    source = source_bytes.decode('utf-8')",
            "    print(json.dumps({",
            "        'status': 'PRESENT',",
            "        'module': scheduler.__name__,",
            "        'source_sha256': hashlib.sha256(source_bytes).hexdigest(),",
            f"        'methods': [name for name in {REQUIRED_DYNAMO_METHODS!r} if hasattr(cls, name)],",
            f"        'fields': [name for name in {REQUIRED_DYNAMO_ARTIFACT_FIELDS!r} if name in source],",
            "        'fpm_fields': [field.name for field in dataclasses.fields(ForwardPassMetrics)],",
            "    }, sort_keys=True))",
        ]
    )
    try:
        dynamo = json.loads(
            _capture(["python3", "-c", dynamo_probe], env=env)
        )
    except json.JSONDecodeError as exc:
        raise PreflightContractError("dynamo_probe_json_invalid") from exc
    if dynamo == {"status": "ABSENT"}:
        dynamo = {
            "module": "",
            "source_sha256": "",
            "methods": [],
            "fields": [],
            "fpm_fields": [],
        }
    elif not isinstance(dynamo, dict) or dynamo.pop("status", None) != "PRESENT":
        raise PreflightContractError("dynamo_probe_contract_invalid")
    aic_probe = "\n".join(
        [
            "import hashlib",
            "import importlib.util",
            "import inspect",
            "import json",
            "import pathlib",
            "try:",
            "    spec = importlib.util.find_spec('collector.fpm_forward.planner')",
            "except ModuleNotFoundError:",
            "    spec = None",
            "if spec is None:",
            "    print(json.dumps({'status': 'ABSENT'}))",
            "else:",
            "    from collector.fpm_forward import planner",
            "    root = pathlib.Path(inspect.getsourcefile(planner)).resolve().parents[2]",
            f"    expected = {PINNED_AIC_RUNTIME_FILES!r}",
            "    files = {",
            "        path: (hashlib.sha256((root / path).read_bytes()).hexdigest()",
            "               if (root / path).is_file() else '')",
            "        for path in expected",
            "    }",
            "    print(json.dumps({",
            "        'status': 'PRESENT',",
            "        'module': planner.__name__,",
            "        'files': files,",
            "    }, sort_keys=True))",
        ]
    )
    try:
        aic = json.loads(_capture(["python3", "-c", aic_probe], env=env))
    except json.JSONDecodeError as exc:
        raise PreflightContractError("aiconfigurator_probe_json_invalid") from exc
    if aic == {"status": "ABSENT"}:
        aic = {"module": "", "files": {}}
    elif not isinstance(aic, dict) or aic.pop("status", None) != "PRESENT":
        raise PreflightContractError("aiconfigurator_probe_contract_invalid")
    return {
        "vllm_version": vllm_version,
        "dynamo_module": dynamo["module"],
        "dynamo_source_sha256": dynamo["source_sha256"],
        "dynamo_methods": dynamo["methods"],
        "dynamo_artifact_fields": dynamo["fields"],
        "dynamo_fpm_fields": dynamo["fpm_fields"],
        "aiconfigurator_runtime_module": aic["module"],
        "aiconfigurator_runtime_files": aic["files"],
    }


def _collect_plan_cells(
    *, model_path: Path, env: dict[str, str]
) -> list[dict[str, str]]:
    script = "\n".join(
        [
            "import json",
            "from collector.fpm_forward import planner",
            "from collector.fpm_forward.config import FPMCollectionOptions",
            f"planner._git_revision = lambda: {AIC_COMMIT!r}",
            "options = FPMCollectionOptions(",
            "    max_gpus=8, gpu_counts=(8,),",
            "    parallel_presets=('tep', 'dep'), parallel_axes=(),",
            "    backend_axes=('auto',), weight_quantizations=(),",
            "    kv_cache_dtypes=('auto',), pp_sizes=(1,), cp_sizes=(1,),",
            "    max_prefill_isl=8192,",
            ")",
            "plan = planner.build_collection_plan(",
            "    backend='vllm', model_path='moonshotai/Kimi-K2.5',",
            "    system='h200_sxm',",
            "    selected_ops={'mla_context_module', 'mla_generation_module'},",
            "    options=options,",
            "    model_architecture='KimiK25ForConditionalGeneration',",
            "    has_model_cases=True,",
            f"    model_config_path={str(model_path / 'config.json')!r},",
            ")",
            "print(json.dumps([",
            "    {'topology': 'tp{tp}pp{pp}dp{dp}moetp{moe_tp}ep{moe_ep}cp{cp}'.format(**cell.topology.to_dict()),",
            "     'workload_kind': cell.workload_kind}",
            "    for cell in plan.cells",
            "], sort_keys=True))",
        ]
    )
    raw = _capture(["python3", "-c", script], env=env)
    try:
        cells = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PreflightContractError("planner_output_invalid") from exc
    if not isinstance(cells, list) or any(not isinstance(row, dict) for row in cells):
        raise PreflightContractError("planner_cells_invalid")
    return [
        {"topology": str(row.get("topology", "")), "workload_kind": str(row.get("workload_kind", ""))}
        for row in cells
    ]


def collect_worker_attestation(
    *,
    execution_root: Path,
    execution_manifest: dict,
    source_bundle_root: Path,
    source_bundle_manifest: dict,
) -> dict:
    validate_source_bundle_manifest(source_bundle_root, source_bundle_manifest)
    target = execution_manifest["target"]
    model_path = Path(target["model_path"])
    env = _runtime_env()
    gpu_before = _gpu_residue(env)
    process_before = _process_residue(env)
    gpu_rows = _gpu_rows(env)
    software = _software_identity(env)
    model_identity = collect_model_identity(
        model_path=model_path,
        aic_config_path=execution_root / AIC_MODEL_CONFIG_RELATIVE_PATH,
    )
    plan: dict[str, Any] = {
        "planner_commit": AIC_COMMIT,
        "cells": [],
        "points": "runtime-determined",
    }
    dynamo_present = (
        software["dynamo_module"] == "dynamo.vllm.instrumented_scheduler"
    )
    dynamo_compatible = (
        dynamo_present
        and software["dynamo_source_sha256"] == PINNED_DYNAMO_SCHEDULER_SHA256
        and set(software["dynamo_methods"]) == set(REQUIRED_DYNAMO_METHODS)
        and set(software["dynamo_artifact_fields"])
        == set(REQUIRED_DYNAMO_ARTIFACT_FIELDS)
        and set(software["dynamo_fpm_fields"]) == set(REQUIRED_FPM_FIELDS)
    )
    aic_present = (
        software["aiconfigurator_runtime_module"]
        == "collector.fpm_forward.planner"
    )
    aic_compatible = (
        aic_present
        and software["aiconfigurator_runtime_files"]
        == PINNED_AIC_RUNTIME_FILES
    )
    runtime_ready = (
        software["vllm_version"] == "0.19.0"
        and dynamo_compatible
        and aic_compatible
    )
    if runtime_ready:
        plan["cells"] = _collect_plan_cells(
            model_path=model_path,
            env=env,
        )
        plan["status"] = "EVALUATED"
    elif not dynamo_present or not aic_present:
        plan["status"] = "NOT_EVALUATED_RUNTIME_UNAVAILABLE"
    else:
        plan["status"] = "NOT_EVALUATED_RUNTIME_INCOMPATIBLE"
    gpu_after = _gpu_residue(env)
    process_after = _process_residue(env)
    return {
        "schema": WORKER_ATTESTATION_SCHEMA,
        "execution_manifest_sha256": execution_manifest["sha256"],
        "host": {
            "ssh_host": target["ssh_host"],
            "hostkey_fingerprint": target["hostkey_fingerprint"],
            "hostname": _capture(["hostname"], env=env),
        },
        "gpu": {
            "rows": gpu_rows,
            "residue_before": gpu_before,
            "residue_after": gpu_after,
        },
        "process_residue": {"before": process_before, "after": process_after},
        "software": software,
        "source_bundle": {
            "manifest_sha256": source_bundle_manifest["sha256"],
            "all_files_match": True,
            "source_attestation_only": True,
            "runtime_installation": False,
        },
        "model_identity": model_identity,
        "quant": {
            "expected_quant_runtime": EXPECTED_QUANT_RUNTIME,
            "actual_quant_runtime": DEFERRED_QUANT_RUNTIME,
        },
        "plan": plan,
        "grid_coverage": "NOT_EVALUATED_RUNTIME_DETERMINED",
        "model_loaded": False,
        "benchmark_executed": False,
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
        "default_aic": "No-Go",
    }


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def evaluate_worker_attestation(
    attestation: dict,
    *,
    execution_manifest: dict,
    source_bundle_manifest: dict,
) -> dict:
    execution_sha = execution_manifest.get("sha256")
    source_sha = source_bundle_manifest.get("sha256")
    target = _dict(execution_manifest.get("target"))
    reasons = []
    host = _dict(attestation.get("host"))
    gpu = _dict(attestation.get("gpu"))
    processes = _dict(attestation.get("process_residue"))
    software = _dict(attestation.get("software"))
    source = _dict(attestation.get("source_bundle"))
    model = _dict(attestation.get("model_identity"))
    quant = _dict(attestation.get("quant"))
    plan = _dict(attestation.get("plan"))
    if (
        attestation.get("schema") != WORKER_ATTESTATION_SCHEMA
        or attestation.get("execution_manifest_sha256")
        != execution_sha
    ):
        reasons.append("execution_manifest")
    if (
        host.get("ssh_host") != target.get("ssh_host")
        or host.get("hostkey_fingerprint") != target.get("hostkey_fingerprint")
    ):
        reasons.append("execution_target")
    rows = gpu.get("rows")
    if (
        not isinstance(rows, list)
        or len(rows) != 8
        or any("NVIDIA H200" not in str(row) for row in rows)
    ):
        reasons.append("gpu_identity")
    if gpu.get("residue_before") or gpu.get("residue_after"):
        reasons.append("gpu_residue")
    if processes.get("before") or processes.get("after"):
        reasons.append("process_residue")
    if software.get("vllm_version") != "0.19.0":
        reasons.append("vllm_version")
    dynamo_absent = (
        software.get("dynamo_module") != "dynamo.vllm.instrumented_scheduler"
    )
    if dynamo_absent:
        reasons.append("dynamo_runtime_absent")
        dynamo_incompatible = False
    else:
        dynamo_incompatible = (
            software.get("dynamo_source_sha256")
            != PINNED_DYNAMO_SCHEDULER_SHA256
            or set(software.get("dynamo_methods") or ())
            != set(REQUIRED_DYNAMO_METHODS)
            or set(software.get("dynamo_artifact_fields") or ())
            != set(REQUIRED_DYNAMO_ARTIFACT_FIELDS)
            or set(software.get("dynamo_fpm_fields") or ())
            != set(REQUIRED_FPM_FIELDS)
        )
        if dynamo_incompatible:
            reasons.append("dynamo_runtime_incompatible")
    aic_absent = (
        software.get("aiconfigurator_runtime_module")
        != "collector.fpm_forward.planner"
    )
    if aic_absent:
        reasons.append("aiconfigurator_runtime_absent")
        aic_incompatible = False
    else:
        aic_incompatible = (
            software.get("aiconfigurator_runtime_files")
            != PINNED_AIC_RUNTIME_FILES
        )
        if aic_incompatible:
            reasons.append("aiconfigurator_runtime_incompatible")
    if (
        source.get("manifest_sha256")
        != source_sha
        or source.get("all_files_match") is not True
        or source.get("source_attestation_only") is not True
        or source.get("runtime_installation") is not False
        or source_bundle_manifest.get("source_attestation_only") is not True
        or source_bundle_manifest.get("runtime_installation") is not False
    ):
        reasons.append("source_bundle")
    if (
        model.get("model_id") != EXPECTED_MODEL_ID
        or model.get("model_load_path") != target.get("model_path")
    ):
        reasons.append("model_target_identity")
    if model.get("aic_model_config_sha256") != target.get(
        "aic_model_config_sha256"
    ):
        reasons.append("aic_model_config_identity")
    if model.get("checkpoint_config_sha256") != target.get(
        "checkpoint_config_sha256"
    ):
        reasons.append("checkpoint_config_identity")
    flat = _dict(model.get("flat_model_identity"))
    if (
        flat.get("schema") != "phase466_flat_model_fingerprint_v1"
        or flat.get("identity_scope") != target.get("flat_model_identity_scope")
        or flat.get("fingerprint_sha256") != target.get("flat_model_fingerprint")
        or len(flat.get("shards") or ()) != 64
    ):
        reasons.append("flat_model_identity")
    semantic = _dict(model.get("model_semantic_identity"))
    if (
        semantic.get("status") != "PASS"
        or semantic.get("aic_projection")
        != target.get("aic_model_semantic_projection")
        or semantic.get("checkpoint_projection")
        != target.get("aic_model_semantic_projection")
    ):
        reasons.append("model_semantic_identity")
    if quant.get("expected_quant_runtime") != EXPECTED_QUANT_RUNTIME:
        reasons.append("expected_quant")
    if quant.get("actual_quant_runtime") != DEFERRED_QUANT_RUNTIME:
        reasons.append("actual_quant_deferred")
    if plan.get("points") != "runtime-determined":
        reasons.append("planner_state_contract")
    runtime_unavailable = dynamo_absent or aic_absent
    if runtime_unavailable:
        if (
            plan.get("status") != "NOT_EVALUATED_RUNTIME_UNAVAILABLE"
            or plan.get("cells") != []
        ):
            reasons.append("planner_state_contract")
    elif (
        dynamo_incompatible
        or aic_incompatible
        or software.get("vllm_version") != "0.19.0"
    ):
        if plan.get("status") != "NOT_EVALUATED_RUNTIME_INCOMPATIBLE":
            reasons.append("planner_state_contract")
    else:
        cells = plan.get("cells")
        normalized_cells = (
            sorted(
                cells,
                key=lambda row: (row.get("topology"), row.get("workload_kind")),
            )
            if isinstance(cells, list)
            and all(isinstance(row, dict) for row in cells)
            else []
        )
        if plan.get("status") != "EVALUATED" or normalized_cells != sorted(
            REQUIRED_PLAN_CELLS,
            key=lambda row: (row["topology"], row["workload_kind"]),
        ):
            reasons.append("planner_contract")
    if attestation.get("grid_coverage") != "NOT_EVALUATED_RUNTIME_DETERMINED":
        reasons.append("grid_coverage_contract")
    if attestation.get("model_loaded") is not False:
        reasons.append("model_load_boundary")
    if attestation.get("benchmark_executed") is not False:
        reasons.append("benchmark_boundary")
    if (
        attestation.get("diagnostic_only") is not True
        or attestation.get("valid_for_default") is not False
        or attestation.get("perf_database") is not False
        or attestation.get("default_aic") != "No-Go"
    ):
        reasons.append("result_boundaries")
    unique_reasons = list(dict.fromkeys(reasons))
    model_reason_names = {
        "model_target_identity",
        "aic_model_config_identity",
        "checkpoint_config_identity",
        "flat_model_identity",
        "model_semantic_identity",
    }
    return {
        "schema": RESULT_SCHEMA,
        "status": (
            "BLOCKED_RUNTIME_CONTRACT"
            if unique_reasons
            else "READY_FOR_TP8_CANARY"
        ),
        "blocking_reasons": unique_reasons,
        "execution_manifest_sha256": execution_sha,
        "source_bundle_manifest_sha256": source_sha,
        "attestation": attestation,
        "model_identity_status": (
            "FAIL"
            if any(reason in model_reason_names for reason in unique_reasons)
            else "PASS"
        ),
        "not_evaluated": [
            item
            for item in (
                "dynamo_source_methods_artifacts_fpm_fields"
                if dynamo_absent
                else None,
                "planner_cells" if runtime_unavailable else None,
                "runtime_grid_coverage",
            )
            if item is not None
        ],
        "grid_coverage": "NOT_EVALUATED_RUNTIME_DETERMINED",
        "model_loaded": False,
        "benchmark_executed": False,
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
        "default_aic": "No-Go",
    }


def run_worker_preflight(
    *,
    output_path: Path,
    execution_manifest: dict,
    source_bundle_root: Path,
    source_bundle_manifest: dict,
    model_path: Path,
    ssh_host: str,
    hostkey_fingerprint: str,
    execution_root: Path | None = None,
) -> int:
    resolved_execution_root = execution_root or Path(__file__).resolve().parent.parent
    validate_coordinator_manifest(
        resolved_execution_root,
        execution_manifest,
    )
    validate_source_bundle_manifest(source_bundle_root, source_bundle_manifest)
    validate_manifest_chain(execution_manifest, source_bundle_manifest)
    target = execution_manifest["target"]
    if (
        ssh_host != target["ssh_host"]
        or hostkey_fingerprint != target["hostkey_fingerprint"]
        or str(model_path) != target["model_path"]
    ):
        raise PreflightContractError("execution_target_mismatch")
    attestation = collect_worker_attestation(
        execution_root=resolved_execution_root,
        execution_manifest=execution_manifest,
        source_bundle_root=source_bundle_root,
        source_bundle_manifest=source_bundle_manifest,
    )
    result = evaluate_worker_attestation(
        attestation,
        execution_manifest=execution_manifest,
        source_bundle_manifest=source_bundle_manifest,
    )
    _write_json(output_path, result)
    return 0 if result["status"] == "READY_FOR_TP8_CANARY" else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--materialize-source-bundle", action="store_true")
    modes.add_argument("--write-execution-manifest", action="store_true")
    modes.add_argument("--worker-preflight", action="store_true")
    parser.add_argument("--aic-source-root", type=Path)
    parser.add_argument("--dynamo-source-root", type=Path)
    parser.add_argument("--source-bundle-root", type=Path)
    parser.add_argument("--source-bundle-manifest", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--base-commit")
    parser.add_argument("--execution-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-path", type=Path, default=Path(DEFAULT_MODEL_PATH))
    parser.add_argument("--ssh-host", default=DEFAULT_SSH_HOST)
    parser.add_argument(
        "--hostkey-fingerprint", default=EXPECTED_HOSTKEY_FINGERPRINT
    )
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.materialize_source_bundle:
        if not all(
            (
                args.aic_source_root,
                args.dynamo_source_root,
                args.source_bundle_root,
                args.source_bundle_manifest,
            )
        ):
            parser.error("source roots, bundle root, and manifest are required")
        manifest = materialize_source_bundle(
            aic_source_root=args.aic_source_root,
            dynamo_source_root=args.dynamo_source_root,
            output_root=args.source_bundle_root,
        )
        _write_json(args.source_bundle_manifest, manifest)
        print(manifest["sha256"])
        return 0
    if args.write_execution_manifest:
        if not all(
            (
                args.repo_root,
                args.base_commit,
                args.source_bundle_manifest,
                args.output,
            )
        ):
            parser.error("repo, base commit, source manifest, and output are required")
        source_manifest = json.loads(args.source_bundle_manifest.read_text())
        manifest = build_coordinator_manifest(
            args.repo_root,
            base_commit=args.base_commit,
            source_bundle_manifest_sha256=source_manifest["sha256"],
            ssh_host=args.ssh_host,
            hostkey_fingerprint=args.hostkey_fingerprint,
            model_path=str(args.model_path),
        )
        _write_json(args.output, manifest)
        print(manifest["sha256"])
        return 0
    if not all(
        (
            args.execution_manifest,
            args.source_bundle_root,
            args.source_bundle_manifest,
            args.output,
        )
    ):
        parser.error("execution/source manifests, bundle root, and output are required")
    return run_worker_preflight(
        output_path=args.output,
        execution_manifest=json.loads(args.execution_manifest.read_text()),
        source_bundle_root=args.source_bundle_root,
        source_bundle_manifest=json.loads(args.source_bundle_manifest.read_text()),
        model_path=args.model_path,
        ssh_host=args.ssh_host,
        hostkey_fingerprint=args.hostkey_fingerprint,
    )


if __name__ == "__main__":
    raise SystemExit(main())
