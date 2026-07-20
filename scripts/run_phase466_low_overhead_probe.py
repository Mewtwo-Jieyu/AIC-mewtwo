#!/usr/bin/env python3
"""Run the fail-closed Phase466 stock-vLLM overhead gate and formal collection."""

from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROCESS_PATTERN = (
    "vllm.entrypoints.cli.main serve|ray::|raylet|gcs_server|"
    "VLLM::APIServer|VLLM::EngineCore"
)
NODE_LOCK_PATH = Path("/tmp/phase466_low_overhead_probe.lock")
COORDINATOR_MANIFEST_SCHEMA = "phase466_coordinator_manifest_v1"
WORKER_ATTESTATION_SCHEMA = "phase466_worker_attestation_v1"
EXECUTION_MANIFEST_SCHEMA = "phase466_execution_manifest_v3"
FLAT_MODEL_FINGERPRINT_SCHEMA = "phase466_flat_model_fingerprint_v1"
SNAPSHOT_MODEL_IDENTITY_SCHEMA = "phase466_snapshot_model_identity_v1"
MAX_SAFETENSORS_HEADER_BYTES = 128 * 1024 * 1024
COORDINATOR_TOOL_RELATIVE_PATHS = {
    "contract": "scripts/analyze_phase466_low_overhead_probe.py",
    "supervisor": "scripts/run_phase466_low_overhead_probe.py",
    "benchmark": "scripts/run_openai_fixed_shape_benchmark.py",
    "rank_analyzer": "scripts/analyze_phase466_rank_timing.py",
}
MODEL_IDENTITY_FILES = (
    "config.json",
    "tokenizer_config.json",
    "tiktoken.model",
    "tokenization_kimi.py",
)
MODEL_REVISION_RE = re.compile(r"[0-9a-f]{40,64}")
LD_LIBRARY_PATH = (
    "/nccl/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:"
    "/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu:"
    "/usr/local/cuda-12.9/compat:"
    "/usr/local/lib/python3.12/dist-packages/torch/lib"
)
PATH = (
    "/opt/py3/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:"
    "/usr/sbin:/usr/bin:/sbin:/bin:/kubebrain:/usr/local/nvidia/bin"
)
ITERATION_ID_RE = re.compile(
    r"EngineCore(?:_DP(?P<rank>\d+))?.*?Iteration\((?P<iteration>\d+)\):"
)
INTEGRITY_ERRORS = {
    "missing_residue_check",
    "nonempty_gpu_residue",
    "nonempty_process_residue",
    "preflight_residue",
    "run_meta_mismatch",
    "vllm_version_mismatch",
    "vllm_source_hash_mismatch",
    "postflight_residue",
    "service_exited_during_benchmark",
    "formal_benchmark_mismatch",
    "formal_benchmark_invalid_throughput",
    "formal_rank_set_mismatch",
    "missing_probe_serve_log",
    "missing_probe_metrics",
    "missing_iteration_rank_rows",
    "execution_manifest",
    "coordinator_manifest",
    "worker_attestation",
    "execution_tool_hash_mismatch",
    "model_identity",
    "flat_model",
    "safetensors",
    "gpu_identity",
    "prompt_cohort_sha256",
}


class AbortRequested(RuntimeError):
    pass


class BenchmarkCommandError(RuntimeError):
    pass


def _load_contract(path: Path):
    spec = importlib.util.spec_from_file_location("phase466_probe_contract_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot_load_contract:{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_execution_plan(plan: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    runs = list(plan.get("runs", []))
    overhead = [run for run in runs if run.get("stage") == "overhead"]
    formal = [run for run in runs if run.get("stage") == "formal"]
    expected_overhead: list[tuple[str, str, int]] = []
    for pair_index in range(1, 7):
        pair_id = f"pair-{pair_index:02d}"
        modes = ("off", "on") if pair_index % 2 else ("on", "off")
        for order_index, mode in enumerate(modes, start=1):
            expected_overhead.append((pair_id, mode, order_index))
    actual_overhead = [
        (str(run.get("pair_id")), str(run.get("probe_mode")), int(run.get("order_index", 0)))
        for run in overhead
    ]
    if actual_overhead != expected_overhead:
        raise ValueError("execution_plan_overhead_order_mismatch")
    expected_formal = [
        "K2.5-tp4ep8dp2-32k3k",
        "K2.5-tp8ep8-8k2k-bt65536",
        "K2.5-tp4ep8dp2-8k2k-bt65536",
    ]
    if [str(run.get("scenario")) for run in formal] != expected_formal:
        raise ValueError("execution_plan_formal_order_mismatch")
    return {"overhead_runs": overhead, "formal_runs": formal}


def formal_runs_for_gate(
    gate: dict[str, Any],
    validated: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if gate.get("status") != "PASS":
        return []
    return validated["formal_runs"]


def failure_action(
    *,
    stage: str,
    cleanup: bool,
    integrity_failure: bool,
    benchmark_failure: bool = False,
) -> str:
    if not cleanup or integrity_failure or not benchmark_failure:
        return "STOP_ALL"
    if stage == "overhead":
        return "CONTINUE_GATE"
    if stage == "formal":
        return "CONTINUE_FORMAL"
    raise ValueError(f"unknown_execution_stage:{stage}")


def is_integrity_failure(error: str) -> bool:
    return any(marker in error for marker in INTEGRITY_ERRORS)


def last_iteration_by_rank(text: str, *, expected_ranks: list[int]) -> dict[int, int]:
    latest: dict[int, int] = {}
    for line in text.splitlines():
        match = ITERATION_ID_RE.search(line)
        if match:
            rank = int(match.group("rank") or 0)
            latest[rank] = max(latest.get(rank, -1), int(match.group("iteration")))
    if sorted(latest) != expected_ranks:
        raise ValueError(f"warmup_iteration_rank_mismatch:{sorted(latest)}!={expected_ranks}")
    return latest


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run(argv: list[str], *, cwd: Path, env: dict[str, str], output: Path | None = None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True) if output else None
    if output is None:
        subprocess.run(argv, cwd=cwd, env=env, check=True)
        return
    with output.open("w", encoding="utf-8") as stream:
        subprocess.run(argv, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def _capture(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    allow_empty: bool = True,
    accepted_returncodes: set[int] | None = None,
) -> str:
    accepted = {0} if accepted_returncodes is None else accepted_returncodes
    completed = subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode not in accepted:
        stderr = completed.stderr.strip().replace("\n", " ")
        raise RuntimeError(
            f"command_failed:{argv[0]}:{completed.returncode}:{stderr}"
        )
    text = completed.stdout.strip()
    if not allow_empty and not text:
        raise RuntimeError(f"command_empty:{argv[0]}")
    return text


def _runtime_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "PATH": PATH,
            "LD_LIBRARY_PATH": LD_LIBRARY_PATH,
            "VLLM_ENABLE_CUDA_COMPATIBILITY": "1",
            "VLLM_LOGGING_LEVEL": "INFO",
            "PYTHONUNBUFFERED": "1",
        }
    )
    env.pop("VLLM_NUM_GPU_BLOCKS_OVERRIDE", None)
    env.pop("NUM_GPU_BLOCKS_OVERRIDE", None)
    return env


def _gpu_residue(cwd: Path, env: dict[str, str]) -> str:
    return _capture(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        cwd=cwd,
        env=env,
    )


def _process_residue(cwd: Path, env: dict[str, str]) -> str:
    return _capture(
        ["pgrep", "-af", PROCESS_PATTERN],
        cwd=cwd,
        env=env,
        accepted_returncodes={0, 1},
    )


def _assert_clean_worker(cwd: Path, env: dict[str, str], *, label: str) -> None:
    gpu = _gpu_residue(cwd, env)
    processes = _process_residue(cwd, env)
    if gpu or processes:
        raise RuntimeError(f"{label}_residue:gpu={gpu!r}:process={processes!r}")


def _source_hashes(contract: Any) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for raw_path in contract.SOURCE_FILES:
        path = Path(raw_path)
        if not path.is_file():
            raise RuntimeError(f"vllm_source_missing:{path}")
        hashes[raw_path] = hashlib.sha256(path.read_bytes()).hexdigest()
    if hashes != contract.SOURCE_FILES:
        raise RuntimeError("vllm_source_hash_mismatch")
    return hashes


def assert_source_identity(contract: Any) -> dict[str, str]:
    hashes = _source_hashes(contract)
    if hashes != contract.SOURCE_FILES:
        raise RuntimeError("vllm_source_hash_mismatch")
    return hashes


def assert_imported_vllm_source_paths(
    contract: Any,
    *,
    cwd: Path,
    env: dict[str, str],
) -> list[str]:
    command = (
        "import json; "
        "from vllm.v1.engine import core; "
        "from vllm.v1.core.sched import scheduler; "
        "print(json.dumps([core.__file__, scheduler.__file__]))"
    )
    raw = _capture(
        ["python3", "-c", command],
        cwd=cwd,
        env=env,
        allow_empty=False,
    )
    paths = json.loads(raw)
    expected = list(contract.SOURCE_FILES)
    if paths != expected:
        raise RuntimeError(f"vllm_import_path_mismatch:{paths}!={expected}")
    return paths


def resolve_model_identity(model_path: str | Path) -> dict[str, Any]:
    root = Path(model_path)
    if not root.exists():
        raise RuntimeError(f"model_identity_path_missing:{root}")
    resolved = root.resolve()
    revision = ""
    snapshot = resolved
    if resolved.parent.name == "snapshots" and MODEL_REVISION_RE.fullmatch(
        resolved.name
    ):
        revision = resolved.name
    else:
        ref = root / "refs" / "main"
        if not ref.is_file():
            return fingerprint_flat_model(root)
        revision = ref.read_text(encoding="utf-8").strip()
        if not MODEL_REVISION_RE.fullmatch(revision):
            raise RuntimeError("model_revision_invalid")
        snapshot = (root / "snapshots" / revision).resolve()
    if not snapshot.is_dir():
        raise RuntimeError(f"model_snapshot_missing:{snapshot}")
    file_hashes: dict[str, str] = {}
    for filename in MODEL_IDENTITY_FILES:
        path = snapshot / filename
        if not path.is_file():
            raise RuntimeError(f"model_identity_file_missing:{filename}")
        file_hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "schema": SNAPSHOT_MODEL_IDENTITY_SCHEMA,
        "model_path": str(root),
        "snapshot_path": str(snapshot),
        "revision": revision,
        "files_sha256": file_hashes,
    }


def _strict_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise RuntimeError(f"json_duplicate_key:{label}:{key}")
            value[key] = item
        return value

    try:
        parsed = json.loads(raw, object_pairs_hook=reject_duplicates)
    except RuntimeError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"json_invalid:{label}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"json_object_required:{label}")
    return parsed


def _safetensors_header_identity(path: Path) -> tuple[int, int, str, set[str]]:
    before = path.stat()
    if before.st_size < 10:
        raise RuntimeError(f"safetensors_header_invalid:{path.name}")
    with path.open("rb") as stream:
        prefix = stream.read(8)
        header_size = int.from_bytes(prefix, "little", signed=False)
        if (
            header_size < 2
            or header_size > MAX_SAFETENSORS_HEADER_BYTES
            or 8 + header_size > before.st_size
        ):
            raise RuntimeError(f"safetensors_header_invalid:{path.name}")
        header = stream.read(header_size)
    try:
        parsed = _strict_json_object(header, label=f"safetensors_header:{path.name}")
    except RuntimeError as exc:
        raise RuntimeError(f"safetensors_header_invalid:{path.name}:{exc}") from exc
    after = path.stat()
    if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
        raise RuntimeError(f"safetensors_changed_during_fingerprint:{path.name}")
    tensor_names = set(parsed) - {"__metadata__"}
    return before.st_size, before.st_mtime_ns, hashlib.sha256(header).hexdigest(), tensor_names


def fingerprint_flat_model(model_path: str | Path) -> dict[str, Any]:
    root = Path(model_path).resolve()
    if not root.is_dir():
        raise RuntimeError(f"model_identity_path_missing:{root}")
    metadata_files: list[dict[str, str]] = []
    for name in (".msc", ".mv"):
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"flat_model_metadata_missing:{name}")
        metadata_files.append(
            {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    runtime_paths = sorted(
        path
        for path in root.iterdir()
        if path.is_file()
        and not path.is_symlink()
        and path.name not in {".msc", ".mv"}
        and path.suffix in {".json", ".py", ".jinja", ".model"}
    )
    index_path = root / "model.safetensors.index.json"
    if index_path not in runtime_paths:
        raise RuntimeError("safetensors_index_missing")
    runtime_files = [
        {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in runtime_paths
    ]
    index = _strict_json_object(
        index_path.read_bytes(), label="model.safetensors.index.json"
    )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("safetensors_index_weight_map_invalid")
    referenced: set[str] = set()
    tensors_by_shard: dict[str, set[str]] = {}
    for tensor_name, shard in weight_map.items():
        if (
            not tensor_name
            or tensor_name == "__metadata__"
            or not isinstance(shard, str)
            or Path(shard).name != shard
            or not shard.endswith(".safetensors")
        ):
            raise RuntimeError("safetensors_index_shard_path_invalid")
        referenced.add(shard)
        tensors_by_shard.setdefault(shard, set()).add(tensor_name)
    shard_paths = list(root.glob("*.safetensors"))
    invalid_shards = sorted(
        path.name for path in shard_paths if path.is_symlink() or not path.is_file()
    )
    if invalid_shards:
        raise RuntimeError(f"flat_model_shard_not_regular:{invalid_shards}")
    actual = {path.name for path in shard_paths}
    missing = sorted(referenced - actual)
    unexpected = sorted(actual - referenced)
    if missing:
        raise RuntimeError(f"flat_model_shards_missing:{missing}")
    if unexpected:
        raise RuntimeError(f"flat_model_shards_unexpected:{unexpected}")
    shards: list[dict[str, Any]] = []
    for name in sorted(referenced):
        path = root / name
        size, mtime_ns, header_sha256, tensor_names = _safetensors_header_identity(path)
        if tensor_names != tensors_by_shard[name]:
            raise RuntimeError(f"safetensors_index_header_mismatch:{name}")
        shards.append(
            {
                "path": name,
                "size": size,
                "mtime_ns": mtime_ns,
                "header_sha256": header_sha256,
            }
        )
    identity: dict[str, Any] = {
        "schema": FLAT_MODEL_FINGERPRINT_SCHEMA,
        "identity_scope": "same_flat_mirror_instance",
        "official_immutable_revision": False,
        "metadata_files": metadata_files,
        "runtime_files": runtime_files,
        "shards": shards,
    }
    identity["fingerprint_sha256"] = execution_manifest_digest(identity)
    return identity


def expected_prompt_cohort_sha256(
    contract: Any, *, benchmark_path: Path
) -> dict[str, dict[str, str]]:
    benchmark = _load_contract(benchmark_path)
    identities: dict[str, dict[str, str]] = {}
    cached: dict[tuple[int, int], str] = {}
    for run in contract.build_run_plan()["runs"]:
        per_run: dict[str, str] = {}
        for label, num_prompts in (
            ("warmup", int(run["warmup_num_prompts"])),
            ("measurement", int(run["num_prompts"])),
        ):
            key = (int(run["input_len"]), num_prompts)
            if key not in cached:
                variants = benchmark.build_prompt_variants(
                    contract.MODEL_PATH,
                    key[0],
                    max(1, key[1]),
                    "fixed",
                )
                cached[key] = benchmark.prompt_cohort_sha256(
                    variants, num_prompts=key[1]
                )
            per_run[label] = cached[key]
        identities[str(run["id"])] = per_run
    return identities


def _write_source_hashes(path: Path, hashes: dict[str, str]) -> None:
    path.write_text(
        "".join(f"{digest}  {source}\n" for source, digest in hashes.items()),
        encoding="utf-8",
    )


def _tool_paths(
    contract: Any,
    *,
    supervisor_path: Path,
    benchmark_path: Path,
    rank_analyzer_path: Path,
) -> dict[str, Path]:
    return {
        "contract": Path(contract.__file__).resolve(),
        "supervisor": supervisor_path.resolve(),
        "benchmark": benchmark_path.resolve(),
        "rank_analyzer": rank_analyzer_path.resolve(),
    }


def _execution_tool_hashes(
    contract: Any,
    workdir: Path,
    *,
    supervisor_path: Path | None = None,
) -> dict[str, str]:
    paths = _tool_paths(
        contract,
        supervisor_path=supervisor_path or Path(__file__),
        benchmark_path=workdir / "scripts" / "run_openai_fixed_shape_benchmark.py",
        rank_analyzer_path=workdir / "scripts" / "analyze_phase466_rank_timing.py",
    )
    hashes: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"execution_tool_missing:{path}")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def build_coordinator_manifest(contract: Any, *, workdir: Path) -> dict[str, Any]:
    workdir = workdir.resolve()
    head = _capture(
        ["git", "rev-parse", "HEAD"],
        cwd=workdir,
        env=dict(os.environ),
        allow_empty=False,
    )
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise RuntimeError("coordinator_head_invalid")
    dirty = _capture(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=workdir,
        env=dict(os.environ),
    )
    if dirty:
        raise RuntimeError("coordinator_checkout_dirty")
    expected_contract = (workdir / COORDINATOR_TOOL_RELATIVE_PATHS["contract"]).resolve()
    if Path(contract.__file__).resolve() != expected_contract:
        raise RuntimeError("coordinator_tool_path_mismatch:contract")
    tools: dict[str, dict[str, str]] = {}
    for name, relative in COORDINATOR_TOOL_RELATIVE_PATHS.items():
        path = workdir / relative
        if not path.is_file():
            raise RuntimeError(f"execution_tool_missing:{path}")
        tools[name] = {
            "path": relative,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return {
        "schema": COORDINATOR_MANIFEST_SCHEMA,
        "contract_schema": contract.SCHEMA,
        "source_commit": head,
        "tools": tools,
    }


def validate_coordinator_manifest_bytes(
    manifest: dict[str, Any],
    *,
    expected_digest: str,
    workdir: Path,
    contract_schema: str,
) -> dict[str, str]:
    if execution_manifest_digest(manifest) != expected_digest:
        raise RuntimeError("coordinator_manifest_digest_mismatch")
    if manifest.get("schema") != COORDINATOR_MANIFEST_SCHEMA:
        raise RuntimeError("coordinator_manifest_schema_mismatch")
    if manifest.get("contract_schema") != contract_schema:
        raise RuntimeError("coordinator_manifest_contract_schema_mismatch")
    if not re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("source_commit", ""))):
        raise RuntimeError("coordinator_manifest_source_commit_invalid")
    tools = manifest.get("tools")
    if not isinstance(tools, dict) or set(tools) != set(
        COORDINATOR_TOOL_RELATIVE_PATHS
    ):
        raise RuntimeError("coordinator_manifest_tools_invalid")
    actual: dict[str, str] = {}
    workdir = workdir.resolve()
    for name, expected_relative in COORDINATOR_TOOL_RELATIVE_PATHS.items():
        item = tools[name]
        if not isinstance(item, dict) or item.get("path") != expected_relative:
            raise RuntimeError(f"coordinator_manifest_tool_path_invalid:{name}")
        expected_hash = str(item.get("sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise RuntimeError(f"coordinator_manifest_tool_hash_invalid:{name}")
        path = (workdir / expected_relative).resolve()
        if path.parent != (workdir / "scripts").resolve() or not path.is_file():
            raise RuntimeError(f"execution_tool_missing:{path}")
        actual[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual[name] != expected_hash:
            raise RuntimeError(f"execution_tool_hash_mismatch:{name}")
    return actual


def execution_manifest_digest(manifest: dict[str, Any]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _validate_prompt_identities(
    prompt_identities: Any, *, contract: Any
) -> dict[str, dict[str, str]]:
    expected_run_ids = {
        str(run["id"]) for run in contract.build_run_plan().get("runs", [])
    }
    if not isinstance(prompt_identities, dict) or set(prompt_identities) != expected_run_ids:
        raise RuntimeError("execution_manifest_prompt_cohort_sha256_missing")
    for run_id, identity in prompt_identities.items():
        if not isinstance(identity, dict) or set(identity) != {"warmup", "measurement"}:
            raise RuntimeError(f"execution_manifest_prompt_cohort_sha256_missing:{run_id}")
        if any(
            not re.fullmatch(r"[0-9a-f]{64}", str(value))
            for value in identity.values()
        ):
            raise RuntimeError(f"execution_manifest_prompt_cohort_sha256_invalid:{run_id}")
    return prompt_identities


def _validate_model_identity_shape(model_identity: Any) -> dict[str, Any]:
    if not isinstance(model_identity, dict):
        raise RuntimeError("execution_manifest_model_identity_missing")
    if model_identity.get("schema") == SNAPSHOT_MODEL_IDENTITY_SCHEMA:
        if (
            not MODEL_REVISION_RE.fullmatch(str(model_identity.get("revision", "")))
            or set(model_identity.get("files_sha256", {})) != set(MODEL_IDENTITY_FILES)
            or any(
                not re.fullmatch(r"[0-9a-f]{64}", str(value))
                for value in model_identity.get("files_sha256", {}).values()
            )
        ):
            raise RuntimeError("execution_manifest_model_identity_missing")
        return model_identity
    if model_identity.get("schema") == FLAT_MODEL_FINGERPRINT_SCHEMA:
        expected = str(model_identity.get("fingerprint_sha256", ""))
        payload = dict(model_identity)
        payload.pop("fingerprint_sha256", None)
        metadata_files = model_identity.get("metadata_files")
        runtime_files = model_identity.get("runtime_files")
        shards = model_identity.get("shards")
        metadata_valid = (
            isinstance(metadata_files, list)
            and [item.get("path") for item in metadata_files if isinstance(item, dict)]
            == [".msc", ".mv"]
            and all(
                isinstance(item, dict)
                and set(item) == {"path", "sha256"}
                and re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", "")))
                for item in metadata_files
            )
        )
        runtime_valid = (
            isinstance(runtime_files, list)
            and bool(runtime_files)
            and all(
                isinstance(item, dict)
                and set(item) == {"path", "sha256"}
                and isinstance(item.get("path"), str)
                and Path(str(item.get("path", ""))).name == item.get("path")
                and re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", "")))
                for item in runtime_files
            )
            and [item["path"] for item in runtime_files]
            == sorted(item["path"] for item in runtime_files)
            and len({item["path"] for item in runtime_files}) == len(runtime_files)
            and "model.safetensors.index.json"
            in {item["path"] for item in runtime_files}
        )
        shard_valid = (
            isinstance(shards, list)
            and bool(shards)
            and all(
                isinstance(item, dict)
                and set(item) == {
                    "path",
                    "size",
                    "mtime_ns",
                    "header_sha256",
                }
                and Path(str(item.get("path", ""))).name == item.get("path")
                and str(item.get("path", "")).endswith(".safetensors")
                and isinstance(item.get("size"), int)
                and item["size"] >= 10
                and isinstance(item.get("mtime_ns"), int)
                and item["mtime_ns"] >= 0
                and re.fullmatch(
                    r"[0-9a-f]{64}", str(item.get("header_sha256", ""))
                )
                for item in shards
            )
            and [item["path"] for item in shards]
            == sorted(item["path"] for item in shards)
            and len({item["path"] for item in shards}) == len(shards)
        )
        if (
            not re.fullmatch(r"[0-9a-f]{64}", expected)
            or execution_manifest_digest(payload) != expected
            or model_identity.get("identity_scope") != "same_flat_mirror_instance"
            or model_identity.get("official_immutable_revision") is not False
            or not metadata_valid
            or not runtime_valid
            or not shard_valid
        ):
            raise RuntimeError("execution_manifest_model_identity_missing")
        return model_identity
    raise RuntimeError("execution_manifest_model_identity_missing")


def build_worker_attestation(
    contract: Any,
    *,
    coordinator_manifest: dict[str, Any],
    coordinator_manifest_sha256: str,
    workdir: Path,
    benchmark_path: Path,
) -> dict[str, Any]:
    tool_hashes = validate_coordinator_manifest_bytes(
        coordinator_manifest,
        expected_digest=coordinator_manifest_sha256,
        workdir=workdir,
        contract_schema=contract.SCHEMA,
    )
    env = _runtime_env()
    version = _capture(
        ["python3", "-c", "import vllm; print(vllm.__version__)"],
        cwd=workdir,
        env=env,
        allow_empty=False,
    )
    if version != "0.19.0":
        raise RuntimeError(f"vllm_version_mismatch:{version}")
    gpu = _capture(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        cwd=workdir,
        env=env,
        allow_empty=False,
    )
    source_hashes = assert_source_identity(contract)
    imported_paths = assert_imported_vllm_source_paths(
        contract, cwd=workdir, env=env
    )
    return {
        "schema": WORKER_ATTESTATION_SCHEMA,
        "contract_schema": contract.SCHEMA,
        "coordinator_manifest_sha256": coordinator_manifest_sha256,
        "tool_sha256": tool_hashes,
        "vllm_version": version,
        "vllm_source_sha256": source_hashes,
        "vllm_import_paths": imported_paths,
        "gpu_identity": {"rows": gpu.splitlines()},
        "model_identity": resolve_model_identity(contract.MODEL_PATH),
        "prompt_cohort_sha256": expected_prompt_cohort_sha256(
            contract, benchmark_path=benchmark_path
        ),
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
    }


def _validate_worker_attestation(
    attestation: Any,
    *,
    expected_digest: str,
    coordinator_manifest: dict[str, Any],
    coordinator_manifest_sha256: str,
    contract: Any,
) -> dict[str, Any]:
    if not isinstance(attestation, dict):
        raise RuntimeError("worker_attestation_missing")
    if execution_manifest_digest(attestation) != expected_digest:
        raise RuntimeError("worker_attestation_digest_mismatch")
    if attestation.get("schema") != WORKER_ATTESTATION_SCHEMA:
        raise RuntimeError("worker_attestation_schema_mismatch")
    if attestation.get("contract_schema") != contract.SCHEMA:
        raise RuntimeError("worker_attestation_contract_schema_mismatch")
    if attestation.get("coordinator_manifest_sha256") != coordinator_manifest_sha256:
        raise RuntimeError("worker_attestation_coordinator_mismatch")
    expected_tools = {
        name: item["sha256"]
        for name, item in coordinator_manifest["tools"].items()
    }
    if attestation.get("tool_sha256") != expected_tools:
        raise RuntimeError("worker_attestation_tool_hash_mismatch")
    if attestation.get("vllm_version") != "0.19.0":
        raise RuntimeError("worker_attestation_vllm_version_mismatch")
    if attestation.get("vllm_source_sha256") != contract.SOURCE_FILES:
        raise RuntimeError("worker_attestation_vllm_source_mismatch")
    if attestation.get("vllm_import_paths") != list(contract.SOURCE_FILES):
        raise RuntimeError("worker_attestation_vllm_import_path_mismatch")
    gpu_identity = attestation.get("gpu_identity")
    if (
        not isinstance(gpu_identity, dict)
        or not isinstance(gpu_identity.get("rows"), list)
        or not gpu_identity["rows"]
        or any(not isinstance(row, str) or not row for row in gpu_identity["rows"])
    ):
        raise RuntimeError("worker_attestation_gpu_identity_missing")
    _validate_model_identity_shape(attestation.get("model_identity"))
    _validate_prompt_identities(attestation.get("prompt_cohort_sha256"), contract=contract)
    for field in ("diagnostic_only", "valid_for_default", "perf_database"):
        expected = field == "diagnostic_only"
        if attestation.get(field) is not expected:
            raise RuntimeError(f"worker_attestation_boundary_invalid:{field}")
    return attestation


def compose_execution_manifest(
    contract: Any,
    *,
    coordinator_manifest: dict[str, Any],
    coordinator_manifest_sha256: str,
    worker_attestation: dict[str, Any],
    worker_attestation_sha256: str,
) -> dict[str, Any]:
    if execution_manifest_digest(coordinator_manifest) != coordinator_manifest_sha256:
        raise RuntimeError("coordinator_manifest_digest_mismatch")
    if coordinator_manifest.get("schema") != COORDINATOR_MANIFEST_SCHEMA:
        raise RuntimeError("coordinator_manifest_schema_mismatch")
    if coordinator_manifest.get("contract_schema") != contract.SCHEMA:
        raise RuntimeError("coordinator_manifest_contract_schema_mismatch")
    _validate_worker_attestation(
        worker_attestation,
        expected_digest=worker_attestation_sha256,
        coordinator_manifest=coordinator_manifest,
        coordinator_manifest_sha256=coordinator_manifest_sha256,
        contract=contract,
    )
    return {
        "schema": EXECUTION_MANIFEST_SCHEMA,
        "contract_schema": contract.SCHEMA,
        "coordinator_manifest_sha256": coordinator_manifest_sha256,
        "worker_attestation_sha256": worker_attestation_sha256,
        "coordinator_manifest": coordinator_manifest,
        "worker_attestation": worker_attestation,
    }


def validate_execution_manifest(
    manifest: dict[str, Any],
    *,
    expected_digest: str,
    contract: Any,
    workdir: Path,
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
        raise RuntimeError("execution_manifest_digest_invalid")
    if execution_manifest_digest(manifest) != expected_digest:
        raise RuntimeError("execution_manifest_digest_mismatch")
    if manifest.get("schema") != EXECUTION_MANIFEST_SCHEMA:
        raise RuntimeError("execution_manifest_schema_mismatch")
    if manifest.get("contract_schema") != contract.SCHEMA:
        raise RuntimeError("execution_manifest_contract_schema_mismatch")
    coordinator = manifest.get("coordinator_manifest")
    coordinator_digest = str(manifest.get("coordinator_manifest_sha256", ""))
    if not isinstance(coordinator, dict):
        raise RuntimeError("coordinator_manifest_missing")
    actual_tools = validate_coordinator_manifest_bytes(
        coordinator,
        expected_digest=coordinator_digest,
        workdir=workdir,
        contract_schema=contract.SCHEMA,
    )
    attestation = _validate_worker_attestation(
        manifest.get("worker_attestation"),
        expected_digest=str(manifest.get("worker_attestation_sha256", "")),
        coordinator_manifest=coordinator,
        coordinator_manifest_sha256=coordinator_digest,
        contract=contract,
    )
    if attestation["tool_sha256"] != actual_tools:
        raise RuntimeError("execution_manifest_tool_hash_mismatch")
    if assert_source_identity(contract) != attestation["vllm_source_sha256"]:
        raise RuntimeError("vllm_source_hash_mismatch")
    actual_model = resolve_model_identity(contract.MODEL_PATH)
    if actual_model != attestation["model_identity"]:
        raise RuntimeError("model_identity_mismatch")
    return attestation


def assert_model_identity(contract: Any, manifest: dict[str, Any]) -> dict[str, Any]:
    actual = resolve_model_identity(contract.MODEL_PATH)
    attestation = manifest.get("worker_attestation")
    expected = attestation.get("model_identity") if isinstance(attestation, dict) else None
    if actual != expected:
        raise RuntimeError("model_identity_mismatch")
    return actual


def model_identity_digest(identity: dict[str, Any]) -> str:
    if identity.get("schema") == FLAT_MODEL_FINGERPRINT_SCHEMA:
        return str(identity["fingerprint_sha256"])
    return execution_manifest_digest(identity)


def assert_benchmark_prompt_identity(
    path: Path, *, expected: str, label: str
) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise RuntimeError(f"prompt_cohort_sha256_expected_invalid:{label}")
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"prompt_cohort_sha256_missing:{label}") from exc
    actual = result.get("prompt_cohort_sha256") if isinstance(result, dict) else None
    if actual != expected:
        raise RuntimeError(f"prompt_cohort_sha256_mismatch:{label}")
    return expected


def acquire_node_lock(path: Path = NODE_LOCK_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        stream.close()
        raise RuntimeError(f"phase466_node_lock_busy:{path}") from exc
    return stream


def release_node_lock(stream: Any) -> None:
    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    stream.close()


def _wait_for_service(process: subprocess.Popen[Any], port: int, timeout_s: int = 2400) -> None:
    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("service_exited_before_ready")
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(5)
    raise RuntimeError("service_ready_timeout")


def _fetch_metrics(port: int) -> str:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"metrics_status:{response.status}")
        return response.read().decode("utf-8", errors="replace")


def _materialize_argv(argv: list[str], *, relative_root: str, absolute_root: Path) -> list[str]:
    prefix = f"{relative_root}/"
    return [
        str(absolute_root / value[len(prefix) :]) if value.startswith(prefix) else value
        for value in argv
    ]


def _process_group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_process_group(
    process: subprocess.Popen[Any],
    *,
    term_timeout_s: float = 120,
    kill_timeout_s: float = 30,
) -> tuple[bool, bool]:
    pgid = process.pid
    had_group = _process_group_alive(pgid)
    if had_group:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    process.poll()
    deadline = time.monotonic() + term_timeout_s
    while _process_group_alive(pgid) and time.monotonic() < deadline:
        process.poll()
        time.sleep(0.1)
    if _process_group_alive(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.poll()
    deadline = time.monotonic() + kill_timeout_s
    while _process_group_alive(pgid) and time.monotonic() < deadline:
        process.poll()
        time.sleep(0.1)
    process.poll()
    clean = not _process_group_alive(pgid)
    if process.poll() is None:
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            clean = False
    return had_group, clean


def _stop_service(process: subprocess.Popen[Any], *, cwd: Path, env: dict[str, str]) -> bool:
    unused_had_group, group_clean = _terminate_process_group(process)
    if not group_clean:
        return False
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if not _gpu_residue(cwd, env) and not _process_residue(cwd, env):
            return True
        time.sleep(3)
    return False


class Supervisor:
    def __init__(
        self,
        *,
        contract: Any,
        workdir: Path,
        artifact_root: Path,
        execution_manifest: dict[str, Any],
        execution_manifest_sha256: str,
    ) -> None:
        coordinator = execution_manifest.get("coordinator_manifest")
        source_commit = (
            coordinator.get("source_commit") if isinstance(coordinator, dict) else None
        )
        if not re.fullmatch(r"[0-9a-f]{40}", str(source_commit or "")):
            raise ValueError("execution_manifest_source_commit_invalid")
        self.contract = contract
        self.workdir = workdir
        self.artifact_root = artifact_root
        self.source_commit = source_commit
        self.execution_manifest = execution_manifest
        self.execution_manifest_sha256 = execution_manifest_sha256
        self.env = _runtime_env()
        self.status_path = artifact_root / "status.json"
        self._status_lock = threading.Lock()
        self._last_status: dict[str, Any] | None = None
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._abort_event = threading.Event()
        self._abort_reason = ""
        self._result_committed = False
        self._active_service: subprocess.Popen[Any] | None = None
        self._active_command: subprocess.Popen[Any] | None = None
        self._previous_signal_handlers: dict[int, Any] = {}
        self.owns_artifact_root = False

    @property
    def abort_requested(self) -> bool:
        return self._abort_event.is_set()

    def request_abort(self, signum: int) -> None:
        if self._result_committed:
            return
        self._abort_reason = f"signal_{signal.Signals(signum).name}"
        self._abort_event.set()
        command = self._active_command
        if command is not None and command.poll() is None:
            try:
                os.killpg(command.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        service = self._active_service
        if service is not None and service.poll() is None:
            try:
                os.killpg(service.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def raise_if_aborted(self) -> None:
        if self.abort_requested:
            raise AbortRequested(self._abort_reason)

    def validate_tool_identity(self) -> dict[str, str]:
        attestation = validate_execution_manifest(
            self.execution_manifest,
            expected_digest=self.execution_manifest_sha256,
            contract=self.contract,
            workdir=self.workdir,
        )
        return dict(attestation["tool_sha256"])

    def validate_model_identity(self) -> dict[str, Any]:
        return assert_model_identity(self.contract, self.execution_manifest)

    def commit_terminal_result(
        self,
        result: dict[str, Any],
        *,
        terminal_status: Any,
    ) -> str:
        while True:
            if self.abort_requested and result.get("status") != "ABORTED":
                result["status"] = "ABORTED"
                result["reason"] = self._abort_reason
            terminal_status(str(result["status"]))
            _write_json(self.artifact_root / "phase466_result.json", result)
            self._result_committed = True
            if not self.abort_requested or result.get("status") == "ABORTED":
                return str(result["status"])
            self._result_committed = False

    def install_signal_handlers(self) -> None:
        for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            self._previous_signal_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, lambda received, frame: self.request_abort(received))

    def restore_signal_handlers(self) -> None:
        for signum, handler in self._previous_signal_handlers.items():
            signal.signal(signum, handler)
        self._previous_signal_handlers.clear()

    def status(self, stage: str, state: str, **extra: Any) -> None:
        with self._status_lock:
            self._last_status = {
                "schema": self.contract.SCHEMA,
                "stage": stage,
                "state": state,
                "updated_at": time.time(),
                **extra,
            }
            self._write_status_locked()

    def _write_status_locked(self) -> None:
        if self._last_status is None:
            return
        _write_json(self.status_path, self._last_status)

    @property
    def heartbeat_alive(self) -> bool:
        return self._heartbeat_thread is not None and self._heartbeat_thread.is_alive()

    def start_heartbeat(self, *, interval_s: float = 30.0) -> None:
        if interval_s <= 0:
            raise ValueError("heartbeat_interval_must_be_positive")
        if self.heartbeat_alive:
            raise RuntimeError("heartbeat_already_running")
        self._heartbeat_stop.clear()

        def heartbeat() -> None:
            while not self._heartbeat_stop.wait(interval_s):
                with self._status_lock:
                    if self._last_status is not None:
                        self._last_status["updated_at"] = time.time()
                        self._write_status_locked()

        self._heartbeat_thread = threading.Thread(
            target=heartbeat,
            name="phase466-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=5)
            if self._heartbeat_thread.is_alive():
                raise RuntimeError("heartbeat_stop_timeout")
        self._heartbeat_thread = None

    def preflight(self) -> dict[str, str]:
        if self.artifact_root.exists():
            raise RuntimeError(f"artifact_already_exists:{self.artifact_root}")
        _assert_clean_worker(self.workdir, self.env, label="preflight")
        tooling_hashes = self.validate_tool_identity()
        model_identity = self.validate_model_identity()
        version = _capture(
            ["python3", "-c", "import vllm; print(vllm.__version__)"],
            cwd=self.workdir,
            env=self.env,
            allow_empty=False,
        )
        if version != "0.19.0":
            raise RuntimeError(f"vllm_version_mismatch:{version}")
        gpu = _capture(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            cwd=self.workdir,
            env=self.env,
            allow_empty=False,
        )
        attestation = self.execution_manifest["worker_attestation"]
        if gpu.splitlines() != attestation["gpu_identity"]["rows"]:
            raise RuntimeError("gpu_identity_mismatch")
        hashes = assert_source_identity(self.contract)
        imported_paths = assert_imported_vllm_source_paths(
            self.contract,
            cwd=self.workdir,
            env=self.env,
        )
        if imported_paths != attestation["vllm_import_paths"]:
            raise RuntimeError("vllm_import_path_mismatch")
        self.artifact_root.mkdir(parents=True, exist_ok=False)
        self.owns_artifact_root = True
        self.status("preflight", "running")
        _write_json(
            self.artifact_root / "expected_execution_manifest.json",
            self.execution_manifest,
        )
        _write_source_hashes(self.artifact_root / "source.sha256", hashes)
        _write_source_hashes(self.artifact_root / "tooling.sha256", tooling_hashes)
        (self.artifact_root / "gpu_compute_apps_before.txt").write_text("", encoding="utf-8")
        (self.artifact_root / "process_residue_before.txt").write_text("", encoding="utf-8")
        _write_json(
            self.artifact_root / "environment.json",
            {
                "source_commit": self.source_commit,
                "execution_manifest_sha256": self.execution_manifest_sha256,
                "gpu": gpu,
                "vllm_version": version,
                "model_identity": model_identity,
                "VLLM_ENABLE_CUDA_COMPATIBILITY": "1",
                "LD_LIBRARY_PATH": LD_LIBRARY_PATH,
            },
        )
        self.status("preflight", "passed")
        return hashes

    def run_command(self, argv: list[str], *, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(
                argv,
                cwd=self.workdir,
                env=self.env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self._active_command = process
            try:
                while True:
                    try:
                        returncode = process.wait(timeout=5)
                        break
                    except subprocess.TimeoutExpired:
                        if not self.abort_requested:
                            continue
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        returncode = process.wait(timeout=30)
                        break
            finally:
                self._active_command = None
        had_group, group_clean = _terminate_process_group(
            process,
            term_timeout_s=5,
            kill_timeout_s=30,
        )
        if not group_clean:
            raise RuntimeError("benchmark_process_group_cleanup_failed")
        if self.abort_requested:
            raise AbortRequested(self._abort_reason)
        if had_group:
            raise RuntimeError("benchmark_process_group_residue")
        if returncode != 0:
            raise BenchmarkCommandError(f"benchmark_command_failed:{returncode}")

    def execute_run(
        self,
        spec: dict[str, Any],
        hashes: dict[str, str],
        *,
        overhead_gate_sha256: str | None = None,
    ) -> Path:
        self.raise_if_aborted()
        run_start_tool_hashes = self.validate_tool_identity()
        run_start_model_identity = self.validate_model_identity()
        prompt_identity = self.execution_manifest["worker_attestation"][
            "prompt_cohort_sha256"
        ].get(
            str(spec["id"])
        )
        if not isinstance(prompt_identity, dict):
            raise RuntimeError(f"prompt_cohort_sha256_missing:{spec['id']}")
        run_dir = self.artifact_root / str(spec["artifact_dir"])
        if run_dir.exists():
            raise RuntimeError(f"artifact_already_exists:{run_dir}")
        run_dir.mkdir(parents=True)
        (run_dir / "warmup").mkdir()
        serve_log = run_dir / "serve.log"
        port = int(spec["serve_argv"][spec["serve_argv"].index("--port") + 1])
        expected_ranks = list(range(int(spec["dp"])))
        run_start_hashes = assert_source_identity(self.contract)
        if run_start_hashes != hashes:
            raise RuntimeError("vllm_source_hash_changed_before_run")
        self.status(spec["stage"], "starting", run_id=spec["id"])
        log_stream = serve_log.open("w", encoding="utf-8")
        process = subprocess.Popen(
            spec["serve_argv"],
            cwd=self.workdir,
            env=self.env,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._active_service = process
        cleanup = False
        run_end_hashes: dict[str, str] | None = None
        run_end_tool_hashes: dict[str, str] | None = None
        run_end_model_identity: dict[str, Any] | None = None
        try:
            _wait_for_service(process, port)
            warmup_argv = _materialize_argv(
                list(spec["warmup_benchmark_argv"]),
                relative_root=f"{spec['artifact_dir']}/warmup",
                absolute_root=run_dir / "warmup",
            )
            self.run_command(warmup_argv, output=run_dir / "warmup" / "bench.log")
            assert_benchmark_prompt_identity(
                run_dir / "warmup" / "bench_result.json",
                expected=str(prompt_identity.get("warmup", "")),
                label=f"{spec['id']}:warmup",
            )
            time.sleep(2)
            log_stream.flush()
            if spec["probe_mode"] == "on":
                cutoffs = last_iteration_by_rank(
                    serve_log.read_text(encoding="utf-8", errors="replace"),
                    expected_ranks=expected_ranks,
                )
            else:
                cutoffs = {rank: -1 for rank in expected_ranks}
            (run_dir / "metrics_before.prom").write_text(_fetch_metrics(port), encoding="utf-8")
            bench_argv = _materialize_argv(
                list(spec["benchmark_argv"]),
                relative_root=str(spec["artifact_dir"]),
                absolute_root=run_dir,
            )
            self.status(spec["stage"], "measuring", run_id=spec["id"])
            self.run_command(bench_argv, output=run_dir / "bench.log")
            assert_benchmark_prompt_identity(
                run_dir / "bench_result.json",
                expected=str(prompt_identity.get("measurement", "")),
                label=f"{spec['id']}:measurement",
            )
            (run_dir / "metrics_after.prom").write_text(_fetch_metrics(port), encoding="utf-8")
            time.sleep(2)
            log_stream.flush()
            if spec["probe_mode"] == "on":
                measurement_ends = last_iteration_by_rank(
                    serve_log.read_text(encoding="utf-8", errors="replace"),
                    expected_ranks=expected_ranks,
                )
            else:
                measurement_ends = {rank: -1 for rank in expected_ranks}
            if process.poll() is not None:
                raise RuntimeError("service_exited_during_benchmark")
        finally:
            cleanup = _stop_service(process, cwd=self.workdir, env=self.env)
            self._active_service = None
            log_stream.close()
            gpu_residue = _gpu_residue(self.workdir, self.env)
            process_residue = _process_residue(self.workdir, self.env)
            (run_dir / "gpu_compute_apps_after.txt").write_text(
                gpu_residue + ("\n" if gpu_residue else ""),
                encoding="utf-8",
            )
            (run_dir / "process_residue_after.txt").write_text(
                process_residue + ("\n" if process_residue else ""),
                encoding="utf-8",
            )
            run_end_hashes = assert_source_identity(self.contract)
            run_end_tool_hashes = self.validate_tool_identity()
            run_end_model_identity = self.validate_model_identity()
        if not cleanup:
            raise RuntimeError("postflight_residue")
        if run_end_hashes != run_start_hashes:
            raise RuntimeError("vllm_source_hash_changed_during_run")
        if run_end_tool_hashes != run_start_tool_hashes:
            raise RuntimeError("execution_tool_hash_changed_during_run")
        if run_end_model_identity != run_start_model_identity:
            raise RuntimeError("model_identity_changed_during_run")
        self.raise_if_aborted()
        meta = self.contract.expected_run_meta(spec)
        meta.update(
            {
                "vllm_version": "0.19.0",
                "execution_manifest_sha256": self.execution_manifest_sha256,
                "execution_tool_sha256": run_end_tool_hashes,
                "model_identity_schema": run_end_model_identity["schema"],
                "model_identity_sha256": model_identity_digest(run_end_model_identity),
                "warmup_prompt_cohort_sha256": prompt_identity["warmup"],
                "prompt_cohort_sha256": prompt_identity["measurement"],
                "measurement_start_after_iteration": {str(rank): value for rank, value in cutoffs.items()},
                "measurement_end_at_iteration": {
                    str(rank): value for rank, value in measurement_ends.items()
                },
            }
        )
        if overhead_gate_sha256 is not None:
            meta["overhead_gate_sha256"] = overhead_gate_sha256
        _write_json(run_dir / "meta.json", meta)
        _write_source_hashes(run_dir / "source.sha256", run_end_hashes)
        _write_source_hashes(run_dir / "tooling.sha256", run_end_tool_hashes)
        _write_json(run_dir / "prompt_identity.json", prompt_identity)
        self.status(spec["stage"], "run_complete", run_id=spec["id"])
        return run_dir


def _compress(path: Path) -> None:
    if not path.is_file() or path.suffix == ".gz":
        return
    with path.open("rb") as source, gzip.open(path.with_suffix(path.suffix + ".gz"), "wb") as target:
        target.write(source.read())
    path.unlink()


def _compress_logs(artifact_root: Path) -> None:
    for path in artifact_root.glob("**/serve.log"):
        _compress(path)


def finalize_result(
    *,
    supervisor: Supervisor,
    result: dict[str, Any],
    final_residue_check: Any,
    terminal_status: Any,
) -> str:
    _compress_logs(supervisor.artifact_root)
    final_residue_check()
    return supervisor.commit_terminal_result(result, terminal_status=terminal_status)


def finalize_failure_result(
    *,
    supervisor: Supervisor,
    result: dict[str, Any],
    terminal_status: Any,
) -> str:
    try:
        _compress_logs(supervisor.artifact_root)
    except Exception as exc:
        result["log_compression_error"] = str(exc)
    try:
        gpu = _gpu_residue(supervisor.workdir, supervisor.env)
        processes = _process_residue(supervisor.workdir, supervisor.env)
        result["cleanup"] = not gpu and not processes
        if gpu:
            result["gpu_residue"] = gpu
        if processes:
            result["process_residue"] = processes
    except Exception as exc:
        result["cleanup"] = False
        result["residue_check_error"] = str(exc)
    return supervisor.commit_terminal_result(result, terminal_status=terminal_status)


def gate_stop_result(gate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": gate.get("schema"),
        "status": "STOPPED_BEFORE_FORMAL",
        "gate_status": gate.get("status"),
        "formal_scenarios": [],
        "formal_failures": [],
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
        "default_readiness": "No-Go",
    }


def _assert_terminal_clean(supervisor: Supervisor, *, label: str) -> None:
    supervisor.raise_if_aborted()
    _assert_clean_worker(supervisor.workdir, supervisor.env, label=label)


def run_all(
    contract: Any,
    *,
    workdir: Path,
    artifact_root: Path,
    execution_manifest: dict[str, Any],
    execution_manifest_sha256: str,
    node_lock_path: Path = NODE_LOCK_PATH,
) -> int:
    plan = contract.build_run_plan()
    validated = validate_execution_plan(plan)
    supervisor = Supervisor(
        contract=contract,
        workdir=workdir,
        artifact_root=artifact_root,
        execution_manifest=execution_manifest,
        execution_manifest_sha256=execution_manifest_sha256,
    )
    node_lock = None
    try:
        node_lock = acquire_node_lock(node_lock_path)
        supervisor.install_signal_handlers()
        hashes = supervisor.preflight()
        supervisor.raise_if_aborted()
        supervisor.start_heartbeat()
        return _execute_plan(
            contract,
            workdir=workdir,
            artifact_root=artifact_root,
            plan=plan,
            validated=validated,
            supervisor=supervisor,
            hashes=hashes,
        )
    except AbortRequested as exc:
        result = {
            "schema": contract.SCHEMA,
            "status": "ABORTED",
            "reason": str(exc),
            "gate_status": "NOT_EVALUATED",
            "formal_scenarios": [],
            "diagnostic_only": True,
            "valid_for_default": False,
            "perf_database": False,
            "default_readiness": "No-Go",
            "execution_manifest_sha256": execution_manifest_sha256,
        }
        if supervisor.owns_artifact_root:
            finalize_failure_result(
                supervisor=supervisor,
                result=result,
                terminal_status=lambda status: supervisor.status("complete", status),
            )
        return 130
    except Exception as exc:
        result = {
            "schema": contract.SCHEMA,
            "status": "FAILED",
            "reason": str(exc),
            "gate_status": "NOT_EVALUATED",
            "formal_scenarios": [],
            "diagnostic_only": True,
            "valid_for_default": False,
            "perf_database": False,
            "default_readiness": "No-Go",
            "execution_manifest_sha256": execution_manifest_sha256,
        }
        if supervisor.owns_artifact_root:
            committed = finalize_failure_result(
                supervisor=supervisor,
                result=result,
                terminal_status=lambda status: supervisor.status("complete", status),
            )
            return 130 if committed == "ABORTED" else 4
        return 130 if supervisor.abort_requested else 4
    finally:
        if supervisor.heartbeat_alive:
            supervisor.stop_heartbeat()
        supervisor.restore_signal_handlers()
        if node_lock is not None:
            release_node_lock(node_lock)


def _execute_plan(
    contract: Any,
    *,
    workdir: Path,
    artifact_root: Path,
    plan: dict[str, Any],
    validated: dict[str, list[dict[str, Any]]],
    supervisor: Supervisor,
    hashes: dict[str, str],
) -> int:
    _write_json(artifact_root / "manifest.json", plan)
    run_dirs: dict[str, Path] = {}
    failures: list[dict[str, Any]] = []
    for run in validated["overhead_runs"]:
        try:
            run_dirs[run["id"]] = supervisor.execute_run(run, hashes)
        except Exception as exc:
            supervisor.raise_if_aborted()
            error = str(exc)
            cleanup = not _gpu_residue(workdir, supervisor.env) and not _process_residue(workdir, supervisor.env)
            failures.append({"run_id": run["id"], "stage": "overhead", "error": error, "cleanup": cleanup})
            action = failure_action(
                stage="overhead",
                cleanup=cleanup,
                integrity_failure=is_integrity_failure(error),
                benchmark_failure=isinstance(exc, BenchmarkCommandError),
            )
            supervisor.status("overhead", "run_failed", run_id=run["id"], error=error, action=action)
            if action == "STOP_ALL":
                break

    pair_results: list[dict[str, Any]] = []
    if not any(is_integrity_failure(item["error"]) for item in failures):
        runs_by_id = {run["id"]: run for run in validated["overhead_runs"]}
        for pair_index in range(1, 7):
            pair_id = f"pair-{pair_index:02d}"
            off_id = f"overhead-{pair_id}-off"
            on_id = f"overhead-{pair_id}-on"
            if off_id not in run_dirs or on_id not in run_dirs:
                continue
            try:
                result = contract.validate_overhead_pair_artifacts(
                    run_dirs[off_id],
                    run_dirs[on_id],
                    off_spec=runs_by_id[off_id],
                    on_spec=runs_by_id[on_id],
                )
                pair_results.append(result)
                _write_json(
                    artifact_root / "overhead" / pair_id / "pair_result.json",
                    result,
                )
            except Exception as exc:
                supervisor.raise_if_aborted()
                error = str(exc)
                failures.append(
                    {
                        "run_id": pair_id,
                        "stage": "overhead_validation",
                        "error": error,
                        "cleanup": True,
                    }
                )
                supervisor.status(
                    "overhead",
                    "pair_validation_failed",
                    pair_id=pair_id,
                    error=error,
                )
                break
    if len(pair_results) == 6 and not failures:
        gate = contract.evaluate_paired_overhead_gate(pair_results)
    else:
        gate = {
            "schema": contract.SCHEMA,
            "status": "INVALID",
            "pair_count": len(pair_results),
            "formal_collection_allowed": False,
            "failures": failures,
        }
    _write_json(artifact_root / "overhead" / "overhead_gate.json", gate)
    if gate.get("status") != "PASS":
        final = gate_stop_result(gate)
        final["execution_manifest_sha256"] = supervisor.execution_manifest_sha256
        committed = finalize_result(
            supervisor=supervisor,
            result=final,
            final_residue_check=lambda: _assert_terminal_clean(
                supervisor, label="gate_postflight"
            ),
            terminal_status=lambda status: supervisor.status(
                "complete", status, gate_status=gate.get("status")
            ),
        )
        return 130 if committed == "ABORTED" else 2

    formal_results: list[dict[str, Any]] = []
    formal_failures: list[dict[str, Any]] = []
    for run in formal_runs_for_gate(gate, validated):
        try:
            run_dir = supervisor.execute_run(
                run,
                hashes,
                overhead_gate_sha256=contract.overhead_gate_digest(gate),
            )
        except AbortRequested:
            raise
        except Exception as exc:
            supervisor.raise_if_aborted()
            error = str(exc)
            cleanup = not _gpu_residue(workdir, supervisor.env) and not _process_residue(workdir, supervisor.env)
            formal_failures.append({"run_id": run["id"], "error": error, "cleanup": cleanup})
            action = failure_action(
                stage="formal",
                cleanup=cleanup,
                integrity_failure=is_integrity_failure(error),
                benchmark_failure=isinstance(exc, BenchmarkCommandError),
            )
            supervisor.status("formal", "run_failed", run_id=run["id"], error=error, action=action)
            if action == "STOP_ALL":
                break
            continue
        try:
            result = contract.validate_formal_artifacts(run_dir, spec=run, gate=gate)
            contract._write_csv(run_dir / "iteration_rows.csv", result["iteration_rows"])
            contract._write_csv(run_dir / "rank_summary.csv", result["rank_summary"])
            result["iteration_rows_identity"] = contract.iteration_csv_identity(
                run_dir / "iteration_rows.csv"
            )
            formal_results.append(result)
            _write_json(
                run_dir / "probe_summary.json",
                {
                    key: value
                    for key, value in result.items()
                    if key not in {"iteration_rows", "rank_summary"}
                },
            )
        except Exception as exc:
            supervisor.raise_if_aborted()
            error = str(exc)
            cleanup = not _gpu_residue(workdir, supervisor.env) and not _process_residue(workdir, supervisor.env)
            formal_failures.append({"run_id": run["id"], "error": error, "cleanup": cleanup})
            supervisor.status(
                "formal",
                "artifact_validation_failed",
                run_id=run["id"],
                error=error,
                action="STOP_ALL",
            )
            break
    rank_timing_report: dict[str, Any] | None = None
    if len(formal_results) == 3 and not formal_failures:
        rank_timing_path = artifact_root / "rank_timing_report.json"
        try:
            _run(
                [
                    "python3",
                    str(workdir / "scripts" / "analyze_phase466_rank_timing.py"),
                    "--artifact-root",
                    str(artifact_root),
                    "--output-json",
                    str(rank_timing_path),
                ],
                cwd=workdir,
                env=supervisor.env,
            )
            rank_timing_report = json.loads(
                rank_timing_path.read_text(encoding="utf-8")
            )
            if rank_timing_report.get("status") != "DIAGNOSTIC_COMPLETE":
                raise RuntimeError("rank_timing_diagnostic_incomplete")
        except Exception as exc:
            formal_failures.append(
                {
                    "run_id": "rank-timing-analysis",
                    "error": str(exc),
                    "cleanup": True,
                }
            )
    diagnostic_complete = (
        len(formal_results) == 3
        and not formal_failures
        and rank_timing_report is not None
    )
    final = {
        "schema": contract.SCHEMA,
        "status": "DIAGNOSTIC_COMPLETE" if diagnostic_complete else "INCOMPLETE",
        "gate_status": gate["status"],
        "formal_scenarios": [result["scenario"] for result in formal_results],
        "formal_failures": formal_failures,
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
        "default_readiness": "No-Go",
        "execution_manifest_sha256": supervisor.execution_manifest_sha256,
        "overhead_gate_sha256": contract.overhead_gate_digest(gate),
        "formal_artifacts": {
            result["scenario"]: result["iteration_rows_identity"]
            for result in formal_results
        },
        "rank_timing_report": (
            "rank_timing_report.json" if rank_timing_report is not None else None
        ),
        "rank_timing_report_sha256": (
            hashlib.sha256(
                (artifact_root / "rank_timing_report.json").read_bytes()
            ).hexdigest()
            if rank_timing_report is not None
            else None
        ),
        "route_selection_executed": False,
        "simulator_rank_rows_generated": False,
    }
    committed = finalize_result(
        supervisor=supervisor,
        result=final,
        final_residue_check=lambda: _assert_terminal_clean(
            supervisor, label="final_postflight"
        ),
        terminal_status=lambda status: supervisor.status("complete", status),
    )
    if committed == "ABORTED":
        return 130
    return 0 if final["status"] == "DIAGNOSTIC_COMPLETE" else 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path)
    write_modes = parser.add_mutually_exclusive_group()
    write_modes.add_argument("--write-coordinator-manifest", type=Path)
    parser.add_argument("--coordinator-manifest", type=Path)
    parser.add_argument("--coordinator-manifest-sha256")
    write_modes.add_argument("--write-worker-attestation", type=Path)
    parser.add_argument("--worker-attestation", type=Path)
    parser.add_argument("--worker-attestation-sha256")
    parser.add_argument("--expected-execution-manifest", type=Path)
    parser.add_argument("--expected-execution-manifest-sha256")
    write_modes.add_argument("--write-execution-manifest", type=Path)
    args = parser.parse_args()
    workdir = args.workdir.resolve()
    supervisor_path = Path(__file__).resolve()
    contract_path = supervisor_path.with_name("analyze_phase466_low_overhead_probe.py")
    benchmark_path = workdir / "scripts" / "run_openai_fixed_shape_benchmark.py"
    contract = _load_contract(contract_path)
    if args.write_coordinator_manifest is not None:
        manifest = build_coordinator_manifest(contract, workdir=workdir)
        _write_json(args.write_coordinator_manifest, manifest)
        print(execution_manifest_digest(manifest))
        return 0
    if args.write_worker_attestation is not None:
        if args.coordinator_manifest is None or args.coordinator_manifest_sha256 is None:
            parser.error(
                "--coordinator-manifest and --coordinator-manifest-sha256 are required"
            )
        coordinator = json.loads(
            args.coordinator_manifest.read_text(encoding="utf-8")
        )
        if not isinstance(coordinator, dict):
            raise ValueError("coordinator_manifest_must_be_object")
        _assert_clean_worker(workdir, _runtime_env(), label="identity_preflight_before")
        attestation = build_worker_attestation(
            contract,
            coordinator_manifest=coordinator,
            coordinator_manifest_sha256=args.coordinator_manifest_sha256,
            workdir=workdir,
            benchmark_path=benchmark_path,
        )
        _assert_clean_worker(workdir, _runtime_env(), label="identity_preflight_after")
        _write_json(args.write_worker_attestation, attestation)
        print(execution_manifest_digest(attestation))
        return 0
    if args.write_execution_manifest is not None:
        if (
            args.coordinator_manifest is None
            or args.coordinator_manifest_sha256 is None
            or args.worker_attestation is None
            or args.worker_attestation_sha256 is None
        ):
            parser.error("coordinator and worker attestation inputs are required")
        coordinator = json.loads(
            args.coordinator_manifest.read_text(encoding="utf-8")
        )
        attestation = json.loads(
            args.worker_attestation.read_text(encoding="utf-8")
        )
        if not isinstance(coordinator, dict) or not isinstance(attestation, dict):
            raise ValueError("manifest_inputs_must_be_objects")
        manifest = compose_execution_manifest(
            contract,
            coordinator_manifest=coordinator,
            coordinator_manifest_sha256=args.coordinator_manifest_sha256,
            worker_attestation=attestation,
            worker_attestation_sha256=args.worker_attestation_sha256,
        )
        _write_json(args.write_execution_manifest, manifest)
        print(execution_manifest_digest(manifest))
        return 0
    if args.artifact_root is None:
        parser.error("--artifact-root is required for execution")
    if args.expected_execution_manifest is None:
        parser.error("--expected-execution-manifest is required for execution")
    if args.expected_execution_manifest_sha256 is None:
        parser.error("--expected-execution-manifest-sha256 is required for execution")
    manifest = json.loads(args.expected_execution_manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("expected_execution_manifest_must_be_object")
    return run_all(
        contract,
        workdir=workdir,
        artifact_root=args.artifact_root.resolve(),
        execution_manifest=manifest,
        execution_manifest_sha256=args.expected_execution_manifest_sha256,
    )


if __name__ == "__main__":
    raise SystemExit(main())
