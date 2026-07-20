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
EXECUTION_MANIFEST_SCHEMA = "phase466_execution_manifest_v2"
MODEL_IDENTITY_FILES = ("config.json", "tokenizer_config.json", "tokenizer.json")
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
    "execution_tool_hash_mismatch",
    "model_identity",
    "model_revision",
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
) -> None:
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
            raise RuntimeError("model_revision_unresolved")
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
        "model_path": str(root),
        "snapshot_path": str(snapshot),
        "revision": revision,
        "files_sha256": file_hashes,
    }


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


def build_execution_manifest(
    contract: Any,
    *,
    source_commit: str,
    supervisor_path: Path,
    benchmark_path: Path,
    rank_analyzer_path: Path,
    model_identity: dict[str, Any] | None = None,
    prompt_identities: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("invalid_source_commit")
    paths = _tool_paths(
        contract,
        supervisor_path=supervisor_path,
        benchmark_path=benchmark_path,
        rank_analyzer_path=rank_analyzer_path,
    )
    tool_hashes: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"execution_tool_missing:{path}")
        tool_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if model_identity is None:
        model_identity = resolve_model_identity(contract.MODEL_PATH)
    if prompt_identities is None:
        prompt_identities = expected_prompt_cohort_sha256(
            contract, benchmark_path=benchmark_path
        )
    return {
        "schema": EXECUTION_MANIFEST_SCHEMA,
        "contract_schema": contract.SCHEMA,
        "source_commit": source_commit,
        "tool_sha256": tool_hashes,
        "vllm_source_sha256": dict(contract.SOURCE_FILES),
        "model_identity": model_identity,
        "prompt_cohort_sha256": prompt_identities,
    }


def execution_manifest_digest(manifest: dict[str, Any]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_execution_manifest(
    manifest: dict[str, Any],
    *,
    expected_digest: str,
    contract: Any,
    source_commit: str,
    supervisor_path: Path,
    benchmark_path: Path,
    rank_analyzer_path: Path,
) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
        raise RuntimeError("execution_manifest_digest_invalid")
    if execution_manifest_digest(manifest) != expected_digest:
        raise RuntimeError("execution_manifest_digest_mismatch")
    if manifest.get("schema") != EXECUTION_MANIFEST_SCHEMA:
        raise RuntimeError("execution_manifest_schema_mismatch")
    if manifest.get("contract_schema") != contract.SCHEMA:
        raise RuntimeError("execution_manifest_contract_schema_mismatch")
    if manifest.get("source_commit") != source_commit:
        raise RuntimeError("execution_manifest_source_commit_mismatch")
    if manifest.get("vllm_source_sha256") != contract.SOURCE_FILES:
        raise RuntimeError("execution_manifest_vllm_source_mismatch")
    model_identity = manifest.get("model_identity")
    if (
        not isinstance(model_identity, dict)
        or not MODEL_REVISION_RE.fullmatch(str(model_identity.get("revision", "")))
        or set(model_identity.get("files_sha256", {})) != set(MODEL_IDENTITY_FILES)
        or any(
            not re.fullmatch(r"[0-9a-f]{64}", str(value))
            for value in model_identity.get("files_sha256", {}).values()
        )
    ):
        raise RuntimeError("execution_manifest_model_identity_missing")
    prompt_identities = manifest.get("prompt_cohort_sha256")
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
    actual = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in _tool_paths(
            contract,
            supervisor_path=supervisor_path,
            benchmark_path=benchmark_path,
            rank_analyzer_path=rank_analyzer_path,
        ).items()
    }
    expected = manifest.get("tool_sha256")
    if not isinstance(expected, dict):
        raise RuntimeError("execution_manifest_tool_hashes_missing")
    for name in ("contract", "supervisor", "benchmark", "rank_analyzer"):
        if actual.get(name) != expected.get(name):
            raise RuntimeError(f"execution_tool_hash_mismatch:{name}")


def assert_model_identity(contract: Any, manifest: dict[str, Any]) -> dict[str, Any]:
    actual = resolve_model_identity(contract.MODEL_PATH)
    if actual != manifest.get("model_identity"):
        raise RuntimeError("model_identity_mismatch")
    return actual


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


def verify_coordinator_checkout(
    workdir: Path,
    source_commit: str,
    *,
    contract_path: Path,
    supervisor_path: Path,
    benchmark_path: Path,
    rank_analyzer_path: Path,
) -> None:
    head = _capture(
        ["git", "rev-parse", "HEAD"],
        cwd=workdir,
        env=dict(os.environ),
        allow_empty=False,
    )
    if head != source_commit:
        raise RuntimeError(f"coordinator_head_mismatch:{head}!={source_commit}")
    expected_paths = {
        "contract": workdir / "scripts" / "analyze_phase466_low_overhead_probe.py",
        "supervisor": workdir / "scripts" / "run_phase466_low_overhead_probe.py",
        "benchmark": workdir / "scripts" / "run_openai_fixed_shape_benchmark.py",
        "rank_analyzer": workdir / "scripts" / "analyze_phase466_rank_timing.py",
    }
    actual_paths = {
        "contract": contract_path,
        "supervisor": supervisor_path,
        "benchmark": benchmark_path,
        "rank_analyzer": rank_analyzer_path,
    }
    for name, expected in expected_paths.items():
        if actual_paths[name].resolve() != expected.resolve():
            raise RuntimeError(f"coordinator_tool_path_mismatch:{name}")
    paths = [str(path.relative_to(workdir)) for path in expected_paths.values()]
    _capture(
        ["git", "diff", "--quiet", "HEAD", "--", *paths],
        cwd=workdir,
        env=dict(os.environ),
    )
    _capture(
        ["git", "diff", "--cached", "--quiet", "HEAD", "--", *paths],
        cwd=workdir,
        env=dict(os.environ),
    )


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
        source_commit: str,
        execution_manifest: dict[str, Any],
        execution_manifest_sha256: str,
    ) -> None:
        if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
            raise ValueError("invalid_source_commit")
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
        validate_execution_manifest(
            self.execution_manifest,
            expected_digest=self.execution_manifest_sha256,
            contract=self.contract,
            source_commit=self.source_commit,
            supervisor_path=Path(__file__),
            benchmark_path=self.workdir / "scripts" / "run_openai_fixed_shape_benchmark.py",
            rank_analyzer_path=self.workdir / "scripts" / "analyze_phase466_rank_timing.py",
        )
        return _execution_tool_hashes(self.contract, self.workdir)

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
        self.artifact_root.mkdir(parents=True, exist_ok=False)
        self.owns_artifact_root = True
        self.status("preflight", "running")
        tooling_hashes = self.validate_tool_identity()
        model_identity = self.validate_model_identity()
        _assert_clean_worker(self.workdir, self.env, label="preflight")
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
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            cwd=self.workdir,
            env=self.env,
            allow_empty=False,
        )
        hashes = assert_source_identity(self.contract)
        assert_imported_vllm_source_paths(
            self.contract,
            cwd=self.workdir,
            env=self.env,
        )
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
        prompt_identity = self.execution_manifest["prompt_cohort_sha256"].get(
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
                "model_revision": run_end_model_identity["revision"],
                "model_files_sha256": run_end_model_identity["files_sha256"],
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
    source_commit: str,
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
        source_commit=source_commit,
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
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--expected-execution-manifest", type=Path)
    parser.add_argument("--expected-execution-manifest-sha256")
    parser.add_argument("--write-execution-manifest", type=Path)
    args = parser.parse_args()
    workdir = args.workdir.resolve()
    supervisor_path = Path(__file__).resolve()
    contract_path = supervisor_path.with_name("analyze_phase466_low_overhead_probe.py")
    benchmark_path = workdir / "scripts" / "run_openai_fixed_shape_benchmark.py"
    rank_analyzer_path = workdir / "scripts" / "analyze_phase466_rank_timing.py"
    contract = _load_contract(contract_path)
    if args.write_execution_manifest is not None:
        verify_coordinator_checkout(
            workdir,
            args.source_commit,
            contract_path=contract_path,
            supervisor_path=supervisor_path,
            benchmark_path=benchmark_path,
            rank_analyzer_path=rank_analyzer_path,
        )
        manifest = build_execution_manifest(
            contract,
            source_commit=args.source_commit,
            supervisor_path=supervisor_path,
            benchmark_path=benchmark_path,
            rank_analyzer_path=rank_analyzer_path,
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
        source_commit=args.source_commit,
        execution_manifest=manifest,
        execution_manifest_sha256=args.expected_execution_manifest_sha256,
    )


if __name__ == "__main__":
    raise SystemExit(main())
