#!/usr/bin/env python3
"""Run the fail-closed Phase466 stock-vLLM overhead gate and formal collection."""

from __future__ import annotations

import argparse
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
    "workdir_dirty",
}


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


def failure_action(*, stage: str, cleanup: bool, integrity_failure: bool) -> str:
    if not cleanup or integrity_failure:
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
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run(argv: list[str], *, cwd: Path, env: dict[str, str], output: Path | None = None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True) if output else None
    if output is None:
        subprocess.run(argv, cwd=cwd, env=env, check=True)
        return
    with output.open("w", encoding="utf-8") as stream:
        subprocess.run(argv, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def _capture(argv: list[str], *, cwd: Path, env: dict[str, str], allow_empty: bool = True) -> str:
    completed = subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode not in (0, 1):
        raise RuntimeError(f"command_failed:{argv[0]}:{completed.returncode}")
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
    return _capture(["pgrep", "-af", PROCESS_PATTERN], cwd=cwd, env=env)


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


def _write_source_hashes(path: Path, hashes: dict[str, str]) -> None:
    path.write_text(
        "".join(f"{digest}  {source}\n" for source, digest in hashes.items()),
        encoding="utf-8",
    )


def _execution_tool_hashes(contract: Any, workdir: Path) -> dict[str, str]:
    paths = (
        Path(contract.__file__).resolve(),
        Path(__file__).resolve(),
        (workdir / "scripts" / "run_openai_fixed_shape_benchmark.py").resolve(),
    )
    hashes: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"execution_tool_missing:{path}")
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


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


def _stop_service(process: subprocess.Popen[Any], *, cwd: Path, env: dict[str, str]) -> bool:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)
    subprocess.run(["ray", "stop", "--force"], cwd=cwd, env=env, check=False, stdout=subprocess.DEVNULL)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if not _gpu_residue(cwd, env) and not _process_residue(cwd, env):
            return True
        time.sleep(3)
    return False


class Supervisor:
    def __init__(self, *, contract: Any, workdir: Path, artifact_root: Path) -> None:
        self.contract = contract
        self.workdir = workdir
        self.artifact_root = artifact_root
        self.env = _runtime_env()
        self.status_path = artifact_root / "status.json"
        self._status_lock = threading.Lock()
        self._last_status: dict[str, Any] | None = None
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None

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
        temporary = self.status_path.with_suffix(".json.tmp")
        _write_json(temporary, self._last_status)
        temporary.replace(self.status_path)

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
        self.status("preflight", "running")
        _assert_clean_worker(self.workdir, self.env, label="preflight")
        workdir_status = _capture(
            ["git", "status", "--porcelain"],
            cwd=self.workdir,
            env=self.env,
        )
        if workdir_status:
            raise RuntimeError(f"workdir_dirty:{workdir_status}")
        version = _capture(
            ["python3", "-c", "import vllm; print(vllm.__version__)"],
            cwd=self.workdir,
            env=self.env,
            allow_empty=False,
        )
        if version != "0.19.0":
            raise RuntimeError(f"vllm_version_mismatch:{version}")
        git_head = _capture(
            ["git", "rev-parse", "HEAD"],
            cwd=self.workdir,
            env=self.env,
            allow_empty=False,
        )
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
        hashes = _source_hashes(self.contract)
        tooling_hashes = _execution_tool_hashes(self.contract, self.workdir)
        _write_source_hashes(self.artifact_root / "source.sha256", hashes)
        _write_source_hashes(self.artifact_root / "tooling.sha256", tooling_hashes)
        (self.artifact_root / "gpu_compute_apps_before.txt").write_text("", encoding="utf-8")
        (self.artifact_root / "process_residue_before.txt").write_text("", encoding="utf-8")
        _write_json(
            self.artifact_root / "environment.json",
            {
                "git_head": git_head,
                "gpu": gpu,
                "vllm_version": version,
                "VLLM_ENABLE_CUDA_COMPATIBILITY": "1",
                "LD_LIBRARY_PATH": LD_LIBRARY_PATH,
            },
        )
        self.status("preflight", "passed")
        return hashes

    def execute_run(self, spec: dict[str, Any], hashes: dict[str, str]) -> Path:
        run_dir = self.artifact_root / str(spec["artifact_dir"])
        if run_dir.exists():
            raise RuntimeError(f"artifact_already_exists:{run_dir}")
        run_dir.mkdir(parents=True)
        (run_dir / "warmup").mkdir()
        serve_log = run_dir / "serve.log"
        port = int(spec["serve_argv"][spec["serve_argv"].index("--port") + 1])
        expected_ranks = list(range(int(spec["dp"])))
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
        cleanup = False
        try:
            _wait_for_service(process, port)
            warmup_argv = _materialize_argv(
                list(spec["warmup_benchmark_argv"]),
                relative_root=f"{spec['artifact_dir']}/warmup",
                absolute_root=run_dir / "warmup",
            )
            _run(warmup_argv, cwd=self.workdir, env=self.env, output=run_dir / "warmup" / "bench.log")
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
            _run(bench_argv, cwd=self.workdir, env=self.env, output=run_dir / "bench.log")
            (run_dir / "metrics_after.prom").write_text(_fetch_metrics(port), encoding="utf-8")
            if process.poll() is not None:
                raise RuntimeError("service_exited_during_benchmark")
        finally:
            cleanup = _stop_service(process, cwd=self.workdir, env=self.env)
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
        if not cleanup:
            raise RuntimeError("postflight_residue")
        meta = self.contract.expected_run_meta(spec)
        meta.update(
            {
                "vllm_version": "0.19.0",
                "measurement_start_after_iteration": {str(rank): value for rank, value in cutoffs.items()},
            }
        )
        _write_json(run_dir / "meta.json", meta)
        _write_source_hashes(run_dir / "source.sha256", hashes)
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


def run_all(contract: Any, *, workdir: Path, artifact_root: Path) -> int:
    plan = contract.build_run_plan()
    validated = validate_execution_plan(plan)
    supervisor = Supervisor(contract=contract, workdir=workdir, artifact_root=artifact_root)
    try:
        hashes = supervisor.preflight()
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
    finally:
        supervisor.stop_heartbeat()


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
            error = str(exc)
            cleanup = not _gpu_residue(workdir, supervisor.env) and not _process_residue(workdir, supervisor.env)
            failures.append({"run_id": run["id"], "stage": "overhead", "error": error, "cleanup": cleanup})
            action = failure_action(
                stage="overhead",
                cleanup=cleanup,
                integrity_failure=is_integrity_failure(error),
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
        _write_json(artifact_root / "phase466_result.json", final)
        _compress_logs(artifact_root)
        supervisor.status("complete", final["status"], gate_status=gate.get("status"))
        return 2

    formal_results: list[dict[str, Any]] = []
    formal_failures: list[dict[str, Any]] = []
    for run in validated["formal_runs"]:
        try:
            run_dir = supervisor.execute_run(run, hashes)
            result = contract.validate_formal_artifacts(run_dir, spec=run, gate=gate)
            formal_results.append(result)
            _write_json(
                run_dir / "probe_summary.json",
                {key: value for key, value in result.items() if key != "iteration_rows"},
            )
            contract._write_csv(run_dir / "iteration_rows.csv", result["iteration_rows"])
            contract._write_csv(run_dir / "rank_summary.csv", result["rank_summary"])
        except Exception as exc:
            error = str(exc)
            cleanup = not _gpu_residue(workdir, supervisor.env) and not _process_residue(workdir, supervisor.env)
            formal_failures.append({"run_id": run["id"], "error": error, "cleanup": cleanup})
            action = failure_action(
                stage="formal",
                cleanup=cleanup,
                integrity_failure=is_integrity_failure(error),
            )
            supervisor.status("formal", "run_failed", run_id=run["id"], error=error, action=action)
            if action == "STOP_ALL":
                break
    final = {
        "schema": contract.SCHEMA,
        "status": "PASS" if len(formal_results) == 3 and not formal_failures else "INCOMPLETE",
        "gate_status": gate["status"],
        "formal_scenarios": [result["scenario"] for result in formal_results],
        "formal_failures": formal_failures,
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
        "default_readiness": "No-Go",
    }
    _write_json(artifact_root / "phase466_result.json", final)
    _compress_logs(artifact_root)
    supervisor.status("complete", final["status"])
    return 0 if final["status"] == "PASS" else 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).with_name("analyze_phase466_low_overhead_probe.py"),
    )
    args = parser.parse_args()
    contract = _load_contract(args.contract.resolve())
    return run_all(
        contract,
        workdir=args.workdir.resolve(),
        artifact_root=args.artifact_root.resolve(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
