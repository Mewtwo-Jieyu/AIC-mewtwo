from __future__ import annotations

import importlib.util
import json
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "run_phase466_low_overhead_probe.py"
CONTRACT_PATH = REPO_ROOT / "scripts" / "analyze_phase466_low_overhead_probe.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_validate_execution_plan_locks_gate_and_formal_order() -> None:
    contract = _load(CONTRACT_PATH, "phase466_contract_for_runner")
    runner = _load(MODULE_PATH, "run_phase466_low_overhead_probe")

    validated = runner.validate_execution_plan(contract.build_run_plan())

    assert len(validated["overhead_runs"]) == 12
    assert [run["probe_mode"] for run in validated["overhead_runs"][:4]] == [
        "off",
        "on",
        "on",
        "off",
    ]
    assert [run["scenario"] for run in validated["formal_runs"]] == list(
        contract.FORMAL_SCENARIOS
    )


def test_validate_execution_plan_rejects_missing_preregistered_run() -> None:
    contract = _load(CONTRACT_PATH, "phase466_contract_missing_run")
    runner = _load(MODULE_PATH, "run_phase466_low_overhead_probe_missing_run")
    plan = contract.build_run_plan()
    plan["runs"].pop(0)

    with pytest.raises(ValueError, match="execution_plan_overhead_order_mismatch"):
        runner.validate_execution_plan(plan)


def test_formal_runs_require_passed_gate_and_keep_exact_scenario_set() -> None:
    contract = _load(CONTRACT_PATH, "phase466_contract_formal_gate")
    runner = _load(MODULE_PATH, "run_phase466_low_overhead_probe_formal_gate")
    validated = runner.validate_execution_plan(contract.build_run_plan())

    assert runner.formal_runs_for_gate(
        {"status": "INCONCLUSIVE"}, validated
    ) == []
    assert [
        run["scenario"]
        for run in runner.formal_runs_for_gate({"status": "PASS"}, validated)
    ] == list(contract.FORMAL_SCENARIOS)


def test_failure_action_is_fail_closed() -> None:
    runner = _load(MODULE_PATH, "run_phase466_low_overhead_probe_failure_action")

    assert runner.failure_action(
        stage="overhead",
        cleanup=True,
        integrity_failure=False,
        benchmark_failure=True,
    ) == "CONTINUE_GATE"
    assert runner.failure_action(
        stage="overhead",
        cleanup=True,
        integrity_failure=False,
        benchmark_failure=False,
    ) == "STOP_ALL"
    assert runner.failure_action(stage="overhead", cleanup=False, integrity_failure=False) == "STOP_ALL"
    assert runner.failure_action(stage="overhead", cleanup=True, integrity_failure=True) == "STOP_ALL"
    assert runner.failure_action(
        stage="formal",
        cleanup=True,
        integrity_failure=False,
        benchmark_failure=True,
    ) == "CONTINUE_FORMAL"
    assert runner.failure_action(
        stage="formal",
        cleanup=True,
        integrity_failure=False,
        benchmark_failure=False,
    ) == "STOP_ALL"
    assert runner.failure_action(stage="formal", cleanup=False, integrity_failure=False) == "STOP_ALL"


def test_integrity_error_detection_matches_nested_validation_errors() -> None:
    runner = _load(MODULE_PATH, "run_phase466_low_overhead_probe_integrity")

    assert runner.is_integrity_failure("validation:vllm_source_hash_mismatch")
    assert runner.is_integrity_failure("formal_benchmark_mismatch:ok_requests")
    assert runner.is_integrity_failure("missing_probe_metrics:/tmp/run")
    assert runner.is_integrity_failure("nonempty_gpu_residue:/tmp/run")
    assert not runner.is_integrity_failure("benchmark_command_failed:1")


def test_capture_only_accepts_explicit_return_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load(MODULE_PATH, "run_phase466_low_overhead_probe_capture")

    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="nvidia failed",
        ),
    )

    with pytest.raises(RuntimeError, match="command_failed:nvidia-smi:1:nvidia failed"):
        runner._capture(["nvidia-smi"], cwd=tmp_path, env={})
    assert runner._capture(
        ["pgrep"], cwd=tmp_path, env={}, accepted_returncodes={0, 1}
    ) == ""


def test_node_lock_is_exclusive_and_released(tmp_path: Path) -> None:
    runner = _load(MODULE_PATH, "run_phase466_low_overhead_probe_lock")
    lock_path = tmp_path / "phase466.lock"

    first = runner.acquire_node_lock(lock_path)
    with pytest.raises(RuntimeError, match="phase466_node_lock_busy"):
        runner.acquire_node_lock(lock_path)
    runner.release_node_lock(first)
    second = runner.acquire_node_lock(lock_path)
    runner.release_node_lock(second)


def test_execution_manifest_binds_commit_tools_and_source_hashes(tmp_path: Path) -> None:
    runner = _load(MODULE_PATH, "run_phase466_low_overhead_probe_manifest")
    contract_path = tmp_path / "contract.py"
    supervisor_path = tmp_path / "supervisor.py"
    benchmark_path = tmp_path / "benchmark.py"
    rank_analyzer_path = tmp_path / "rank_analyzer.py"
    for path, value in (
        (contract_path, "contract"),
        (supervisor_path, "supervisor"),
        (benchmark_path, "benchmark"),
        (rank_analyzer_path, "rank analyzer"),
    ):
        path.write_text(value, encoding="utf-8")
    contract = SimpleNamespace(
        SCHEMA="unit-schema",
        SOURCE_FILES={"/vllm/core.py": "a" * 64},
        build_run_plan=lambda: {"runs": [{"id": "unit-run"}]},
        __file__=str(contract_path),
    )
    model_identity = {
        "revision": "c" * 40,
        "files_sha256": {
            "config.json": "1" * 64,
            "tokenizer_config.json": "2" * 64,
            "tiktoken.model": "3" * 64,
            "tokenization_kimi.py": "4" * 64,
        },
    }
    prompt_identities = {
        "unit-run": {"warmup": "4" * 64, "measurement": "5" * 64}
    }

    manifest = runner.build_execution_manifest(
        contract,
        source_commit="b" * 40,
        supervisor_path=supervisor_path,
        benchmark_path=benchmark_path,
        rank_analyzer_path=rank_analyzer_path,
        model_identity=model_identity,
        prompt_identities=prompt_identities,
    )
    digest = runner.execution_manifest_digest(manifest)

    runner.validate_execution_manifest(
        manifest,
        expected_digest=digest,
        contract=contract,
        source_commit="b" * 40,
        supervisor_path=supervisor_path,
        benchmark_path=benchmark_path,
        rank_analyzer_path=rank_analyzer_path,
    )
    supervisor_path.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="execution_tool_hash_mismatch:supervisor"):
        runner.validate_execution_manifest(
            manifest,
            expected_digest=digest,
            contract=contract,
            source_commit="b" * 40,
            supervisor_path=supervisor_path,
            benchmark_path=benchmark_path,
            rank_analyzer_path=rank_analyzer_path,
        )


def test_model_revision_and_tokenizer_hash_are_fail_closed(tmp_path: Path) -> None:
    runner = _load(MODULE_PATH, "run_phase466_low_overhead_probe_model_identity")
    model = tmp_path / "models--moonshotai--Kimi-K2.5"
    revision = "a" * 40
    snapshot = model / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (model / "refs").mkdir()
    (model / "refs" / "main").write_text(revision + "\n")
    for filename in runner.MODEL_IDENTITY_FILES:
        (snapshot / filename).write_text(filename)

    identity = runner.resolve_model_identity(model)

    assert identity["revision"] == revision
    assert set(identity["files_sha256"]) == set(runner.MODEL_IDENTITY_FILES)
    contract = SimpleNamespace(MODEL_PATH=str(model))
    assert runner.assert_model_identity(contract, {"model_identity": identity}) == identity
    (snapshot / "tiktoken.model").write_text("changed")
    with pytest.raises(RuntimeError, match="model_identity_mismatch"):
        runner.assert_model_identity(contract, {"model_identity": identity})


def test_model_revision_must_resolve_to_snapshot(tmp_path: Path) -> None:
    runner = _load(MODULE_PATH, "run_phase466_low_overhead_probe_model_revision")
    model = tmp_path / "model"
    model.mkdir()

    with pytest.raises(RuntimeError, match="model_revision_unresolved"):
        runner.resolve_model_identity(model)


def test_flat_composite_model_mirror_is_not_an_immutable_snapshot(
    tmp_path: Path,
) -> None:
    runner = _load(MODULE_PATH, "run_phase466_low_overhead_probe_flat_mirror")
    model = tmp_path / "models--moonshotai--Kimi-K2.5"
    metadata = model / ".cache" / "huggingface" / "download"
    metadata.mkdir(parents=True)
    for filename in runner.MODEL_IDENTITY_FILES:
        (model / filename).write_text(filename, encoding="utf-8")
    (metadata / "config.json.metadata").write_text(
        f"{'a' * 40}\netag\n123.0\n", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="model_revision_unresolved"):
        runner.resolve_model_identity(model)


def test_prompt_digest_missing_or_drift_fails_fast(tmp_path: Path) -> None:
    runner = _load(MODULE_PATH, "run_phase466_low_overhead_probe_prompt_identity")
    result = tmp_path / "bench_result.json"
    with pytest.raises(RuntimeError, match="prompt_cohort_sha256_missing"):
        runner.assert_benchmark_prompt_identity(
            result, expected="a" * 64, label="unit"
        )
    result.write_text(json.dumps({"prompt_cohort_sha256": "b" * 64}))
    with pytest.raises(RuntimeError, match="prompt_cohort_sha256_mismatch"):
        runner.assert_benchmark_prompt_identity(
            result, expected="a" * 64, label="unit"
        )


def test_coordinator_manifest_requires_exact_clean_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load(MODULE_PATH, "run_phase466_low_overhead_probe_coordinator")
    monkeypatch.setattr(
        runner,
        "_capture",
        lambda *args, **kwargs: "c" * 40,
    )

    with pytest.raises(RuntimeError, match="coordinator_head_mismatch"):
        runner.verify_coordinator_checkout(
            tmp_path,
            "d" * 40,
            contract_path=tmp_path / "scripts" / "analyze_phase466_low_overhead_probe.py",
            supervisor_path=tmp_path / "scripts" / "run_phase466_low_overhead_probe.py",
            benchmark_path=tmp_path / "scripts" / "run_openai_fixed_shape_benchmark.py",
            rank_analyzer_path=tmp_path / "scripts" / "analyze_phase466_rank_timing.py",
        )


def test_coordinator_rejects_tool_outside_exact_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load(MODULE_PATH, "run_phase466_low_overhead_probe_tool_path")
    monkeypatch.setattr(
        runner,
        "_capture",
        lambda *args, **kwargs: "a" * 40 if args[0][1:3] == ["rev-parse", "HEAD"] else "",
    )

    with pytest.raises(RuntimeError, match="coordinator_tool_path_mismatch:contract"):
        runner.verify_coordinator_checkout(
            tmp_path,
            "a" * 40,
            contract_path=tmp_path / "outside-contract.py",
            supervisor_path=tmp_path / "scripts" / "run_phase466_low_overhead_probe.py",
            benchmark_path=tmp_path / "scripts" / "run_openai_fixed_shape_benchmark.py",
            rank_analyzer_path=tmp_path / "scripts" / "analyze_phase466_rank_timing.py",
        )


def test_source_identity_is_resampled_for_each_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load(MODULE_PATH, "run_phase466_low_overhead_probe_source_identity")
    contract = SimpleNamespace(SOURCE_FILES={"/vllm/core.py": "a" * 64})
    samples = iter(
        [
            {"/vllm/core.py": "a" * 64},
            {"/vllm/core.py": "b" * 64},
        ]
    )
    monkeypatch.setattr(runner, "_source_hashes", lambda unused: next(samples))

    assert runner.assert_source_identity(contract) == {"/vllm/core.py": "a" * 64}
    with pytest.raises(RuntimeError, match="vllm_source_hash_mismatch"):
        runner.assert_source_identity(contract)


def test_signal_abort_terminates_active_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load(MODULE_PATH, "run_phase466_low_overhead_probe_signal")
    supervisor = runner.Supervisor(
        contract=SimpleNamespace(SCHEMA="unit-schema", SOURCE_FILES={}),
        workdir=tmp_path,
        artifact_root=tmp_path / "artifact",
        source_commit="a" * 40,
        execution_manifest={},
        execution_manifest_sha256="b" * 64,
    )
    service = SimpleNamespace(pid=123, poll=lambda: None)
    command = SimpleNamespace(pid=456, poll=lambda: None)
    supervisor._active_service = service
    supervisor._active_command = command
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(runner.os, "killpg", lambda pid, sig: killed.append((pid, sig)))

    supervisor.request_abort(signal.SIGTERM)

    assert supervisor.abort_requested
    assert killed == [(456, signal.SIGTERM), (123, signal.SIGTERM)]
    with pytest.raises(runner.AbortRequested, match="signal_SIGTERM"):
        supervisor.raise_if_aborted()


def test_process_group_cleanup_does_not_trust_exited_leader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load(MODULE_PATH, "run_phase466_low_overhead_probe_group_cleanup")
    killed = False
    reaped = False
    sent: list[int] = []

    class Process:
        pid = 321

        def poll(self):
            nonlocal reaped
            if killed:
                reaped = True
                return 0
            return None

        def wait(self, timeout: float):
            return self.poll()

    def killpg(unused_pid: int, signum: int) -> None:
        nonlocal killed
        if signum == 0:
            if reaped:
                raise ProcessLookupError
            return
        sent.append(signum)
        if signum == signal.SIGKILL:
            killed = True

    monkeypatch.setattr(runner.os, "killpg", killpg)

    had_group, clean = runner._terminate_process_group(
        Process(),
        term_timeout_s=0,
        kill_timeout_s=0,
    )

    assert had_group
    assert clean
    assert sent == [signal.SIGTERM, signal.SIGKILL]


def test_tool_identity_is_rechecked_against_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load(MODULE_PATH, "run_phase466_low_overhead_probe_tool_recheck")
    supervisor = runner.Supervisor(
        contract=SimpleNamespace(SCHEMA="unit-schema"),
        workdir=tmp_path,
        artifact_root=tmp_path / "artifact",
        source_commit="a" * 40,
        execution_manifest={},
        execution_manifest_sha256="b" * 64,
    )
    calls: list[bool] = []
    monkeypatch.setattr(
        runner,
        "validate_execution_manifest",
        lambda *args, **kwargs: calls.append(True),
    )
    monkeypatch.setattr(
        runner,
        "_execution_tool_hashes",
        lambda *args, **kwargs: {"contract": "c" * 64},
    )

    assert supervisor.validate_tool_identity() == {"contract": "c" * 64}
    assert calls == [True]


def test_finalize_writes_pass_result_only_after_log_compression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load(MODULE_PATH, "run_phase466_low_overhead_probe_finalize")
    supervisor = runner.Supervisor(
        contract=SimpleNamespace(SCHEMA="unit-schema", SOURCE_FILES={}),
        workdir=tmp_path,
        artifact_root=tmp_path,
        source_commit="a" * 40,
        execution_manifest={},
        execution_manifest_sha256="b" * 64,
    )
    result = {"status": "PASS"}
    monkeypatch.setattr(
        runner,
        "_compress_logs",
        lambda root: (_ for _ in ()).throw(RuntimeError("compression_failed")),
    )

    with pytest.raises(RuntimeError, match="compression_failed"):
        runner.finalize_result(
            supervisor=supervisor,
            result=result,
            final_residue_check=lambda: None,
            terminal_status=lambda unused_status: None,
        )
    assert not (tmp_path / "phase466_result.json").exists()


def test_terminal_commit_rewrites_pass_as_aborted_when_signal_arrives_during_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load(MODULE_PATH, "run_phase466_low_overhead_probe_terminal_race")
    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()
    supervisor = runner.Supervisor(
        contract=SimpleNamespace(SCHEMA="unit-schema", SOURCE_FILES={}),
        workdir=tmp_path,
        artifact_root=artifact_root,
        source_commit="a" * 40,
        execution_manifest={},
        execution_manifest_sha256="b" * 64,
    )
    original_write = runner._write_json
    injected = False

    def write_with_signal(path: Path, value: object) -> None:
        nonlocal injected
        if path.name == "phase466_result.json" and not injected:
            injected = True
            supervisor.request_abort(signal.SIGTERM)
        original_write(path, value)

    monkeypatch.setattr(runner, "_write_json", write_with_signal)
    statuses: list[str] = []

    committed = supervisor.commit_terminal_result(
        {"status": "PASS", "gate_status": "PASS"},
        terminal_status=statuses.append,
    )

    result = json.loads((artifact_root / "phase466_result.json").read_text())
    assert committed == "ABORTED"
    assert statuses == ["PASS", "ABORTED"]
    assert result["status"] == "ABORTED"
    assert result["reason"] == "signal_SIGTERM"


def test_supervisor_heartbeat_updates_status_and_stops(tmp_path: Path) -> None:
    runner = _load(MODULE_PATH, "run_phase466_low_overhead_probe_heartbeat")
    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()
    supervisor = runner.Supervisor(
        contract=SimpleNamespace(SCHEMA="unit-schema", SOURCE_FILES={}),
        workdir=tmp_path,
        artifact_root=artifact_root,
        source_commit="a" * 40,
        execution_manifest={},
        execution_manifest_sha256="b" * 64,
    )
    supervisor.status("formal", "measuring", run_id="run-1")
    first = json.loads(supervisor.status_path.read_text())

    supervisor.start_heartbeat(interval_s=0.01)
    deadline = time.monotonic() + 1.0
    current = first
    while time.monotonic() < deadline and current["updated_at"] == first["updated_at"]:
        time.sleep(0.01)
        current = json.loads(supervisor.status_path.read_text())
    supervisor.stop_heartbeat()

    assert current["updated_at"] > first["updated_at"]
    assert current["stage"] == "formal"
    assert current["state"] == "measuring"
    assert current["run_id"] == "run-1"
    assert not supervisor.heartbeat_alive


def test_preflight_captures_source_commit_gpu_and_runtime_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load(MODULE_PATH, "run_phase466_low_overhead_probe_preflight")
    artifact_root = tmp_path / "artifact"
    supervisor = runner.Supervisor(
        contract=SimpleNamespace(SCHEMA="unit-schema", SOURCE_FILES={}),
        workdir=tmp_path,
        artifact_root=artifact_root,
        source_commit="a" * 40,
        execution_manifest={},
        execution_manifest_sha256="b" * 64,
    )
    monkeypatch.setattr(runner, "validate_execution_manifest", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_assert_clean_worker", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runner,
        "assert_imported_vllm_source_paths",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(runner, "_source_hashes", lambda contract: {})
    model_identity = {
        "revision": "a" * 40,
        "files_sha256": {
            "config.json": "1" * 64,
            "tokenizer_config.json": "2" * 64,
            "tiktoken.model": "3" * 64,
            "tokenization_kimi.py": "4" * 64,
        },
    }
    monkeypatch.setattr(
        runner, "assert_model_identity", lambda contract, manifest: model_identity
    )
    monkeypatch.setattr(
        runner,
        "_execution_tool_hashes",
        lambda contract, workdir: {"tool.py": "b" * 64},
    )

    def capture(argv, **kwargs):
        if argv[0] == "nvidia-smi":
            return "NVIDIA H200, 143771, 575.57.08"
        if argv[:2] == ["python3", "-c"]:
            return "0.19.0"
        raise AssertionError(argv)

    monkeypatch.setattr(runner, "_capture", capture)

    supervisor.preflight()
    environment = json.loads((artifact_root / "environment.json").read_text())

    assert environment["source_commit"] == "a" * 40
    assert environment["gpu"] == "NVIDIA H200, 143771, 575.57.08"
    assert environment["vllm_version"] == "0.19.0"
    assert environment["model_identity"] == model_identity
    assert (artifact_root / "tooling.sha256").read_text() == f"{'b' * 64}  tool.py\n"


def test_supervisor_rejects_invalid_source_commit(tmp_path: Path) -> None:
    runner = _load(MODULE_PATH, "run_phase466_low_overhead_probe_commit")

    with pytest.raises(ValueError, match="invalid_source_commit"):
        runner.Supervisor(
            contract=SimpleNamespace(SCHEMA="unit-schema"),
            workdir=tmp_path,
            artifact_root=tmp_path / "artifact",
            source_commit="2f6ad73d",
            execution_manifest={},
            execution_manifest_sha256="b" * 64,
        )


def test_gate_stop_result_is_explicit_and_never_default_evidence() -> None:
    runner = _load(MODULE_PATH, "run_phase466_low_overhead_probe_gate_result")

    result = runner.gate_stop_result({"status": "INCONCLUSIVE"})

    assert result["status"] == "STOPPED_BEFORE_FORMAL"
    assert result["gate_status"] == "INCONCLUSIVE"
    assert result["formal_scenarios"] == []
    assert result["diagnostic_only"] is True
    assert result["valid_for_default"] is False
    assert result["perf_database"] is False
    assert result["default_readiness"] == "No-Go"


def test_stop_service_does_not_require_ray_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load(MODULE_PATH, "run_phase466_low_overhead_probe_stop")

    class ExitedProcess:
        pid = 123

        def poll(self):
            return 0

    monkeypatch.setattr(runner, "_gpu_residue", lambda *args, **kwargs: "")
    monkeypatch.setattr(runner, "_process_residue", lambda *args, **kwargs: "")
    monkeypatch.setattr(
        runner,
        "_terminate_process_group",
        lambda *args, **kwargs: (False, True),
    )

    def forbidden_run(*args, **kwargs):
        raise AssertionError("cleanup must not invoke unavailable ray CLI")

    monkeypatch.setattr(runner.subprocess, "run", forbidden_run)

    assert runner._stop_service(ExitedProcess(), cwd=tmp_path, env={}) is True


def test_last_iteration_by_rank_requires_every_expected_rank() -> None:
    runner = _load(MODULE_PATH, "run_phase466_low_overhead_probe_cutoff")
    log = "\n".join(
        [
            "(EngineCore_DP0 pid=10) INFO Iteration(3): 1 context requests, "
            "8 context tokens, 0 generation requests, 0 generation tokens, "
            "iteration elapsed time: 1.0 ms",
            "(EngineCore_DP1 pid=11) INFO Iteration(4): 0 context requests, "
            "0 context tokens, 1 generation requests, 1 generation tokens, "
            "iteration elapsed time: 1.0 ms",
        ]
    )

    assert runner.last_iteration_by_rank(log, expected_ranks=[0, 1]) == {0: 3, 1: 4}
    with pytest.raises(ValueError, match="warmup_iteration_rank_mismatch"):
        runner.last_iteration_by_rank(log, expected_ranks=[0, 1, 2])
