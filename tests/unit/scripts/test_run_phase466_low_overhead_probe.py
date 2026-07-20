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


def test_manifest_write_modes_are_mutually_exclusive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load(MODULE_PATH, "phase466_manifest_mode_exclusive")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(MODULE_PATH),
            "--workdir",
            str(tmp_path),
            "--write-coordinator-manifest",
            str(tmp_path / "coordinator.json"),
            "--write-worker-attestation",
            str(tmp_path / "worker.json"),
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        runner.main()


def _supervisor_manifest(source_commit: str = "a" * 40) -> dict:
    return {"coordinator_manifest": {"source_commit": source_commit}}


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


def test_two_stage_manifest_binds_local_and_worker_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load(MODULE_PATH, "run_phase466_low_overhead_probe_manifest")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    tools = {}
    for name, relative in runner.COORDINATOR_TOOL_RELATIVE_PATHS.items():
        path = tmp_path / relative
        path.write_text(name, encoding="utf-8")
        tools[name] = {
            "path": relative,
            "sha256": runner.hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    source_path = tmp_path / "vllm_core.py"
    source_path.write_text("source", encoding="utf-8")
    source_hash = runner.hashlib.sha256(source_path.read_bytes()).hexdigest()
    contract = SimpleNamespace(
        SCHEMA="unit-schema",
        SOURCE_FILES={str(source_path): source_hash},
        MODEL_PATH=str(tmp_path / "model"),
        build_run_plan=lambda: {"runs": [{"id": "unit-run"}]},
        __file__=str(tmp_path / runner.COORDINATOR_TOOL_RELATIVE_PATHS["contract"]),
    )
    coordinator = {
        "schema": runner.COORDINATOR_MANIFEST_SCHEMA,
        "contract_schema": contract.SCHEMA,
        "source_commit": "b" * 40,
        "tools": tools,
    }
    coordinator_digest = runner.execution_manifest_digest(coordinator)
    model_identity_payload = {
        "schema": runner.FLAT_MODEL_FINGERPRINT_SCHEMA,
        "identity_scope": "same_flat_mirror_instance",
        "official_immutable_revision": False,
        "metadata_files": [
            {"path": ".msc", "sha256": "6" * 64},
            {"path": ".mv", "sha256": "7" * 64},
        ],
        "runtime_files": [
            {"path": "model.safetensors.index.json", "sha256": "8" * 64}
        ],
        "shards": [
            {
                "path": "model-00001-of-00001.safetensors",
                "size": 10,
                "mtime_ns": 1,
                "header_sha256": "9" * 64,
            }
        ],
    }
    model_identity = dict(model_identity_payload)
    model_identity["fingerprint_sha256"] = runner.execution_manifest_digest(
        model_identity_payload
    )
    prompt_identities = {
        "unit-run": {"warmup": "4" * 64, "measurement": "5" * 64}
    }
    monkeypatch.setattr(runner, "resolve_model_identity", lambda path: model_identity)
    monkeypatch.setattr(
        runner,
        "expected_prompt_cohort_sha256",
        lambda contract, benchmark_path: prompt_identities,
    )
    monkeypatch.setattr(
        runner,
        "assert_imported_vllm_source_paths",
        lambda contract, cwd, env: list(contract.SOURCE_FILES),
    )
    monkeypatch.setattr(
        runner,
        "_capture",
        lambda argv, **kwargs: (
            "0.19.0" if argv[:2] == ["python3", "-c"] else "H200, 141312, 570.133.20"
        ),
    )

    attestation = runner.build_worker_attestation(
        contract,
        coordinator_manifest=coordinator,
        coordinator_manifest_sha256=coordinator_digest,
        workdir=tmp_path,
        benchmark_path=tmp_path / runner.COORDINATOR_TOOL_RELATIVE_PATHS["benchmark"],
    )
    attestation_digest = runner.execution_manifest_digest(attestation)
    manifest = runner.compose_execution_manifest(
        contract,
        coordinator_manifest=coordinator,
        coordinator_manifest_sha256=coordinator_digest,
        worker_attestation=attestation,
        worker_attestation_sha256=attestation_digest,
    )
    manifest_digest = runner.execution_manifest_digest(manifest)

    validated = runner.validate_execution_manifest(
        manifest,
        expected_digest=manifest_digest,
        contract=contract,
        workdir=tmp_path,
    )
    assert validated["model_identity"] == model_identity
    assert validated["gpu_identity"]["rows"] == ["H200, 141312, 570.133.20"]
    assert manifest["coordinator_manifest_sha256"] == coordinator_digest
    assert manifest["worker_attestation_sha256"] == attestation_digest

    (tmp_path / runner.COORDINATOR_TOOL_RELATIVE_PATHS["supervisor"]).write_text(
        "changed", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="execution_tool_hash_mismatch:supervisor"):
        runner.validate_execution_manifest(
            manifest,
            expected_digest=manifest_digest,
            contract=contract,
            workdir=tmp_path,
        )


def test_coordinator_manifest_requires_clean_checkout_and_relative_tool_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load(MODULE_PATH, "run_phase466_coordinator_manifest")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    tool_names = {
        "contract": "analyze_phase466_low_overhead_probe.py",
        "supervisor": "run_phase466_low_overhead_probe.py",
        "benchmark": "run_openai_fixed_shape_benchmark.py",
        "rank_analyzer": "analyze_phase466_rank_timing.py",
    }
    for name, filename in tool_names.items():
        (scripts / filename).write_text(name, encoding="utf-8")
    contract = SimpleNamespace(
        SCHEMA="unit-schema",
        __file__=str(scripts / tool_names["contract"]),
    )

    def clean_capture(argv, **kwargs):
        if argv[1:] == ["rev-parse", "HEAD"]:
            return "a" * 40
        if argv[1:] == ["status", "--porcelain", "--untracked-files=all"]:
            return ""
        raise AssertionError(argv)

    monkeypatch.setattr(runner, "_capture", clean_capture)
    manifest = runner.build_coordinator_manifest(contract, workdir=tmp_path)

    assert manifest["schema"] == "phase466_coordinator_manifest_v1"
    assert manifest["source_commit"] == "a" * 40
    assert {
        name: item["path"] for name, item in manifest["tools"].items()
    } == {name: f"scripts/{filename}" for name, filename in tool_names.items()}

    monkeypatch.setattr(
        runner,
        "_capture",
        lambda argv, **kwargs: (
            "a" * 40
            if argv[1:] == ["rev-parse", "HEAD"]
            else " M scripts/run_phase466_low_overhead_probe.py"
        ),
    )
    with pytest.raises(RuntimeError, match="coordinator_checkout_dirty"):
        runner.build_coordinator_manifest(contract, workdir=tmp_path)


def test_flat_worker_identity_requires_complete_fingerprint_sections() -> None:
    runner = _load(MODULE_PATH, "run_phase466_flat_identity_shape")
    payload = {
        "schema": runner.FLAT_MODEL_FINGERPRINT_SCHEMA,
        "identity_scope": "same_flat_mirror_instance",
        "official_immutable_revision": False,
        "metadata_files": [],
        "runtime_files": [],
        "shards": [],
    }
    identity = dict(payload)
    identity["fingerprint_sha256"] = runner.execution_manifest_digest(payload)

    with pytest.raises(RuntimeError, match="execution_manifest_model_identity_missing"):
        runner._validate_model_identity_shape(identity)


def test_worker_validates_uploaded_tool_bytes_without_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load(MODULE_PATH, "run_phase466_worker_tool_validation")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    tool_names = {
        "contract": "analyze_phase466_low_overhead_probe.py",
        "supervisor": "run_phase466_low_overhead_probe.py",
        "benchmark": "run_openai_fixed_shape_benchmark.py",
        "rank_analyzer": "analyze_phase466_rank_timing.py",
    }
    tools = {}
    for name, filename in tool_names.items():
        path = scripts / filename
        path.write_text(name, encoding="utf-8")
        tools[name] = {
            "path": f"scripts/{filename}",
            "sha256": runner.hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest = {
        "schema": "phase466_coordinator_manifest_v1",
        "contract_schema": "unit-schema",
        "source_commit": "a" * 40,
        "tools": tools,
    }
    digest = runner.execution_manifest_digest(manifest)
    monkeypatch.setattr(
        runner,
        "_capture",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("git called")),
    )

    assert runner.validate_coordinator_manifest_bytes(
        manifest,
        expected_digest=digest,
        workdir=tmp_path,
        contract_schema="unit-schema",
    ) == {name: item["sha256"] for name, item in tools.items()}
    (scripts / tool_names["benchmark"]).write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="execution_tool_hash_mismatch:benchmark"):
        runner.validate_coordinator_manifest_bytes(
            manifest,
            expected_digest=digest,
            workdir=tmp_path,
            contract_schema="unit-schema",
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
    manifest = {"worker_attestation": {"model_identity": identity}}
    assert runner.assert_model_identity(contract, manifest) == identity
    (snapshot / "tiktoken.model").write_text("changed")
    with pytest.raises(RuntimeError, match="model_identity_mismatch"):
        runner.assert_model_identity(contract, manifest)


def test_model_path_without_snapshot_or_flat_metadata_fails(tmp_path: Path) -> None:
    runner = _load(MODULE_PATH, "run_phase466_low_overhead_probe_model_identity_missing")
    model = tmp_path / "model"
    model.mkdir()

    with pytest.raises(RuntimeError, match="flat_model_metadata_missing"):
        runner.resolve_model_identity(model)


def test_flat_composite_model_mirror_uses_instance_fingerprint(
    tmp_path: Path,
) -> None:
    runner = _load(MODULE_PATH, "run_phase466_low_overhead_probe_flat_mirror")
    model = tmp_path / "models--moonshotai--Kimi-K2.5"
    model.mkdir()
    (model / ".msc").write_bytes(b"multiple revisions")
    (model / ".mv").write_text("Revision:master", encoding="utf-8")
    (model / "config.json").write_text("{}", encoding="utf-8")
    shard = "model-00001-of-00001.safetensors"
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": shard}}), encoding="utf-8"
    )
    header = json.dumps({"layer": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}).encode()
    (model / shard).write_bytes(len(header).to_bytes(8, "little") + header + b"xx")

    identity = runner.resolve_model_identity(model)

    assert identity["schema"] == runner.FLAT_MODEL_FINGERPRINT_SCHEMA
    assert identity["official_immutable_revision"] is False


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
        execution_manifest=_supervisor_manifest(),
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
        execution_manifest=_supervisor_manifest(),
        execution_manifest_sha256="b" * 64,
    )
    calls: list[bool] = []
    monkeypatch.setattr(
        runner,
        "validate_execution_manifest",
        lambda *args, **kwargs: (
            calls.append(True) or {"tool_sha256": {"contract": "c" * 64}}
        ),
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
        execution_manifest=_supervisor_manifest(),
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
        execution_manifest=_supervisor_manifest(),
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
        execution_manifest=_supervisor_manifest(),
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
    execution_manifest = _supervisor_manifest()
    execution_manifest["worker_attestation"] = {
        "gpu_identity": {"rows": ["NVIDIA H200, 143771, 575.57.08"]},
        "vllm_import_paths": [],
    }
    supervisor = runner.Supervisor(
        contract=SimpleNamespace(SCHEMA="unit-schema", SOURCE_FILES={}),
        workdir=tmp_path,
        artifact_root=artifact_root,
        execution_manifest=execution_manifest,
        execution_manifest_sha256="b" * 64,
    )
    monkeypatch.setattr(
        runner,
        "validate_execution_manifest",
        lambda *args, **kwargs: {"tool_sha256": {"tool.py": "b" * 64}},
    )
    monkeypatch.setattr(runner, "_assert_clean_worker", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runner,
        "assert_imported_vllm_source_paths",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(runner, "_source_hashes", lambda contract: {})
    model_identity = {
        "schema": runner.SNAPSHOT_MODEL_IDENTITY_SCHEMA,
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


def test_execution_preflight_identity_failure_creates_no_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load(MODULE_PATH, "run_phase466_preflight_no_artifact")
    artifact_root = tmp_path / "artifact"
    supervisor = runner.Supervisor(
        contract=SimpleNamespace(SCHEMA="unit-schema", SOURCE_FILES={}),
        workdir=tmp_path,
        artifact_root=artifact_root,
        execution_manifest=_supervisor_manifest(),
        execution_manifest_sha256="b" * 64,
    )
    monkeypatch.setattr(
        supervisor,
        "validate_tool_identity",
        lambda: (_ for _ in ()).throw(RuntimeError("execution_manifest_digest_mismatch")),
    )
    monkeypatch.setattr(runner, "_assert_clean_worker", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="execution_manifest_digest_mismatch"):
        supervisor.preflight()

    assert not artifact_root.exists()
    assert supervisor.owns_artifact_root is False


def test_supervisor_rejects_invalid_coordinator_source_commit(tmp_path: Path) -> None:
    runner = _load(MODULE_PATH, "run_phase466_low_overhead_probe_commit")

    with pytest.raises(ValueError, match="execution_manifest_source_commit_invalid"):
        runner.Supervisor(
            contract=SimpleNamespace(SCHEMA="unit-schema"),
            workdir=tmp_path,
            artifact_root=tmp_path / "artifact",
            execution_manifest=_supervisor_manifest("2f6ad73d"),
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
