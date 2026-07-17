from __future__ import annotations

import importlib.util
import json
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


def test_failure_action_is_fail_closed() -> None:
    runner = _load(MODULE_PATH, "run_phase466_low_overhead_probe_failure_action")

    assert runner.failure_action(stage="overhead", cleanup=True, integrity_failure=False) == "CONTINUE_GATE"
    assert runner.failure_action(stage="overhead", cleanup=False, integrity_failure=False) == "STOP_ALL"
    assert runner.failure_action(stage="overhead", cleanup=True, integrity_failure=True) == "STOP_ALL"
    assert runner.failure_action(stage="formal", cleanup=True, integrity_failure=False) == "CONTINUE_FORMAL"
    assert runner.failure_action(stage="formal", cleanup=False, integrity_failure=False) == "STOP_ALL"


def test_integrity_error_detection_matches_nested_validation_errors() -> None:
    runner = _load(MODULE_PATH, "run_phase466_low_overhead_probe_integrity")

    assert runner.is_integrity_failure("validation:vllm_source_hash_mismatch")
    assert runner.is_integrity_failure("nonempty_gpu_residue:/tmp/run")
    assert not runner.is_integrity_failure("benchmark_command_failed:1")


def test_supervisor_heartbeat_updates_status_and_stops(tmp_path: Path) -> None:
    runner = _load(MODULE_PATH, "run_phase466_low_overhead_probe_heartbeat")
    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()
    supervisor = runner.Supervisor(
        contract=SimpleNamespace(SCHEMA="unit-schema"),
        workdir=tmp_path,
        artifact_root=artifact_root,
        source_commit="a" * 40,
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
        contract=SimpleNamespace(SCHEMA="unit-schema"),
        workdir=tmp_path,
        artifact_root=artifact_root,
        source_commit="a" * 40,
    )
    monkeypatch.setattr(runner, "_assert_clean_worker", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_source_hashes", lambda contract: {})
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
    assert (artifact_root / "tooling.sha256").read_text() == f"{'b' * 64}  tool.py\n"


def test_supervisor_rejects_invalid_source_commit(tmp_path: Path) -> None:
    runner = _load(MODULE_PATH, "run_phase466_low_overhead_probe_commit")

    with pytest.raises(ValueError, match="invalid_source_commit"):
        runner.Supervisor(
            contract=SimpleNamespace(SCHEMA="unit-schema"),
            workdir=tmp_path,
            artifact_root=tmp_path / "artifact",
            source_commit="2f6ad73d",
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
        def poll(self):
            return 0

    monkeypatch.setattr(runner, "_gpu_residue", lambda *args, **kwargs: "")
    monkeypatch.setattr(runner, "_process_residue", lambda *args, **kwargs: "")

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
