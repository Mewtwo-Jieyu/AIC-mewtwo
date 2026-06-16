from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "collector" / "vllm" / "run_phase272_deeper_trace_source_map.sh"

EXPECTED_SCENARIOS = {
    "tp8ep8-12k2k-bt12000",
    "tp8ep8-12k2k-bt65536",
}

PHASE234_FIELDS = {
    "scenario",
    "iteration",
    "phase",
    "scheduled_context_tokens",
    "scheduled_decode_tokens",
    "scheduled_total_tokens",
    "scheduled_context_reqs",
    "scheduled_decode_reqs",
    "max_num_batched_tokens",
    "max_num_seqs",
    "forward_token_count",
    "tp",
    "dp",
    "ep",
    "topology_key",
    "shape_key",
}

DEEPER_FIELDS = {
    "iteration_start_ns",
    "forward_start_ns",
    "forward_end_ns",
    "iteration_end_ns",
    "iteration_elapsed_ns",
    "forward_elapsed_ns",
    "scheduler_running_reqs",
    "scheduler_waiting_reqs",
    "scheduler_scheduled_new_reqs",
    "scheduler_scheduled_decode_reqs",
    "scheduler_scheduled_prefill_reqs",
    "active_request_count",
    "context_drain_tokens",
    "mixed_iteration_index",
    "tail_decode_tokens",
}


def _run_runner(
    *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(RUNNER), *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_runner_is_valid_bash() -> None:
    result = subprocess.run(
        ["bash", "-n", str(RUNNER)],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_source_check_prints_deeper_trace_source_map(tmp_path: Path) -> None:
    source = tmp_path / "vllm" / "vllm" / "v1" / "worker" / "gpu_model_runner.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "scheduler_output\nbatch_desc.num_tokens\n",
        encoding="utf-8",
    )

    result = _run_runner("source-check", env={"WORKDIR": str(tmp_path)})

    assert result.returncode == 0, result.stderr
    assert "phase272_mode=source-check" in result.stdout
    assert "source_check=PASS" in result.stdout
    assert "vllm_source_present=true" in result.stdout
    assert "patch_anchor_compute_iteration_details=true" in result.stdout
    assert "patch_anchor_scheduler_output=true" in result.stdout
    assert "source_map_forward_timing=available" in result.stdout
    assert "source_map_queue_state=available" in result.stdout
    assert "source_map_boundary_timeline=available" in result.stdout
    assert "gpu_benchmark_started=false" in result.stdout


def test_preflight_prints_two_tp8_12k2k_scenarios_and_deeper_schema() -> None:
    result = _run_runner("preflight")

    assert result.returncode == 0, result.stderr
    assert "phase272_mode=preflight" in result.stdout
    assert "scenario_count=2" in result.stdout
    assert "gpu_benchmark_started=false" in result.stdout
    scenario_lines = [
        line.removeprefix("scenario=")
        for line in result.stdout.splitlines()
        if line.startswith("scenario=")
    ]
    assert set(scenario_lines) == EXPECTED_SCENARIOS
    assert result.stdout.count("serve_command=") == 2
    assert result.stdout.count("benchmark_command=") == 2

    schema_line = next(
        line for line in result.stdout.splitlines() if line.startswith("trace_schema=")
    )
    trace_fields = set(schema_line.removeprefix("trace_schema=").split(","))
    assert PHASE234_FIELDS <= trace_fields
    assert DEEPER_FIELDS <= trace_fields


def test_single_scenario_preflight_validates_exact_tp8_12k2k_scenario() -> None:
    result = _run_runner("preflight", "tp8ep8-12k2k-bt65536")

    assert result.returncode == 0, result.stderr
    assert "scenario_count=1" in result.stdout
    assert result.stdout.count("scenario=") == 1
    assert "scenario=tp8ep8-12k2k-bt65536" in result.stdout
    assert "topology_key=tp8_dp1_ep8" in result.stdout
    assert "shape_key=isl12000_osl2000_batch128" in result.stdout
    assert "max_num_batched_tokens=65536" in result.stdout


def test_unknown_scenario_fails_fast() -> None:
    result = _run_runner("preflight", "tp4dp2ep8-12k2k-bt65536")

    assert result.returncode == 2
    assert "unknown_scenario=tp4dp2ep8-12k2k-bt65536" in result.stderr


def test_run_one_requires_explicit_gpu_authorization() -> None:
    result = _run_runner("run-one", "tp8ep8-12k2k-bt12000")

    assert result.returncode == 2
    assert "phase272_allow_gpu_run_required=true" in result.stderr
