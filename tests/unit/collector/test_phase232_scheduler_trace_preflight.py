from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "collector" / "vllm" / "run_phase232_scheduler_trace_preflight.sh"


EXPECTED_SCENARIOS = {
    "tp8ep8-4k2k-bt4000",
    "tp8ep8-4k2k-bt65536",
    "tp4dp2ep8-4k2k-bt4000",
    "tp4dp2ep8-4k2k-bt65536",
    "tp8ep8-12k2k-bt12000",
    "tp8ep8-12k2k-bt65536",
    "tp4dp2ep8-12k2k-bt12000",
    "tp4dp2ep8-12k2k-bt65536",
}

EXPECTED_TRACE_FIELDS = {
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


def _run_runner(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(RUNNER), *args],
        cwd=REPO_ROOT,
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


def test_preflight_prints_eight_scenarios_and_trace_schema() -> None:
    result = _run_runner("preflight")

    assert result.returncode == 0, result.stderr
    assert "phase232_mode=preflight" in result.stdout
    assert "scenario_count=8" in result.stdout
    assert "scheduler_trace_started=false" in result.stdout
    assert "gpu_benchmark_started=false" in result.stdout
    assert "compute_iteration_details(scheduler_output)" in result.stdout
    scenario_lines = [
        line.removeprefix("scenario=")
        for line in result.stdout.splitlines()
        if line.startswith("scenario=")
    ]
    assert set(scenario_lines) == EXPECTED_SCENARIOS
    schema_line = next(
        line for line in result.stdout.splitlines() if line.startswith("trace_schema=")
    )
    trace_fields = set(schema_line.removeprefix("trace_schema=").split(","))
    assert EXPECTED_TRACE_FIELDS <= trace_fields


def test_single_scenario_preflight_validates_exact_scenario() -> None:
    result = _run_runner("preflight", "tp8ep8-4k2k-bt65536")

    assert result.returncode == 0, result.stderr
    assert "scenario_count=1" in result.stdout
    assert result.stdout.count("scenario=") == 1
    assert "scenario=tp8ep8-4k2k-bt65536" in result.stdout
    assert "topology_key=tp8_dp1_ep8" in result.stdout
    assert "shape_key=isl4000_osl2000_batch128" in result.stdout
    assert "max_num_batched_tokens=65536" in result.stdout


def test_unknown_scenario_fails_fast() -> None:
    result = _run_runner("preflight", "tp8ep8-4k2k-bt8000")

    assert result.returncode == 2
    assert "unknown_scenario=tp8ep8-4k2k-bt8000" in result.stderr
