from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "collector" / "vllm" / "run_phase234_scheduler_trace.sh"

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


def _run_runner(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(RUNNER), *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _fake_cleanup_path(tmp_path: Path) -> tuple[Path, Path]:
    marker = tmp_path / "cleanup_marker.log"
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    for name in ("pkill", "ray"):
        script = fakebin / name
        script.write_text(
            "#!/usr/bin/env bash\n"
            f"printf '%s %s\\n' '{name}' \"$*\" >> '{marker}'\n"
            "exit 0\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
    return fakebin, marker


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
    assert "phase234_mode=preflight" in result.stdout
    assert "scenario_count=8" in result.stdout
    assert "scheduler_trace_started=false" in result.stdout
    assert "gpu_benchmark_started=false" in result.stdout
    scenario_lines = [
        line.removeprefix("scenario=")
        for line in result.stdout.splitlines()
        if line.startswith("scenario=")
    ]
    assert set(scenario_lines) == EXPECTED_SCENARIOS
    assert result.stdout.count("serve_command=") == 8
    assert result.stdout.count("benchmark_command=") == 8
    schema_line = next(
        line for line in result.stdout.splitlines() if line.startswith("trace_schema=")
    )
    trace_fields = set(schema_line.removeprefix("trace_schema=").split(","))
    assert EXPECTED_TRACE_FIELDS <= trace_fields


def test_source_check_confirms_patch_contract() -> None:
    result = _run_runner("source-check")

    assert result.returncode == 0, result.stderr
    assert "source_check=PASS" in result.stdout
    assert "patch_anchor_compute_iteration_details=true" in result.stdout
    assert "patch_anchor_scheduler_output=true" in result.stdout
    assert "trace_field_scheduled_total_tokens=true" in result.stdout
    assert "trace_field_phase=true" in result.stdout


def test_run_one_requires_explicit_gpu_authorization() -> None:
    result = _run_runner("run-one", "tp8ep8-4k2k-bt4000")

    assert result.returncode == 2
    assert "phase234_allow_gpu_run_required=true" in result.stderr


def test_unknown_scenario_fails_fast() -> None:
    result = _run_runner("preflight", "tp8ep8-4k2k-bt8000")

    assert result.returncode == 2
    assert "unknown_scenario=tp8ep8-4k2k-bt8000" in result.stderr


def test_cleanup_mode_calls_cleanup_commands(tmp_path: Path) -> None:
    fakebin, marker = _fake_cleanup_path(tmp_path)
    env = dict(os.environ)
    env["PATH"] = f"{fakebin}:{env['PATH']}"

    result = _run_runner("cleanup", env=env)

    assert result.returncode == 0, result.stderr
    marker_text = marker.read_text(encoding="utf-8")
    assert "pkill -f vllm.entrypoints.cli.main serve" in marker_text
    assert "ray stop --force" in marker_text


def _write_synthetic_artifacts(tmp_path: Path, trace_row: dict[str, object]) -> None:
    bench = {
        "failed_requests": 0,
        "ok_requests": 128,
        "output_len": 2000,
        "max_concurrency": 128,
        "request_throughput": 1.0,
        "output_throughput": 100.0,
        "total_token_throughput": 300.0,
    }
    (tmp_path / "bench_result.json").write_text(
        json.dumps(bench),
        encoding="utf-8",
    )
    (tmp_path / "scheduler_trace.jsonl").write_text(
        json.dumps(trace_row) + "\n",
        encoding="utf-8",
    )


def _valid_trace_row() -> dict[str, object]:
    return {
        "source": "phase234_vllm_scheduler_trace",
        "scenario": "tp8ep8-4k2k-bt4000",
        "iteration": 0,
        "phase": "mixed",
        "scheduled_context_tokens": 4000,
        "scheduled_decode_tokens": 128,
        "scheduled_total_tokens": 4128,
        "scheduled_context_reqs": 1,
        "scheduled_decode_reqs": 128,
        "scheduled_total_reqs": 129,
        "max_num_batched_tokens": 4000,
        "max_num_seqs": 256,
        "forward_token_count": 4128,
        "tp": 8,
        "dp": 1,
        "ep": 8,
        "topology_key": "tp8_dp1_ep8",
        "shape_key": "isl4000_osl2000_batch128",
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
    }


def _run_trace_guard(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.update(
        {
            "PHASE234_TEST_BENCH_JSON": str(tmp_path / "bench_result.json"),
            "PHASE234_TEST_TRACE_JSONL": str(tmp_path / "scheduler_trace.jsonl"),
            "PHASE234_TEST_OUT_JSON": str(tmp_path / "phase234_result.json"),
        }
    )
    return _run_runner("__test-validate-trace", "tp8ep8-4k2k-bt4000", env=env)


def test_trace_guard_accepts_complete_synthetic_trace(tmp_path: Path) -> None:
    _write_synthetic_artifacts(tmp_path, _valid_trace_row())

    result = _run_trace_guard(tmp_path)

    assert result.returncode == 0, result.stderr
    assert "phase234_trace_guard=PASS" in result.stdout
    payload = json.loads((tmp_path / "phase234_result.json").read_text(encoding="utf-8"))
    assert payload["scheduler_trace"]["trace_rows"] == 1


def test_trace_guard_rejects_scenario_tamper(tmp_path: Path) -> None:
    row = _valid_trace_row()
    row["scenario"] = "tp8ep8-4k2k-bt65536"
    _write_synthetic_artifacts(tmp_path, row)

    result = _run_trace_guard(tmp_path)

    assert result.returncode == 1
    assert "trace_row_scenario_mismatch" in result.stderr


def test_trace_guard_rejects_flag_tamper(tmp_path: Path) -> None:
    row = _valid_trace_row()
    row["valid_for_default"] = True
    _write_synthetic_artifacts(tmp_path, row)

    result = _run_trace_guard(tmp_path)

    assert result.returncode == 1
    assert "trace_row_flag_mismatch" in result.stderr


def test_trace_guard_rejects_token_sum_tamper(tmp_path: Path) -> None:
    row = _valid_trace_row()
    row["scheduled_total_tokens"] = 9999
    _write_synthetic_artifacts(tmp_path, row)

    result = _run_trace_guard(tmp_path)

    assert result.returncode == 1
    assert "trace_row_token_sum_mismatch" in result.stderr
