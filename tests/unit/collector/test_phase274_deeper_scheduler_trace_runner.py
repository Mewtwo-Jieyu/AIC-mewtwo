from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "collector" / "vllm" / "run_phase274_deeper_scheduler_trace.sh"

EXPECTED_SCENARIOS = {
    "tp8ep8-12k2k-bt12000",
    "tp8ep8-12k2k-bt65536",
}

RAW_TRACE_FIELDS = {
    "source",
    "scenario",
    "iteration",
    "phase",
    "scheduled_context_tokens",
    "scheduled_decode_tokens",
    "scheduled_total_tokens",
    "scheduled_context_reqs",
    "scheduled_decode_reqs",
    "scheduled_total_reqs",
    "max_num_batched_tokens",
    "max_num_seqs",
    "forward_token_count",
    "tp",
    "dp",
    "ep",
    "topology_key",
    "shape_key",
    "iteration_start_ns",
    "forward_start_ns",
    "forward_end_ns",
    "iteration_end_ns",
    "iteration_elapsed_ns",
    "forward_elapsed_ns",
    "active_request_count",
    "scheduled_new_req_count",
    "scheduled_cached_req_count",
    "scheduled_resumed_req_count",
    "finished_req_count",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
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


def _write_fake_vllm_source(tmp_path: Path) -> None:
    source = tmp_path / "vllm" / "v1" / "worker" / "gpu_model_runner.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "\n".join(
            [
                "import time",
                "scheduler_output",
                "batch_desc.num_tokens",
                "maybe_create_ubatch_slices",
                "",
            ]
        ),
        encoding="utf-8",
    )
    utils = tmp_path / "vllm" / "v1" / "utils.py"
    utils.parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "vllm" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "vllm" / "v1" / "__init__.py").write_text("", encoding="utf-8")
    utils.write_text(
        "def compute_iteration_details(_scheduler_output):\n"
        "    return None\n",
        encoding="utf-8",
    )
    output_module = tmp_path / "vllm" / "v1" / "core" / "sched" / "output.py"
    output_module.parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "vllm" / "v1" / "core" / "__init__.py").write_text(
        "", encoding="utf-8"
    )
    (tmp_path / "vllm" / "v1" / "core" / "sched" / "__init__.py").write_text(
        "", encoding="utf-8"
    )
    output_module.write_text(
        "class CachedRequestData:\n"
        "    num_reqs = 0\n"
        "    resumed_req_ids = ()\n",
        encoding="utf-8",
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


def test_source_check_finds_vllm_source_anchor(tmp_path: Path) -> None:
    _write_fake_vllm_source(tmp_path)

    env = dict(os.environ)
    env["WORKDIR"] = str(tmp_path)
    env["PYTHONPATH"] = str(tmp_path)
    result = _run_runner("source-check", env=env)

    assert result.returncode == 0, result.stderr
    assert "phase274_mode=source-check" in result.stdout
    assert "source_check=PASS" in result.stdout
    assert "vllm_source_present=true" in result.stdout
    assert "compute_iteration_details_import=true" in result.stdout
    assert "cached_request_data_num_reqs=true" in result.stdout
    assert "cached_request_data_resumed_req_ids=true" in result.stdout
    assert "patch_anchor_scheduler_output=true" in result.stdout
    assert "patch_anchor_forward_timing=true" in result.stdout
    assert "patch_anchor_maybe_create_ubatch_slices=true" in result.stdout
    assert "gpu_benchmark_started=false" in result.stdout


def test_runner_uses_real_cached_request_data_fields() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert "scheduler_output.scheduled_cached_reqs.num_reqs" in text
    assert "scheduler_output.scheduled_cached_reqs.resumed_req_ids" in text
    assert "len(scheduler_output.scheduled_cached_reqs)" not in text
    assert "scheduler_output.scheduled_resumed_reqs" not in text


def test_preflight_prints_two_tp8_12k2k_scenarios_and_raw_schema() -> None:
    result = _run_runner("preflight")

    assert result.returncode == 0, result.stderr
    assert "phase274_mode=preflight" in result.stdout
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
        line for line in result.stdout.splitlines() if line.startswith("raw_trace_schema=")
    )
    trace_fields = set(schema_line.removeprefix("raw_trace_schema=").split(","))
    assert RAW_TRACE_FIELDS <= trace_fields
    assert "result_derived_fields=boundary_timeline,mixed_sequence,tail_decode_tokens" in result.stdout


def test_unknown_scenario_fails_fast() -> None:
    result = _run_runner("preflight", "tp4dp2ep8-12k2k-bt65536")

    assert result.returncode == 2
    assert "unknown_scenario=tp4dp2ep8-12k2k-bt65536" in result.stderr


def test_run_one_requires_explicit_gpu_authorization() -> None:
    result = _run_runner("run-one", "tp8ep8-12k2k-bt12000")

    assert result.returncode == 2
    assert "phase274_allow_gpu_run_required=true" in result.stderr


def _valid_trace_row(iteration: int = 0, phase: str = "mixed") -> dict[str, object]:
    context_tokens = 12000 if phase in {"prefill", "mixed"} else 0
    decode_tokens = 128 if phase in {"mixed", "pure_decode"} else 0
    context_reqs = 1 if context_tokens else 0
    decode_reqs = 128 if decode_tokens else 0
    return {
        "source": "phase274_deeper_scheduler_trace",
        "scenario": "tp8ep8-12k2k-bt12000",
        "iteration": iteration,
        "phase": phase,
        "scheduled_context_tokens": context_tokens,
        "scheduled_decode_tokens": decode_tokens,
        "scheduled_total_tokens": context_tokens + decode_tokens,
        "scheduled_context_reqs": context_reqs,
        "scheduled_decode_reqs": decode_reqs,
        "scheduled_total_reqs": context_reqs + decode_reqs,
        "max_num_batched_tokens": 12000,
        "max_num_seqs": 256,
        "forward_token_count": context_tokens + decode_tokens,
        "tp": 8,
        "dp": 1,
        "ep": 8,
        "topology_key": "tp8_dp1_ep8",
        "shape_key": "isl12000_osl2000_batch128",
        "iteration_start_ns": iteration * 1000,
        "forward_start_ns": iteration * 1000 + 100,
        "forward_end_ns": iteration * 1000 + 700,
        "iteration_end_ns": iteration * 1000 + 900,
        "iteration_elapsed_ns": 900,
        "forward_elapsed_ns": 600,
        "active_request_count": 128,
        "scheduled_new_req_count": context_reqs,
        "scheduled_cached_req_count": decode_reqs,
        "scheduled_resumed_req_count": 0,
        "finished_req_count": 0,
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
    }


def _write_synthetic_artifacts(tmp_path: Path, rows: list[dict[str, object]]) -> None:
    bench = {
        "failed_requests": 0,
        "ok_requests": 128,
        "output_len": 2000,
        "max_concurrency": 128,
        "request_throughput": 1.0,
        "output_throughput": 100.0,
        "total_token_throughput": 300.0,
    }
    (tmp_path / "bench_result.json").write_text(json.dumps(bench), encoding="utf-8")
    (tmp_path / "deeper_scheduler_trace.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _run_trace_guard(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.update(
        {
            "PHASE274_TEST_BENCH_JSON": str(tmp_path / "bench_result.json"),
            "PHASE274_TEST_TRACE_JSONL": str(tmp_path / "deeper_scheduler_trace.jsonl"),
            "PHASE274_TEST_OUT_JSON": str(tmp_path / "phase274_result.json"),
        }
    )
    return _run_runner("__test-validate-trace", "tp8ep8-12k2k-bt12000", env=env)


def test_trace_guard_accepts_complete_synthetic_trace(tmp_path: Path) -> None:
    rows = [
        _valid_trace_row(0, "prefill"),
        _valid_trace_row(1, "mixed"),
        _valid_trace_row(2, "pure_decode"),
        _valid_trace_row(3, "pure_decode"),
    ]
    _write_synthetic_artifacts(tmp_path, rows)

    result = _run_trace_guard(tmp_path)

    assert result.returncode == 0, result.stderr
    assert "phase274_trace_guard=PASS" in result.stdout
    payload = json.loads((tmp_path / "phase274_result.json").read_text(encoding="utf-8"))
    assert payload["scheduler_trace"]["trace_rows"] == 4
    assert payload["boundary_timeline"]["mixed_iterations"] == [1]
    assert payload["boundary_timeline"]["first_pure_decode_iteration"] == 2
    assert payload["boundary_timeline"]["tail_decode_tokens"] == [128, 128]
    assert payload["diagnostic_only"] is True
    assert payload["valid_for_default"] is False
    assert payload["perf_database"] is False


def test_trace_guard_rejects_token_sum_tamper(tmp_path: Path) -> None:
    row = _valid_trace_row()
    row["scheduled_total_tokens"] = 9999
    _write_synthetic_artifacts(tmp_path, [row])

    result = _run_trace_guard(tmp_path)

    assert result.returncode == 1
    assert "trace_row_token_sum_mismatch" in result.stderr


def test_trace_guard_rejects_negative_timing(tmp_path: Path) -> None:
    row = _valid_trace_row()
    row["forward_elapsed_ns"] = -1
    _write_synthetic_artifacts(tmp_path, [row])

    result = _run_trace_guard(tmp_path)

    assert result.returncode == 1
    assert "trace_row_non_negative_integer_mismatch" in result.stderr


def test_trace_guard_rejects_negative_queue_count(tmp_path: Path) -> None:
    row = _valid_trace_row()
    row["scheduled_cached_req_count"] = -1
    _write_synthetic_artifacts(tmp_path, [row])

    result = _run_trace_guard(tmp_path)

    assert result.returncode == 1
    assert "trace_row_non_negative_integer_mismatch" in result.stderr


def test_trace_guard_rejects_flag_tamper(tmp_path: Path) -> None:
    row = _valid_trace_row()
    row["valid_for_default"] = True
    _write_synthetic_artifacts(tmp_path, [row])

    result = _run_trace_guard(tmp_path)

    assert result.returncode == 1
    assert "trace_row_flag_mismatch" in result.stderr
