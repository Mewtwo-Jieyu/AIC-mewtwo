from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "collector" / "vllm" / "run_phase274_deeper_scheduler_trace.sh"

EXPECTED_SCENARIOS = {
    "tp8ep8-12k2k-bt12000",
    "tp8ep8-12k2k-bt65536",
    "tp4dp2ep8-12k2k-bt12000",
    "tp4dp2ep8-12k2k-bt65536",
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


def _write_h_style_patchable_vllm_source(tmp_path: Path) -> Path:
    source = tmp_path / "gpu_model_runner.py"
    source.write_text(
        "\n".join(
            [
                "import functools",
                "from vllm.v1.worker.utils import is_residual_scattered_for_sp",
                "",
                "def execute_model(self):",
                "        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens",
                "        with (",
                "            context_manager(),",
                "        ):",
                "            pass",
                "",
                "            num_tokens_padded = batch_desc.num_tokens",
                "            num_reqs_padded = (",
                "                batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs",
                "            )",
                "            ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(",
                "                batch_desc,",
                "            )",
                "",
                "            model_output = self._model_forward(",
                "                input_ids=input_ids,",
                "                positions=positions,",
                "                intermediate_tensors=intermediate_tensors,",
                "                inputs_embeds=inputs_embeds,",
                "                **model_kwargs,",
                "            )",
                "",
                "        if deferred_state_corrections_fn:",
                "            deferred_state_corrections_fn()",
                "",
                "        return None",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return source


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


def test_runner_uses_h_current_forward_end_anchor() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert '"                inputs_embeds=inputs_embeds,\\n"' in text
    assert '"                **model_kwargs,\\n"' in text
    assert (
        '"                intermediate_tensors=intermediate_tensors,\\n"\n'
        '    "            )\\n"'
    ) not in text


def test_patch_source_accepts_h_current_model_forward_anchor(tmp_path: Path) -> None:
    source = _write_h_style_patchable_vllm_source(tmp_path)
    env = dict(os.environ)
    env["PHASE274_GPU_MODEL_RUNNER_PATH"] = str(source)

    result = _run_runner("__test-patch-source", env=env)

    assert result.returncode == 0, result.stderr
    assert "phase274_patch_source=PASS" in result.stdout
    restored = source.read_text(encoding="utf-8")
    assert "AIC_PHASE274_DEEPER_TRACE_ROW" not in restored
    assert not Path(f"{source}.phase274.bak").exists()


def test_preflight_prints_four_12k2k_scenarios_and_raw_schema() -> None:
    result = _run_runner("preflight")

    assert result.returncode == 0, result.stderr
    assert "phase274_mode=preflight" in result.stdout
    assert "scenario_count=4" in result.stdout
    assert "gpu_benchmark_started=false" in result.stdout
    scenario_lines = [
        line.removeprefix("scenario=")
        for line in result.stdout.splitlines()
        if line.startswith("scenario=")
    ]
    assert set(scenario_lines) == EXPECTED_SCENARIOS
    assert result.stdout.count("serve_command=") == 4
    assert result.stdout.count("benchmark_command=") == 4
    schema_line = next(
        line for line in result.stdout.splitlines() if line.startswith("raw_trace_schema=")
    )
    trace_fields = set(schema_line.removeprefix("raw_trace_schema=").split(","))
    assert RAW_TRACE_FIELDS <= trace_fields
    assert "result_derived_fields=boundary_timeline,mixed_sequence,tail_decode_tokens" in result.stdout


@pytest.mark.parametrize(
    ("scenario", "topology_key", "max_bt"),
    [
        ("tp4dp2ep8-12k2k-bt12000", "tp4_dp2_ep8", 12000),
        ("tp4dp2ep8-12k2k-bt65536", "tp4_dp2_ep8", 65536),
    ],
)
def test_preflight_prints_tp4dp2_12k2k_scenarios(
    scenario: str,
    topology_key: str,
    max_bt: int,
) -> None:
    result = _run_runner("preflight", scenario)

    assert result.returncode == 0, result.stderr
    assert "scenario_count=1" in result.stdout
    assert f"scenario={scenario}" in result.stdout
    assert f"topology_key={topology_key}" in result.stdout
    assert "tp=4 dp=2 ep=8" in result.stdout
    assert f"max_num_batched_tokens={max_bt}" in result.stdout
    assert "--data-parallel-size 2" in result.stdout
    assert "gpu_benchmark_started=false" in result.stdout


def test_unknown_scenario_fails_fast() -> None:
    result = _run_runner("preflight", "tp8ep8-8k2k-bt65536")

    assert result.returncode == 2
    assert "unknown_scenario=tp8ep8-8k2k-bt65536" in result.stderr


def test_run_one_requires_explicit_gpu_authorization() -> None:
    result = _run_runner("run-one", "tp8ep8-12k2k-bt12000")

    assert result.returncode == 2
    assert "phase274_allow_gpu_run_required=true" in result.stderr


def _write_fake_ray_logs(ray_root: Path) -> None:
    logs = ray_root / "session_2026" / "logs"
    logs.mkdir(parents=True)
    (logs / "raylet.err").write_text(
        "\n".join(f"raylet err line {index}" for index in range(5)),
        encoding="utf-8",
    )
    (logs / "worker.out").write_text(
        "\n".join(f"worker out line {index}" for index in range(5)),
        encoding="utf-8",
    )


def test_startup_diagnostics_writes_failure_artifacts(tmp_path: Path) -> None:
    out_dir = tmp_path / "case"
    ray_root = tmp_path / "ray"
    _write_fake_ray_logs(ray_root)
    env = dict(os.environ)
    env.update(
        {
            "PHASE274_TEST_OUT_DIR": str(out_dir),
            "PHASE274_TEST_RAY_ROOT": str(ray_root),
            "WORKDIR": str(tmp_path),
        }
    )

    result = _run_runner(
        "__test-capture-startup-diagnostics",
        "tp8ep8-12k2k-bt12000",
        env=env,
    )

    assert result.returncode == 0, result.stderr
    expected_files = {
        "startup_failure_summary.txt",
        "startup_process_snapshot.txt",
        "startup_gpu_compute_apps.txt",
        "startup_nvidia_smi.txt",
        "startup_ray_logs_tail.txt",
        "startup_serve_tail.txt",
        "startup_ready_tail.txt",
    }
    assert expected_files <= {path.name for path in out_dir.iterdir()}
    summary = (out_dir / "startup_failure_summary.txt").read_text(encoding="utf-8")
    assert "scenario=tp8ep8-12k2k-bt12000" in summary
    assert "reason=test_startup_failure" in summary
    assert f"workdir={tmp_path}" in summary
    assert "service_pid=" in summary
    assert "port=18500" in summary
    assert "runner_path=" in summary
    ray_tail = (out_dir / "startup_ray_logs_tail.txt").read_text(encoding="utf-8")
    assert "ray_logs_found=2" in ray_tail
    assert "worker out line 4" in ray_tail
    assert "phase274_startup_diagnostics=PASS" in result.stdout


def test_startup_diagnostics_handles_missing_ray_logs(tmp_path: Path) -> None:
    out_dir = tmp_path / "case"
    env = dict(os.environ)
    env.update(
        {
            "PHASE274_TEST_OUT_DIR": str(out_dir),
            "PHASE274_TEST_RAY_ROOT": str(tmp_path / "missing-ray"),
            "WORKDIR": str(tmp_path),
        }
    )

    result = _run_runner(
        "__test-capture-startup-diagnostics",
        "tp8ep8-12k2k-bt65536",
        env=env,
    )

    assert result.returncode == 0, result.stderr
    ray_tail = (out_dir / "startup_ray_logs_tail.txt").read_text(encoding="utf-8")
    assert "ray_logs_found=0" in ray_tail


def test_wait_for_service_failure_path_references_startup_diagnostics() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert (
        'capture_startup_diagnostics "${out_dir}" "${scenario}" '
        '"service_exited_before_ready"'
    ) in text
    assert (
        'capture_startup_diagnostics "${out_dir}" "${scenario}" '
        '"service_ready_timeout"'
    ) in text


def test_startup_diagnostics_does_not_mask_primary_failure() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert (
        'capture_startup_diagnostics "${out_dir}" "${scenario}" '
        '"service_exited_before_ready" || true'
    ) in text
    assert (
        'capture_startup_diagnostics "${out_dir}" "${scenario}" '
        '"service_ready_timeout" || true'
    ) in text


def _valid_trace_row(
    iteration: int = 0,
    phase: str = "mixed",
    *,
    scenario: str = "tp8ep8-12k2k-bt12000",
    tp: int = 8,
    dp: int = 1,
    max_bt: int = 12000,
    topology_key: str = "tp8_dp1_ep8",
) -> dict[str, object]:
    context_tokens = 12000 if phase in {"prefill", "mixed"} else 0
    decode_tokens = 128 if phase in {"mixed", "pure_decode"} else 0
    context_reqs = 1 if context_tokens else 0
    decode_reqs = 128 if decode_tokens else 0
    return {
        "source": "phase274_deeper_scheduler_trace",
        "scenario": scenario,
        "iteration": iteration,
        "phase": phase,
        "scheduled_context_tokens": context_tokens,
        "scheduled_decode_tokens": decode_tokens,
        "scheduled_total_tokens": context_tokens + decode_tokens,
        "scheduled_context_reqs": context_reqs,
        "scheduled_decode_reqs": decode_reqs,
        "scheduled_total_reqs": context_reqs + decode_reqs,
        "max_num_batched_tokens": max_bt,
        "max_num_seqs": 256,
        "forward_token_count": context_tokens + decode_tokens,
        "tp": tp,
        "dp": dp,
        "ep": 8,
        "topology_key": topology_key,
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


def _worker_rows(row: dict[str, object], count: int) -> list[dict[str, object]]:
    rows = []
    for worker in range(count):
        worker_row = dict(row)
        worker_row["iteration_start_ns"] = int(worker_row["iteration_start_ns"]) + worker
        worker_row["forward_start_ns"] = int(worker_row["forward_start_ns"]) + worker
        worker_row["forward_end_ns"] = int(worker_row["forward_end_ns"]) + worker
        worker_row["iteration_end_ns"] = int(worker_row["iteration_end_ns"]) + worker
        worker_row["forward_elapsed_ns"] = int(worker_row["forward_elapsed_ns"]) + worker
        worker_row["iteration_elapsed_ns"] = int(worker_row["iteration_elapsed_ns"]) + worker
        rows.append(worker_row)
    return rows


def _tp8_iteration_rows(iteration: int, phase: str) -> list[dict[str, object]]:
    return _worker_rows(_valid_trace_row(iteration, phase), 8)


def _dp2_payload_row(
    iteration: int,
    phase: str,
    *,
    context_tokens: int,
    decode_tokens: int,
) -> dict[str, object]:
    row = _valid_trace_row(
        iteration,
        phase,
        scenario="tp4dp2ep8-12k2k-bt12000",
        tp=4,
        dp=2,
        max_bt=12000,
        topology_key="tp4_dp2_ep8",
    )
    context_reqs = 1 if context_tokens else 0
    decode_reqs = 64 if decode_tokens else 0
    row["scheduled_context_tokens"] = context_tokens
    row["scheduled_decode_tokens"] = decode_tokens
    row["scheduled_total_tokens"] = context_tokens + decode_tokens
    row["scheduled_context_reqs"] = context_reqs
    row["scheduled_decode_reqs"] = decode_reqs
    row["scheduled_total_reqs"] = context_reqs + decode_reqs
    row["forward_token_count"] = context_tokens + decode_tokens
    row["scheduled_new_req_count"] = context_reqs
    row["scheduled_cached_req_count"] = decode_reqs
    return row


def _tp4dp2_iteration_rows(
    iteration: int,
    phase: str,
    payloads: list[tuple[int, int]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for context_tokens, decode_tokens in payloads:
        rows.extend(
            _worker_rows(
                _dp2_payload_row(
                    iteration,
                    phase,
                    context_tokens=context_tokens,
                    decode_tokens=decode_tokens,
                ),
                4,
            )
        )
    return rows


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


def _run_trace_guard(
    tmp_path: Path,
    scenario: str = "tp8ep8-12k2k-bt12000",
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.update(
        {
            "PHASE274_TEST_BENCH_JSON": str(tmp_path / "bench_result.json"),
            "PHASE274_TEST_TRACE_JSONL": str(tmp_path / "deeper_scheduler_trace.jsonl"),
            "PHASE274_TEST_OUT_JSON": str(tmp_path / "phase274_result.json"),
        }
    )
    return _run_runner("__test-validate-trace", scenario, env=env)


def test_trace_guard_accepts_complete_synthetic_trace(tmp_path: Path) -> None:
    rows = [
        *_tp8_iteration_rows(0, "prefill"),
        *_tp8_iteration_rows(1, "mixed"),
        *_tp8_iteration_rows(2, "pure_decode"),
        *_tp8_iteration_rows(3, "pure_decode"),
    ]
    _write_synthetic_artifacts(tmp_path, rows)

    result = _run_trace_guard(tmp_path)

    assert result.returncode == 0, result.stderr
    assert "phase274_trace_guard=PASS" in result.stdout
    payload = json.loads((tmp_path / "phase274_result.json").read_text(encoding="utf-8"))
    assert payload["scheduler_trace"]["trace_rows"] == 32
    assert payload["scheduler_trace"]["unique_iterations"] == 4
    assert payload["scheduler_trace"]["max_payloads_per_iteration"] == 1
    assert payload["boundary_timeline"]["mixed_iterations"] == [1]
    assert payload["boundary_timeline"]["first_pure_decode_iteration"] == 2
    assert payload["boundary_timeline"]["tail_decode_tokens"] == [128, 128]
    assert payload["diagnostic_only"] is True
    assert payload["valid_for_default"] is False
    assert payload["perf_database"] is False


def test_trace_guard_rejects_token_sum_tamper(tmp_path: Path) -> None:
    rows = _tp8_iteration_rows(0, "mixed")
    rows[0]["scheduled_total_tokens"] = 9999
    _write_synthetic_artifacts(tmp_path, rows)

    result = _run_trace_guard(tmp_path)

    assert result.returncode == 1
    assert "trace_row_token_sum_mismatch" in result.stderr


def test_trace_guard_rejects_negative_timing(tmp_path: Path) -> None:
    rows = _tp8_iteration_rows(0, "mixed")
    rows[0]["forward_elapsed_ns"] = -1
    _write_synthetic_artifacts(tmp_path, rows)

    result = _run_trace_guard(tmp_path)

    assert result.returncode == 1
    assert "trace_row_non_negative_integer_mismatch" in result.stderr


def test_trace_guard_rejects_negative_queue_count(tmp_path: Path) -> None:
    rows = _tp8_iteration_rows(0, "mixed")
    rows[0]["scheduled_cached_req_count"] = -1
    _write_synthetic_artifacts(tmp_path, rows)

    result = _run_trace_guard(tmp_path)

    assert result.returncode == 1
    assert "trace_row_non_negative_integer_mismatch" in result.stderr


def test_trace_guard_rejects_flag_tamper(tmp_path: Path) -> None:
    rows = _tp8_iteration_rows(0, "mixed")
    rows[0]["valid_for_default"] = True
    _write_synthetic_artifacts(tmp_path, rows)

    result = _run_trace_guard(tmp_path)

    assert result.returncode == 1
    assert "trace_row_flag_mismatch" in result.stderr


def test_dp2_trace_guard_accepts_two_payloads_and_rank_sum_aggregation(
    tmp_path: Path,
) -> None:
    rows = [
        *_tp4dp2_iteration_rows(0, "prefill", [(7000, 0), (5000, 0)]),
        *_tp4dp2_iteration_rows(1, "mixed", [(6000, 32), (4000, 96)]),
        *_tp4dp2_iteration_rows(2, "pure_decode", [(0, 64), (0, 64)]),
    ]
    _write_synthetic_artifacts(tmp_path, rows)

    result = _run_trace_guard(tmp_path, "tp4dp2ep8-12k2k-bt12000")

    assert result.returncode == 0, result.stderr
    payload = json.loads((tmp_path / "phase274_result.json").read_text(encoding="utf-8"))
    assert payload["parallelism"] == {"tp": 4, "dp": 2, "ep": 8}
    assert payload["scheduler_trace"]["trace_rows"] == 24
    assert payload["scheduler_trace"]["unique_iterations"] == 3
    assert payload["scheduler_trace"]["max_payloads_per_iteration"] == 2
    assert payload["scheduler_trace"]["max_scheduled_total_tokens"] == 12000
    assert payload["scheduler_trace"]["max_forward_token_count"] == 12000
    assert payload["boundary_timeline"]["mixed_iterations"] == [1]
    assert payload["boundary_timeline"]["tail_decode_tokens"] == [128]


def test_dp2_trace_guard_rejects_three_payloads(tmp_path: Path) -> None:
    payload_a = _worker_rows(
        _dp2_payload_row(0, "mixed", context_tokens=6000, decode_tokens=32),
        3,
    )
    payload_b = _worker_rows(
        _dp2_payload_row(0, "mixed", context_tokens=4000, decode_tokens=96),
        3,
    )
    payload_c = _worker_rows(
        _dp2_payload_row(0, "mixed", context_tokens=1000, decode_tokens=16),
        2,
    )
    rows = [*payload_a, *payload_b, *payload_c]
    _write_synthetic_artifacts(tmp_path, rows)

    result = _run_trace_guard(tmp_path, "tp4dp2ep8-12k2k-bt12000")

    assert result.returncode == 1
    assert "trace_iteration_payload_count_mismatch" in result.stderr
