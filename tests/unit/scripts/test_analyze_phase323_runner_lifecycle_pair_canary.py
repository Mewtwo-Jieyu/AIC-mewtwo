from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "analyze_phase323_runner_lifecycle_pair_canary.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase323_runner_lifecycle_pair_canary",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


CONTROL = "tp8ep8-12k2k-bt12000"
HOLDOUT = "tp8ep8-12k2k-bt65536"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _trace_row(scenario: str, iteration: int, phase: str, max_bt: int, worker_offset: int) -> dict[str, object]:
    context_tokens = max_bt if phase == "prefill" else 0
    decode_tokens = 0 if phase == "prefill" else 128
    context_reqs = 1 if phase == "prefill" else 0
    decode_reqs = 0 if phase == "prefill" else 128
    start = 1_000_000 + iteration * 10_000 + worker_offset
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
        "tp": 8,
        "dp": 1,
        "ep": 8,
        "topology_key": "tp8_dp1_ep8",
        "shape_key": "isl12000_osl2000_batch128",
        "iteration_start_ns": start,
        "forward_start_ns": start + 100,
        "forward_end_ns": start + 200,
        "iteration_end_ns": start + 300,
        "iteration_elapsed_ns": 300,
        "forward_elapsed_ns": 100,
        "active_request_count": 128,
        "scheduled_new_req_count": context_reqs,
        "scheduled_cached_req_count": decode_reqs,
        "scheduled_resumed_req_count": 0,
        "finished_req_count": 0,
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
    }


def _write_scenario(root: Path, scenario: str, *, max_bt: int, output_tok_s: float, phases: list[str]) -> None:
    directory = root / scenario
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(
        directory / "bench_result.json",
        {
            "ok_requests": 128,
            "failed_requests": 0,
            "output_tok_s": output_tok_s,
        },
    )
    (directory / "bench_records.jsonl").write_text(
        "".join(json.dumps({"ok": True, "i": i}) + "\n" for i in range(128)),
        encoding="utf-8",
    )
    with (directory / "deeper_scheduler_trace.jsonl").open("w", encoding="utf-8") as handle:
        for iteration, phase in enumerate(phases):
            for worker_offset in range(8):
                handle.write(json.dumps(_trace_row(scenario, iteration, phase, max_bt, worker_offset)) + "\n")
    _write_json(
        directory / "phase274_result.json",
        {
            "scenario": scenario,
            "shape_key": "isl12000_osl2000_batch128",
            "topology_key": "tp8_dp1_ep8",
            "diagnostic_only": True,
            "valid_for_default": False,
            "perf_database": False,
        },
    )
    (directory / f"run_one_{scenario}.log").write_text(
        "run_id=test-run\nrunner_pid=123\nbenchmark_exit_code=0\n",
        encoding="utf-8",
    )
    (directory / f"cleanup_{scenario}.log").write_text("cleanup_trap_done=now\n", encoding="utf-8")
    (directory / "gpu_compute_apps_after.txt").write_text("", encoding="utf-8")
    (directory / "gpu_compute_apps_drain.log").write_text("gpu_compute_apps_drain_stable=true\n", encoding="utf-8")


def _write_pair(root: Path) -> None:
    _write_scenario(
        root,
        CONTROL,
        max_bt=12000,
        output_tok_s=2689.6564050452644,
        phases=["prefill", "mixed", "mixed", "pure_decode", "pure_decode"],
    )
    _write_scenario(
        root,
        HOLDOUT,
        max_bt=65536,
        output_tok_s=2719.944278826508,
        phases=["prefill", "mixed", "mixed", "mixed", "mixed", "mixed", "pure_decode"],
    )


def test_runner_lifecycle_pair_canary_outputs_one_row(tmp_path: Path) -> None:
    _write_pair(tmp_path)

    rows = analyzer.analyze_runner_lifecycle_pair_canary(tmp_path)

    assert len(rows) == 1
    row = rows[0]
    assert row["source_sha"] == "13cbb5c650fd622e2655fed844d11464b5c2371c"
    assert row["worker"] == "worker-wrzh8"
    assert row["control_scenario"] == CONTROL
    assert row["holdout_scenario"] == HOLDOUT
    assert row["control_ok_requests"] == "128"
    assert row["holdout_ok_requests"] == "128"
    assert row["control_failed_requests"] == "0"
    assert row["holdout_failed_requests"] == "0"
    assert row["output_ratio"] == "1.011261"
    assert row["control_max_bt"] == "12000"
    assert row["holdout_max_bt"] == "65536"
    assert row["lock_released"] == "true"
    assert row["benchmark_failure_count"] == "0"
    assert row["cleanup_trap_done"] == "true"
    assert row["verdict"] == "runner_lifecycle_pair_canary_pass"
    assert row["evidence_status"] == "not_replacement_evidence"
    assert row["diagnostic_only"] == "true"
    assert row["valid_for_default"] == "false"
    assert row["perf_database"] == "false"


def test_writer_outputs_csv_and_doc(tmp_path: Path) -> None:
    _write_pair(tmp_path)
    rows = analyzer.analyze_runner_lifecycle_pair_canary(tmp_path)
    csv_path = tmp_path / "out.csv"
    doc_path = tmp_path / "out.md"

    analyzer.write_runner_lifecycle_pair_canary_csv(csv_path, rows)
    analyzer.write_runner_lifecycle_pair_canary_doc(doc_path, rows)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 1
    assert written[0]["output_ratio"] == "1.011261"
    doc = doc_path.read_text(encoding="utf-8")
    assert "runner_lifecycle_pair_canary_pass" in doc
    assert "not_replacement_evidence" in doc
    assert "Default AIC" in doc
    assert "No-Go" in doc


def test_bad_flags_fail_fast(tmp_path: Path) -> None:
    _write_pair(tmp_path)
    trace_path = tmp_path / HOLDOUT / "deeper_scheduler_trace.jsonl"
    lines = trace_path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["valid_for_default"] = True
    lines[0] = json.dumps(row)
    trace_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="flags"):
        analyzer.analyze_runner_lifecycle_pair_canary(tmp_path)


def test_lock_or_benchmark_failure_fails_fast(tmp_path: Path) -> None:
    _write_pair(tmp_path)
    (tmp_path / HOLDOUT / ".phase274_run.lock").mkdir()

    with pytest.raises(ValueError, match="lock"):
        analyzer.analyze_runner_lifecycle_pair_canary(tmp_path)

    (tmp_path / HOLDOUT / ".phase274_run.lock").rmdir()
    (tmp_path / HOLDOUT / "benchmark_failure_summary.txt").write_text("failed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="benchmark failure"):
        analyzer.analyze_runner_lifecycle_pair_canary(tmp_path)


def test_worker_payload_divergence_fails_fast(tmp_path: Path) -> None:
    _write_pair(tmp_path)
    trace_path = tmp_path / CONTROL / "deeper_scheduler_trace.jsonl"
    lines = trace_path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[1])
    row["scheduled_decode_tokens"] = 127
    row["scheduled_total_tokens"] = row["scheduled_context_tokens"] + row["scheduled_decode_tokens"]
    row["forward_token_count"] = row["scheduled_total_tokens"]
    lines[1] = json.dumps(row)
    trace_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="payload"):
        analyzer.analyze_runner_lifecycle_pair_canary(tmp_path)
