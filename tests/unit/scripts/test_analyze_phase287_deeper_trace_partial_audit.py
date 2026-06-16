from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "analyze_phase287_deeper_trace_partial_audit.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase287_deeper_trace_partial_audit",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


CONTROL = "tp8ep8-12k2k-bt12000"
HOLDOUT = "tp8ep8-12k2k-bt65536"


def _phase(iteration: int, mixed_iterations: set[int]) -> str:
    if iteration == 0:
        return "prefill"
    if iteration in mixed_iterations:
        return "mixed"
    return "pure_decode"


def _tokens(iteration: int, total_iterations: int, mixed_iterations: set[int]) -> tuple[int, int]:
    if iteration == 0:
        return 12000, 0
    if iteration in mixed_iterations:
        return 64, 64
    tail = {
        total_iterations - 5: 128,
        total_iterations - 4: 128,
        total_iterations - 3: 127,
        total_iterations - 2: 127,
        total_iterations - 1: 12,
    }
    return 0, tail.get(iteration, 128)


def _write_case(
    root: Path,
    scenario: str,
    max_bt: int,
    output_tok_s: float,
    total_iterations: int,
    mixed_iterations: set[int],
    timing_base: int,
) -> None:
    case_dir = root / scenario
    case_dir.mkdir(parents=True)
    (case_dir / "bench_result.json").write_text(
        json.dumps(
            {
                "ok_requests": 128,
                "failed_requests": 0,
                "output_tok_s": output_tok_s,
            }
        ),
        encoding="utf-8",
    )
    (case_dir / "bench_records.jsonl").write_text(
        "\n".join(json.dumps({"ok": 1, "request_index": idx}) for idx in range(128)) + "\n",
        encoding="utf-8",
    )
    (case_dir / "phase274_result.json").write_text(
        json.dumps(
            {
                "source": "phase274_deeper_scheduler_trace",
                "scenario": scenario,
                "topology_key": "tp8_dp1_ep8",
                "shape_key": "isl12000_osl2000_batch128",
                "parallelism": {"tp": 8, "dp": 1, "ep": 8},
                "shape": {
                    "isl": 12000,
                    "osl": 2000,
                    "batch_size": 128,
                    "max_num_batched_tokens": max_bt,
                    "max_num_seqs": 256,
                },
                "diagnostic_only": True,
                "valid_for_default": False,
                "perf_database": False,
            }
        ),
        encoding="utf-8",
    )

    rows = []
    for iteration in range(total_iterations):
        context_tokens, decode_tokens = _tokens(iteration, total_iterations, mixed_iterations)
        total_tokens = context_tokens + decode_tokens
        for worker in range(8):
            forward_elapsed = timing_base + iteration * 10 + worker
            iteration_elapsed = timing_base + 1000 + iteration * 20 + worker * 3
            rows.append(
                {
                    "source": "phase274_deeper_scheduler_trace",
                    "scenario": scenario,
                    "iteration": iteration,
                    "phase": _phase(iteration, mixed_iterations),
                    "scheduled_context_tokens": context_tokens,
                    "scheduled_decode_tokens": decode_tokens,
                    "scheduled_total_tokens": total_tokens,
                    "scheduled_context_reqs": 1 if context_tokens else 0,
                    "scheduled_decode_reqs": decode_tokens if decode_tokens else 0,
                    "scheduled_total_reqs": (1 if context_tokens else 0) + (decode_tokens if decode_tokens else 0),
                    "max_num_batched_tokens": max_bt,
                    "max_num_seqs": 256,
                    "forward_token_count": total_tokens,
                    "tp": 8,
                    "dp": 1,
                    "ep": 8,
                    "topology_key": "tp8_dp1_ep8",
                    "shape_key": "isl12000_osl2000_batch128",
                    "iteration_start_ns": 1000000 + iteration * 10000 + worker,
                    "forward_start_ns": 1001000 + iteration * 10000 + worker,
                    "forward_end_ns": 1001000 + iteration * 10000 + worker + forward_elapsed,
                    "iteration_end_ns": 1001000 + iteration * 10000 + worker + iteration_elapsed,
                    "iteration_elapsed_ns": iteration_elapsed,
                    "forward_elapsed_ns": forward_elapsed,
                    "active_request_count": 128,
                    "scheduled_new_req_count": 128 if iteration == 0 else 0,
                    "scheduled_cached_req_count": 128 if iteration else 0,
                    "scheduled_resumed_req_count": 0,
                    "finished_req_count": 1 if iteration == total_iterations - 1 else 0,
                    "diagnostic_only": True,
                    "valid_for_default": False,
                    "perf_database": False,
                }
            )
    (case_dir / "deeper_scheduler_trace.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_pair(root: Path) -> None:
    _write_case(
        root,
        CONTROL,
        12000,
        2849.909890929967,
        10,
        {2, 3},
        100,
    )
    _write_case(
        root,
        HOLDOUT,
        65536,
        2762.4186955329096,
        16,
        {2, 3, 4, 8, 9},
        200,
    )


def test_deeper_trace_partial_audit_outputs_pair_row(tmp_path: Path) -> None:
    _write_pair(tmp_path)
    rows = analyzer.analyze_deeper_trace_partial_audit(tmp_path)

    assert len(rows) == 1
    row = rows[0]
    assert row["source"] == "phase287_deeper_trace_partial_audit"
    assert row["pair_key"] == "tp8_dp1_ep8:isl12000_osl2000_batch128:control_bt12000:holdout_bt65536"
    assert row["control_scenario"] == CONTROL
    assert row["holdout_scenario"] == HOLDOUT
    assert row["output_ratio"] == "0.969300"
    assert row["unique_iteration_delta"] == "6"
    assert row["mixed_iteration_delta"] == "3"
    assert row["holdout_max_fill"] == "0.183105"
    assert row["verdict"] == "partial_only"
    assert row["mechanism_conclusion"] == "boundary_mixed_overhead_partial_only"
    assert row["default_readiness"] == "No-Go"
    assert row["diagnostic_only"] == "true"
    assert row["valid_for_default"] == "false"
    assert row["perf_database"] == "false"


def test_rank_max_timing_is_used_for_delta(tmp_path: Path) -> None:
    _write_pair(tmp_path)
    row = analyzer.analyze_deeper_trace_partial_audit(tmp_path)[0]

    assert row["forward_elapsed_p50_delta_ns"] == "130"
    assert row["iteration_elapsed_p50_delta_ns"] == "160"
    assert row["overhead_p50_delta_ns"] == "30"


def test_write_csv_and_doc(tmp_path: Path) -> None:
    _write_pair(tmp_path)
    rows = analyzer.analyze_deeper_trace_partial_audit(tmp_path)
    out_csv = tmp_path / "phase287.csv"
    out_doc = tmp_path / "phase287.md"

    analyzer.write_deeper_trace_partial_audit_csv(out_csv, rows)
    analyzer.write_deeper_trace_partial_audit_doc(out_doc, rows)

    persisted = list(csv.DictReader(out_csv.open(newline="", encoding="utf-8")))
    assert len(persisted) == 1
    assert persisted[0]["verdict"] == "partial_only"
    doc = out_doc.read_text(encoding="utf-8")
    assert "Deeper Trace Partial Audit" in doc
    assert "boundary_mixed_overhead_partial_only" in doc
    assert "Default AIC | No-Go" in doc


def test_wrong_scenario_fails_fast(tmp_path: Path) -> None:
    _write_pair(tmp_path)
    trace = tmp_path / HOLDOUT / "deeper_scheduler_trace.jsonl"
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    rows[0]["scenario"] = "tp8ep8-8k2k-bt65536"
    trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="trace scenario mismatch"):
        analyzer.analyze_deeper_trace_partial_audit(tmp_path)


def test_trace_flags_fail_fast(tmp_path: Path) -> None:
    _write_pair(tmp_path)
    trace = tmp_path / CONTROL / "deeper_scheduler_trace.jsonl"
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    rows[0]["valid_for_default"] = True
    trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="trace flag mismatch"):
        analyzer.analyze_deeper_trace_partial_audit(tmp_path)


def test_token_sum_fails_fast(tmp_path: Path) -> None:
    _write_pair(tmp_path)
    trace = tmp_path / CONTROL / "deeper_scheduler_trace.jsonl"
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    rows[0]["scheduled_total_tokens"] += 1
    trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="trace token sum mismatch"):
        analyzer.analyze_deeper_trace_partial_audit(tmp_path)


def test_iteration_gap_fails_fast(tmp_path: Path) -> None:
    _write_pair(tmp_path)
    trace = tmp_path / CONTROL / "deeper_scheduler_trace.jsonl"
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if row["iteration"] != 5]
    trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="trace iterations must be contiguous"):
        analyzer.analyze_deeper_trace_partial_audit(tmp_path)


def test_worker_payload_divergence_fails_fast(tmp_path: Path) -> None:
    _write_pair(tmp_path)
    trace = tmp_path / CONTROL / "deeper_scheduler_trace.jsonl"
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    rows[1]["scheduled_decode_tokens"] += 1
    rows[1]["scheduled_total_tokens"] += 1
    rows[1]["scheduled_total_reqs"] += 1
    rows[1]["scheduled_decode_reqs"] += 1
    rows[1]["forward_token_count"] += 1
    trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="scheduler payload divergence"):
        analyzer.analyze_deeper_trace_partial_audit(tmp_path)
