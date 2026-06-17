from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "analyze_phase301_boundary_timeline_cadence_diagnostic.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase301_boundary_timeline_cadence_diagnostic",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


CONTROL = "tp4dp2ep8-12k2k-bt12000"
HOLDOUT = "tp4dp2ep8-12k2k-bt65536"
TRACE_SOURCE = "phase274_deeper_scheduler_trace"
TOPOLOGY_KEY = "tp4_dp2_ep8"
SHAPE_KEY = "isl12000_osl2000_batch128"


def _payloads(
    scenario: str,
    max_bt: int,
    iteration: int,
    role: str,
) -> list[tuple[int, int, int]]:
    if iteration == 0:
        if role == "holdout":
            return [(12000, 0, 12000), (12736, 0, 12736)]
        return [(12000, 0, 12000), (12000, 0, 12000)]
    if role == "control" and iteration in {1, 2}:
        return [(432, 1, 866), (576, 55, 1208)]
    if role == "holdout" and iteration == 2:
        return [(1776, 2, 1778), (0, 0, 128)]
    if role == "control":
        return [(0, 64, 160), (0, 64, 160)]
    return [(0, 64, 144), (0, 64, 144)]


def _phase(iteration: int, role: str) -> str:
    if iteration == 0:
        return "prefill"
    if role == "control" and iteration in {1, 2}:
        return "mixed"
    if role == "holdout" and iteration == 2:
        return "mixed"
    return "pure_decode"


def _write_case(
    deploy_root: Path,
    source_sha_name: str,
    source_sha: str,
    scenario: str,
    max_bt: int,
    output_tok_s: float,
    role: str,
    start_step_ns: int,
) -> Path:
    deploy_root.mkdir(parents=True)
    (deploy_root / source_sha_name).write_text(source_sha, encoding="utf-8")
    case_dir = (
        deploy_root
        / "docs"
        / "iter_gap_investigation"
        / "phase274_deeper_scheduler_trace"
        / scenario
    )
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
        "\n".join(json.dumps({"ok": 1, "request_index": index}) for index in range(128)) + "\n",
        encoding="utf-8",
    )
    (case_dir / "phase274_result.json").write_text(
        json.dumps(
            {
                "source": TRACE_SOURCE,
                "scenario": scenario,
                "topology_key": TOPOLOGY_KEY,
                "shape_key": SHAPE_KEY,
                "parallelism": {"tp": 4, "dp": 2, "ep": 8},
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
    for iteration in range(6):
        phase = _phase(iteration, role)
        start = 1_000_000_000 + iteration * start_step_ns
        for rank, (context_tokens, decode_tokens, forward_tokens) in enumerate(
            _payloads(scenario, max_bt, iteration, role)
        ):
            total_tokens = context_tokens + decode_tokens
            context_reqs = 1 if context_tokens else 0
            decode_reqs = 1 if decode_tokens else 0
            for worker in range(4):
                worker_offset = rank * 4 + worker
                forward_elapsed = 600_000 + worker_offset
                iteration_elapsed = 3_000_000 + worker_offset
                rows.append(
                    {
                        "source": TRACE_SOURCE,
                        "scenario": scenario,
                        "iteration": iteration,
                        "phase": phase,
                        "scheduled_context_tokens": context_tokens,
                        "scheduled_decode_tokens": decode_tokens,
                        "scheduled_total_tokens": total_tokens,
                        "scheduled_context_reqs": context_reqs,
                        "scheduled_decode_reqs": decode_reqs,
                        "scheduled_total_reqs": context_reqs + decode_reqs,
                        "max_num_batched_tokens": max_bt,
                        "max_num_seqs": 256,
                        "forward_token_count": forward_tokens,
                        "tp": 4,
                        "dp": 2,
                        "ep": 8,
                        "topology_key": TOPOLOGY_KEY,
                        "shape_key": SHAPE_KEY,
                        "iteration_start_ns": start + worker_offset,
                        "forward_start_ns": start + 100_000 + worker_offset,
                        "forward_end_ns": start + 100_000 + worker_offset + forward_elapsed,
                        "iteration_end_ns": start + worker_offset + iteration_elapsed,
                        "iteration_elapsed_ns": iteration_elapsed,
                        "forward_elapsed_ns": forward_elapsed,
                        "active_request_count": 64,
                        "scheduled_new_req_count": 64 if iteration == 0 else 0,
                        "scheduled_cached_req_count": 64 if iteration else 0,
                        "scheduled_resumed_req_count": 0,
                        "finished_req_count": 1 if iteration == 5 else 0,
                        "diagnostic_only": True,
                        "valid_for_default": False,
                        "perf_database": False,
                    }
                )
    (case_dir / "deeper_scheduler_trace.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    return case_dir


def _write_pair(tmp_path: Path) -> tuple[Path, Path]:
    control_dir = _write_case(
        tmp_path / "phase291_deeper_trace_preflight_65372730",
        ".phase291_source_sha",
        "65372730f59a99cbb237e951a9a27871a0f78bdb",
        CONTROL,
        12000,
        2039.8091937406505,
        "control",
        59_000_000,
    )
    holdout_dir = _write_case(
        tmp_path / "phase298_deeper_trace_preflight_d357f2d4",
        ".phase298_source_sha",
        "d357f2d428a35272acc3c5e0d9af09b792dfa4b0",
        HOLDOUT,
        65536,
        2696.6185770670454,
        "holdout",
        43_000_000,
    )
    return control_dir, holdout_dir


def test_boundary_timeline_cadence_outputs_pair_row(tmp_path: Path) -> None:
    control_dir, holdout_dir = _write_pair(tmp_path)

    rows = analyzer.analyze_boundary_timeline_cadence_diagnostic(control_dir, holdout_dir)

    assert len(rows) == 1
    row = rows[0]
    assert row["source"] == "phase301_boundary_timeline_cadence_diagnostic"
    assert row["pair_key"] == "tp4_dp2_ep8:isl12000_osl2000_batch128:control_bt12000:holdout_bt65536"
    assert row["source_sha_comparison"] == "startup_diagnostics_only"
    assert row["output_ratio"] == "1.321996"
    assert row["verdict"] == "boundary_timeline_explains_direction"
    assert row["mechanism_conclusion"] == "wall_span_iteration_cadence_diagnostic"
    assert row["default_readiness"] == "No-Go"
    assert row["diagnostic_only"] == "true"
    assert row["valid_for_default"] == "false"
    assert row["perf_database"] == "false"


def test_cadence_and_wall_span_fields_are_present(tmp_path: Path) -> None:
    control_dir, holdout_dir = _write_pair(tmp_path)

    row = analyzer.analyze_boundary_timeline_cadence_diagnostic(control_dir, holdout_dir)[0]

    assert row["control_trace_wall_span_s"] == "0.298000"
    assert row["holdout_trace_wall_span_s"] == "0.218000"
    assert row["control_iteration_start_delta_p50_ms"] == "59.000000"
    assert row["holdout_iteration_start_delta_p50_ms"] == "43.000000"
    assert row["overhead_p50_delta_ms"] == "0.000000"
    assert "overhead_reduction" not in row["mechanism_conclusion"]


def test_write_csv_and_doc(tmp_path: Path) -> None:
    control_dir, holdout_dir = _write_pair(tmp_path)
    rows = analyzer.analyze_boundary_timeline_cadence_diagnostic(control_dir, holdout_dir)
    out_csv = tmp_path / "phase301.csv"
    out_doc = tmp_path / "phase301.md"

    analyzer.write_boundary_timeline_cadence_csv(out_csv, rows)
    analyzer.write_boundary_timeline_cadence_doc(out_doc, rows)

    persisted = list(csv.DictReader(out_csv.open(newline="", encoding="utf-8")))
    assert len(persisted) == 1
    assert persisted[0]["verdict"] == "boundary_timeline_explains_direction"
    doc = out_doc.read_text(encoding="utf-8")
    assert "Boundary Timeline Cadence Diagnostic" in doc
    assert "wall_span_iteration_cadence_diagnostic" in doc
    assert "Default AIC | No-Go" in doc


def test_wrong_control_scenario_fails_fast(tmp_path: Path) -> None:
    control_dir, holdout_dir = _write_pair(tmp_path)
    result = control_dir / "phase274_result.json"
    value = json.loads(result.read_text(encoding="utf-8"))
    value["scenario"] = "tp4dp2ep8-8k2k-bt12000"
    result.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="result scenario mismatch"):
        analyzer.analyze_boundary_timeline_cadence_diagnostic(control_dir, holdout_dir)


def test_wrong_holdout_scenario_fails_fast(tmp_path: Path) -> None:
    control_dir, holdout_dir = _write_pair(tmp_path)
    result = holdout_dir / "phase274_result.json"
    value = json.loads(result.read_text(encoding="utf-8"))
    value["scenario"] = "tp4dp2ep8-8k2k-bt65536"
    result.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="result scenario mismatch"):
        analyzer.analyze_boundary_timeline_cadence_diagnostic(control_dir, holdout_dir)


def test_flags_fail_fast(tmp_path: Path) -> None:
    control_dir, holdout_dir = _write_pair(tmp_path)
    trace = control_dir / "deeper_scheduler_trace.jsonl"
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    rows[0]["valid_for_default"] = True
    trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="trace flag mismatch"):
        analyzer.analyze_boundary_timeline_cadence_diagnostic(control_dir, holdout_dir)


def test_dp2_payload_count_over_two_fails_fast(tmp_path: Path) -> None:
    control_dir, holdout_dir = _write_pair(tmp_path)
    trace = holdout_dir / "deeper_scheduler_trace.jsonl"
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    rows[0]["scheduled_decode_tokens"] = 7
    rows[0]["scheduled_total_tokens"] = rows[0]["scheduled_context_tokens"] + 7
    trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="payload count mismatch"):
        analyzer.analyze_boundary_timeline_cadence_diagnostic(control_dir, holdout_dir)


def test_worker_row_count_fails_fast(tmp_path: Path) -> None:
    control_dir, holdout_dir = _write_pair(tmp_path)
    trace = holdout_dir / "deeper_scheduler_trace.jsonl"
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    rows = rows[1:]
    trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="worker count mismatch"):
        analyzer.analyze_boundary_timeline_cadence_diagnostic(control_dir, holdout_dir)
