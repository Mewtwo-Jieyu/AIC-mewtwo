from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "analyze_phase242_actual_scheduled_token_mechanism.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase242_actual_scheduled_token_mechanism",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


CONTROL_SCENARIO = "tp8ep8-4k2k-bt4000"
HOLDOUT_SCENARIO = "tp8ep8-4k2k-bt65536"


def _trace_row(
    scenario: str,
    max_bt: int,
    iteration: int,
    phase: str,
    context_tokens: int,
    decode_tokens: int,
    context_reqs: int,
    decode_reqs: int,
) -> dict[str, object]:
    return {
        "source": "phase234_vllm_scheduler_trace",
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
        "shape_key": "isl4000_osl2000_batch128",
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
    }


def _trace_rows(scenario: str, max_bt: int) -> list[dict[str, object]]:
    return [
        _trace_row(scenario, max_bt, 0, "prefill", 4000, 0, 1, 0),
        _trace_row(scenario, max_bt, 1, "mixed", 2032, 1, 127, 1),
        _trace_row(scenario, max_bt, 2, "pure_decode", 0, 128, 0, 128),
        _trace_row(scenario, max_bt, 3, "pure_decode", 0, 128, 0, 128),
    ]


def _write_case(
    root: Path,
    scenario: str,
    max_bt: int,
    output_tok_s: float,
    rows: list[dict[str, object]] | None = None,
    *,
    result_overrides: dict[str, object] | None = None,
) -> None:
    case_dir = root / scenario
    case_dir.mkdir(parents=True)
    trace_rows = rows if rows is not None else _trace_rows(scenario, max_bt)
    bench = {
        "ok_requests": 128,
        "failed_requests": 0,
        "output_tok_s": output_tok_s,
        "total_tok_s": output_tok_s * 3,
    }
    result = {
        "scenario": scenario,
        "topology_key": "tp8_dp1_ep8",
        "shape_key": "isl4000_osl2000_batch128",
        "shape": {
            "isl": 4000,
            "osl": 2000,
            "batch_size": 128,
            "max_num_batched_tokens": max_bt,
            "max_num_seqs": 256,
        },
        "parallelism": {"tp": 8, "dp": 1, "ep": 8},
        "bench_result": {"ok_requests": 128, "failed_requests": 0},
        "scheduler_trace": {"trace_rows": len(trace_rows)},
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
    }
    result.update(result_overrides or {})
    (case_dir / "bench_result.json").write_text(json.dumps(bench), encoding="utf-8")
    (case_dir / "bench_records.jsonl").write_text("\n".join("{}" for _ in range(128)) + "\n", encoding="utf-8")
    (case_dir / "phase234_result.json").write_text(json.dumps(result), encoding="utf-8")
    (case_dir / "scheduler_trace.jsonl").write_text(
        "\n".join(json.dumps(row) for row in trace_rows) + "\n",
        encoding="utf-8",
    )


def _write_pair(root: Path) -> None:
    _write_case(root, CONTROL_SCENARIO, 4000, 100.0)
    _write_case(root, HOLDOUT_SCENARIO, 65536, 99.0)


def test_actual_scheduled_token_mechanism_outputs_pair_rows(tmp_path: Path) -> None:
    _write_pair(tmp_path)
    out_csv = tmp_path / "phase242.csv"
    out_doc = tmp_path / "phase242.md"

    rows = analyzer.analyze_actual_scheduled_token_mechanism(tmp_path)
    analyzer.write_actual_scheduled_token_csv(out_csv, rows)
    analyzer.write_actual_scheduled_token_doc(out_doc, rows)

    assert len(rows) == 2
    control, holdout = rows
    assert control["source"] == "phase242_actual_scheduled_token_mechanism"
    assert control["scenario"] == CONTROL_SCENARIO
    assert control["max_num_batched_tokens"] == "4000"
    assert control["max_scheduled_total_tokens"] == "4000"
    assert control["output_ratio_vs_control"] == "1.000000"
    assert holdout["scenario"] == HOLDOUT_SCENARIO
    assert holdout["max_num_batched_tokens"] == "65536"
    assert holdout["max_scheduled_total_tokens"] == "4000"
    assert holdout["max_budget_fill_ratio"] == "0.061035"
    assert holdout["output_ratio_vs_control"] == "0.990000"
    assert {row["mechanism_hypothesis"] for row in rows} == {
        "actual_scheduled_tokens_not_configured_budget"
    }
    assert {row["default_readiness"] for row in rows} == {"No-Go"}
    assert {row["diagnostic_only"] for row in rows} == {"true"}
    assert {row["valid_for_default"] for row in rows} == {"false"}
    assert {row["perf_database"] for row in rows} == {"false"}
    assert out_csv.read_text(encoding="utf-8").count("\n") == 3
    doc = out_doc.read_text(encoding="utf-8")
    assert "max_num_batched_tokens is a ceiling" in doc
    assert "actual_scheduled_tokens_not_configured_budget" in doc
    assert "Default AIC | No-Go" in doc


def test_pair_must_be_complete(tmp_path: Path) -> None:
    _write_case(tmp_path, CONTROL_SCENARIO, 4000, 100.0)

    with pytest.raises(ValueError, match="missing scenario"):
        analyzer.analyze_actual_scheduled_token_mechanism(tmp_path)


def test_topology_shape_must_match(tmp_path: Path) -> None:
    _write_case(tmp_path, CONTROL_SCENARIO, 4000, 100.0)
    rows = _trace_rows(HOLDOUT_SCENARIO, 65536)
    for row in rows:
        row["shape_key"] = "isl12000_osl2000_batch128"
    _write_case(
        tmp_path,
        HOLDOUT_SCENARIO,
        65536,
        99.0,
        rows,
        result_overrides={"shape_key": "isl12000_osl2000_batch128"},
    )

    with pytest.raises(ValueError, match="control and holdout must share topology/shape"):
        analyzer.analyze_actual_scheduled_token_mechanism(tmp_path)


def test_flag_tamper_fails_fast(tmp_path: Path) -> None:
    _write_case(tmp_path, CONTROL_SCENARIO, 4000, 100.0)
    rows = _trace_rows(HOLDOUT_SCENARIO, 65536)
    rows[0]["valid_for_default"] = True
    _write_case(tmp_path, HOLDOUT_SCENARIO, 65536, 99.0, rows)

    with pytest.raises(ValueError, match="trace flag mismatch"):
        analyzer.analyze_actual_scheduled_token_mechanism(tmp_path)


def test_holdout_must_not_fill_configured_budget(tmp_path: Path) -> None:
    _write_case(tmp_path, CONTROL_SCENARIO, 4000, 100.0)
    rows = _trace_rows(HOLDOUT_SCENARIO, 65536)
    rows[0]["scheduled_context_tokens"] = 50000
    rows[0]["scheduled_total_tokens"] = 50000
    rows[0]["forward_token_count"] = 50000
    _write_case(tmp_path, HOLDOUT_SCENARIO, 65536, 99.0, rows)

    with pytest.raises(ValueError, match="holdout max scheduled tokens must stay far below configured budget"):
        analyzer.analyze_actual_scheduled_token_mechanism(tmp_path)


def test_positive_token_distribution_required(tmp_path: Path) -> None:
    _write_case(tmp_path, CONTROL_SCENARIO, 4000, 100.0)
    rows = [_trace_row(HOLDOUT_SCENARIO, 65536, 0, "pure_decode", 0, 0, 0, 0)]
    _write_case(tmp_path, HOLDOUT_SCENARIO, 65536, 99.0, rows)

    with pytest.raises(ValueError, match="p99_scheduled_total_tokens must be positive"):
        analyzer.analyze_actual_scheduled_token_mechanism(tmp_path)
