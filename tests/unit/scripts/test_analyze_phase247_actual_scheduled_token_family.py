from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "analyze_phase247_actual_scheduled_token_family.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase247_actual_scheduled_token_family",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


TP8_CONTROL = "tp8ep8-4k2k-bt4000"
TP8_HOLDOUT = "tp8ep8-4k2k-bt65536"
TP4_CONTROL = "tp4dp2ep8-4k2k-bt4000"
TP4_HOLDOUT = "tp4dp2ep8-4k2k-bt65536"


def _trace_row(
    scenario: str,
    *,
    iteration: int,
    phase: str,
    context_tokens: int,
    decode_tokens: int,
    context_reqs: int,
    decode_reqs: int,
    max_bt: int,
    tp: int,
    dp: int,
    ep: int,
    topology_key: str,
    forward_token_count: int | None = None,
) -> dict[str, object]:
    total_tokens = context_tokens + decode_tokens
    return {
        "source": "phase234_vllm_scheduler_trace",
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
        "forward_token_count": total_tokens if forward_token_count is None else forward_token_count,
        "tp": tp,
        "dp": dp,
        "ep": ep,
        "topology_key": topology_key,
        "shape_key": "isl4000_osl2000_batch128",
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
    }


def _tp8_rows(scenario: str, max_bt: int) -> list[dict[str, object]]:
    rows = [
        _trace_row(
            scenario,
            iteration=0,
            phase="prefill",
            context_tokens=4000,
            decode_tokens=0,
            context_reqs=1,
            decode_reqs=0,
            max_bt=max_bt,
            tp=8,
            dp=1,
            ep=8,
            topology_key="tp8_dp1_ep8",
        ),
        _trace_row(
            scenario,
            iteration=1,
            phase="pure_decode",
            context_tokens=0,
            decode_tokens=128,
            context_reqs=0,
            decode_reqs=128,
            max_bt=max_bt,
            tp=8,
            dp=1,
            ep=8,
            topology_key="tp8_dp1_ep8",
        ),
    ]
    return [row for row in rows for _ in range(8)]


def _tp4_rows(scenario: str, max_bt: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rank_a = _trace_row(
        scenario,
        iteration=0,
        phase="prefill",
        context_tokens=4000,
        decode_tokens=0,
        context_reqs=1,
        decode_reqs=0,
        max_bt=max_bt,
        tp=4,
        dp=2,
        ep=8,
        topology_key="tp4_dp2_ep8",
    )
    rank_b = _trace_row(
        scenario,
        iteration=0,
        phase="prefill",
        context_tokens=4240 if max_bt == 65536 else 4000,
        decode_tokens=0,
        context_reqs=16 if max_bt == 65536 else 1,
        decode_reqs=0,
        max_bt=max_bt,
        tp=4,
        dp=2,
        ep=8,
        topology_key="tp4_dp2_ep8",
    )
    decode_a = _trace_row(
        scenario,
        iteration=1,
        phase="pure_decode",
        context_tokens=0,
        decode_tokens=63,
        context_reqs=0,
        decode_reqs=63,
        max_bt=max_bt,
        tp=4,
        dp=2,
        ep=8,
        topology_key="tp4_dp2_ep8",
    )
    decode_b = _trace_row(
        scenario,
        iteration=1,
        phase="pure_decode",
        context_tokens=0,
        decode_tokens=65,
        context_reqs=0,
        decode_reqs=65,
        max_bt=max_bt,
        tp=4,
        dp=2,
        ep=8,
        topology_key="tp4_dp2_ep8",
    )
    for row in (rank_a, rank_b, decode_a, decode_b):
        rows.extend([row] * 4)
    return rows


def _write_case(
    root: Path,
    scenario: str,
    max_bt: int,
    output_tok_s: float,
    rows: list[dict[str, object]],
    *,
    topology_key: str,
    parallelism: dict[str, int],
    result_overrides: dict[str, object] | None = None,
) -> None:
    case_dir = root / scenario
    case_dir.mkdir(parents=True)
    bench = {
        "ok_requests": 128,
        "failed_requests": 0,
        "output_tok_s": output_tok_s,
        "total_tok_s": output_tok_s * 3,
    }
    result = {
        "scenario": scenario,
        "topology_key": topology_key,
        "shape_key": "isl4000_osl2000_batch128",
        "shape": {
            "isl": 4000,
            "osl": 2000,
            "batch_size": 128,
            "max_num_batched_tokens": max_bt,
            "max_num_seqs": 256,
        },
        "parallelism": parallelism,
        "bench_result": {"ok_requests": 128, "failed_requests": 0},
        "scheduler_trace": {"trace_rows": len(rows)},
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
    }
    result.update(result_overrides or {})
    (case_dir / "bench_result.json").write_text(json.dumps(bench), encoding="utf-8")
    (case_dir / "bench_records.jsonl").write_text("\n".join("{}" for _ in range(128)) + "\n", encoding="utf-8")
    (case_dir / "phase234_result.json").write_text(json.dumps(result), encoding="utf-8")
    (case_dir / "scheduler_trace.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_family(root: Path) -> None:
    _write_case(
        root,
        TP8_CONTROL,
        4000,
        100.0,
        _tp8_rows(TP8_CONTROL, 4000),
        topology_key="tp8_dp1_ep8",
        parallelism={"tp": 8, "dp": 1, "ep": 8},
    )
    _write_case(
        root,
        TP8_HOLDOUT,
        65536,
        99.0,
        _tp8_rows(TP8_HOLDOUT, 65536),
        topology_key="tp8_dp1_ep8",
        parallelism={"tp": 8, "dp": 1, "ep": 8},
    )
    _write_case(
        root,
        TP4_CONTROL,
        4000,
        80.0,
        _tp4_rows(TP4_CONTROL, 4000),
        topology_key="tp4_dp2_ep8",
        parallelism={"tp": 4, "dp": 2, "ep": 8},
    )
    _write_case(
        root,
        TP4_HOLDOUT,
        65536,
        88.0,
        _tp4_rows(TP4_HOLDOUT, 65536),
        topology_key="tp4_dp2_ep8",
        parallelism={"tp": 4, "dp": 2, "ep": 8},
    )


def test_actual_scheduled_token_family_outputs_four_rows(tmp_path: Path) -> None:
    _write_family(tmp_path)
    out_csv = tmp_path / "phase247.csv"
    out_doc = tmp_path / "phase247.md"

    rows = analyzer.analyze_actual_scheduled_token_family(tmp_path)
    analyzer.write_actual_scheduled_token_family_csv(out_csv, rows)
    analyzer.write_actual_scheduled_token_family_doc(out_doc, rows)

    assert [row["scenario"] for row in rows] == [TP8_CONTROL, TP8_HOLDOUT, TP4_CONTROL, TP4_HOLDOUT]
    assert {row["source"] for row in rows} == {"phase247_actual_scheduled_token_family"}
    assert {row["mechanism_hypothesis"] for row in rows} == {
        "actual_scheduled_tokens_not_configured_budget"
    }
    assert {row["default_readiness"] for row in rows} == {"No-Go"}
    assert {row["diagnostic_only"] for row in rows} == {"true"}
    assert {row["valid_for_default"] for row in rows} == {"false"}
    assert {row["perf_database"] for row in rows} == {"false"}

    tp8_holdout = rows[1]
    assert tp8_holdout["topology_key"] == "tp8_dp1_ep8"
    assert tp8_holdout["role"] == "holdout"
    assert tp8_holdout["max_bt"] == "65536"
    assert tp8_holdout["output_ratio_vs_control"] == "0.990000"
    assert tp8_holdout["rank_sum_p50_scheduled_total_tokens"] == "128.000000"
    assert tp8_holdout["rank_sum_max_scheduled_total_tokens"] == "4000"
    assert tp8_holdout["rank_sum_max_fill_ratio"] == "0.061035"

    tp4_holdout = rows[3]
    assert tp4_holdout["topology_key"] == "tp4_dp2_ep8"
    assert tp4_holdout["output_ratio_vs_control"] == "1.100000"
    assert tp4_holdout["rank_min_p50_scheduled_total_tokens"] == "63.000000"
    assert tp4_holdout["rank_max_p50_scheduled_total_tokens"] == "65.000000"
    assert tp4_holdout["rank_sum_p50_scheduled_total_tokens"] == "128.000000"
    assert tp4_holdout["rank_sum_max_scheduled_total_tokens"] == "8240"
    assert tp4_holdout["rank_sum_max_fill_ratio"] == "0.125732"
    assert out_csv.read_text(encoding="utf-8").count("\n") == 5
    doc = out_doc.read_text(encoding="utf-8")
    assert "DP=2 rows are aggregated as rank min/max/sum" in doc
    assert "Default AIC | No-Go" in doc


def test_pair_must_be_complete(tmp_path: Path) -> None:
    _write_family(tmp_path)
    case_dir = tmp_path / TP4_HOLDOUT
    for child in case_dir.iterdir():
        child.unlink()
    case_dir.rmdir()

    with pytest.raises(ValueError, match="missing scenario"):
        analyzer.analyze_actual_scheduled_token_family(tmp_path)


def test_tp8_duplicate_payload_mismatch_fails_fast(tmp_path: Path) -> None:
    _write_family(tmp_path)
    trace_path = tmp_path / TP8_HOLDOUT / "scheduler_trace.jsonl"
    lines = trace_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["scheduled_context_tokens"] = 3999
    first["scheduled_total_tokens"] = 3999
    first["forward_token_count"] = 3999
    lines[0] = json.dumps(first)
    trace_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="tp8 duplicate worker trace payload mismatch"):
        analyzer.analyze_actual_scheduled_token_family(tmp_path)


def test_tp4dp2_allows_distinct_dp_payloads(tmp_path: Path) -> None:
    _write_family(tmp_path)

    rows = analyzer.analyze_actual_scheduled_token_family(tmp_path)

    tp4_holdout = rows[3]
    assert tp4_holdout["rank_min_max_scheduled_total_tokens"] == "4000"
    assert tp4_holdout["rank_max_max_scheduled_total_tokens"] == "4240"
    assert tp4_holdout["rank_sum_max_scheduled_total_tokens"] == "8240"


def test_flag_tamper_fails_fast(tmp_path: Path) -> None:
    _write_family(tmp_path)
    trace_path = tmp_path / TP4_HOLDOUT / "scheduler_trace.jsonl"
    lines = trace_path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["valid_for_default"] = True
    lines[0] = json.dumps(row)
    trace_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="trace flag mismatch"):
        analyzer.analyze_actual_scheduled_token_family(tmp_path)


def test_token_sum_tamper_fails_fast(tmp_path: Path) -> None:
    _write_family(tmp_path)
    trace_path = tmp_path / TP4_HOLDOUT / "scheduler_trace.jsonl"
    lines = trace_path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["scheduled_total_tokens"] = 9999
    lines[0] = json.dumps(row)
    trace_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="trace token sum mismatch"):
        analyzer.analyze_actual_scheduled_token_family(tmp_path)


def test_holdout_must_not_fill_configured_budget(tmp_path: Path) -> None:
    _write_family(tmp_path)
    trace_path = tmp_path / TP4_HOLDOUT / "scheduler_trace.jsonl"
    rows = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["iteration"] == 0:
            row["scheduled_context_tokens"] = 40000
            row["scheduled_total_tokens"] = 40000
            row["forward_token_count"] = 40000
        rows.append(row)
    trace_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="holdout rank-sum max scheduled tokens must stay far below configured budget"):
        analyzer.analyze_actual_scheduled_token_family(tmp_path)
