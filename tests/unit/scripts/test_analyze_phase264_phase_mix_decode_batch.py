from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "analyze_phase264_phase_mix_decode_batch.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase264_phase_mix_decode_batch",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


TP8_4K_CONTROL = "tp8ep8-4k2k-bt4000"
TP8_4K_HOLDOUT = "tp8ep8-4k2k-bt65536"
TP4_4K_CONTROL = "tp4dp2ep8-4k2k-bt4000"
TP4_4K_HOLDOUT = "tp4dp2ep8-4k2k-bt65536"
TP8_12K_CONTROL = "tp8ep8-12k2k-bt12000"
TP8_12K_HOLDOUT = "tp8ep8-12k2k-bt65536"
TP4_12K_CONTROL = "tp4dp2ep8-12k2k-bt12000"
TP4_12K_HOLDOUT = "tp4dp2ep8-12k2k-bt65536"

EXPECTED_SCENARIOS = [
    TP8_4K_CONTROL,
    TP8_4K_HOLDOUT,
    TP4_4K_CONTROL,
    TP4_4K_HOLDOUT,
    TP8_12K_CONTROL,
    TP8_12K_HOLDOUT,
    TP4_12K_CONTROL,
    TP4_12K_HOLDOUT,
]


def _trace_row(
    scenario: str,
    *,
    iteration: int,
    phase: str,
    context_tokens: int,
    decode_tokens: int,
    max_bt: int,
    tp: int,
    dp: int,
    topology_key: str,
    shape_key: str,
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
        "scheduled_context_reqs": 1 if context_tokens else 0,
        "scheduled_decode_reqs": decode_tokens if decode_tokens else 0,
        "scheduled_total_reqs": (1 if context_tokens else 0) + (decode_tokens if decode_tokens else 0),
        "max_num_batched_tokens": max_bt,
        "max_num_seqs": 256,
        "forward_token_count": total_tokens,
        "tp": tp,
        "dp": dp,
        "ep": 8,
        "topology_key": topology_key,
        "shape_key": shape_key,
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
    }


def _repeat(row: dict[str, object], count: int) -> list[dict[str, object]]:
    return [dict(row) for _ in range(count)]


def _tp8_rows(scenario: str, max_bt: int, shape_key: str, prefill_tokens: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in (
        _trace_row(
            scenario,
            iteration=0,
            phase="prefill",
            context_tokens=prefill_tokens,
            decode_tokens=0,
            max_bt=max_bt,
            tp=8,
            dp=1,
            topology_key="tp8_dp1_ep8",
            shape_key=shape_key,
        ),
        _trace_row(
            scenario,
            iteration=1,
            phase="mixed",
            context_tokens=prefill_tokens // 2,
            decode_tokens=64,
            max_bt=max_bt,
            tp=8,
            dp=1,
            topology_key="tp8_dp1_ep8",
            shape_key=shape_key,
        ),
        _trace_row(
            scenario,
            iteration=2,
            phase="pure_decode",
            context_tokens=0,
            decode_tokens=128,
            max_bt=max_bt,
            tp=8,
            dp=1,
            topology_key="tp8_dp1_ep8",
            shape_key=shape_key,
        ),
        _trace_row(
            scenario,
            iteration=3,
            phase="pure_decode",
            context_tokens=0,
            decode_tokens=96,
            max_bt=max_bt,
            tp=8,
            dp=1,
            topology_key="tp8_dp1_ep8",
            shape_key=shape_key,
        ),
    ):
        rows.extend(_repeat(row, 8))
    return rows


def _tp4_rows(
    scenario: str,
    max_bt: int,
    shape_key: str,
    prefill_a: int,
    prefill_b: int,
    decode_a: int = 63,
    decode_b: int = 65,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    per_rank_rows = [
        _trace_row(
            scenario,
            iteration=0,
            phase="prefill",
            context_tokens=prefill_a,
            decode_tokens=0,
            max_bt=max_bt,
            tp=4,
            dp=2,
            topology_key="tp4_dp2_ep8",
            shape_key=shape_key,
        ),
        _trace_row(
            scenario,
            iteration=0,
            phase="prefill",
            context_tokens=prefill_b,
            decode_tokens=0,
            max_bt=max_bt,
            tp=4,
            dp=2,
            topology_key="tp4_dp2_ep8",
            shape_key=shape_key,
        ),
        _trace_row(
            scenario,
            iteration=1,
            phase="prefill",
            context_tokens=prefill_a // 2,
            decode_tokens=0,
            max_bt=max_bt,
            tp=4,
            dp=2,
            topology_key="tp4_dp2_ep8",
            shape_key=shape_key,
        ),
        _trace_row(
            scenario,
            iteration=1,
            phase="pure_decode",
            context_tokens=0,
            decode_tokens=decode_b,
            max_bt=max_bt,
            tp=4,
            dp=2,
            topology_key="tp4_dp2_ep8",
            shape_key=shape_key,
        ),
        _trace_row(
            scenario,
            iteration=2,
            phase="pure_decode",
            context_tokens=0,
            decode_tokens=decode_a,
            max_bt=max_bt,
            tp=4,
            dp=2,
            topology_key="tp4_dp2_ep8",
            shape_key=shape_key,
        ),
        _trace_row(
            scenario,
            iteration=2,
            phase="pure_decode",
            context_tokens=0,
            decode_tokens=decode_b,
            max_bt=max_bt,
            tp=4,
            dp=2,
            topology_key="tp4_dp2_ep8",
            shape_key=shape_key,
        ),
    ]
    for row in per_rank_rows:
        rows.extend(_repeat(row, 4))
    return rows


def _write_case(
    root: Path,
    scenario: str,
    max_bt: int,
    output_tok_s: float,
    rows: list[dict[str, object]],
    *,
    topology_key: str,
    shape_key: str,
    parallelism: dict[str, int],
    isl: int,
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
        "shape_key": shape_key,
        "shape": {
            "isl": isl,
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
        TP8_4K_CONTROL,
        4000,
        100.0,
        _tp8_rows(TP8_4K_CONTROL, 4000, "isl4000_osl2000_batch128", 4000),
        topology_key="tp8_dp1_ep8",
        shape_key="isl4000_osl2000_batch128",
        parallelism={"tp": 8, "dp": 1, "ep": 8},
        isl=4000,
    )
    _write_case(
        root,
        TP8_4K_HOLDOUT,
        65536,
        99.0,
        _tp8_rows(TP8_4K_HOLDOUT, 65536, "isl4000_osl2000_batch128", 4000),
        topology_key="tp8_dp1_ep8",
        shape_key="isl4000_osl2000_batch128",
        parallelism={"tp": 8, "dp": 1, "ep": 8},
        isl=4000,
    )
    _write_case(
        root,
        TP4_4K_CONTROL,
        4000,
        80.0,
        _tp4_rows(TP4_4K_CONTROL, 4000, "isl4000_osl2000_batch128", 4000, 4000),
        topology_key="tp4_dp2_ep8",
        shape_key="isl4000_osl2000_batch128",
        parallelism={"tp": 4, "dp": 2, "ep": 8},
        isl=4000,
    )
    _write_case(
        root,
        TP4_4K_HOLDOUT,
        65536,
        88.0,
        _tp4_rows(TP4_4K_HOLDOUT, 65536, "isl4000_osl2000_batch128", 4000, 4240),
        topology_key="tp4_dp2_ep8",
        shape_key="isl4000_osl2000_batch128",
        parallelism={"tp": 4, "dp": 2, "ep": 8},
        isl=4000,
    )
    _write_case(
        root,
        TP8_12K_CONTROL,
        12000,
        70.0,
        _tp8_rows(TP8_12K_CONTROL, 12000, "isl12000_osl2000_batch128", 12000),
        topology_key="tp8_dp1_ep8",
        shape_key="isl12000_osl2000_batch128",
        parallelism={"tp": 8, "dp": 1, "ep": 8},
        isl=12000,
    )
    _write_case(
        root,
        TP8_12K_HOLDOUT,
        65536,
        67.0,
        _tp8_rows(TP8_12K_HOLDOUT, 65536, "isl12000_osl2000_batch128", 12000),
        topology_key="tp8_dp1_ep8",
        shape_key="isl12000_osl2000_batch128",
        parallelism={"tp": 8, "dp": 1, "ep": 8},
        isl=12000,
    )
    _write_case(
        root,
        TP4_12K_CONTROL,
        12000,
        60.0,
        _tp4_rows(TP4_12K_CONTROL, 12000, "isl12000_osl2000_batch128", 12000, 12000),
        topology_key="tp4_dp2_ep8",
        shape_key="isl12000_osl2000_batch128",
        parallelism={"tp": 4, "dp": 2, "ep": 8},
        isl=12000,
    )
    _write_case(
        root,
        TP4_12K_HOLDOUT,
        65536,
        81.0,
        _tp4_rows(TP4_12K_HOLDOUT, 65536, "isl12000_osl2000_batch128", 12000, 12000),
        topology_key="tp4_dp2_ep8",
        shape_key="isl12000_osl2000_batch128",
        parallelism={"tp": 4, "dp": 2, "ep": 8},
        isl=12000,
    )


def test_phase_mix_decode_batch_outputs_eight_rows(tmp_path: Path) -> None:
    _write_family(tmp_path)
    out_csv = tmp_path / "phase264.csv"
    out_doc = tmp_path / "phase264.md"

    rows = analyzer.analyze_phase_mix_decode_batch(tmp_path)
    analyzer.write_phase_mix_decode_batch_csv(out_csv, rows)
    analyzer.write_phase_mix_decode_batch_doc(out_doc, rows)

    assert [row["scenario"] for row in rows] == EXPECTED_SCENARIOS
    assert {row["source"] for row in rows} == {"phase264_phase_mix_decode_batch"}
    assert {row["mechanism_hypothesis"] for row in rows} == {
        "phase_mix_decode_batch_diagnostic"
    }
    assert {row["default_readiness"] for row in rows} == {"No-Go"}
    assert {row["diagnostic_only"] for row in rows} == {"true"}
    assert {row["valid_for_default"] for row in rows} == {"false"}
    assert {row["perf_database"] for row in rows} == {"false"}

    by_scenario = {row["scenario"]: row for row in rows}
    tp4_holdout = by_scenario[TP4_12K_HOLDOUT]
    assert tp4_holdout["configured_budget_aggregate"] == "131072"
    assert tp4_holdout["prefill_iterations"] == "1"
    assert tp4_holdout["mixed_iterations"] == "1"
    assert tp4_holdout["pure_decode_iterations"] == "1"
    assert tp4_holdout["mixed_ratio"] == "0.333333"
    assert tp4_holdout["pure_decode_rank_sum_p99_decode_tokens"] == "128.000000"
    assert tp4_holdout["pure_decode_rank_sum_max_decode_tokens"] == "128"
    assert tp4_holdout["mixed_rank_sum_p99_total_tokens"] == "6065.000000"
    assert tp4_holdout["mixed_rank_sum_max_total_tokens"] == "6065"
    assert tp4_holdout["rank_sum_max_fill_ratio"] == "0.183105"

    assert out_csv.read_text(encoding="utf-8").count("\n") == 9
    doc = out_doc.read_text(encoding="utf-8")
    assert "Phase264: Phase Mix / Decode Batch Diagnostic" in doc
    assert "Default AIC | No-Go" in doc
    assert "phase mix and decode batch" in doc


def test_missing_scenario_fails_fast(tmp_path: Path) -> None:
    _write_family(tmp_path)
    case_dir = tmp_path / TP4_12K_HOLDOUT
    for child in case_dir.iterdir():
        child.unlink()
    case_dir.rmdir()

    with pytest.raises(ValueError, match="missing scenario"):
        analyzer.analyze_phase_mix_decode_batch(tmp_path)


def test_invalid_phase_fails_fast(tmp_path: Path) -> None:
    _write_family(tmp_path)
    trace_path = tmp_path / TP4_12K_HOLDOUT / "scheduler_trace.jsonl"
    lines = trace_path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["phase"] = "unknown"
    lines[0] = json.dumps(row)
    trace_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="trace phase mismatch"):
        analyzer.analyze_phase_mix_decode_batch(tmp_path)


def test_tp4dp2_too_many_distinct_payloads_fails_fast(tmp_path: Path) -> None:
    _write_family(tmp_path)
    trace_path = tmp_path / TP4_12K_HOLDOUT / "scheduler_trace.jsonl"
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    extra_a = dict(rows[0])
    extra_a["scheduled_context_tokens"] = 11999
    extra_a["scheduled_total_tokens"] = 11999
    extra_a["forward_token_count"] = 11999
    extra_b = dict(rows[0])
    extra_b["scheduled_context_tokens"] = 11998
    extra_b["scheduled_total_tokens"] = 11998
    extra_b["forward_token_count"] = 11998
    rows.extend(dict(extra_a) for _ in range(4))
    rows.extend(dict(extra_b) for _ in range(4))
    trace_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="more distinct DP payloads than dp"):
        analyzer.analyze_phase_mix_decode_batch(tmp_path)


def test_holdout_budget_ceiling_guard_fails_fast(tmp_path: Path) -> None:
    _write_family(tmp_path)
    trace_path = tmp_path / TP4_12K_HOLDOUT / "scheduler_trace.jsonl"
    rows = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["iteration"] == 0:
            row["scheduled_context_tokens"] = 65536
            row["scheduled_total_tokens"] = 65536
            row["forward_token_count"] = 65536
        rows.append(row)
    trace_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="holdout rank-sum max scheduled tokens must stay below aggregate configured budget"):
        analyzer.analyze_phase_mix_decode_batch(tmp_path)


def test_actual_phase264_csv_can_be_loaded_if_present() -> None:
    csv_path = (
        Path(__file__).resolve().parents[3]
        / "docs"
        / "iter_gap_investigation"
        / "phase264_phase_mix_decode_batch.csv"
    )
    if not csv_path.exists():
        pytest.skip("Phase264 output is generated after analyzer implementation")

    rows = analyzer.read_phase_mix_decode_batch_csv(csv_path)

    assert len(rows) == 8
    assert rows[-1]["scenario"] == TP4_12K_HOLDOUT
    assert rows[-1]["default_readiness"] == "No-Go"
    assert rows[-1]["valid_for_default"] == "false"
