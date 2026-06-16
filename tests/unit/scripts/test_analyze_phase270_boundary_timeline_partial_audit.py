from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "analyze_phase270_boundary_timeline_partial_audit.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase270_boundary_timeline_partial_audit",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


EXPECTED_PAIR_KEYS = [
    "tp8_dp1_ep8:isl4000_osl2000_batch128:control_bt4000:holdout_bt65536",
    "tp4_dp2_ep8:isl4000_osl2000_batch128:control_bt4000:holdout_bt65536",
    "tp8_dp1_ep8:isl12000_osl2000_batch128:control_bt12000:holdout_bt65536",
    "tp4_dp2_ep8:isl12000_osl2000_batch128:control_bt12000:holdout_bt65536",
]

EXPECTED_VERDICTS = {
    EXPECTED_PAIR_KEYS[0]: "boundary_explains",
    EXPECTED_PAIR_KEYS[1]: "boundary_explains",
    EXPECTED_PAIR_KEYS[2]: "needs_deeper_trace",
    EXPECTED_PAIR_KEYS[3]: "boundary_partial",
}


SCENARIO_SPECS = {
    "tp8ep8-4k2k-bt4000": {
        "tp": 8,
        "dp": 1,
        "ep": 8,
        "max_bt": 4000,
        "topology_key": "tp8_dp1_ep8",
        "shape_key": "isl4000_osl2000_batch128",
        "output_tok_s": 4071.651448,
        "total_iterations": 2002,
        "context_iters": [(0, 4000, 0), (2, 2032, 1)],
        "tail_decode": [128, 128, 128, 127, 127],
    },
    "tp8ep8-4k2k-bt65536": {
        "tp": 8,
        "dp": 1,
        "ep": 8,
        "max_bt": 65536,
        "topology_key": "tp8_dp1_ep8",
        "shape_key": "isl4000_osl2000_batch128",
        "output_tok_s": 4027.060291,
        "total_iterations": 2003,
        "context_iters": [(0, 4000, 0), (2, 1552, 1), (3, 480, 98)],
        "tail_decode": [128, 128, 127, 127, 30],
    },
    "tp4dp2ep8-4k2k-bt4000": {
        "tp": 4,
        "dp": 2,
        "ep": 8,
        "max_bt": 4000,
        "topology_key": "tp4_dp2_ep8",
        "shape_key": "isl4000_osl2000_batch128",
        "output_tok_s": 3143.891608,
        "total_iterations": 2002,
        "context_iters": [(0, 8000, 0), (1, 240, 2), (2, 1776, 17)],
        "tail_decode": [128, 128, 128, 126, 111],
    },
    "tp4dp2ep8-4k2k-bt65536": {
        "tp": 4,
        "dp": 2,
        "ep": 8,
        "max_bt": 65536,
        "topology_key": "tp4_dp2_ep8",
        "shape_key": "isl4000_osl2000_batch128",
        "output_tok_s": 3445.355421,
        "total_iterations": 2002,
        "context_iters": [(0, 8240, 0), (2, 1776, 17)],
        "tail_decode": [128, 128, 128, 111, 111],
    },
    "tp8ep8-12k2k-bt12000": {
        "tp": 8,
        "dp": 1,
        "ep": 8,
        "max_bt": 12000,
        "topology_key": "tp8_dp1_ep8",
        "shape_key": "isl12000_osl2000_batch128",
        "output_tok_s": 2777.513790,
        "total_iterations": 2008,
        "context_iters": [(0, 12000, 0), (2, 1040, 1), (4, 512, 66), (7, 160, 98), (8, 320, 108)],
        "tail_decode": [62, 30, 30, 30, 20],
    },
    "tp8ep8-12k2k-bt65536": {
        "tp": 8,
        "dp": 1,
        "ep": 8,
        "max_bt": 65536,
        "topology_key": "tp8_dp1_ep8",
        "shape_key": "isl12000_osl2000_batch128",
        "output_tok_s": 2687.912821,
        "total_iterations": 2004,
        "context_iters": [(0, 12000, 0), (2, 1264, 1), (3, 288, 80), (4, 480, 98)],
        "tail_decode": [128, 127, 127, 48, 30],
    },
    "tp4dp2ep8-12k2k-bt12000": {
        "tp": 4,
        "dp": 2,
        "ep": 8,
        "max_bt": 12000,
        "topology_key": "tp4_dp2_ep8",
        "shape_key": "isl12000_osl2000_batch128",
        "output_tok_s": 2091.977129,
        "total_iterations": 2002,
        "context_iters": [(0, 24000, 0), (1, 864, 2), (2, 1152, 56)],
        "tail_decode": [128, 128, 128, 126, 72],
    },
    "tp4dp2ep8-12k2k-bt65536": {
        "tp": 4,
        "dp": 2,
        "ep": 8,
        "max_bt": 65536,
        "topology_key": "tp4_dp2_ep8",
        "shape_key": "isl12000_osl2000_batch128",
        "output_tok_s": 2802.211032,
        "total_iterations": 2003,
        "context_iters": [(0, 24000, 0), (2, 1776, 2), (3, 240, 113)],
        "tail_decode": [128, 128, 126, 126, 15],
    },
}


def _actual_phase264_csv() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "docs"
        / "iter_gap_investigation"
        / "phase264_phase_mix_decode_batch.csv"
    )


def _actual_phase267_csv() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "docs"
        / "iter_gap_investigation"
        / "phase267_mixed_phase_partial_audit.csv"
    )


def _trace_row(
    scenario: str,
    spec: dict[str, object],
    iteration: int,
    context_tokens: int,
    decode_tokens: int,
) -> dict[str, object]:
    return {
        "source": "phase234_vllm_scheduler_trace",
        "scenario": scenario,
        "iteration": iteration,
        "phase": "mixed" if context_tokens and decode_tokens else "prefill" if context_tokens else "pure_decode",
        "scheduled_context_tokens": context_tokens,
        "scheduled_decode_tokens": decode_tokens,
        "scheduled_total_tokens": context_tokens + decode_tokens,
        "scheduled_context_reqs": 1 if context_tokens else 0,
        "scheduled_decode_reqs": decode_tokens if decode_tokens else 0,
        "scheduled_total_reqs": (1 if context_tokens else 0) + (decode_tokens if decode_tokens else 0),
        "max_num_batched_tokens": spec["max_bt"],
        "max_num_seqs": 256,
        "forward_token_count": context_tokens + decode_tokens,
        "tp": spec["tp"],
        "dp": spec["dp"],
        "ep": spec["ep"],
        "topology_key": spec["topology_key"],
        "shape_key": spec["shape_key"],
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
    }


def _split_rank_sum(total: int, dp: int) -> list[int]:
    base = total // dp
    values = [base] * dp
    for idx in range(total - base * dp):
        values[idx] += 1
    return values


def _write_trace_artifact(root: Path, scenario: str, spec: dict[str, object]) -> None:
    case_dir = root / scenario
    case_dir.mkdir(parents=True)
    total_iterations = int(spec["total_iterations"])
    tail_start = total_iterations - 5
    context_by_iter = {
        int(iteration): (int(context_tokens), int(decode_tokens))
        for iteration, context_tokens, decode_tokens in spec["context_iters"]
    }
    tail_by_iter = {
        iteration: int(decode)
        for iteration, decode in zip(range(tail_start, total_iterations), spec["tail_decode"])
    }
    rows = []
    for iteration in range(total_iterations):
        if iteration in context_by_iter:
            context_tokens, decode_tokens = context_by_iter[iteration]
        else:
            context_tokens = 0
            decode_tokens = tail_by_iter.get(iteration, 128)
        if int(spec["dp"]) == 1:
            row = _trace_row(scenario, spec, iteration, context_tokens, decode_tokens)
            rows.extend(dict(row) for _ in range(int(spec["tp"])))
        else:
            context_parts = _split_rank_sum(context_tokens, int(spec["dp"]))
            decode_parts = _split_rank_sum(decode_tokens, int(spec["dp"]))
            for context_part, decode_part in zip(context_parts, decode_parts):
                row = _trace_row(scenario, spec, iteration, context_part, decode_part)
                rows.extend(dict(row) for _ in range(int(spec["tp"])))

    bench = {
        "ok_requests": 128,
        "failed_requests": 0,
        "output_tok_s": spec["output_tok_s"],
    }
    result = {
        "scenario": scenario,
        "topology_key": spec["topology_key"],
        "shape_key": spec["shape_key"],
        "parallelism": {"tp": spec["tp"], "dp": spec["dp"], "ep": spec["ep"]},
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
    }
    (case_dir / "bench_result.json").write_text(json.dumps(bench), encoding="utf-8")
    (case_dir / "phase234_result.json").write_text(json.dumps(result), encoding="utf-8")
    (case_dir / "scheduler_trace.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_trace_family(root: Path) -> None:
    for scenario, spec in SCENARIO_SPECS.items():
        _write_trace_artifact(root, scenario, spec)


def test_boundary_timeline_partial_audit_outputs_four_rows(tmp_path: Path) -> None:
    trace_root = tmp_path / "trace"
    _write_trace_family(trace_root)
    out_csv = tmp_path / "phase270.csv"
    out_doc = tmp_path / "phase270.md"

    rows = analyzer.analyze_boundary_timeline_partial_audit(
        trace_root,
        _actual_phase264_csv(),
        _actual_phase267_csv(),
    )
    analyzer.write_boundary_timeline_partial_audit_csv(out_csv, rows)
    analyzer.write_boundary_timeline_partial_audit_doc(out_doc, rows)

    assert [row["pair_key"] for row in rows] == EXPECTED_PAIR_KEYS
    assert {row["source"] for row in rows} == {"phase270_boundary_timeline_partial_audit"}
    assert {row["conclusion"] for row in rows} == {"boundary_timeline_partial_only"}
    assert {row["default_readiness"] for row in rows} == {"No-Go"}
    assert {row["diagnostic_only"] for row in rows} == {"true"}
    assert {row["valid_for_default"] for row in rows} == {"false"}
    assert {row["perf_database"] for row in rows} == {"false"}

    by_key = {row["pair_key"]: row for row in rows}
    assert {key: row["boundary_verdict"] for key, row in by_key.items()} == EXPECTED_VERDICTS

    tp8_12k = by_key[EXPECTED_PAIR_KEYS[2]]
    assert tp8_12k["control_mixed_iterations"] == "2,4,7,8"
    assert tp8_12k["holdout_mixed_iterations"] == "2,3,4"
    assert tp8_12k["control_first_pure_iteration"] == "1"
    assert tp8_12k["holdout_first_pure_iteration"] == "1"
    assert tp8_12k["total_iteration_delta"] == "-4"
    assert tp8_12k["tail_decode_tokens_control"] == "62,30,30,30,20"
    assert tp8_12k["tail_decode_tokens_holdout"] == "128,127,127,48,30"
    assert tp8_12k["next_mechanism"] == "deeper_trace_for_tp8_12k2k"

    tp4_12k = by_key[EXPECTED_PAIR_KEYS[3]]
    assert tp4_12k["control_first_pure_iteration"] == "3"
    assert tp4_12k["holdout_first_pure_iteration"] == "1"
    assert tp4_12k["boundary_verdict"] == "boundary_partial"

    persisted = list(csv.DictReader(out_csv.open(newline="", encoding="utf-8")))
    assert len(persisted) == 4
    doc = out_doc.read_text(encoding="utf-8")
    assert "Boundary / Timeline Partial Audit" in doc
    assert "tp8 12k2k needs deeper trace" in doc
    assert "Default AIC | No-Go" in doc


def test_verdict_distribution_is_fixed(tmp_path: Path) -> None:
    trace_root = tmp_path / "trace"
    _write_trace_family(trace_root)

    rows = analyzer.analyze_boundary_timeline_partial_audit(
        trace_root,
        _actual_phase264_csv(),
        _actual_phase267_csv(),
    )
    verdicts = [row["boundary_verdict"] for row in rows]

    assert verdicts.count("boundary_explains") == 2
    assert verdicts.count("boundary_partial") == 1
    assert verdicts.count("needs_deeper_trace") == 1


def test_phase267_verdict_change_fails_fast(tmp_path: Path) -> None:
    trace_root = tmp_path / "trace"
    _write_trace_family(trace_root)
    rows = list(csv.DictReader(_actual_phase267_csv().open(newline="", encoding="utf-8")))
    rows[0]["verdict"] = "explains"
    mutated = tmp_path / "phase267_bad.csv"
    with mutated.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="phase267 verdict mismatch"):
        analyzer.analyze_boundary_timeline_partial_audit(
            trace_root,
            _actual_phase264_csv(),
            mutated,
        )


def test_tail_decode_change_fails_fast(tmp_path: Path) -> None:
    trace_root = tmp_path / "trace"
    _write_trace_family(trace_root)
    trace_path = trace_root / "tp8ep8-12k2k-bt65536" / "scheduler_trace.jsonl"
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row["iteration"] == 2003:
            row["scheduled_decode_tokens"] = 31
            row["scheduled_total_tokens"] = row["scheduled_context_tokens"] + 31
            row["forward_token_count"] = row["scheduled_total_tokens"]
    trace_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="tail decode mismatch"):
        analyzer.analyze_boundary_timeline_partial_audit(
            trace_root,
            _actual_phase264_csv(),
            _actual_phase267_csv(),
        )


def test_trace_flags_fail_fast(tmp_path: Path) -> None:
    trace_root = tmp_path / "trace"
    _write_trace_family(trace_root)
    trace_path = trace_root / "tp4dp2ep8-12k2k-bt65536" / "scheduler_trace.jsonl"
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["valid_for_default"] = True
    trace_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="trace flag mismatch"):
        analyzer.analyze_boundary_timeline_partial_audit(
            trace_root,
            _actual_phase264_csv(),
            _actual_phase267_csv(),
        )
