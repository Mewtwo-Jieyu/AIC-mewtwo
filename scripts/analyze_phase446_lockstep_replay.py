#!/usr/bin/env python3
"""Phase446 Step0: replay DP lockstep with graph-outer forward busy events."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = "phase446_lockstep_replay"
SCENARIO = "K2.5-tp4ep8dp2-8k2k"
DEFAULT_EVENT_JSONL = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase442_b2_event_timing/"
    / "overhead_gate_20260708_164017/overhead_on/event_timing.jsonl.gz"
)
DEFAULT_SERVE_LOG = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase442_b2_event_timing/"
    / "overhead_gate_20260708_164017/overhead_on/K2.5-tp4ep8dp2-8k2k/serve.log.gz"
)
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase446_lockstep_replay.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase446_lockstep_replay.md"

DEFAULT_READINESS = "No-Go"
DEFAULT_TP_WIDTH = 4
GATE_ERROR_PCT = 10.0
FIXTURE_LIMIT = 20

ITER_RE = re.compile(
    r"EngineCore_DP(?P<engine>\d+).*?"
    r"Iteration\((?P<iteration>\d+)\): "
    r"(?P<ctx_req>\d+) context requests, "
    r"(?P<ctx_tok>\d+) context tokens, "
    r"(?P<gen_req>\d+) generation requests, "
    r"(?P<gen_tok>\d+) generation tokens, "
    r"iteration elapsed time: (?P<elapsed>[0-9.]+) ms"
)

CSV_FIELDS = [
    "source",
    "row_type",
    "scenario",
    "window",
    "cycle_index",
    "cycle_count",
    "engine0_ctx_tokens",
    "engine0_decode_batch",
    "engine1_ctx_tokens",
    "engine1_decode_batch",
    "engine0_busy_ms",
    "engine1_busy_ms",
    "coupled_busy_ms",
    "engine0_wall_ms",
    "engine1_wall_ms",
    "replay_wall_ms",
    "real_wall_ms",
    "sum_max_wall_ms",
    "replay_vs_real_wall_error_pct",
    "replay_vs_sum_max_wall_error_pct",
    "event_steps_rank0",
    "event_steps_rank1",
    "wall_steps_rank0",
    "wall_steps_rank1",
    "event_rows_per_step_rank0",
    "event_rows_per_step_rank1",
    "event_group_mismatch_count",
    "event_wall_key_mismatch_count",
    "gate",
    "verdict",
    "metric",
    "value",
    "note",
    "gpu_allowed",
    "ssh_allowed",
    "runtime_modified",
    "perf_database",
    "valid_for_default",
    "diagnostic_only",
    "default_readiness",
]


@dataclass(frozen=True)
class EventRecord:
    dp_rank: str
    ctx_tokens: int
    generation_requests: int
    forward_busy_ms: float


@dataclass(frozen=True)
class EventStep:
    dp_rank: str
    index: int
    ctx_tokens: int
    decode_batch: int
    busy_ms: float
    source_row_count: int


@dataclass(frozen=True)
class GroupedEvents:
    steps: list[EventStep]
    mismatch_count: int
    remainder_count: int
    rows_per_step: int


@dataclass(frozen=True)
class WallStep:
    engine: str
    iteration: int
    ctx_tokens: int
    decode_batch: int
    elapsed_ms: float


@dataclass(frozen=True)
class CoupledCycle:
    index: int
    engine0_ctx_tokens: int
    engine0_decode_batch: int
    engine0_busy_ms: float
    engine0_wall_ms: float
    engine1_ctx_tokens: int
    engine1_decode_batch: int
    engine1_busy_ms: float
    engine1_wall_ms: float

    @property
    def coupled_busy_ms(self) -> float:
        return max(self.engine0_busy_ms, self.engine1_busy_ms)

    @property
    def sum_max_wall_ms(self) -> float:
        return max(self.engine0_wall_ms, self.engine1_wall_ms)


@dataclass(frozen=True)
class ReplayResult:
    window: str
    cycle_count: int
    replay_wall_ms: float
    real_wall_ms: float
    sum_max_wall_ms: float
    replay_vs_real_wall_error_pct: float
    replay_vs_sum_max_wall_error_pct: float
    gate: str


@dataclass(frozen=True)
class ReplayBundle:
    cycles: list[CoupledCycle]
    results: list[ReplayResult]
    event_steps_rank0: int
    event_steps_rank1: int
    wall_steps_rank0: int
    wall_steps_rank1: int
    event_rows_per_step_rank0: int
    event_rows_per_step_rank1: int
    event_group_mismatch_count: int
    event_wall_key_mismatch_count: int
    verdict: str


def _iter_text(path: Path) -> Iterable[str]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            yield from handle
        return
    with path.open("rt", encoding="utf-8", errors="replace") as handle:
        yield from handle


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def _base_row() -> dict[str, object]:
    return {
        "source": SOURCE,
        "scenario": SCENARIO,
        "gpu_allowed": False,
        "ssh_allowed": False,
        "runtime_modified": False,
        "perf_database": False,
        "valid_for_default": False,
        "diagnostic_only": True,
        "default_readiness": DEFAULT_READINESS,
    }


def _pct_delta(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return math.inf
    return (numerator / denominator - 1.0) * 100.0


def parse_event_records(path: Path) -> list[EventRecord]:
    rows: list[EventRecord] = []
    for line in _iter_text(path):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("schema") != "phase442_graph_outer_event_v1":
            continue
        busy = record.get("forward_busy_ms")
        if busy is None:
            continue
        rows.append(
            EventRecord(
                dp_rank=str(record.get("dp_rank")),
                ctx_tokens=int(record.get("ctx_tokens") or 0),
                generation_requests=int(record.get("generation_requests") or 0),
                forward_busy_ms=float(busy),
            )
        )
    if not rows:
        raise ValueError(f"missing phase442 event rows in {path}")
    return rows


def parse_wall_steps(path: Path) -> dict[str, list[WallStep]]:
    by_engine: dict[str, list[WallStep]] = defaultdict(list)
    for line in _iter_text(path):
        match = ITER_RE.search(line)
        if not match:
            continue
        ctx_tokens = int(match.group("ctx_tok"))
        decode_batch = int(match.group("gen_req"))
        elapsed = float(match.group("elapsed"))
        if ctx_tokens == 0 and decode_batch == 0 and elapsed == 0.0:
            continue
        by_engine[match.group("engine")].append(
            WallStep(
                engine=match.group("engine"),
                iteration=int(match.group("iteration")),
                ctx_tokens=ctx_tokens,
                decode_batch=decode_batch,
                elapsed_ms=elapsed,
            )
        )
    if not by_engine:
        raise ValueError(f"missing iteration details in {path}")
    return dict(by_engine)


def group_event_steps(records: list[EventRecord], tp_width: int = DEFAULT_TP_WIDTH) -> GroupedEvents:
    if tp_width <= 0:
        raise ValueError("tp_width must be positive")
    steps: list[EventStep] = []
    mismatch_count = 0
    remainder_count = len(records) % tp_width
    for start in range(0, len(records) - remainder_count, tp_width):
        chunk = records[start : start + tp_width]
        keys = {(row.ctx_tokens, row.generation_requests) for row in chunk}
        if len(keys) != 1:
            mismatch_count += 1
            continue
        ctx_tokens, decode_batch = next(iter(keys))
        steps.append(
            EventStep(
                dp_rank=chunk[0].dp_rank,
                index=len(steps),
                ctx_tokens=ctx_tokens,
                decode_batch=decode_batch,
                busy_ms=max(row.forward_busy_ms for row in chunk),
                source_row_count=len(chunk),
            )
        )
    return GroupedEvents(
        steps=steps,
        mismatch_count=mismatch_count,
        remainder_count=remainder_count,
        rows_per_step=tp_width,
    )


def group_events_by_rank(records: list[EventRecord], tp_width: int) -> dict[str, GroupedEvents]:
    by_rank: dict[str, list[EventRecord]] = defaultdict(list)
    for record in records:
        by_rank[record.dp_rank].append(record)
    return {rank: group_event_steps(rank_rows, tp_width=tp_width) for rank, rank_rows in by_rank.items()}


def build_cycles(grouped: dict[str, GroupedEvents], wall_steps: dict[str, list[WallStep]]) -> tuple[list[CoupledCycle], int]:
    required = {"0", "1"}
    if not required <= set(grouped):
        raise ValueError(f"event rows must contain DP ranks 0 and 1, got {sorted(grouped)}")
    if not required <= set(wall_steps):
        raise ValueError(f"wall rows must contain engines 0 and 1, got {sorted(wall_steps)}")

    event0 = grouped["0"].steps
    event1 = grouped["1"].steps
    wall0 = wall_steps["0"]
    wall1 = wall_steps["1"]
    cycle_count = min(len(event0), len(event1), len(wall0), len(wall1))
    key_mismatches = 0
    cycles: list[CoupledCycle] = []
    for idx in range(cycle_count):
        if (event0[idx].ctx_tokens, event0[idx].decode_batch) != (wall0[idx].ctx_tokens, wall0[idx].decode_batch):
            key_mismatches += 1
            continue
        if (event1[idx].ctx_tokens, event1[idx].decode_batch) != (wall1[idx].ctx_tokens, wall1[idx].decode_batch):
            key_mismatches += 1
            continue
        cycles.append(
            CoupledCycle(
                index=idx,
                engine0_ctx_tokens=event0[idx].ctx_tokens,
                engine0_decode_batch=event0[idx].decode_batch,
                engine0_busy_ms=event0[idx].busy_ms,
                engine0_wall_ms=wall0[idx].elapsed_ms,
                engine1_ctx_tokens=event1[idx].ctx_tokens,
                engine1_decode_batch=event1[idx].decode_batch,
                engine1_busy_ms=event1[idx].busy_ms,
                engine1_wall_ms=wall1[idx].elapsed_ms,
            )
        )
    return cycles, key_mismatches


def replay_window(name: str, cycles: list[CoupledCycle]) -> ReplayResult:
    replay_wall = sum(cycle.coupled_busy_ms for cycle in cycles)
    engine0_wall = sum(cycle.engine0_wall_ms for cycle in cycles)
    engine1_wall = sum(cycle.engine1_wall_ms for cycle in cycles)
    real_wall = max(engine0_wall, engine1_wall)
    sum_max_wall = sum(cycle.sum_max_wall_ms for cycle in cycles)
    real_error = _pct_delta(replay_wall, real_wall)
    sum_max_error = _pct_delta(replay_wall, sum_max_wall)
    gate = "passed" if abs(real_error) <= GATE_ERROR_PCT else "failed"
    return ReplayResult(
        window=name,
        cycle_count=len(cycles),
        replay_wall_ms=replay_wall,
        real_wall_ms=real_wall,
        sum_max_wall_ms=sum_max_wall,
        replay_vs_real_wall_error_pct=real_error,
        replay_vs_sum_max_wall_error_pct=sum_max_error,
        gate=gate,
    )


def select_windows(cycles: list[CoupledCycle]) -> dict[str, list[CoupledCycle]]:
    return {
        "all_paired": list(cycles),
        "steady_decode_ge_40": [
            cycle
            for cycle in cycles
            if max(cycle.engine0_decode_batch, cycle.engine1_decode_batch) >= 40
        ],
        "high_decode_ge_45": [
            cycle
            for cycle in cycles
            if max(cycle.engine0_decode_batch, cycle.engine1_decode_batch) >= 45
        ],
        "peak_decode_ge_56": [
            cycle
            for cycle in cycles
            if max(cycle.engine0_decode_batch, cycle.engine1_decode_batch) >= 56
        ],
    }


def decide_verdict(results: list[ReplayResult], event_group_mismatch_count: int, key_mismatch_count: int) -> str:
    result_by_window = {result.window: result for result in results}
    steady = result_by_window.get("steady_decode_ge_40")
    if event_group_mismatch_count or key_mismatch_count:
        return "lockstep_replay_alignment_failed"
    if steady is None or steady.cycle_count == 0:
        return "lockstep_replay_missing_steady_window"
    if steady.gate == "passed":
        return "lockstep_busy_replay_passes"
    return "lockstep_busy_replay_fails"


def run_replay(event_jsonl: Path, serve_log: Path, tp_width: int) -> ReplayBundle:
    event_records = parse_event_records(event_jsonl)
    grouped = group_events_by_rank(event_records, tp_width=tp_width)
    wall_steps = parse_wall_steps(serve_log)
    cycles, key_mismatches = build_cycles(grouped, wall_steps)
    windows = select_windows(cycles)
    results = [replay_window(name, window_cycles) for name, window_cycles in windows.items()]
    event_group_mismatches = sum(item.mismatch_count + item.remainder_count for item in grouped.values())
    verdict = decide_verdict(results, event_group_mismatches, key_mismatches)
    return ReplayBundle(
        cycles=cycles,
        results=results,
        event_steps_rank0=len(grouped["0"].steps),
        event_steps_rank1=len(grouped["1"].steps),
        wall_steps_rank0=len(wall_steps["0"]),
        wall_steps_rank1=len(wall_steps["1"]),
        event_rows_per_step_rank0=grouped["0"].rows_per_step,
        event_rows_per_step_rank1=grouped["1"].rows_per_step,
        event_group_mismatch_count=event_group_mismatches,
        event_wall_key_mismatch_count=key_mismatches,
        verdict=verdict,
    )


def _summary_csv_rows(bundle: ReplayBundle) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in bundle.results:
        row = _base_row()
        row.update(
            {
                "row_type": "window",
                "window": result.window,
                "cycle_count": result.cycle_count,
                "replay_wall_ms": result.replay_wall_ms,
                "real_wall_ms": result.real_wall_ms,
                "sum_max_wall_ms": result.sum_max_wall_ms,
                "replay_vs_real_wall_error_pct": result.replay_vs_real_wall_error_pct,
                "replay_vs_sum_max_wall_error_pct": result.replay_vs_sum_max_wall_error_pct,
                "gate": result.gate,
                "verdict": bundle.verdict,
                "event_steps_rank0": bundle.event_steps_rank0,
                "event_steps_rank1": bundle.event_steps_rank1,
                "wall_steps_rank0": bundle.wall_steps_rank0,
                "wall_steps_rank1": bundle.wall_steps_rank1,
                "event_rows_per_step_rank0": bundle.event_rows_per_step_rank0,
                "event_rows_per_step_rank1": bundle.event_rows_per_step_rank1,
                "event_group_mismatch_count": bundle.event_group_mismatch_count,
                "event_wall_key_mismatch_count": bundle.event_wall_key_mismatch_count,
                "note": "primary gate is steady_decode_ge_40 replay_vs_real_wall_error_pct <= 10%",
            }
        )
        rows.append(row)
    for metric, value in {
        "event_steps_rank0": bundle.event_steps_rank0,
        "event_steps_rank1": bundle.event_steps_rank1,
        "wall_steps_rank0": bundle.wall_steps_rank0,
        "wall_steps_rank1": bundle.wall_steps_rank1,
        "event_group_mismatch_count": bundle.event_group_mismatch_count,
        "event_wall_key_mismatch_count": bundle.event_wall_key_mismatch_count,
    }.items():
        row = _base_row()
        row.update(
            {
                "row_type": "summary",
                "metric": metric,
                "value": value,
                "verdict": bundle.verdict,
                "note": "coupling granularity audit",
            }
        )
        rows.append(row)
    return rows


def _fixture_rows(bundle: ReplayBundle) -> list[dict[str, object]]:
    ranked = sorted(
        bundle.cycles,
        key=lambda cycle: cycle.coupled_busy_ms,
        reverse=True,
    )
    rows: list[dict[str, object]] = []
    for cycle in ranked[:FIXTURE_LIMIT]:
        row = _base_row()
        row.update(
            {
                "row_type": "fixture_cycle",
                "cycle_index": cycle.index,
                "engine0_ctx_tokens": cycle.engine0_ctx_tokens,
                "engine0_decode_batch": cycle.engine0_decode_batch,
                "engine1_ctx_tokens": cycle.engine1_ctx_tokens,
                "engine1_decode_batch": cycle.engine1_decode_batch,
                "engine0_busy_ms": cycle.engine0_busy_ms,
                "engine1_busy_ms": cycle.engine1_busy_ms,
                "coupled_busy_ms": cycle.coupled_busy_ms,
                "engine0_wall_ms": cycle.engine0_wall_ms,
                "engine1_wall_ms": cycle.engine1_wall_ms,
                "verdict": bundle.verdict,
                "note": "highest coupled-busy cycles; future runtime fixture seed",
            }
        )
        rows.append(row)
    return rows


def write_csv(bundle: ReplayBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in _summary_csv_rows(bundle) + _fixture_rows(bundle):
            writer.writerow({key: _fmt(row.get(key, "")) for key in CSV_FIELDS})


def _md_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(value) for value in row) + " |")
    return lines


def write_markdown(bundle: ReplayBundle, path: Path) -> None:
    result_rows = [
        [
            result.window,
            result.cycle_count,
            result.replay_wall_ms,
            result.real_wall_ms,
            result.replay_vs_real_wall_error_pct,
            result.sum_max_wall_ms,
            result.replay_vs_sum_max_wall_error_pct,
            result.gate,
        ]
        for result in bundle.results
    ]
    fixture_rows = [
        [
            cycle.index,
            cycle.engine0_ctx_tokens,
            cycle.engine0_decode_batch,
            cycle.engine1_ctx_tokens,
            cycle.engine1_decode_batch,
            cycle.engine0_busy_ms,
            cycle.engine1_busy_ms,
            cycle.coupled_busy_ms,
        ]
        for cycle in sorted(bundle.cycles, key=lambda item: item.coupled_busy_ms, reverse=True)[:10]
    ]
    lines = [
        "# Phase446 Lockstep Replay",
        "",
        f"- Verdict: `{bundle.verdict}`.",
        f"- Coupling granularity: event rows group into `{bundle.event_rows_per_step_rank0}` TP rows per DP0 step and `{bundle.event_rows_per_step_rank1}` TP rows per DP1 step.",
        f"- Event grouping mismatches: `{bundle.event_group_mismatch_count}`; event/wall key mismatches: `{bundle.event_wall_key_mismatch_count}`.",
        "- Primary gate: `steady_decode_ge_40` must reconstruct real window wall within 10%.",
        "- Source basis: vLLM 0.19.0 `vllm/v1/worker/dp_utils.py:78-90` pads each DP rank to the max token count; `dp_utils.py:153-159` enables that padding when cudagraph/ubatching requires it; `gpu_model_runner.py:3615-3639` coordinates DP ranks each step and re-dispatches cudagraph with the padded token count.",
        "",
        "## Replay Windows",
    ]
    lines.extend(
        _md_table(
            [
                "window",
                "cycles",
                "replay_busy_ms",
                "real_wall_ms",
                "error_pct",
                "sum_max_wall_ms",
                "sum_max_error_pct",
                "gate",
            ],
            result_rows,
        )
    )
    lines.extend(["", "## Fixture Seed Cycles"])
    lines.extend(
        _md_table(
            [
                "cycle",
                "e0_ctx",
                "e0_decode",
                "e1_ctx",
                "e1_decode",
                "e0_busy",
                "e1_busy",
                "coupled_busy",
            ],
            fixture_rows,
        )
    )
    lines.extend(
        [
            "",
            "## Next Gate",
            "",
            "Step 0 passes only the offline mechanism gate. It does not make Phase442 event rows ingestible because that run failed the overhead gate. Runtime DP lockstep coupling may proceed to a red/green implementation only if the same fixture-style max coupling is used and the next B2b run passes its overhead and reproducibility gates.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(event_jsonl: Path, serve_log: Path, tp_width: int, csv_out: Path, md_out: Path) -> str:
    bundle = run_replay(event_jsonl, serve_log, tp_width)
    write_csv(bundle, csv_out)
    write_markdown(bundle, md_out)
    return bundle.verdict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-jsonl", type=Path, default=DEFAULT_EVENT_JSONL)
    parser.add_argument("--serve-log", type=Path, default=DEFAULT_SERVE_LOG)
    parser.add_argument("--tp-width", type=int, default=DEFAULT_TP_WIDTH)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    verdict = run_analysis(args.event_jsonl, args.serve_log, args.tp_width, args.csv_out, args.md_out)
    print(f"wrote {args.csv_out}")
    print(f"wrote {args.md_out}")
    print(f"verdict={verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
