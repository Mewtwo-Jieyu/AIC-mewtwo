#!/usr/bin/env python3
"""Phase443-A: compare graph-outer event busy, serve wall time, and cb_sim charge."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SOURCE = "phase443_b2_diagnostic"
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
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase443_b2_diagnostic.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase443_b2_diagnostic.md"

MIN_CONFIDENCE_ROWS = 20
SIM_BUSY_CLOSE_PCT = 10.0
MATERIAL_GAP_PCT = 30.0
OVERHEAD_GATE_PCT = 5.6761764274547915
DEFAULT_READINESS = "No-Go"

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
    "ctx_tokens",
    "decode_batch",
    "ctx_requests_median",
    "generation_tokens_median",
    "event_count",
    "wall_count",
    "event_to_wall_row_ratio",
    "dp_ranks",
    "engines",
    "event_forward_busy_median_ms",
    "event_forward_busy_p90_ms",
    "wall_elapsed_median_ms",
    "wall_elapsed_p90_ms",
    "sim_charge_ms",
    "sim_context_non_attention_ms",
    "sim_context_attention_ms",
    "sim_generation_non_attention_ms",
    "sim_generation_attention_ms",
    "busy_vs_sim_pct",
    "wall_vs_busy_pct",
    "wall_minus_busy_ms",
    "busy_minus_sim_ms",
    "classification",
    "confidence",
    "b2b_gate",
    "primary_gap",
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
class EventRow:
    ctx_requests: int
    ctx_tokens: int
    generation_requests: int
    generation_tokens: int
    dp_rank: str
    forward_busy_ms: float
    cudagraph_mode: str

    @property
    def key(self) -> tuple[int, int]:
        return (self.ctx_tokens, self.generation_requests)


@dataclass(frozen=True)
class WallStep:
    engine: str
    iteration: int
    ctx_requests: int
    ctx_tokens: int
    generation_requests: int
    generation_tokens: int
    elapsed_ms: float

    @property
    def key(self) -> tuple[int, int]:
        return (self.ctx_tokens, self.generation_requests)


@dataclass(frozen=True)
class SimCharge:
    total_ms: float
    context_non_attention_ms: float
    context_attention_ms: float
    generation_non_attention_ms: float
    generation_attention_ms: float


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


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _median(values: list[float]) -> float:
    return _percentile(values, 0.5)


def _safe_pct(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return math.inf
    return (numerator / denominator - 1.0) * 100.0


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


def parse_event_rows(path: Path) -> list[EventRow]:
    rows: list[EventRow] = []
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
            EventRow(
                ctx_requests=int(record.get("ctx_requests") or 0),
                ctx_tokens=int(record.get("ctx_tokens") or 0),
                generation_requests=int(record.get("generation_requests") or 0),
                generation_tokens=int(record.get("generation_tokens") or 0),
                dp_rank=str(record.get("dp_rank")),
                forward_busy_ms=float(busy),
                cudagraph_mode=str(record.get("cudagraph_mode") or ""),
            )
        )
    if not rows:
        raise ValueError(f"missing phase442 event rows in {path}")
    return rows


def parse_wall_steps(path: Path) -> list[WallStep]:
    steps: list[WallStep] = []
    for line in _iter_text(path):
        match = ITER_RE.search(line)
        if not match:
            continue
        steps.append(
            WallStep(
                engine=match.group("engine"),
                iteration=int(match.group("iteration")),
                ctx_requests=int(match.group("ctx_req")),
                ctx_tokens=int(match.group("ctx_tok")),
                generation_requests=int(match.group("gen_req")),
                generation_tokens=int(match.group("gen_tok")),
                elapsed_ms=float(match.group("elapsed")),
            )
        )
    if not steps:
        raise ValueError(f"missing iteration details in {path}")
    return steps


def classify_bucket(
    *,
    event_count: int,
    wall_count: int,
    busy_ms: float,
    sim_charge_ms: float,
    wall_ms: float,
) -> str:
    if event_count < MIN_CONFIDENCE_ROWS or wall_count < MIN_CONFIDENCE_ROWS:
        return "low_confidence"
    busy_vs_sim_pct = _safe_pct(busy_ms, sim_charge_ms)
    wall_vs_busy_pct = _safe_pct(wall_ms, busy_ms)
    if abs(busy_vs_sim_pct) <= SIM_BUSY_CLOSE_PCT and wall_vs_busy_pct >= MATERIAL_GAP_PCT:
        return "wall_gap_dominates_b2b_not_useful"
    if busy_vs_sim_pct >= MATERIAL_GAP_PCT and abs(wall_vs_busy_pct) <= SIM_BUSY_CLOSE_PCT:
        return "busy_gap_dominates_b2b_candidate"
    if busy_vs_sim_pct >= MATERIAL_GAP_PCT and wall_vs_busy_pct >= MATERIAL_GAP_PCT:
        return "mixed_busy_and_wall_gap"
    if abs(busy_vs_sim_pct) <= SIM_BUSY_CLOSE_PCT and abs(wall_vs_busy_pct) <= SIM_BUSY_CLOSE_PCT:
        return "no_material_gap"
    return "mixed_or_inconclusive"


def summarize_b2b_gate(rows: list[dict[str, object]]) -> dict[str, object]:
    bucket_rows = [row for row in rows if row.get("row_type") == "bucket"]
    counts = Counter(str(row.get("classification")) for row in bucket_rows)
    weighted_gap = 0.0
    weighted_wall_over_busy = 0.0
    weighted_busy_over_sim = 0.0
    for row in bucket_rows:
        wall_count = int(row.get("wall_count") or 0)
        sim_ms = float(row.get("sim_charge_ms") or 0.0)
        busy_ms = float(row.get("event_forward_busy_median_ms") or 0.0)
        wall_ms = float(row.get("wall_elapsed_median_ms") or 0.0)
        gap = max(0.0, wall_ms - sim_ms) * wall_count
        weighted_gap += gap
        weighted_wall_over_busy += max(0.0, wall_ms - busy_ms) * wall_count
        weighted_busy_over_sim += max(0.0, busy_ms - sim_ms) * wall_count

    wall_share = weighted_wall_over_busy / weighted_gap if weighted_gap > 0 else 0.0
    busy_share = weighted_busy_over_sim / weighted_gap if weighted_gap > 0 else 0.0
    if wall_share >= 0.60 and counts["wall_gap_dominates_b2b_not_useful"] >= counts["busy_gap_dominates_b2b_candidate"]:
        gate = "not_recommended"
        primary = "wall_over_busy"
    elif busy_share >= 0.60 and counts["busy_gap_dominates_b2b_candidate"] > 0:
        gate = "recommended"
        primary = "busy_over_sim"
    else:
        gate = "inconclusive"
        primary = "mixed"
    return {
        "b2b_gate": gate,
        "primary_gap": primary,
        "bucket_rows": len(bucket_rows),
        "wall_gap_share": wall_share,
        "busy_gap_share": busy_share,
        "wall_gap_buckets": counts["wall_gap_dominates_b2b_not_useful"],
        "busy_gap_buckets": counts["busy_gap_dominates_b2b_candidate"],
        "mixed_buckets": counts["mixed_busy_and_wall_gap"] + counts["mixed_or_inconclusive"],
        "low_confidence_buckets": counts["low_confidence"],
    }


def _load_calc():
    from aiconfigurator.sdk.backends.cb_simulator.iteration_latency import (  # noqa: PLC0415
        IterationLatencyCalculator,
    )
    from scripts import validate_cb_simulator as validate_cb  # noqa: PLC0415

    model, db, backend = validate_cb._load_model_and_db(tp=4, dp=2, moe_tp=1, moe_ep=8)
    return IterationLatencyCalculator(
        backend,
        model,
        db,
        overlap_factor=0.0,
        per_iteration_overhead_ms=0.0,
    )


def _compute_sim_charges(
    keys: list[tuple[int, int]],
    *,
    ctx_requests_by_key: dict[tuple[int, int], int],
    generation_tokens_by_key: dict[tuple[int, int], int],
) -> dict[tuple[int, int], SimCharge]:
    calc = _load_calc()
    out: dict[tuple[int, int], SimCharge] = {}
    for ctx_tokens, decode_batch in sorted(keys):
        ctx_requests = ctx_requests_by_key.get((ctx_tokens, decode_batch), 1 if ctx_tokens > 0 else 0)
        calc.compute(
            prefill_tokens=ctx_tokens,
            prefill_batch_size=max(1, ctx_requests) if ctx_tokens > 0 else 0,
            prefill_seq_len=8000,
            decode_batch_size=decode_batch,
            decode_avg_kv_len=8000,
        )
        breakdown = calc.get_last_breakdown()
        if breakdown is None:
            raise RuntimeError("missing iteration latency breakdown")
        out[(ctx_tokens, decode_batch)] = SimCharge(
            total_ms=breakdown.total_ms,
            context_non_attention_ms=breakdown.context_non_attention_ms,
            context_attention_ms=breakdown.context_attention_ms,
            generation_non_attention_ms=breakdown.generation_non_attention_ms,
            generation_attention_ms=breakdown.generation_attention_ms,
        )
        # Keep the argument in the signature for audit symmetry. The current
        # calculator keys generation cost by decode batch, not generation token count.
        _ = generation_tokens_by_key
    return out


def build_bucket_rows(
    event_rows: list[EventRow],
    wall_steps: list[WallStep],
    *,
    with_sim: bool = True,
) -> list[dict[str, object]]:
    event_values: dict[tuple[int, int], list[float]] = defaultdict(list)
    event_ctx_req: dict[tuple[int, int], list[int]] = defaultdict(list)
    event_gen_tok: dict[tuple[int, int], list[int]] = defaultdict(list)
    event_dp: dict[tuple[int, int], Counter[str]] = defaultdict(Counter)
    for row in event_rows:
        event_values[row.key].append(row.forward_busy_ms)
        event_ctx_req[row.key].append(row.ctx_requests)
        event_gen_tok[row.key].append(row.generation_tokens)
        event_dp[row.key][row.dp_rank] += 1

    wall_values: dict[tuple[int, int], list[float]] = defaultdict(list)
    wall_ctx_req: dict[tuple[int, int], list[int]] = defaultdict(list)
    wall_gen_tok: dict[tuple[int, int], list[int]] = defaultdict(list)
    wall_engines: dict[tuple[int, int], Counter[str]] = defaultdict(Counter)
    for step in wall_steps:
        wall_values[step.key].append(step.elapsed_ms)
        wall_ctx_req[step.key].append(step.ctx_requests)
        wall_gen_tok[step.key].append(step.generation_tokens)
        wall_engines[step.key][step.engine] += 1

    keys = sorted(set(event_values) & set(wall_values))
    ctx_requests_by_key = {
        key: int(round(_median([float(v) for v in wall_ctx_req.get(key) or event_ctx_req.get(key) or [0]])))
        for key in keys
    }
    gen_tokens_by_key = {
        key: int(round(_median([float(v) for v in wall_gen_tok.get(key) or event_gen_tok.get(key) or [0]])))
        for key in keys
    }
    sim_by_key: dict[tuple[int, int], SimCharge] = {}
    if with_sim:
        sim_by_key = _compute_sim_charges(
            keys,
            ctx_requests_by_key=ctx_requests_by_key,
            generation_tokens_by_key=gen_tokens_by_key,
        )

    rows: list[dict[str, object]] = []
    for key in keys:
        ctx_tokens, decode_batch = key
        busy_values = event_values[key]
        elapsed_values = wall_values[key]
        busy_median = _median(busy_values)
        wall_median = _median(elapsed_values)
        sim = sim_by_key.get(key) or SimCharge(
            total_ms=math.nan,
            context_non_attention_ms=math.nan,
            context_attention_ms=math.nan,
            generation_non_attention_ms=math.nan,
            generation_attention_ms=math.nan,
        )
        classification = classify_bucket(
            event_count=len(busy_values),
            wall_count=len(elapsed_values),
            busy_ms=busy_median,
            sim_charge_ms=sim.total_ms,
            wall_ms=wall_median,
        )
        row = _base_row()
        row.update(
            {
                "row_type": "bucket",
                "ctx_tokens": ctx_tokens,
                "decode_batch": decode_batch,
                "ctx_requests_median": ctx_requests_by_key[key],
                "generation_tokens_median": gen_tokens_by_key[key],
                "event_count": len(busy_values),
                "wall_count": len(elapsed_values),
                "event_to_wall_row_ratio": len(busy_values) / len(elapsed_values) if elapsed_values else math.inf,
                "dp_ranks": ";".join(f"{rank}:{count}" for rank, count in sorted(event_dp[key].items())),
                "engines": ";".join(f"{engine}:{count}" for engine, count in sorted(wall_engines[key].items())),
                "event_forward_busy_median_ms": busy_median,
                "event_forward_busy_p90_ms": _percentile(busy_values, 0.90),
                "wall_elapsed_median_ms": wall_median,
                "wall_elapsed_p90_ms": _percentile(elapsed_values, 0.90),
                "sim_charge_ms": sim.total_ms,
                "sim_context_non_attention_ms": sim.context_non_attention_ms,
                "sim_context_attention_ms": sim.context_attention_ms,
                "sim_generation_non_attention_ms": sim.generation_non_attention_ms,
                "sim_generation_attention_ms": sim.generation_attention_ms,
                "busy_vs_sim_pct": _safe_pct(busy_median, sim.total_ms),
                "wall_vs_busy_pct": _safe_pct(wall_median, busy_median),
                "wall_minus_busy_ms": wall_median - busy_median,
                "busy_minus_sim_ms": busy_median - sim.total_ms,
                "classification": classification,
                "confidence": "low" if classification == "low_confidence" else "high",
            }
        )
        rows.append(row)
    return rows


def build_summary_rows(rows: list[dict[str, object]], event_rows: list[EventRow], wall_steps: list[WallStep]) -> list[dict[str, object]]:
    summary = summarize_b2b_gate(rows)
    out: list[dict[str, object]] = []
    for metric, value in summary.items():
        row = _base_row()
        row.update(
            {
                "row_type": "summary",
                "metric": metric,
                "value": value,
                "b2b_gate": summary["b2b_gate"],
                "primary_gap": summary["primary_gap"],
            }
        )
        out.append(row)
    for metric, value in {
        "event_rows": len(event_rows),
        "wall_rows": len(wall_steps),
        "event_to_wall_row_ratio": len(event_rows) / len(wall_steps) if wall_steps else math.inf,
        "overhead_gate_failed_pct": OVERHEAD_GATE_PCT,
    }.items():
        row = _base_row()
        row.update(
            {
                "row_type": "inventory",
                "metric": metric,
                "value": value,
                "b2b_gate": summary["b2b_gate"],
                "primary_gap": summary["primary_gap"],
            }
        )
        out.append(row)
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CSV_FIELDS,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field, "")) for field in CSV_FIELDS})


def _top_rows(rows: list[dict[str, object]], limit: int = 12) -> list[dict[str, object]]:
    buckets = [row for row in rows if row.get("row_type") == "bucket"]
    return sorted(
        buckets,
        key=lambda row: abs(float(row.get("wall_minus_busy_ms") or 0.0)) * int(row.get("wall_count") or 0),
        reverse=True,
    )[:limit]


def _top_busy_sim_rows(rows: list[dict[str, object]], limit: int = 12) -> list[dict[str, object]]:
    buckets = [row for row in rows if row.get("row_type") == "bucket"]
    return sorted(
        buckets,
        key=lambda row: max(0.0, float(row.get("busy_minus_sim_ms") or 0.0))
        * int(row.get("wall_count") or 0),
        reverse=True,
    )[:limit]


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    summary = summarize_b2b_gate(rows)
    top_wall = _top_rows(rows)
    top_busy = _top_busy_sim_rows(rows)
    gate_text = {
        "not_recommended": "B2b 不建议立项: event busy 与 sim 计费接近,主要缺口在 wall-busy。",
        "recommended": "B2b 候选: event busy 已明显高于 sim 计费,需要继续测低开销 busy。",
        "inconclusive": "B2b 暂无单一结论: busy 与 wall 两侧都有缺口或样本不足。",
    }[str(summary["b2b_gate"])]
    lines = [
        "# Phase443-A B2 event timing diagnostic",
        "",
        gate_text,
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in (
        "b2b_gate",
        "primary_gap",
        "bucket_rows",
        "wall_gap_share",
        "busy_gap_share",
        "wall_gap_buckets",
        "busy_gap_buckets",
        "mixed_buckets",
        "low_confidence_buckets",
    ):
        lines.append(f"| {key} | {_fmt(summary[key])} |")
    lines.extend(
        [
            "",
            "## Largest Busy-Sim Buckets",
            "",
            "| ctx_tokens | decode_batch | event rows | wall rows | sim ms | busy ms | wall ms | class |",
            "|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in top_busy:
        lines.append(
            "| "
            f"{row.get('ctx_tokens')} | {row.get('decode_batch')} | "
            f"{row.get('event_count')} | {row.get('wall_count')} | "
            f"{_fmt(row.get('sim_charge_ms'))} | "
            f"{_fmt(row.get('event_forward_busy_median_ms'))} | "
            f"{_fmt(row.get('wall_elapsed_median_ms'))} | "
            f"{row.get('classification')} |"
        )
    lines.extend(
        [
            "",
            "## Largest Wall-Busy Buckets",
            "",
            "| ctx_tokens | decode_batch | event rows | wall rows | sim ms | busy ms | wall ms | class |",
            "|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in top_wall:
        lines.append(
            "| "
            f"{row.get('ctx_tokens')} | {row.get('decode_batch')} | "
            f"{row.get('event_count')} | {row.get('wall_count')} | "
            f"{_fmt(row.get('sim_charge_ms'))} | "
            f"{_fmt(row.get('event_forward_busy_median_ms'))} | "
            f"{_fmt(row.get('wall_elapsed_median_ms'))} | "
            f"{row.get('classification')} |"
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "- diagnostic_only=true; valid_for_default=false; perf_database=false.",
            "- event rows are grouped by `(ctx_tokens, generation_requests)` because event rows are approximately 4x serve iteration rows.",
            "- The 5.676% event timing overhead gate failure is recorded as measurement risk; no data is ingested into PerfDB.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(event_jsonl: Path, serve_log: Path, csv_path: Path, md_path: Path, *, with_sim: bool = True) -> dict[str, object]:
    events = parse_event_rows(event_jsonl)
    wall_steps = parse_wall_steps(serve_log)
    bucket_rows = build_bucket_rows(events, wall_steps, with_sim=with_sim)
    rows = bucket_rows + build_summary_rows(bucket_rows, events, wall_steps)
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    return summarize_b2b_gate(bucket_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-jsonl", type=Path, default=DEFAULT_EVENT_JSONL)
    parser.add_argument("--serve-log", type=Path, default=DEFAULT_SERVE_LOG)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--no-sim", action="store_true", help="skip cb_sim charge computation")
    args = parser.parse_args()
    summary = run(args.event_jsonl, args.serve_log, args.csv, args.md, with_sim=not args.no_sim)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
