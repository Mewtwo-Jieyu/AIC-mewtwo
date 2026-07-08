#!/usr/bin/env python3
"""Phase446: derive peer-phase tagged forward_total serving-state rows."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_8K_EVENT = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase446_b2b_event_timing/"
    / "overhead_gate_20260708_075726/overhead_on/event_timing.jsonl"
)
DEFAULT_32K_EVENT = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase446_b2b_event_timing/"
    / "event_32k_20260708_083632/event_on_32k/event_timing.jsonl"
)
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase446_b2b_forward_total.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase446_b2b_forward_total.md"
DEFAULT_PERFDB = REPO_ROOT / "docs/iter_gap_investigation/phase446_b2b_forward_total_perfdb_rows.txt"

MODEL = "kimi-k2.5"
TOPOLOGY = "tp4dp2ep8"
QUANT_RUNTIME = "CompressedTensorsWNA16MarlinMoEMethod"
HIDDEN_SIZE = 7168
TOPK = 8
MOE_EP_SIZE = 8
KERNEL_SOURCE = "phase446_b2b_event_timing"
DEFAULT_READINESS = "No-Go"

CSV_FIELDS = [
    "source",
    "scenario",
    "phase",
    "row_kind",
    "category",
    "bucket_tokens",
    "decode_batch",
    "latency_ms",
    "sample_count",
    "peer_phase",
    "latency_p50_ms",
    "latency_p90_ms",
    "latency_min_ms",
    "latency_max_ms",
    "provenance",
    "valid_for_default",
    "default_readiness",
]

PERFDB_FIELDS = [
    "framework",
    "version",
    "device",
    "model",
    "topology",
    "phase",
    "row_kind",
    "category",
    "kernel_source",
    "bucket_tokens",
    "decode_batch",
    "hidden_size",
    "topk",
    "moe_ep_size",
    "quant_runtime",
    "latency",
    "provenance",
]


@dataclass(frozen=True)
class EventRecord:
    dp_rank: str
    ctx_tokens: int
    generation_requests: int
    forward_busy_ms: float
    generation_tokens: int = 0
    num_tokens_unpadded: int = 0
    num_tokens_padded: int = 0
    cudagraph_mode: str = ""

    @property
    def bucket_tokens(self) -> int:
        return self.num_tokens_unpadded if self.num_tokens_unpadded > 0 else self.ctx_tokens + self.generation_requests


@dataclass(frozen=True)
class EngineStep:
    dp_rank: str
    local_step: int
    ctx_tokens: int
    generation_requests: int
    forward_busy_ms: float
    generation_tokens: int = 0
    num_tokens_unpadded: int = 0
    num_tokens_padded: int = 0
    cudagraph_mode: str = ""

    @property
    def bucket_tokens(self) -> int:
        return self.num_tokens_unpadded if self.num_tokens_unpadded > 0 else self.ctx_tokens + self.generation_requests

    @property
    def phase(self) -> str:
        has_prefill_work = self.bucket_tokens > self.generation_requests
        if has_prefill_work and self.generation_requests > 0:
            return "mixed_prefill"
        if has_prefill_work:
            return "prefill"
        if self.generation_requests > 0:
            return "decode"
        return "empty"

    @property
    def is_clean_decode(self) -> bool:
        return self.phase == "decode" and self.cudagraph_mode == "FULL"


@dataclass(frozen=True)
class AnnotatedStep:
    step: EngineStep
    peer_phase: str


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
    return values[idx]


def read_event_records(path: Path) -> list[EventRecord]:
    records: list[EventRecord] = []
    with _open_text(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != "phase446_graph_outer_event_v2":
                continue
            records.append(
                EventRecord(
                    dp_rank=str(row.get("dp_rank")),
                    ctx_tokens=int(row.get("ctx_tokens") or 0),
                    generation_requests=int(row.get("generation_requests") or 0),
                    forward_busy_ms=float(row["forward_busy_ms"]),
                    generation_tokens=int(row.get("generation_tokens") or 0),
                    num_tokens_unpadded=int(row.get("num_tokens_unpadded") or 0),
                    num_tokens_padded=int(row.get("num_tokens_padded") or 0),
                    cudagraph_mode=str(row.get("cudagraph_mode") or ""),
                )
            )
    if not records:
        raise ValueError(f"no phase446 event records found in {path}")
    return records


def group_event_steps(records: Iterable[EventRecord], *, tp_width: int = 4) -> list[EngineStep]:
    by_dp: dict[str, list[EventRecord]] = defaultdict(list)
    for record in records:
        by_dp[record.dp_rank].append(record)

    steps: list[EngineStep] = []
    for dp_rank, items in sorted(by_dp.items()):
        if len(items) % tp_width != 0:
            raise ValueError(f"dp_rank={dp_rank} row count {len(items)} not divisible by tp_width={tp_width}")
        local_step = 0
        for idx in range(0, len(items), tp_width):
            chunk = items[idx : idx + tp_width]
            keys = {
                (
                    item.ctx_tokens,
                    item.generation_requests,
                    item.generation_tokens,
                    item.num_tokens_unpadded,
                    item.num_tokens_padded,
                    item.cudagraph_mode,
                )
                for item in chunk
            }
            if len(keys) != 1:
                raise ValueError(f"dp_rank={dp_rank} mixed TP chunk at offset={idx}: {sorted(keys)}")
            (
                ctx_tokens,
                generation_requests,
                generation_tokens,
                num_tokens_unpadded,
                num_tokens_padded,
                cudagraph_mode,
            ) = next(iter(keys))
            steps.append(
                EngineStep(
                    dp_rank=dp_rank,
                    local_step=local_step,
                    ctx_tokens=ctx_tokens,
                    generation_requests=generation_requests,
                    forward_busy_ms=max(item.forward_busy_ms for item in chunk),
                    generation_tokens=generation_tokens,
                    num_tokens_unpadded=num_tokens_unpadded,
                    num_tokens_padded=num_tokens_padded,
                    cudagraph_mode=cudagraph_mode,
                )
            )
            local_step += 1
    return steps


def annotate_peer_phase(steps: Iterable[EngineStep]) -> list[AnnotatedStep]:
    by_key = {(step.dp_rank, step.local_step): step for step in steps}
    annotated: list[AnnotatedStep] = []
    for step in sorted(steps, key=lambda item: (item.local_step, item.dp_rank)):
        peer_rank = "1" if step.dp_rank == "0" else "0"
        peer = by_key.get((peer_rank, step.local_step))
        if peer is None:
            peer_phase = "peer_missing"
        elif peer.phase in {"mixed_prefill", "prefill"}:
            peer_phase = "peer_prefill"
        elif peer.phase == "decode":
            peer_phase = "peer_decode"
        else:
            peer_phase = "peer_empty"
        annotated.append(AnnotatedStep(step=step, peer_phase=peer_phase))
    return annotated


def build_forward_total_rows(annotated_steps: Iterable[AnnotatedStep], *, scenario: str) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, int, str], list[float]] = defaultdict(list)
    for annotated in annotated_steps:
        step = annotated.step
        if step.phase == "mixed_prefill":
            grouped[(step.phase, step.bucket_tokens, step.generation_requests, "any_peer")].append(
                step.forward_busy_ms
            )
        elif step.is_clean_decode and annotated.peer_phase == "peer_decode":
            grouped[(step.phase, step.bucket_tokens, step.generation_requests, annotated.peer_phase)].append(
                step.forward_busy_ms
            )

    rows: list[dict[str, object]] = []
    for (phase, bucket_tokens, decode_batch, peer_phase), values in sorted(grouped.items()):
        rows.append(
            {
                "source": "phase446_b2b_ingest",
                "scenario": scenario,
                "phase": phase,
                "row_kind": "forward_total",
                "category": "forward_total",
                "bucket_tokens": bucket_tokens,
                "decode_batch": decode_batch,
                "latency_ms": statistics.median(values),
                "sample_count": len(values),
                "peer_phase": peer_phase,
                "latency_p50_ms": statistics.median(values),
                "latency_p90_ms": _percentile(values, 0.90),
                "latency_min_ms": min(values),
                "latency_max_ms": max(values),
                "provenance": "phase446_b2b_event_step_bucket_peer_tagged",
                "valid_for_default": False,
                "default_readiness": DEFAULT_READINESS,
            }
        )
    return rows


def _perfdb_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "framework": "VLLM",
        "version": "0.19.0",
        "device": "NVIDIA H200",
        "model": MODEL,
        "topology": TOPOLOGY,
        "phase": row["phase"],
        "row_kind": row["row_kind"],
        "category": row["category"],
        "kernel_source": KERNEL_SOURCE,
        "bucket_tokens": row["bucket_tokens"],
        "decode_batch": row["decode_batch"],
        "hidden_size": HIDDEN_SIZE,
        "topk": TOPK,
        "moe_ep_size": MOE_EP_SIZE,
        "quant_runtime": QUANT_RUNTIME,
        "latency": row["latency_ms"],
        "provenance": row["provenance"],
    }


def select_perfdb_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for row in rows:
        # The 32k run is collected to supply 32k mixed-prefill rows. Decode
        # latency depends on KV length, but the current serving-state key has
        # no KV axis, so 32k decode rows must not overwrite the 8k decode curve.
        if row["phase"] == "decode" and row["scenario"] != "K2.5-tp4ep8dp2-8k2k":
            continue
        key = (
            row["phase"],
            row["row_kind"],
            row["category"],
            row["bucket_tokens"],
            row["decode_batch"],
        )
        if key in seen:
            raise ValueError(f"duplicate selected PerfDB row key: {key}")
        seen.add(key)
        selected.append(row)
    return selected


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_md(path: Path, rows: list[dict[str, object]], summaries: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase446 B2b forward_total candidate rows",
        "",
        "结论: B2b event 行可生成 `forward_total` 候选表。`bucket_tokens` 使用 vLLM 实际 forward token 数 `num_tokens_unpadded`; `num_tokens_unpadded > generation_requests` 的步按 mixed/prefill 计,避免把 chunked-prefill 误放进 decode。",
        "",
        "decode 入表只接受 `peer_decode` 且 `cudagraph_mode=FULL` 的纯 decode 步。`peer_prefill` 下的 decode 步和非 FULL decode 步交给 runtime lockstep 耦合计费,不进入 intrinsic decode 表,避免双计。",
        "",
        "## Summary",
        "",
        "| scenario | event steps | candidate rows | mixed rows | decode rows | excluded peer-prefill decode steps | excluded non-FULL decode steps |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['scenario']} | {item['event_steps']} | {item['candidate_rows']} | "
            f"{item['mixed_rows']} | {item['decode_rows']} | {item['excluded_peer_prefill_decode_steps']} | "
            f"{item['excluded_non_full_decode_steps']} |"
        )
    lines.extend(
        [
            "",
            "## Candidate Rows",
            "",
            "| scenario | phase | bucket | decode batch | median ms | samples | peer phase |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    preview = sorted(rows, key=lambda row: (str(row["scenario"]), str(row["phase"]), int(row["bucket_tokens"]), int(row["decode_batch"])))
    for row in preview[:80]:
        lines.append(
            f"| {row['scenario']} | {row['phase']} | {row['bucket_tokens']} | {row['decode_batch']} | "
            f"{float(row['latency_ms']):.3f} | {row['sample_count']} | {row['peer_phase']} |"
        )
    if len(preview) > 80:
        lines.append(f"| ... | ... | ... | ... | ... | ... | {len(preview) - 80} more rows |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_sources(
    sources: list[tuple[str, Path]],
    *,
    tp_width: int = 1,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for scenario, path in sources:
        records = read_event_records(path)
        steps = group_event_steps(records, tp_width=tp_width)
        annotated = annotate_peer_phase(steps)
        rows = build_forward_total_rows(annotated, scenario=scenario)
        phase_counts = Counter(row["phase"] for row in rows)
        excluded = sum(
            1
            for item in annotated
            if item.step.phase == "decode" and item.peer_phase == "peer_prefill"
        )
        excluded_non_full = sum(
            1
            for item in annotated
            if item.step.phase == "decode" and item.peer_phase == "peer_decode" and not item.step.is_clean_decode
        )
        summaries.append(
            {
                "scenario": scenario,
                "event_steps": len(steps),
                "candidate_rows": len(rows),
                "mixed_rows": phase_counts["mixed_prefill"],
                "decode_rows": phase_counts["decode"],
                "excluded_peer_prefill_decode_steps": excluded,
                "excluded_non_full_decode_steps": excluded_non_full,
            }
        )
        all_rows.extend(rows)
    return all_rows, summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-8k", type=Path, default=DEFAULT_8K_EVENT)
    parser.add_argument("--event-32k", type=Path, default=DEFAULT_32K_EVENT)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    parser.add_argument("--perfdb-out", type=Path, default=DEFAULT_PERFDB)
    parser.add_argument("--tp-width", type=int, default=1)
    args = parser.parse_args()

    rows, summaries = analyze_sources(
        [
            ("K2.5-tp4ep8dp2-8k2k", args.event_8k),
            ("K2.5-tp4ep8dp2-32k3k", args.event_32k),
        ],
        tp_width=args.tp_width,
    )
    write_csv(args.csv_out, rows, CSV_FIELDS)
    write_csv(args.perfdb_out, [_perfdb_row(row) for row in select_perfdb_rows(rows)], PERFDB_FIELDS)
    write_md(args.md_out, rows, summaries)
    print(f"wrote {args.csv_out}")
    print(f"wrote {args.perfdb_out}")
    print(f"wrote {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
