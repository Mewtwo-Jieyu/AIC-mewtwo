#!/usr/bin/env python3
"""Phase447: extract DP phase and step-composition fingerprints from B2b events."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE446_PATH = REPO_ROOT / "scripts" / "analyze_phase446_b2b_ingest.py"
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
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase447_sequence_fingerprint.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase447_sequence_fingerprint.md"


def _load_phase446():
    spec = importlib.util.spec_from_file_location("analyze_phase446_b2b_ingest", PHASE446_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


phase446 = _load_phase446()


@dataclass(frozen=True)
class Fingerprint:
    scenario: str
    engine_summary_rows: list[dict[str, object]]
    mixed_distribution_rows: list[dict[str, object]]
    phase_joint_rows: list[dict[str, object]]
    wave_rows: list[dict[str, object]]
    tolerance_rows: list[dict[str, object]]

    @property
    def rows(self) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for section, rows in (
            ("engine_summary", self.engine_summary_rows),
            ("mixed_distribution", self.mixed_distribution_rows),
            ("phase_joint", self.phase_joint_rows),
            ("mixed_wave", self.wave_rows),
            ("fingerprint_tolerance", self.tolerance_rows),
        ):
            for row in rows:
                result.append({"scenario": self.scenario, "section": section, **row})
        return result


def _phase(step) -> str:
    return step.phase


def _percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
    return float(values[idx])


def _decode_run_lengths(steps: list[object]) -> list[int]:
    lengths: list[int] = []
    current = 0
    seen_mixed = False
    for step in sorted(steps, key=lambda item: item.local_step):
        if _phase(step) == "decode":
            if seen_mixed:
                current += 1
            continue
        if _phase(step) == "mixed_prefill":
            if seen_mixed:
                lengths.append(current)
            seen_mixed = True
            current = 0
    if seen_mixed:
        lengths.append(current)
    return lengths


def _summarize_engine(steps: list[object], *, scenario: str) -> list[dict[str, object]]:
    by_engine: dict[str, list[object]] = defaultdict(list)
    for step in steps:
        by_engine[step.dp_rank].append(step)

    rows: list[dict[str, object]] = []
    for dp_rank, engine_steps in sorted(by_engine.items()):
        phase_counts = Counter(_phase(step) for step in engine_steps)
        run_lengths = _decode_run_lengths(engine_steps)
        rows.append(
            {
                "dp_rank": dp_rank,
                "total_steps": len(engine_steps),
                "mixed_steps": phase_counts["mixed_prefill"],
                "decode_steps": phase_counts["decode"],
                "mixed_frequency": phase_counts["mixed_prefill"] / max(len(engine_steps), 1),
                "decode_run_count": len(run_lengths),
                "decode_run_p50": statistics.median(run_lengths) if run_lengths else 0,
                "decode_run_p90": _percentile(run_lengths, 0.90),
                "decode_run_max": max(run_lengths) if run_lengths else 0,
            }
        )
    return rows


def _mixed_distribution(steps: list[object]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for step in steps:
        if _phase(step) == "mixed_prefill":
            grouped[(step.dp_rank, step.bucket_tokens, step.generation_requests)].append(
                step.forward_busy_ms
            )

    rows: list[dict[str, object]] = []
    total_mixed = sum(len(values) for values in grouped.values())
    for (dp_rank, bucket_tokens, decode_batch), values in sorted(grouped.items()):
        rows.append(
            {
                "dp_rank": dp_rank,
                "bucket_tokens": bucket_tokens,
                "decode_batch": decode_batch,
                "count": len(values),
                "share": len(values) / max(total_mixed, 1),
                "busy_p50_ms": statistics.median(values),
                "busy_p90_ms": phase446._percentile(values, 0.90),
            }
        )
    return rows


def _phase_joint(steps: list[object]) -> list[dict[str, object]]:
    by_step: dict[int, dict[str, object]] = defaultdict(dict)
    for step in steps:
        by_step[step.local_step][step.dp_rank] = step

    counts: Counter[str] = Counter()
    for _, pair in sorted(by_step.items()):
        if "0" not in pair or "1" not in pair:
            continue
        phase_pair = f"{_phase(pair['0'])}+{_phase(pair['1'])}"
        counts[phase_pair] += 1

    total = sum(counts.values())
    return [
        {"phase_pair": phase_pair, "count": count, "share": count / max(total, 1)}
        for phase_pair, count in sorted(counts.items())
    ]


def _mixed_waves(steps: list[object]) -> list[dict[str, object]]:
    by_step: dict[int, list[object]] = defaultdict(list)
    for step in steps:
        by_step[step.local_step].append(step)

    waves: list[dict[str, object]] = []
    active: list[object] = []
    start_step: int | None = None
    previous_step: int | None = None
    for local_step in sorted(by_step):
        mixed_steps = [step for step in by_step[local_step] if _phase(step) == "mixed_prefill"]
        if mixed_steps:
            if start_step is None:
                start_step = local_step
            active.extend(mixed_steps)
            previous_step = local_step
            continue
        if active and start_step is not None:
            waves.append(_wave_row(start_step, previous_step if previous_step is not None else start_step, active))
            active = []
            start_step = None
            previous_step = None
    if active and start_step is not None:
        waves.append(_wave_row(start_step, previous_step if previous_step is not None else start_step, active))
    return waves


def _wave_row(start_step: int, end_step: int, mixed_steps: list[object]) -> dict[str, object]:
    return {
        "wave_start_step": start_step,
        "wave_end_step": end_step,
        "wave_span_steps": end_step - start_step + 1,
        "wave_mixed_steps": len(mixed_steps),
        "wave_prefill_tokens": sum(step.bucket_tokens - step.generation_requests for step in mixed_steps),
        "wave_decode_batch_p50": statistics.median([step.generation_requests for step in mixed_steps]),
    }


def _tolerance_rows(fingerprint: Fingerprint | None, steps: list[object]) -> list[dict[str, object]]:
    mixed_batches = [step.generation_requests for step in steps if _phase(step) == "mixed_prefill"]
    mixed_buckets = [step.bucket_tokens for step in steps if _phase(step) == "mixed_prefill"]
    return [
        {
            "metric": "mixed_decode_batch_p10_p90",
            "target_low": _percentile(mixed_batches, 0.10),
            "target_high": _percentile(mixed_batches, 0.90),
            "tolerance": "sim p50 must fall inside real p10-p90; distribution EMD <= 0.20",
        },
        {
            "metric": "mixed_bucket_tokens_p10_p90",
            "target_low": _percentile(mixed_buckets, 0.10),
            "target_high": _percentile(mixed_buckets, 0.90),
            "tolerance": "sim p50 must fall inside real p10-p90; no manual phase offset",
        },
        {
            "metric": "phase_joint_share",
            "target_low": 0.0,
            "target_high": 1.0,
            "tolerance": "each major phase-pair share within +/-0.10 absolute",
        },
    ]


def build_fingerprint(steps: list[object], *, scenario: str) -> Fingerprint:
    fingerprint = Fingerprint(
        scenario=scenario,
        engine_summary_rows=_summarize_engine(steps, scenario=scenario),
        mixed_distribution_rows=_mixed_distribution(steps),
        phase_joint_rows=_phase_joint(steps),
        wave_rows=_mixed_waves(steps),
        tolerance_rows=[],
    )
    return Fingerprint(
        scenario=fingerprint.scenario,
        engine_summary_rows=fingerprint.engine_summary_rows,
        mixed_distribution_rows=fingerprint.mixed_distribution_rows,
        phase_joint_rows=fingerprint.phase_joint_rows,
        wave_rows=fingerprint.wave_rows,
        tolerance_rows=_tolerance_rows(fingerprint, steps),
    )


def _read_steps(path: Path, *, tp_width: int) -> list[object]:
    records = phase446.read_event_records(path)
    return phase446.group_event_steps(records, tp_width=tp_width)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    preferred = ["scenario", "section", "dp_rank", "phase_pair", "bucket_tokens", "decode_batch", "count"]
    fields = [field for field in preferred if field in fields] + [
        field for field in fields if field not in preferred
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_md(path: Path, fingerprints: list[Fingerprint]) -> None:
    lines = [
        "# Phase447 DP sequence fingerprint",
        "",
        "结论: 真实 B2b 序列的指纹门必须先于 runtime 修复。门的核心不是均值,而是 mixed 构成、跨 engine 相位联合和 mixed 波大小分布。",
        "",
        "## Engine Summary",
        "",
        "| scenario | dp | total steps | mixed steps | mixed freq | decode run p50 | decode run p90 | decode run max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fp in fingerprints:
        for row in fp.engine_summary_rows:
            lines.append(
                f"| {fp.scenario} | {row['dp_rank']} | {row['total_steps']} | {row['mixed_steps']} | "
                f"{float(row['mixed_frequency']):.4f} | {row['decode_run_p50']} | {row['decode_run_p90']} | {row['decode_run_max']} |"
            )
    lines.extend([
        "",
        "## Phase Joint",
        "",
        "| scenario | phase pair | count | share |",
        "|---|---|---:|---:|",
    ])
    for fp in fingerprints:
        for row in fp.phase_joint_rows:
            lines.append(
                f"| {fp.scenario} | {row['phase_pair']} | {row['count']} | {float(row['share']):.4f} |"
            )
    lines.extend([
        "",
        "## Mixed Composition Top Rows",
        "",
        "| scenario | dp | bucket | decode batch | count | share | busy p50 ms | busy p90 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for fp in fingerprints:
        rows = sorted(fp.mixed_distribution_rows, key=lambda row: int(row["count"]), reverse=True)
        for row in rows[:20]:
            lines.append(
                f"| {fp.scenario} | {row['dp_rank']} | {row['bucket_tokens']} | {row['decode_batch']} | "
                f"{row['count']} | {float(row['share']):.4f} | {float(row['busy_p50_ms']):.3f} | {float(row['busy_p90_ms']):.3f} |"
            )
    lines.extend([
        "",
        "## Gate Tolerances",
        "",
        "| scenario | metric | target low | target high | tolerance |",
        "|---|---|---:|---:|---|",
    ])
    for fp in fingerprints:
        for row in fp.tolerance_rows:
            lines.append(
                f"| {fp.scenario} | {row['metric']} | {row['target_low']} | {row['target_high']} | {row['tolerance']} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(event_8k: Path, event_32k: Path, *, tp_width: int = 1) -> list[Fingerprint]:
    return [
        build_fingerprint(_read_steps(event_8k, tp_width=tp_width), scenario="K2.5-tp4ep8dp2-8k2k"),
        build_fingerprint(_read_steps(event_32k, tp_width=tp_width), scenario="K2.5-tp4ep8dp2-32k3k"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-8k", type=Path, default=DEFAULT_8K_EVENT)
    parser.add_argument("--event-32k", type=Path, default=DEFAULT_32K_EVENT)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    parser.add_argument("--tp-width", type=int, default=1)
    args = parser.parse_args()

    fingerprints = analyze(args.event_8k, args.event_32k, tp_width=args.tp_width)
    rows = [row for fp in fingerprints for row in fp.rows]
    write_csv(args.csv_out, rows)
    write_md(args.md_out, fingerprints)
    print(f"wrote {args.csv_out}")
    print(f"wrote {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
