#!/usr/bin/env python3
"""Phase445-B: recheck prefill gap attribution from Phase443 busy/wall buckets."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "docs/iter_gap_investigation/phase443_b2_diagnostic.csv"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase445_perlayer_prefill_recheck.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase445_perlayer_prefill_recheck.md"

SOURCE = "phase445_perlayer_prefill_recheck"
DEFAULT_READINESS = "No-Go"
MATERIAL_GAP_PCT = 30.0
CLOSE_GAP_PCT = 10.0

CSV_FIELDS = [
    "source",
    "row_type",
    "scenario",
    "ctx_tokens",
    "decode_batch",
    "event_count",
    "confidence",
    "sim_charge_ms",
    "event_forward_busy_median_ms",
    "wall_elapsed_median_ms",
    "busy_minus_sim_ms",
    "wall_minus_busy_ms",
    "busy_gap_share",
    "wall_gap_share",
    "mechanism_candidate",
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
class BucketRow:
    scenario: str
    ctx_tokens: int
    decode_batch: int
    event_count: int
    confidence: str
    sim_charge_ms: float
    event_forward_busy_median_ms: float
    wall_elapsed_median_ms: float

    @property
    def is_mixed_prefill(self) -> bool:
        return self.ctx_tokens > 0 and self.decode_batch > 0


@dataclass(frozen=True)
class ClassifiedBucket:
    scenario: str
    ctx_tokens: int
    decode_batch: int
    event_count: int
    confidence: str
    sim_charge_ms: float
    event_forward_busy_median_ms: float
    wall_elapsed_median_ms: float
    busy_minus_sim_ms: float
    wall_minus_busy_ms: float
    busy_gap_share: float
    wall_gap_share: float
    mechanism_candidate: str
    note: str

    @property
    def is_mixed_prefill(self) -> bool:
        return self.ctx_tokens > 0 and self.decode_batch > 0


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
        "gpu_allowed": False,
        "ssh_allowed": False,
        "runtime_modified": False,
        "perf_database": False,
        "valid_for_default": False,
        "diagnostic_only": True,
        "default_readiness": DEFAULT_READINESS,
    }


def _positive_share(left: float, right: float) -> tuple[float, float]:
    left_pos = max(left, 0.0)
    right_pos = max(right, 0.0)
    total = left_pos + right_pos
    if total <= 0.0:
        return 0.0, 0.0
    return left_pos / total, right_pos / total


def _pct_delta(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return math.inf
    return (numerator / denominator - 1.0) * 100.0


def classify_bucket(row: BucketRow) -> ClassifiedBucket:
    busy_minus_sim = row.event_forward_busy_median_ms - row.sim_charge_ms
    wall_minus_busy = row.wall_elapsed_median_ms - row.event_forward_busy_median_ms
    busy_share, wall_share = _positive_share(busy_minus_sim, wall_minus_busy)

    if row.confidence != "high":
        mechanism = "low_confidence"
        note = "sample count below Phase443 confidence gate"
    elif not row.is_mixed_prefill:
        mechanism = "decode_or_pure_prefill_control"
        note = "not a mixed prefill bucket"
    elif _pct_delta(row.event_forward_busy_median_ms, row.sim_charge_ms) >= MATERIAL_GAP_PCT:
        mechanism = "busy_over_sim"
        note = "forward busy exceeds cb_sim charge; per-layer/kernel charge remains the next gate"
    elif (
        abs(_pct_delta(row.event_forward_busy_median_ms, row.sim_charge_ms)) <= CLOSE_GAP_PCT
        and _pct_delta(row.wall_elapsed_median_ms, row.event_forward_busy_median_ms) >= MATERIAL_GAP_PCT
    ):
        mechanism = "wall_over_busy"
        note = "forward busy matches sim, but wall time is materially higher"
    else:
        mechanism = "mixed_or_inconclusive"
        note = "no single gap source dominates this bucket"

    return ClassifiedBucket(
        scenario=row.scenario,
        ctx_tokens=row.ctx_tokens,
        decode_batch=row.decode_batch,
        event_count=row.event_count,
        confidence=row.confidence,
        sim_charge_ms=row.sim_charge_ms,
        event_forward_busy_median_ms=row.event_forward_busy_median_ms,
        wall_elapsed_median_ms=row.wall_elapsed_median_ms,
        busy_minus_sim_ms=busy_minus_sim,
        wall_minus_busy_ms=wall_minus_busy,
        busy_gap_share=busy_share,
        wall_gap_share=wall_share,
        mechanism_candidate=mechanism,
        note=note,
    )


def read_phase443_buckets(path: Path) -> list[BucketRow]:
    rows: list[BucketRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("row_type") != "bucket":
                continue
            rows.append(
                BucketRow(
                    scenario=row["scenario"],
                    ctx_tokens=int(row["ctx_tokens"]),
                    decode_batch=int(row["decode_batch"]),
                    event_count=int(row["event_count"]),
                    confidence=row["confidence"],
                    sim_charge_ms=float(row["sim_charge_ms"]),
                    event_forward_busy_median_ms=float(row["event_forward_busy_median_ms"]),
                    wall_elapsed_median_ms=float(row["wall_elapsed_median_ms"]),
                )
            )
    if not rows:
        raise ValueError(f"no Phase443 bucket rows in {path}")
    return rows


def classify_buckets(rows: list[BucketRow]) -> list[ClassifiedBucket]:
    return [classify_bucket(row) for row in rows]


def _mixed_high(rows: list[ClassifiedBucket]) -> list[ClassifiedBucket]:
    return [row for row in rows if row.is_mixed_prefill and row.confidence == "high"]


def decide_verdict(rows: list[ClassifiedBucket]) -> str:
    mixed = _mixed_high(rows)
    busy_total = sum(max(row.busy_minus_sim_ms, 0.0) for row in mixed)
    wall_total = sum(max(row.wall_minus_busy_ms, 0.0) for row in mixed)
    total = busy_total + wall_total
    if not mixed or total <= 0.0:
        return "prefill_gap_not_established"
    busy_share = busy_total / total
    wall_share = wall_total / total
    if busy_share >= 0.7:
        return "prefill_gap_is_busy_charge_not_wall_only"
    if wall_share >= 0.7:
        return "prefill_gap_is_wall_over_busy"
    return "prefill_gap_is_mixed"


def summarize(rows: list[ClassifiedBucket]) -> dict[str, float | int | str]:
    mixed = _mixed_high(rows)
    busy_total = sum(max(row.busy_minus_sim_ms, 0.0) for row in mixed)
    wall_total = sum(max(row.wall_minus_busy_ms, 0.0) for row in mixed)
    total = busy_total + wall_total
    counts: dict[str, int] = {}
    for row in mixed:
        counts[row.mechanism_candidate] = counts.get(row.mechanism_candidate, 0) + 1
    return {
        "verdict": decide_verdict(rows),
        "mixed_high_bucket_count": len(mixed),
        "busy_gap_total_ms": busy_total,
        "wall_gap_total_ms": wall_total,
        "busy_gap_share": busy_total / total if total else 0.0,
        "wall_gap_share": wall_total / total if total else 0.0,
        "busy_over_sim_bucket_count": counts.get("busy_over_sim", 0),
        "wall_over_busy_bucket_count": counts.get("wall_over_busy", 0),
        "mixed_or_inconclusive_bucket_count": counts.get("mixed_or_inconclusive", 0),
    }


def _bucket_csv_row(row: ClassifiedBucket) -> dict[str, object]:
    output = _base_row()
    output.update(
        {
            "row_type": "bucket",
            "scenario": row.scenario,
            "ctx_tokens": row.ctx_tokens,
            "decode_batch": row.decode_batch,
            "event_count": row.event_count,
            "confidence": row.confidence,
            "sim_charge_ms": row.sim_charge_ms,
            "event_forward_busy_median_ms": row.event_forward_busy_median_ms,
            "wall_elapsed_median_ms": row.wall_elapsed_median_ms,
            "busy_minus_sim_ms": row.busy_minus_sim_ms,
            "wall_minus_busy_ms": row.wall_minus_busy_ms,
            "busy_gap_share": row.busy_gap_share,
            "wall_gap_share": row.wall_gap_share,
            "mechanism_candidate": row.mechanism_candidate,
            "note": row.note,
        }
    )
    return output


def write_csv(rows: list[ClassifiedBucket], path: Path) -> None:
    summary = summarize(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _fmt(value) for key, value in _bucket_csv_row(row).items()})
        for metric, value in summary.items():
            output = _base_row()
            output.update(
                {
                    "row_type": "summary",
                    "metric": metric,
                    "value": value,
                    "note": "high-confidence mixed prefill buckets only",
                }
            )
            writer.writerow({key: _fmt(output.get(key, "")) for key in CSV_FIELDS})


def _md_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(value) for value in row) + " |")
    return lines


def write_markdown(rows: list[ClassifiedBucket], path: Path) -> None:
    summary = summarize(rows)
    mixed = sorted(
        _mixed_high(rows),
        key=lambda row: max(row.busy_minus_sim_ms, 0.0) + max(row.wall_minus_busy_ms, 0.0),
        reverse=True,
    )
    top_rows = [
        [
            row.ctx_tokens,
            row.decode_batch,
            row.event_count,
            row.sim_charge_ms,
            row.event_forward_busy_median_ms,
            row.wall_elapsed_median_ms,
            row.busy_minus_sim_ms,
            row.wall_minus_busy_ms,
            row.mechanism_candidate,
        ]
        for row in mixed[:12]
    ]

    lines = [
        "# Phase445-B Per-Layer Prefill Recheck",
        "",
        f"- Verdict: `{summary['verdict']}`.",
        f"- High-confidence mixed prefill buckets: `{summary['mixed_high_bucket_count']}`.",
        f"- Busy-gap share: `{summary['busy_gap_share']:.3f}`; wall-gap share: `{summary['wall_gap_share']:.3f}`.",
        "- The current evidence is whole-forward event timing, not per-layer timing. It can reject a wall-only explanation, but it cannot split the busy excess by layer without a lower-level map.",
        "",
        "## Summary",
    ]
    lines.extend(
        _md_table(
            ["metric", "value"],
            [[key, value] for key, value in summary.items()],
        )
    )
    lines.extend(["", "## Largest Mixed-Prefill Buckets"])
    lines.extend(
        _md_table(
            [
                "ctx_tokens",
                "decode_batch",
                "event_count",
                "sim_ms",
                "busy_ms",
                "wall_ms",
                "busy_minus_sim",
                "wall_minus_busy",
                "candidate",
            ],
            top_rows,
        )
    )
    lines.extend(
        [
            "",
            "## Next Gate",
            "",
            "Do not model the 8k prefill residual as a generic host or DP wall term. The high-confidence mixed prefill buckets show the forward-busy event is already materially above cb_sim charge. The next valid gate is either a lower-overhead B2b busy measurement with finer attribution, or a category/layer mapping that explains which part of the busy charge is missing.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(input_csv: Path, csv_out: Path, md_out: Path) -> str:
    buckets = read_phase443_buckets(input_csv)
    classified = classify_buckets(buckets)
    verdict = decide_verdict(classified)
    write_csv(classified, csv_out)
    write_markdown(classified, md_out)
    return verdict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    verdict = run_analysis(args.input, args.csv_out, args.md_out)
    print(f"wrote {args.csv_out}")
    print(f"wrote {args.md_out}")
    print(f"verdict={verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
