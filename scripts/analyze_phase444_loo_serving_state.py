#!/usr/bin/env python3
"""Phase444-B: leave-one-out error audit for vLLM serving-state perf rows."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    REPO_ROOT
    / "src/aiconfigurator/systems/data/h200_sxm/vllm/0.19.0/vllm_serving_state_perf.txt"
)
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase444_loo_serving_state.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase444_loo_serving_state.md"

INTERIOR_MEDIAN_GATE_PCT = 10.0
INTERIOR_P90_GATE_PCT = 20.0
DEFAULT_READINESS = "No-Go"

CSV_FIELDS = [
    "source",
    "row_type",
    "phase",
    "row_kind",
    "category",
    "bucket_tokens",
    "decode_batch",
    "actual_ms",
    "predicted_ms",
    "error_pct",
    "fold",
    "gate",
    "sample_count",
    "median_error_pct",
    "p90_error_pct",
    "loo_gate",
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
class ServingStateRow:
    model: str
    topology: str
    phase: str
    row_kind: str
    category: str
    bucket_tokens: int
    decode_batch: int
    hidden_size: int
    topk: int
    moe_ep_size: int
    quant_runtime: str
    latency_ms: float
    kernel_source: str
    provenance: str

    @property
    def scope_key(self) -> tuple[object, ...]:
        return (
            self.model,
            self.topology,
            self.phase,
            self.row_kind,
            self.category,
            self.hidden_size,
            self.topk,
            self.moe_ep_size,
            self.quant_runtime,
        )


@dataclass(frozen=True)
class LOORow:
    phase: str
    row_kind: str
    category: str
    bucket_tokens: int
    decode_batch: int
    actual_ms: float
    predicted_ms: float | None
    error_pct: float | None
    fold: str
    gate: str
    note: str


def _base_row() -> dict[str, object]:
    return {
        "source": "phase444_loo_serving_state",
        "gpu_allowed": False,
        "ssh_allowed": False,
        "runtime_modified": False,
        "perf_database": False,
        "valid_for_default": False,
        "diagnostic_only": True,
        "default_readiness": DEFAULT_READINESS,
    }


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
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def _interp_1d(left: int, right: int, left_value: float, right_value: float, value: int) -> float:
    if left == right:
        return left_value
    ratio = (value - left) / (right - left)
    return left_value * (1.0 - ratio) + right_value * ratio


def _bracket(value: int, values: list[int]) -> tuple[int, int] | None:
    if value in values:
        return value, value
    smaller = [item for item in values if item < value]
    larger = [item for item in values if item > value]
    if not smaller or not larger:
        return None
    return max(smaller), min(larger)


def _predict(table: dict[int, dict[int, float]], bucket_tokens: int, decode_batch: int) -> float | None:
    token_bracket = _bracket(bucket_tokens, sorted(table))
    if token_bracket is None:
        return None
    token_left, token_right = token_bracket
    token_values: list[float] = []
    for token in (token_left, token_right):
        batch_table = table[token]
        batch_bracket = _bracket(decode_batch, sorted(batch_table))
        if batch_bracket is None:
            return None
        batch_left, batch_right = batch_bracket
        token_values.append(
            _interp_1d(
                batch_left,
                batch_right,
                batch_table[batch_left],
                batch_table[batch_right],
                decode_batch,
            )
        )
    return _interp_1d(token_left, token_right, token_values[0], token_values[1], bucket_tokens)


def _fold_for(row: ServingStateRow, table: dict[int, dict[int, float]], predicted: float | None) -> str:
    all_buckets = sorted(table)
    all_batches = sorted({batch for batch_table in table.values() for batch in batch_table})
    if predicted is None:
        if row.bucket_tokens <= min(all_buckets, default=row.bucket_tokens) or row.bucket_tokens >= max(
            all_buckets,
            default=row.bucket_tokens,
        ):
            return "frontier"
        if row.decode_batch <= min(all_batches, default=row.decode_batch) or row.decode_batch >= max(
            all_batches,
            default=row.decode_batch,
        ):
            return "frontier"
        return "interpolation_gap"
    if (
        min(all_buckets, default=row.bucket_tokens) < row.bucket_tokens < max(all_buckets, default=row.bucket_tokens)
        and min(all_batches, default=row.decode_batch) < row.decode_batch < max(all_batches, default=row.decode_batch)
    ):
        return "interior"
    return "frontier"


def read_serving_rows(path: Path) -> list[ServingStateRow]:
    rows: list[ServingStateRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                ServingStateRow(
                    model=row["model"],
                    topology=row["topology"],
                    phase=row["phase"],
                    row_kind=row.get("row_kind") or "category",
                    category=row["category"],
                    bucket_tokens=int(row["bucket_tokens"]),
                    decode_batch=int(row["decode_batch"]),
                    hidden_size=int(row["hidden_size"]),
                    topk=int(row["topk"]),
                    moe_ep_size=int(row["moe_ep_size"]),
                    quant_runtime=row["quant_runtime"],
                    latency_ms=float(row["latency"]),
                    kernel_source=row["kernel_source"],
                    provenance=row["provenance"],
                )
            )
    if not rows:
        raise ValueError(f"no serving-state rows in {path}")
    return rows


def run_loo(rows: list[ServingStateRow]) -> list[LOORow]:
    by_scope: dict[tuple[object, ...], list[ServingStateRow]] = defaultdict(list)
    for row in rows:
        by_scope[row.scope_key].append(row)

    results: list[LOORow] = []
    for scope_rows in by_scope.values():
        for held in scope_rows:
            table: dict[int, dict[int, float]] = defaultdict(dict)
            for row in scope_rows:
                if row is held:
                    continue
                table[row.bucket_tokens][row.decode_batch] = row.latency_ms
            predicted = _predict(table, held.bucket_tokens, held.decode_batch) if table else None
            error_pct = (
                abs(predicted / held.latency_ms - 1.0) * 100.0
                if predicted is not None and held.latency_ms > 0
                else None
            )
            fold = _fold_for(held, table, predicted)
            gate = "not_counted"
            if fold == "interior" and error_pct is not None:
                gate = "passed" if error_pct <= INTERIOR_P90_GATE_PCT else "failed"
            results.append(
                LOORow(
                    phase=held.phase,
                    row_kind=held.row_kind,
                    category=held.category,
                    bucket_tokens=held.bucket_tokens,
                    decode_batch=held.decode_batch,
                    actual_ms=held.latency_ms,
                    predicted_ms=predicted,
                    error_pct=error_pct,
                    fold=fold,
                    gate=gate,
                    note="leave_one_out_inner_only",
                )
            )
    return results


def summarize_loo(rows: list[LOORow]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[LOORow]] = defaultdict(list)
    for row in rows:
        grouped[(row.phase, row.row_kind, row.category, row.fold)].append(row)
    out: list[dict[str, object]] = []
    for (phase, row_kind, category, fold), items in sorted(grouped.items()):
        errors = [row.error_pct for row in items if row.error_pct is not None]
        median = _percentile(errors, 0.50) if errors else math.nan
        p90 = _percentile(errors, 0.90) if errors else math.nan
        if fold == "interior" and errors:
            loo_gate = (
                "passed"
                if median <= INTERIOR_MEDIAN_GATE_PCT and p90 <= INTERIOR_P90_GATE_PCT
                else "failed"
            )
        else:
            loo_gate = "not_counted"
        out.append(
            {
                "phase": phase,
                "row_kind": row_kind,
                "category": category,
                "fold": fold,
                "sample_count": len(items),
                "median_error_pct": median,
                "p90_error_pct": p90,
                "loo_gate": loo_gate,
            }
        )
    return out


def _detail_dict(row: LOORow) -> dict[str, object]:
    base = _base_row()
    base.update(
        {
            "row_type": "loo_detail",
            "phase": row.phase,
            "row_kind": row.row_kind,
            "category": row.category,
            "bucket_tokens": row.bucket_tokens,
            "decode_batch": row.decode_batch,
            "actual_ms": row.actual_ms,
            "predicted_ms": row.predicted_ms,
            "error_pct": row.error_pct,
            "fold": row.fold,
            "gate": row.gate,
            "note": row.note,
        }
    )
    return base


def _summary_dict(row: dict[str, object]) -> dict[str, object]:
    base = _base_row()
    base.update({"row_type": "loo_summary", **row})
    return base


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field, "")) for field in CSV_FIELDS})


def write_markdown(path: Path, summaries: list[dict[str, object]]) -> None:
    failed = [
        row
        for row in summaries
        if row.get("fold") == "interior" and row.get("loo_gate") == "failed"
    ]
    verdict = (
        "LOO 不支持本 phase 改 query 规则: interior median 可接受,但 p90 超过 20% 的分层仍存在。"
        if failed
        else "LOO 支持后续把对应 interior interpolation_gap 改成有证据的插值命中。"
    )
    lines = [
        "# Phase444-B Serving-State LOO",
        "",
        verdict,
        "",
        "结论按 `(phase, row_kind, category, fold)` 分层;只有 interior 计入门。",
        "",
        "| phase | row_kind | category | fold | n | median err % | p90 err % | gate |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            "| "
            f"{row['phase']} | {row['row_kind']} | {row['category']} | {row['fold']} | "
            f"{row['sample_count']} | {_fmt(row['median_error_pct'])} | "
            f"{_fmt(row['p90_error_pct'])} | {row['loo_gate']} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- diagnostic_only=true; valid_for_default=false.",
            "- This phase measures interpolation error only; it does not change PerfDB query behavior.",
            "- Frontier and interpolation-gap rows are reported but not counted in the LOO gate.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(input_path: Path, csv_path: Path, md_path: Path) -> list[dict[str, object]]:
    rows = read_serving_rows(input_path)
    loo_rows = run_loo(rows)
    summaries = summarize_loo(loo_rows)
    output = [_detail_dict(row) for row in loo_rows] + [_summary_dict(row) for row in summaries]
    write_csv(csv_path, output)
    write_markdown(md_path, summaries)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    summaries = run(args.input, args.csv, args.md)
    failed = [row for row in summaries if row["fold"] == "interior" and row["loo_gate"] == "failed"]
    print(f"wrote {args.csv} and {args.md}; interior_failed={len(failed)}")


if __name__ == "__main__":
    main()
