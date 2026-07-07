#!/usr/bin/env python3
"""Phase440: derive non-attention total serving-state rows from iteration logs."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import analyze_phase413_steady_decompose as phase413  # noqa: E402


SOURCE = "phase440_nonattn_total"
KERNEL_SOURCE = "phase440_nonattn_total"
MODEL = "kimi-k2.5"
TOPOLOGY = "tp4dp2ep8"
QUANT_RUNTIME = "CompressedTensorsWNA16MarlinMoEMethod"
HIDDEN_SIZE = 7168
TOPK = 8
MOE_EP_SIZE = 8
DEFAULT_READINESS = "No-Go"
TARGET_SCENARIO = "K2.5-tp4ep8dp2-8k2k"
TARGET_MIN_DECODE_BATCH = 56
TARGET_MIN_BUCKET_TOKENS = 7000
TARGET_CATEGORIES = ("ep_a2a", "moe_gemm_or_aux", "other_cuda", "collective_other")

DEFAULT_ARTIFACT = (
    REPO_ROOT / "docs/iter_gap_investigation/phase425_8k2k_sweep/K2.5-tp4ep8dp2-8k2k"
)
DEFAULT_PERFDB = (
    REPO_ROOT
    / "src/aiconfigurator/systems/data/h200_sxm/vllm/0.19.0/vllm_serving_state_perf.txt"
)
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase440_nonattn_total.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase440_nonattn_total.md"

REPORT_FIELDS = [
    "source",
    "row_type",
    "scenario",
    "artifact_dir",
    "phase",
    "row_kind",
    "category",
    "bucket_tokens",
    "decode_batch",
    "step_count",
    "latency_ms",
    "elapsed_ms_mean",
    "attention_ms_mean",
    "baseline_value",
    "candidate_value",
    "reconstruction_error_pct",
    "reconstruction_gate",
    "coverage_gate",
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


class ServingRow(NamedTuple):
    source: str
    phase: str
    category: str
    bucket_tokens: int
    decode_batch: int
    latency_ms: float
    provenance: str
    row_kind: str = "category"

    @property
    def key(self) -> tuple[str, str, str, int, int]:
        return (self.phase, self.row_kind, self.category, self.bucket_tokens, self.decode_batch)


@dataclass(frozen=True)
class StepRecord:
    source: str
    bucket_tokens: int
    decode_batch: int
    elapsed_ms: float
    context_attention_ms: float
    generation_attention_ms: float

    @property
    def non_attn_total_ms(self) -> float:
        return self.elapsed_ms - self.context_attention_ms - self.generation_attention_ms


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


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _serve_log_path(artifact_dir: Path) -> Path:
    for name in ("serve.log", "serve.log.gz"):
        path = artifact_dir / name
        if path.exists():
            return path
    raise ValueError(f"missing serve.log under {artifact_dir}")


def _base_row() -> dict[str, object]:
    return {
        "source": SOURCE,
        "scenario": TARGET_SCENARIO,
        "gpu_allowed": False,
        "ssh_allowed": False,
        "runtime_modified": True,
        "perf_database": True,
        "valid_for_default": False,
        "diagnostic_only": False,
        "default_readiness": DEFAULT_READINESS,
    }


def derive_non_attn_total_rows(steps: list[StepRecord]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int], list[StepRecord]] = defaultdict(list)
    for step in steps:
        if step.non_attn_total_ms <= 0:
            continue
        grouped[(step.bucket_tokens, step.decode_batch)].append(step)

    rows: list[dict[str, object]] = []
    for (bucket_tokens, decode_batch), group in sorted(grouped.items()):
        latency = _mean([step.non_attn_total_ms for step in group])
        elapsed = _mean([step.elapsed_ms for step in group])
        attention = _mean(
            [step.context_attention_ms + step.generation_attention_ms for step in group]
        )
        row = _base_row()
        row.update(
            {
                "row_type": "non_attn_total_curve",
                "artifact_dir": ",".join(sorted({step.source for step in group})),
                "phase": "mixed_prefill",
                "row_kind": "non_attn_total",
                "category": "non_attn_total",
                "bucket_tokens": bucket_tokens,
                "decode_batch": decode_batch,
                "step_count": len(group),
                "latency_ms": latency,
                "elapsed_ms_mean": elapsed,
                "attention_ms_mean": attention,
                "coverage_gate": (
                    "target_covered"
                    if decode_batch >= TARGET_MIN_DECODE_BATCH and bucket_tokens >= TARGET_MIN_BUCKET_TOKENS
                    else "below_target_working_point"
                ),
                "note": "derived_from_iteration_elapsed_minus_sim_attention",
            }
        )
        rows.append(row)
    return rows


def read_serving_perfdb_rows(path: Path) -> list[ServingRow]:
    rows: list[ServingRow] = []
    for row in _read_rows(path):
        rows.append(
            ServingRow(
                source=row.get("kernel_source") or "current_perfdb",
                phase=row["phase"],
                row_kind=row.get("row_kind") or "category",
                category=row["category"],
                bucket_tokens=int(row["bucket_tokens"]),
                decode_batch=int(row["decode_batch"]),
                latency_ms=float(row["latency"]),
                provenance=row.get("provenance", ""),
            )
        )
    return rows


def calibration_gate_rows(
    non_attn_rows: list[dict[str, object]],
    category_rows: list[ServingRow],
    *,
    tolerance_pct: float = 10.0,
) -> list[dict[str, object]]:
    category_by_key: dict[tuple[int, int], dict[str, float]] = defaultdict(dict)
    for row in category_rows:
        if row.phase != "mixed_prefill" or row.row_kind != "category":
            continue
        if row.category not in TARGET_CATEGORIES:
            continue
        category_by_key[(row.bucket_tokens, row.decode_batch)][row.category] = row.latency_ms

    gate_rows: list[dict[str, object]] = []
    for row in non_attn_rows:
        if row.get("row_type") != "non_attn_total_curve":
            continue
        key = (int(row["bucket_tokens"]), int(row["decode_batch"]))
        categories = category_by_key.get(key, {})
        if set(categories) != set(TARGET_CATEGORIES):
            continue
        baseline = sum(categories.values())
        candidate = float(row["latency_ms"])
        error_pct = abs(candidate / baseline - 1.0) * 100.0 if baseline > 0 else math.inf
        gate = _base_row()
        gate.update(
            {
                "row_type": "calibration_gate",
                "phase": "mixed_prefill",
                "row_kind": "non_attn_total",
                "category": "non_attn_total",
                "bucket_tokens": key[0],
                "decode_batch": key[1],
                "baseline_value": baseline,
                "candidate_value": candidate,
                "reconstruction_error_pct": error_pct,
                "reconstruction_gate": "passed" if error_pct <= tolerance_pct else "failed",
                "note": "overlap_non_attn_total_vs_category_sum",
            }
        )
        gate_rows.append(gate)
    if not gate_rows:
        gate = _base_row()
        gate.update(
            {
                "row_type": "calibration_gate",
                "phase": "mixed_prefill",
                "row_kind": "non_attn_total",
                "category": "non_attn_total",
                "reconstruction_gate": "failed",
                "note": "no_overlap_rows_between_non_attn_total_and_category_sum",
            }
        )
        gate_rows.append(gate)
    return gate_rows


def _load_validate_calc():
    from aiconfigurator.sdk.backends.cb_simulator.iteration_latency import (  # noqa: PLC0415
        IterationLatencyCalculator,
    )
    from scripts import validate_cb_simulator as validate_cb  # noqa: PLC0415

    model, db, backend = validate_cb._load_model_and_db(tp=4, dp=2, moe_tp=1, moe_ep=8)
    calc = IterationLatencyCalculator(
        backend,
        model,
        db,
        overlap_factor=0.0,
        per_iteration_overhead_ms=0.0,
    )
    return calc


def build_step_records_from_artifact(artifact_dir: Path) -> list[StepRecord]:
    steps = phase413.parse_iteration_steps(_serve_log_path(artifact_dir))
    timelines = phase413.build_wallclock_timelines(steps)
    calc = _load_validate_calc()
    attention_cache: dict[tuple[int, int, int, int], tuple[float, float]] = {}
    out: list[StepRecord] = []
    for timeline in timelines.values():
        for timed in timeline:
            step = timed.step
            if step.ctx_tokens <= 0 or step.generation_tokens <= 0:
                continue
            bucket_tokens = step.ctx_tokens + step.generation_tokens
            key = (
                step.ctx_tokens,
                step.ctx_requests,
                step.generation_requests,
                bucket_tokens,
            )
            if key not in attention_cache:
                calc.compute(
                    prefill_tokens=step.ctx_tokens,
                    prefill_batch_size=max(1, step.ctx_requests),
                    prefill_seq_len=8000,
                    decode_batch_size=step.generation_requests,
                    decode_avg_kv_len=8000,
                )
                breakdown = calc.get_last_breakdown()
                if breakdown is None:
                    raise RuntimeError("missing iteration latency breakdown")
                attention_cache[key] = (
                    breakdown.context_attention_ms,
                    breakdown.generation_attention_ms,
                )
            context_attention_ms, generation_attention_ms = attention_cache[key]
            out.append(
                StepRecord(
                    source=str(artifact_dir.relative_to(REPO_ROOT)),
                    bucket_tokens=bucket_tokens,
                    decode_batch=step.generation_requests,
                    elapsed_ms=step.elapsed_ms,
                    context_attention_ms=context_attention_ms,
                    generation_attention_ms=generation_attention_ms,
                )
            )
    return out


def inventory_rows(non_attn_rows: list[dict[str, object]], *, artifact_dir: Path) -> list[dict[str, object]]:
    batches = [
        int(row["decode_batch"])
        for row in non_attn_rows
        if row.get("row_type") == "non_attn_total_curve"
    ]
    buckets = [
        int(row["bucket_tokens"])
        for row in non_attn_rows
        if row.get("row_type") == "non_attn_total_curve"
    ]
    row = _base_row()
    max_batch = max(batches) if batches else 0
    target_batches = [
        int(row["decode_batch"])
        for row in non_attn_rows
        if row.get("row_type") == "non_attn_total_curve"
        and int(row["bucket_tokens"]) >= TARGET_MIN_BUCKET_TOKENS
    ]
    max_target_batch = max(target_batches) if target_batches else 0
    row.update(
        {
            "row_type": "inventory",
            "artifact_dir": str(artifact_dir.relative_to(REPO_ROOT)),
            "phase": "mixed_prefill",
            "metric": "existing_iteration_log_coverage",
            "value": max_target_batch,
            "step_count": sum(
                int(item.get("step_count") or 0)
                for item in non_attn_rows
                if item.get("row_type") == "non_attn_total_curve"
            ),
            "bucket_tokens": min(buckets) if buckets else "",
            "decode_batch": max_batch,
            "coverage_gate": (
                "passed" if max_target_batch >= TARGET_MIN_DECODE_BATCH else "needs_pure_log_gpu_collect"
            ),
            "note": (
                f"target_bucket_tokens_min={TARGET_MIN_BUCKET_TOKENS};"
                f"target_decode_batch_min={TARGET_MIN_DECODE_BATCH};"
                f"raw_max_decode_batch={max_batch}"
            ),
        }
    )
    return [row]


def build_phase440_rows(
    *,
    artifact_dir: Path = DEFAULT_ARTIFACT,
    perfdb: Path = DEFAULT_PERFDB,
) -> list[dict[str, object]]:
    step_records = build_step_records_from_artifact(artifact_dir)
    non_attn_rows = derive_non_attn_total_rows(step_records)
    category_rows = read_serving_perfdb_rows(perfdb)
    rows: list[dict[str, object]] = []
    rows.extend(inventory_rows(non_attn_rows, artifact_dir=artifact_dir))
    rows.extend(non_attn_rows)
    rows.extend(calibration_gate_rows(non_attn_rows, category_rows))
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field, "")) for field in REPORT_FIELDS})


def _row_from_report(row: dict[str, object]) -> ServingRow:
    return ServingRow(
        source=KERNEL_SOURCE,
        phase=str(row["phase"]),
        row_kind="non_attn_total",
        category="non_attn_total",
        bucket_tokens=int(row["bucket_tokens"]),
        decode_batch=int(row["decode_batch"]),
        latency_ms=float(row["latency_ms"]),
        provenance=str(row.get("note") or "derived_from_iteration_elapsed_minus_sim_attention"),
    )


def write_candidate_perfdb(
    *,
    base_rows: list[ServingRow],
    report_rows: list[dict[str, object]],
    path: Path,
) -> None:
    merged = {row.key: row for row in base_rows}
    for row in report_rows:
        if row.get("row_type") != "non_attn_total_curve":
            continue
        serving_row = _row_from_report(row)
        if serving_row.key not in merged:
            merged[serving_row.key] = serving_row

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PERFDB_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in sorted(merged.values(), key=lambda item: item.key):
            writer.writerow(
                {
                    "framework": "VLLM",
                    "version": "0.19.0",
                    "device": "NVIDIA H200",
                    "model": MODEL,
                    "topology": TOPOLOGY,
                    "phase": row.phase,
                    "row_kind": row.row_kind,
                    "category": row.category,
                    "kernel_source": row.source,
                    "bucket_tokens": row.bucket_tokens,
                    "decode_batch": row.decode_batch,
                    "hidden_size": HIDDEN_SIZE,
                    "topk": TOPK,
                    "moe_ep_size": MOE_EP_SIZE,
                    "quant_runtime": QUANT_RUNTIME,
                    "latency": _fmt(row.latency_ms),
                    "provenance": row.provenance,
                }
            )


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    inventory = next((row for row in rows if row.get("row_type") == "inventory"), {})
    gate_rows = [row for row in rows if row.get("row_type") == "calibration_gate"]
    curve_rows = [row for row in rows if row.get("row_type") == "non_attn_total_curve"]
    high_rows = [
        row
        for row in curve_rows
        if int(row.get("decode_batch") or 0) >= TARGET_MIN_DECODE_BATCH
        and int(row.get("bucket_tokens") or 0) >= TARGET_MIN_BUCKET_TOKENS
    ]
    gate_passed = bool(gate_rows) and all(row.get("reconstruction_gate") == "passed" for row in gate_rows)
    coverage_passed = bool(high_rows)
    if not gate_passed:
        verdict = "calibration_gate_failed_do_not_ingest"
    elif gate_passed and coverage_passed:
        verdict = "non_attn_total_rows_ready_for_candidate_ab"
    elif not coverage_passed:
        verdict = "existing_logs_do_not_cover_target_batch_run_pure_log_collect"
    else:
        verdict = "calibration_gate_failed_do_not_ingest"

    lines = [
        "# Phase440 non-attention total rows",
        "",
        f"- verdict: `{verdict}`.",
        f"- existing log max decode_batch: {inventory.get('value', '')}.",
        f"- derived curve rows: {len(curve_rows)}.",
        f"- target decode_batch minimum: {TARGET_MIN_DECODE_BATCH}.",
        f"- target bucket_tokens minimum: {TARGET_MIN_BUCKET_TOKENS}.",
        "- GPU pure-log collection is skipped while the calibration hard gate is failed.",
        "- Default AIC remains No-Go until A/B validation is clean.",
        "",
        "## Calibration",
        "",
        "| bucket_tokens | decode_batch | category sum ms | non_attn_total ms | error % | gate |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in gate_rows:
        lines.append(
            "| {bucket} | {batch} | {baseline} | {candidate} | {error} | {gate} |".format(
                bucket=row.get("bucket_tokens", ""),
                batch=row.get("decode_batch", ""),
                baseline=_fmt(row.get("baseline_value", "")),
                candidate=_fmt(row.get("candidate_value", "")),
                error=_fmt(row.get("reconstruction_error_pct", "")),
                gate=row.get("reconstruction_gate", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Coverage",
            "",
            "| decode_batch | bucket_tokens | steps | latency ms | coverage |",
            "|---:|---:|---:|---:|---|",
        ]
    )
    for row in curve_rows:
        lines.append(
            f"| {row.get('decode_batch', '')} | {row.get('bucket_tokens', '')} | "
            f"{row.get('step_count', '')} | {_fmt(row.get('latency_ms', ''))} | "
            f"{row.get('coverage_gate', '')} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--perfdb", type=Path, default=DEFAULT_PERFDB)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--perfdb-out", type=Path, default=None)
    args = parser.parse_args()

    rows = build_phase440_rows(artifact_dir=args.artifact_dir, perfdb=args.perfdb)
    write_csv(rows, args.csv)
    write_markdown(rows, args.md)
    if args.perfdb_out is not None:
        base_rows = read_serving_perfdb_rows(args.perfdb)
        write_candidate_perfdb(base_rows=base_rows, report_rows=rows, path=args.perfdb_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
