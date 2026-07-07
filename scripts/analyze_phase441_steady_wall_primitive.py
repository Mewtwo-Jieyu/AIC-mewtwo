#!/usr/bin/env python3
"""Phase441: audit whether whole-step non-attention rows can be a steady wall-clock primitive."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import analyze_phase413_steady_decompose as phase413  # noqa: E402
from scripts import validate_cb_simulator as validate_cb  # noqa: E402


SOURCE = "phase441_steady_wall_primitive"
SCENARIO = "K2.5-tp4ep8dp2-8k2k"
DEFAULT_ARTIFACT = (
    REPO_ROOT / "docs/iter_gap_investigation/phase425_8k2k_sweep/K2.5-tp4ep8dp2-8k2k"
)
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase441_steady_wall_primitive.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase441_steady_wall_primitive.md"
DEFAULT_READINESS = "No-Go"

MODEL = "kimi-k2.5"
TOPOLOGY = "tp4dp2ep8"
QUANT_RUNTIME = "CompressedTensorsWNA16MarlinMoEMethod"
HIDDEN_SIZE = 7168
TOPK = 8
MOE_EP_SIZE = 8
TARGET_CATEGORIES = ("ep_a2a", "moe_gemm_or_aux", "other_cuda", "collective_other")

HIGH_BATCH_MIN = 45
TARGET_MIN_BUCKET_TOKENS = 7000
MIN_EXPLAINED_SHARE = 0.80
HIGH_BATCH_ERROR_LIMIT_PCT = 15.0
ATTENTION_ERROR_UPPER_FRACTION = 0.10
PHASE433_HOST_GAP_UPPER_MS = 20.0

CSV_FIELDS = [
    "source",
    "row_type",
    "scenario",
    "artifact_dir",
    "bucket_tokens",
    "decode_batch",
    "step_count",
    "category_sum_ms",
    "non_attn_total_ms",
    "candidate_gap_ms",
    "candidate_error_pct",
    "attention_ms",
    "peer_wait_ms",
    "host_gap_upper_ms",
    "attention_error_upper_ms",
    "explained_ms",
    "unexplained_ms",
    "explained_share",
    "coverage_band",
    "missing_categories",
    "explain_gate",
    "high_batch_gate",
    "life_gate",
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
class AuditPoint:
    bucket_tokens: int
    decode_batch: int
    step_count: int
    category_sum_ms: float
    non_attn_total_ms: float
    attention_ms: float
    peer_wait_ms: float
    host_gap_upper_ms: float
    attention_error_upper_ms: float

    @property
    def candidate_gap_ms(self) -> float:
        return self.non_attn_total_ms - self.category_sum_ms

    @property
    def candidate_error_pct(self) -> float:
        if self.category_sum_ms <= 0:
            return math.inf
        return abs(self.non_attn_total_ms / self.category_sum_ms - 1.0) * 100.0

    @property
    def explained_ms(self) -> float:
        if self.candidate_gap_ms <= 0:
            return 0.0
        direct = self.peer_wait_ms + self.host_gap_upper_ms + self.attention_error_upper_ms
        return min(self.candidate_gap_ms, max(0.0, direct))

    @property
    def unexplained_ms(self) -> float:
        return max(0.0, self.candidate_gap_ms - self.explained_ms)

    @property
    def explained_share(self) -> float:
        if self.candidate_gap_ms <= 0:
            return 1.0
        return self.explained_ms / self.candidate_gap_ms

    @property
    def is_high_batch(self) -> bool:
        return self.bucket_tokens >= TARGET_MIN_BUCKET_TOKENS and self.decode_batch >= HIGH_BATCH_MIN


@dataclass(frozen=True)
class _StepAudit:
    bucket_tokens: int
    decode_batch: int
    elapsed_ms: float
    attention_ms: float
    peer_wait_ms: float

    @property
    def non_attn_total_ms(self) -> float:
        return self.elapsed_ms - self.attention_ms


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


def _safe_ratio(a: float, b: float) -> float:
    if b <= 0:
        return math.inf
    return a / b


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


def summarize_life_gate(points: list[AuditPoint]) -> dict[str, float | int | str]:
    positive = [point for point in points if point.candidate_gap_ms > 0]
    total_gap = sum(point.candidate_gap_ms * point.step_count for point in positive)
    total_explained = sum(point.explained_ms * point.step_count for point in positive)
    explained_share = _safe_ratio(total_explained, total_gap) if total_gap > 0 else 1.0

    high = [point for point in points if point.is_high_batch]
    high_errors = [point.candidate_error_pct for point in high]
    high_batch_max_error_pct = max(high_errors) if high_errors else math.inf
    high_batch_mean_error_pct = (
        sum(point.candidate_error_pct * point.step_count for point in high)
        / sum(point.step_count for point in high)
        if high
        else math.inf
    )

    explain_gate = "passed" if explained_share >= MIN_EXPLAINED_SHARE else "failed"
    high_batch_gate = (
        "passed"
        if high and high_batch_max_error_pct <= HIGH_BATCH_ERROR_LIMIT_PCT
        else "failed"
    )
    life_gate = "passed" if explain_gate == "passed" and high_batch_gate == "passed" else "failed"
    return {
        "total_gap_ms_weighted": total_gap,
        "total_explained_ms_weighted": total_explained,
        "explained_share": explained_share,
        "high_batch_row_count": len(high),
        "high_batch_max_error_pct": high_batch_max_error_pct,
        "high_batch_mean_error_pct": high_batch_mean_error_pct,
        "explain_gate": explain_gate,
        "high_batch_gate": high_batch_gate,
        "life_gate": life_gate,
    }


def _serve_log_path(artifact_dir: Path) -> Path:
    for name in ("serve.log", "serve.log.gz"):
        path = artifact_dir / name
        if path.exists():
            return path
    raise ValueError(f"missing serve.log under {artifact_dir}")


def _metrics_path(artifact_dir: Path) -> Path:
    for name in ("metrics.jsonl", "metrics.jsonl.gz"):
        path = artifact_dir / name
        if path.exists():
            return path
    raise ValueError(f"missing metrics under {artifact_dir}")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_calc_and_db():
    from aiconfigurator.sdk.backends.cb_simulator.iteration_latency import (  # noqa: PLC0415
        IterationLatencyCalculator,
    )

    model, db, backend = validate_cb._load_model_and_db(tp=4, dp=2, moe_tp=1, moe_ep=8)
    calc = IterationLatencyCalculator(
        backend,
        model,
        db,
        overlap_factor=0.0,
        per_iteration_overhead_ms=0.0,
    )
    return calc, db


def _attention_ms(calc, *, ctx_tokens: int, ctx_requests: int, decode_batch: int) -> float:
    calc.compute(
        prefill_tokens=ctx_tokens,
        prefill_batch_size=max(1, ctx_requests),
        prefill_seq_len=8000,
        decode_batch_size=decode_batch,
        decode_avg_kv_len=8000,
    )
    breakdown = calc.get_last_breakdown()
    if breakdown is None:
        raise RuntimeError("missing iteration latency breakdown")
    return breakdown.context_attention_ms + breakdown.generation_attention_ms


def _category_sum_ms(db, *, bucket_tokens: int, decode_batch: int) -> float | None:
    total = 0.0
    for category in TARGET_CATEGORIES:
        result = db.query_vllm_serving_state(
            model=MODEL,
            topology=TOPOLOGY,
            phase="mixed_prefill",
            row_kind="category",
            category=category,
            bucket_tokens=bucket_tokens,
            decode_batch=decode_batch,
            hidden_size=HIDDEN_SIZE,
            topk=TOPK,
            moe_ep_size=MOE_EP_SIZE,
            quant_runtime=QUANT_RUNTIME,
        )
        if result is None:
            return None
        total += float(result)
    return total


def _peer_active_at(timeline: list[phase413.TimedStep], midpoint_ms: float) -> phase413.TimedStep | None:
    for item in timeline:
        if item.start_ms <= midpoint_ms < item.end_ms:
            return item
    return None


def _peer_wait_upper_ms(
    timed: phase413.TimedStep,
    *,
    engine: str,
    timelines: dict[str, list[phase413.TimedStep]],
) -> float:
    peer = "1" if engine == "0" else "0"
    midpoint = (timed.start_ms + timed.end_ms) / 2.0
    peer_step = _peer_active_at(timelines[peer], midpoint)
    if peer_step is None:
        return 0.0
    # Conservative log-visible bound: only count time where this rank could be
    # waiting for a slower peer step. Anything else stays unexplained.
    return max(0.0, peer_step.step.elapsed_ms - timed.step.elapsed_ms)


def _build_step_audits(artifact_dir: Path) -> list[_StepAudit]:
    steps = phase413.parse_iteration_steps(_serve_log_path(artifact_dir))
    timelines = phase413.build_wallclock_timelines(steps)
    calc, _db = _load_calc_and_db()
    attention_cache: dict[tuple[int, int, int], float] = {}
    audits: list[_StepAudit] = []
    for engine, timeline in timelines.items():
        for timed in timeline:
            step = timed.step
            if step.ctx_tokens <= 0 or step.generation_tokens <= 0:
                continue
            key = (step.ctx_tokens, step.ctx_requests, step.generation_requests)
            if key not in attention_cache:
                attention_cache[key] = _attention_ms(
                    calc,
                    ctx_tokens=step.ctx_tokens,
                    ctx_requests=step.ctx_requests,
                    decode_batch=step.generation_requests,
                )
            audits.append(
                _StepAudit(
                    bucket_tokens=step.ctx_tokens + step.generation_tokens,
                    decode_batch=step.generation_requests,
                    elapsed_ms=step.elapsed_ms,
                    attention_ms=attention_cache[key],
                    peer_wait_ms=_peer_wait_upper_ms(timed, engine=engine, timelines=timelines),
                )
            )
    return audits


def build_audit_points(artifact_dir: Path = DEFAULT_ARTIFACT) -> list[AuditPoint]:
    points, _missing_rows = _build_audit_points_and_missing(artifact_dir)
    return points


def _build_audit_points_and_missing(
    artifact_dir: Path = DEFAULT_ARTIFACT,
) -> tuple[list[AuditPoint], list[dict[str, object]]]:
    step_audits = _build_step_audits(artifact_dir)
    _calc, db = _load_calc_and_db()
    grouped: dict[tuple[int, int], list[_StepAudit]] = {}
    for step in step_audits:
        grouped.setdefault((step.bucket_tokens, step.decode_batch), []).append(step)

    points: list[AuditPoint] = []
    missing_rows: list[dict[str, object]] = []
    for (bucket_tokens, decode_batch), group in sorted(grouped.items()):
        category_sum = _category_sum_ms(db, bucket_tokens=bucket_tokens, decode_batch=decode_batch)
        if category_sum is None:
            row = _base_row()
            row.update(
                {
                    "row_type": "missing_category_coverage",
                    "artifact_dir": _display_path(artifact_dir),
                    "bucket_tokens": bucket_tokens,
                    "decode_batch": decode_batch,
                    "step_count": len(group),
                    "non_attn_total_ms": _mean([step.non_attn_total_ms for step in group]),
                    "attention_ms": _mean([step.attention_ms for step in group]),
                    "coverage_band": (
                        "high_batch"
                        if bucket_tokens >= TARGET_MIN_BUCKET_TOKENS and decode_batch >= HIGH_BATCH_MIN
                        else "ramp_or_low_batch"
                    ),
                    "missing_categories": ",".join(TARGET_CATEGORIES),
                    "note": "candidate_log_row_exists_but_serving_state_category_sum_missing",
                }
            )
            missing_rows.append(row)
            continue
        attention = _mean([step.attention_ms for step in group])
        points.append(
            AuditPoint(
                bucket_tokens=bucket_tokens,
                decode_batch=decode_batch,
                step_count=len(group),
                category_sum_ms=category_sum,
                non_attn_total_ms=_mean([step.non_attn_total_ms for step in group]),
                attention_ms=attention,
                peer_wait_ms=_mean([step.peer_wait_ms for step in group]),
                host_gap_upper_ms=PHASE433_HOST_GAP_UPPER_MS,
                attention_error_upper_ms=attention * ATTENTION_ERROR_UPPER_FRACTION,
            )
        )
    return points, missing_rows


def _point_to_row(point: AuditPoint, *, artifact_dir: Path) -> dict[str, object]:
    coverage_band = "high_batch" if point.is_high_batch else "ramp_or_low_batch"
    row = _base_row()
    row.update(
        {
            "row_type": "audit_point",
            "artifact_dir": _display_path(artifact_dir),
            "bucket_tokens": point.bucket_tokens,
            "decode_batch": point.decode_batch,
            "step_count": point.step_count,
            "category_sum_ms": point.category_sum_ms,
            "non_attn_total_ms": point.non_attn_total_ms,
            "candidate_gap_ms": point.candidate_gap_ms,
            "candidate_error_pct": point.candidate_error_pct,
            "attention_ms": point.attention_ms,
            "peer_wait_ms": point.peer_wait_ms,
            "host_gap_upper_ms": point.host_gap_upper_ms,
            "attention_error_upper_ms": point.attention_error_upper_ms,
            "explained_ms": point.explained_ms,
            "unexplained_ms": point.unexplained_ms,
            "explained_share": point.explained_share,
            "coverage_band": coverage_band,
            "note": "log_visible_peer_wait_plus_phase433_host_gap_bound_plus_attention_bound",
        }
    )
    return row


def build_phase441_rows(artifact_dir: Path = DEFAULT_ARTIFACT) -> list[dict[str, object]]:
    points, missing_rows = _build_audit_points_and_missing(artifact_dir)
    summary = summarize_life_gate(points)
    rows = [_point_to_row(point, artifact_dir=artifact_dir) for point in points]
    rows.extend(missing_rows)

    summary_row = _base_row()
    summary_row.update(
        {
            "row_type": "summary",
            "artifact_dir": _display_path(artifact_dir),
            "step_count": sum(point.step_count for point in points),
            "explained_share": summary["explained_share"],
            "explain_gate": summary["explain_gate"],
            "high_batch_gate": summary["high_batch_gate"],
            "life_gate": summary["life_gate"],
            "metric": "b1_life_gate",
            "value": summary["high_batch_max_error_pct"],
            "note": (
                "B1_survives_only_if_explained_share_ge_80pct_and_high_batch_error_le_15pct"
                f";missing_category_rows={len(missing_rows)}"
            ),
        }
    )
    rows.insert(0, summary_row)
    for metric, value in summary.items():
        metric_row = _base_row()
        metric_row.update(
            {
                "row_type": "metric",
                "artifact_dir": _display_path(artifact_dir),
                "metric": metric,
                "value": value,
                "life_gate": summary["life_gate"],
                "note": "phase441_step0_offline_audit",
            }
        )
        rows.append(metric_row)
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field, "")) for field in CSV_FIELDS})


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    summary = next(row for row in rows if row.get("row_type") == "summary")
    points = [row for row in rows if row.get("row_type") == "audit_point"]
    missing = [row for row in rows if row.get("row_type") == "missing_category_coverage"]
    high = [row for row in points if row.get("coverage_band") == "high_batch"]
    verdict = (
        "steady_wall_primitive_gate_passed_collect_logs_next"
        if summary.get("life_gate") == "passed"
        else "b1_failed_turn_to_b2_or_redefine_with_new_evidence"
    )
    lines = [
        "# Phase441 steady wall-clock primitive audit",
        "",
        f"- verdict: `{verdict}`.",
        f"- life_gate: `{summary.get('life_gate', '')}`.",
        f"- explained_share: `{_fmt(summary.get('explained_share', ''))}`.",
        f"- high_batch_gate: `{summary.get('high_batch_gate', '')}`.",
        f"- high_batch_max_error_pct: `{_fmt(summary.get('value', ''))}`.",
        "- No PerfDB rows are ingested by this report.",
        "- GPU pure-log collection remains blocked unless this audit passes.",
        "- Default AIC remains No-Go.",
        "",
        "## Audit Points",
        "",
        "| bucket | batch | steps | category ms | wall non-attn ms | error % | explained % | unexplained ms | band |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in points:
        lines.append(
            "| {bucket} | {batch} | {steps} | {cat} | {nonattn} | {err} | {expl} | {unexpl} | {band} |".format(
                bucket=row.get("bucket_tokens", ""),
                batch=row.get("decode_batch", ""),
                steps=row.get("step_count", ""),
                cat=_fmt(row.get("category_sum_ms", "")),
                nonattn=_fmt(row.get("non_attn_total_ms", "")),
                err=_fmt(row.get("candidate_error_pct", "")),
                expl=_fmt(float(row.get("explained_share") or 0.0) * 100.0),
                unexpl=_fmt(row.get("unexplained_ms", "")),
                band=row.get("coverage_band", ""),
            )
        )
    lines.extend(
        [
            "",
            "## High Batch",
            "",
            "| bucket | batch | steps | error % |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in high:
        lines.append(
            f"| {row.get('bucket_tokens', '')} | {row.get('decode_batch', '')} | "
            f"{row.get('step_count', '')} | {_fmt(row.get('candidate_error_pct', ''))} |"
        )
    lines.extend(
        [
            "",
            "## Missing Category Coverage",
            "",
            "| bucket | batch | steps | wall non-attn ms | band |",
            "|---:|---:|---:|---:|---|",
        ]
    )
    for row in missing:
        lines.append(
            f"| {row.get('bucket_tokens', '')} | {row.get('decode_batch', '')} | "
            f"{row.get('step_count', '')} | {_fmt(row.get('non_attn_total_ms', ''))} | "
            f"{row.get('coverage_band', '')} |"
        )
    lines.extend(
        [
            "",
            "## B2 Trigger",
            "",
            "- B1 does not pass the offline life gate.",
            "- The overlap delta is not explained by log-visible peer wait, Phase433 host-gap bound, and attention-error bound.",
            "- The high-batch segment has iteration-log candidates but lacks complete serving-state category coverage, so the steady convergence gate cannot be proven.",
            "- A pure-log GPU run may add coverage, but it cannot fix the failed overlap explanation gate by itself.",
            "- B2 must time graph outer boundaries only; graph-internal CUDA events are invalid for replay timing and must not be used as the acceptance signal.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows = build_phase441_rows(args.artifact_dir)
    write_csv(rows, args.csv)
    write_markdown(rows, args.md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
