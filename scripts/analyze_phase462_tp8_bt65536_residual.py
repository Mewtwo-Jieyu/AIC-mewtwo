#!/usr/bin/env python3
"""Decompose the TP8 bt65536 non-recompute sim-only residual."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import scripts.analyze_phase461_step4_cell_triage as phase461_cells  # noqa: E402
import scripts.validate_cb_simulator as validate  # noqa: E402
from aiconfigurator.sdk.backends.cb_simulator import CBSimulator  # noqa: E402
from aiconfigurator.sdk.backends.cb_simulator.datatypes import (  # noqa: E402
    RequestState,
)
from aiconfigurator.sdk.backends.cb_simulator.iteration_latency import (  # noqa: E402
    IterationLatencyCalculator,
)
from aiconfigurator.sdk.backends.cb_simulator import iteration_latency as latency_module  # noqa: E402
from aiconfigurator.sdk.backends.cb_simulator.scheduler import CBScheduler  # noqa: E402


TP8_BT = "K2.5-tp8ep8-8k2k-bt65536"
DEFAULT_CSV = REPO_ROOT / (
    "docs/iter_gap_investigation/phase462_tp8_bt65536_residual.csv"
)
DEFAULT_REPORT = REPO_ROOT / (
    "docs/iter_gap_investigation/phase462_tp8_bt65536_residual.md"
)


@dataclass(frozen=True)
class AuditHit:
    hit: bool
    miss_reason: str


@dataclass(frozen=True)
class PrefillPart:
    origin: str
    extent: str
    tokens: int


@dataclass(frozen=True)
class ResidualStep:
    iteration: int
    bucket_tokens: int
    decode_batch: int
    wall_ms: float
    prefill_tokens: int
    recompute_tokens: int
    chunk_composition: str
    serving_coverage: str
    serving_miss_reasons: tuple[str, ...]
    new_admit_count: int


@dataclass(frozen=True)
class ResidualDecomposition:
    total_wall_ms: float
    chunk_wall_ms: dict[str, float]
    coverage_wall_ms: dict[str, float]
    admit_count_wall_ms: dict[int, float]
    streak_length_wall_ms: dict[int, float]


@dataclass(frozen=True)
class CaptureResult:
    steps: list[ResidualStep]
    preemptions: int
    num_requests: int


def chunk_composition(parts: Iterable[PrefillPart]) -> str:
    labels = sorted({f"{part.origin}_{part.extent}" for part in parts})
    if not labels:
        raise AssertionError("non-recompute prefill step has no chunk parts")
    return "+".join(labels)


def serving_coverage(audits: Iterable[AuditHit]) -> tuple[str, tuple[str, ...]]:
    audits = list(audits)
    if not audits:
        return "scope_disabled", ()
    hit_count = sum(audit.hit for audit in audits)
    reasons = tuple(sorted({audit.miss_reason for audit in audits if not audit.hit}))
    if hit_count == len(audits):
        return "all_hit", ()
    if hit_count == 0:
        return "all_miss", reasons
    return "partial_hit", reasons


def serving_cache_key(
    prefill_tokens: int,
    prefill_batch_size: int,
    prefill_seq_len: int,
    decode_batch_size: int,
    decode_avg_kv_len: int,
) -> tuple[int, int, int, int, int]:
    kv_bucket = (
        latency_module._bucket(  # noqa: SLF001
            decode_avg_kv_len, latency_module._KV_LEN_BUCKET  # noqa: SLF001
        )
        if decode_avg_kv_len > 0
        else 0
    )
    return (
        prefill_tokens + decode_batch_size,
        prefill_batch_size,
        prefill_seq_len,
        decode_batch_size,
        kv_bucket,
    )


def _sum_wall(steps: Iterable[ResidualStep], key) -> dict[object, float]:
    result: dict[object, float] = defaultdict(float)
    for step in steps:
        result[key(step)] += step.wall_ms
    return dict(result)


def decompose_residual_steps(steps: Sequence[ResidualStep]) -> ResidualDecomposition:
    ordered = sorted(steps, key=lambda step: step.iteration)
    if len({step.iteration for step in ordered}) != len(ordered):
        raise AssertionError("duplicate residual iteration")
    streak_by_iteration: dict[int, int] = {}
    for start in range(len(ordered)):
        if start > 0 and ordered[start].iteration == ordered[start - 1].iteration + 1:
            continue
        stop = start + 1
        while (
            stop < len(ordered)
            and ordered[stop].iteration == ordered[stop - 1].iteration + 1
        ):
            stop += 1
        streak_length = stop - start
        for step in ordered[start:stop]:
            streak_by_iteration[step.iteration] = streak_length
    total = sum(step.wall_ms for step in ordered)
    result = ResidualDecomposition(
        total_wall_ms=total,
        chunk_wall_ms={
            str(key): value
            for key, value in _sum_wall(
                ordered, lambda step: step.chunk_composition
            ).items()
        },
        coverage_wall_ms={
            str(key): value
            for key, value in _sum_wall(
                ordered, lambda step: step.serving_coverage
            ).items()
        },
        admit_count_wall_ms={
            int(key): value
            for key, value in _sum_wall(
                ordered, lambda step: step.new_admit_count
            ).items()
        },
        streak_length_wall_ms={
            int(key): value
            for key, value in _sum_wall(
                ordered, lambda step: streak_by_iteration[step.iteration]
            ).items()
        },
    )
    for name, dimension in (
        ("chunk", result.chunk_wall_ms),
        ("coverage", result.coverage_wall_ms),
        ("admit_count", result.admit_count_wall_ms),
        ("streak", result.streak_length_wall_ms),
    ):
        if not math.isclose(sum(dimension.values()), total, rel_tol=0.0, abs_tol=1e-9):
            raise AssertionError(f"{name}_wall_conservation_failed")
    return result


def _point():
    return next(point for point in validate.MULTI_CONFIG_DATA if point.name == TP8_BT)


def run_capture(*, num_requests: int | None = None) -> CaptureResult:
    point = _point()
    model, database, backend = validate._load_model_and_db(
        tp=point.tp,
        dp=point.dp,
        moe_tp=point.moe_tp,
        moe_ep=point.moe_ep,
    )
    config = validate._make_multi_config_cb_config(
        point,
        overlap_factor=0.0,
        ep8_per_iteration_overhead_ms=validate.DEFAULT_EP8_PER_ITERATION_OVERHEAD_MS,
    )
    if num_requests is not None:
        config = replace(
            config,
            num_requests=num_requests,
            warmup_requests=point.batch_size,
        )

    pending: deque[dict[str, object]] = deque()
    steps: list[ResidualStep] = []
    audit_cache: dict[tuple[int, int, int, int, int], tuple[AuditHit, ...]] = {}
    original_schedule = CBScheduler.schedule
    original_preempt = CBScheduler._preempt
    original_compute = IterationLatencyCalculator.compute
    iteration = 0
    preemptions = 0

    def wrapped_schedule(self, waiting, running):
        nonlocal iteration
        iteration += 1
        before = {
            request.request_id: (
                request.state,
                int(request.prefill_tokens_remaining),
                int(request.isl),
                int(request.num_preemptions),
            )
            for request in [*waiting, *running]
        }
        result = original_schedule(self, waiting, running)
        if result.is_empty:
            return result
        parts: list[PrefillPart] = []
        recompute_tokens = 0
        new_admits = 0
        for request in result.prefill_reqs:
            state, remaining, isl, request_preemptions = before[request.request_id]
            tokens = int(result.prefill_tokens[request.request_id])
            if request_preemptions > 0 or state == RequestState.PREEMPTED:
                recompute_tokens += tokens
                continue
            origin = (
                "fresh"
                if state == RequestState.WAITING and remaining == isl
                else "continued"
            )
            extent = "complete" if tokens == remaining else "partial"
            parts.append(PrefillPart(origin, extent, tokens))
            new_admits += origin == "fresh"
        pending.append(
            {
                "iteration": iteration,
                "prefill_tokens": int(result.total_prefill_tokens),
                "prefill_batch": len(result.prefill_reqs),
                "decode_batch": len(result.decode_reqs),
                "recompute_tokens": recompute_tokens,
                "parts": parts,
                "new_admits": new_admits,
            }
        )
        return result

    def wrapped_preempt(self, victim, waiting, running, result, preempted_ids):
        nonlocal preemptions
        preemptions += 1
        return original_preempt(self, victim, waiting, running, result, preempted_ids)

    def wrapped_compute(
        self,
        prefill_tokens: int,
        prefill_batch_size: int,
        prefill_seq_len: int,
        decode_batch_size: int,
        decode_avg_kv_len: int,
    ) -> float:
        audit_before = len(self.get_serving_state_query_audit())
        wall_ms = original_compute(
            self,
            prefill_tokens,
            prefill_batch_size,
            prefill_seq_len,
            decode_batch_size,
            decode_avg_kv_len,
        )
        audit_after = self.get_serving_state_query_audit()
        key = serving_cache_key(
            prefill_tokens,
            prefill_batch_size,
            prefill_seq_len,
            decode_batch_size,
            decode_avg_kv_len,
        )
        new_audits = tuple(
            AuditHit(bool(audit.hit), str(audit.miss_reason))
            for audit in audit_after[audit_before:]
        )
        if new_audits:
            audit_cache[key] = new_audits
        audits = audit_cache.get(key, ())
        if not pending:
            # Simulator startup estimates decode-skip latency before scheduling.
            return wall_ms
        current = pending.popleft()
        observed = (prefill_tokens, prefill_batch_size, decode_batch_size)
        expected = (
            current["prefill_tokens"],
            current["prefill_batch"],
            current["decode_batch"],
        )
        if observed != expected:
            raise RuntimeError(
                f"schedule/latency signature mismatch:{expected=}:{observed=}"
            )
        if prefill_tokens > 0 and decode_batch_size > 0:
            coverage, miss_reasons = serving_coverage(audits)
            parts = list(current["parts"])
            recompute_tokens = int(current["recompute_tokens"])
            composition = (
                chunk_composition(parts)
                if parts
                else "recompute_only"
            )
            steps.append(
                ResidualStep(
                    iteration=int(current["iteration"]),
                    bucket_tokens=prefill_tokens + decode_batch_size,
                    decode_batch=decode_batch_size,
                    wall_ms=wall_ms,
                    prefill_tokens=prefill_tokens,
                    recompute_tokens=recompute_tokens,
                    chunk_composition=composition,
                    serving_coverage=coverage,
                    serving_miss_reasons=miss_reasons,
                    new_admit_count=int(current["new_admits"]),
                )
            )
        return wall_ms

    with (
        patch.object(CBScheduler, "schedule", wrapped_schedule),
        patch.object(CBScheduler, "_preempt", wrapped_preempt),
        patch.object(IterationLatencyCalculator, "compute", wrapped_compute),
    ):
        sim = CBSimulator(backend, model, database, config)
        sim.run(
            isl=point.isl,
            osl=point.osl,
            concurrency=math.ceil(point.batch_size / point.dp),
            num_gpus=point.tp,
        )
    if pending:
        raise RuntimeError(f"unpaired schedule rows:{len(pending)}")
    return CaptureResult(steps, preemptions, config.num_requests)


def _real_cells() -> set[tuple[int, int]]:
    scenario = next(item for item in phase461_cells.SCENARIOS if item[0] == TP8_BT)
    return set(phase461_cells._real_mixed_cells(scenario[2]))


def _rows(capture: CaptureResult) -> tuple[list[dict[str, object]], dict[str, float]]:
    real_cells = _real_cells()
    mixed = capture.steps
    sim_only = [
        step
        for step in mixed
        if (step.bucket_tokens, step.decode_batch) not in real_cells
    ]
    recompute = [step for step in sim_only if step.recompute_tokens > 0]
    residual = [step for step in sim_only if step.recompute_tokens == 0]
    decomposition = decompose_residual_steps(residual)
    total_mixed = sum(step.wall_ms for step in mixed)
    sim_only_wall = sum(step.wall_ms for step in sim_only)
    recompute_wall = sum(step.wall_ms for step in recompute)
    summary = {
        "mixed_wall_ms": total_mixed,
        "sim_only_wall_ms": sim_only_wall,
        "sim_only_wall_share": sim_only_wall / total_mixed,
        "recompute_sim_only_wall_ms": recompute_wall,
        "recompute_sim_only_wall_share": recompute_wall / sim_only_wall,
        "non_recompute_sim_only_wall_ms": decomposition.total_wall_ms,
        "non_recompute_sim_only_wall_share": decomposition.total_wall_ms / sim_only_wall,
        "non_recompute_total_mixed_wall_share": decomposition.total_wall_ms / total_mixed,
    }
    rows: list[dict[str, object]] = []
    for metric, value in summary.items():
        rows.append(
            {
                "section": "summary",
                "category": metric,
                "wall_ms": value if metric.endswith("wall_ms") else "",
                "share_of_residual": "",
                "value": value,
            }
        )
    dimensions = (
        ("chunk_composition", decomposition.chunk_wall_ms),
        ("serving_coverage", decomposition.coverage_wall_ms),
        ("new_admit_count", decomposition.admit_count_wall_ms),
        ("cadence_streak_length", decomposition.streak_length_wall_ms),
    )
    for section, values in dimensions:
        for category, wall_ms in sorted(values.items(), key=lambda item: str(item[0])):
            rows.append(
                {
                    "section": section,
                    "category": category,
                    "wall_ms": wall_ms,
                    "share_of_residual": wall_ms / decomposition.total_wall_ms,
                    "value": "",
                }
            )
    miss_wall: dict[str, float] = defaultdict(float)
    for step in residual:
        for reason in step.serving_miss_reasons:
            miss_wall[reason] += step.wall_ms
    for reason, wall_ms in sorted(miss_wall.items()):
        rows.append(
            {
                "section": "serving_miss_reason_overlap",
                "category": reason,
                "wall_ms": wall_ms,
                "share_of_residual": wall_ms / decomposition.total_wall_ms,
                "value": "",
            }
        )
    return rows, summary


def write_outputs(
    rows: Sequence[dict[str, object]],
    summary: dict[str, float],
    *,
    capture: CaptureResult,
    output_csv: Path,
    output_report: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=["section", "category", "wall_ms", "share_of_residual", "value"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    def dimension(section: str) -> str:
        selected = [row for row in rows if row["section"] == section]
        return "\n".join(
            f"| {row['category']} | {float(row['wall_ms']):.6f} | "
            f"{float(row['share_of_residual']):.2%} |"
            for row in selected
        )

    output_report.write_text(
        f"""# Phase462 TP8-bt65536 非 recompute 残差分解

结论：null-only N{capture.num_requests} 下，sim-only mixed wall 中 recompute 关联占 {summary['recompute_sim_only_wall_share']:.2%}，剩余 {summary['non_recompute_sim_only_wall_share']:.2%}（占全部 mixed wall {summary['non_recompute_total_mixed_wall_share']:.2%}）。以下三维是同一残差的并列签名视图，不能彼此相加。`Default AIC=No-Go`。

| 口径 | wall ms | 占比 |
|---|---:|---:|
| 全部 mixed | {summary['mixed_wall_ms']:.6f} | 100.00% |
| sim-only mixed | {summary['sim_only_wall_ms']:.6f} | {summary['sim_only_wall_share']:.2%} |
| sim-only 且含 recompute | {summary['recompute_sim_only_wall_ms']:.6f} | {summary['recompute_sim_only_wall_share']:.2%} of sim-only |
| sim-only 且无 recompute | {summary['non_recompute_sim_only_wall_ms']:.6f} | {summary['non_recompute_sim_only_wall_share']:.2%} of sim-only |

## Chunk 构成

| 构成 | wall ms | 残差占比 |
|---|---:|---:|
{dimension('chunk_composition')}

## Serving-state 覆盖

| 覆盖 | wall ms | 残差占比 |
|---|---:|---:|
{dimension('serving_coverage')}

| miss reason | wall ms | 残差占比 |
|---|---:|---:|
{dimension('serving_miss_reason_overlap')}

## 调度节奏

| 单步新 admit 数 | wall ms | 残差占比 |
|---|---:|---:|
{dimension('new_admit_count')}

| 连续残差步长度 | wall ms | 残差占比 |
|---|---:|---:|
{dimension('cadence_streak_length')}

本步仅离线分解，不改 runtime、PerfDB 或 gate；抢占数为 {capture.preemptions}，用于和 null-only 基线核对。
""",
        encoding="utf-8",
    )


def run_analysis(
    *,
    output_csv: Path = DEFAULT_CSV,
    output_report: Path = DEFAULT_REPORT,
) -> dict[str, object]:
    capture = run_capture()
    rows, summary = _rows(capture)
    write_outputs(
        rows,
        summary,
        capture=capture,
        output_csv=output_csv,
        output_report=output_report,
    )
    return {
        "status": "completed_report_only",
        "num_requests": capture.num_requests,
        "preemptions": capture.preemptions,
        **summary,
        "runtime_changed": False,
        "perfdb_changed": False,
        "gate_changed": False,
        "gpu_used": False,
        "default_aic": "No-Go",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    print(
        json.dumps(
            run_analysis(output_csv=args.output_csv, output_report=args.output_report),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
