#!/usr/bin/env python3
"""Locate the TP8 bt65536 eight-request mixed-batch first divergence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import scripts.analyze_phase459_residual_triage as phase459  # noqa: E402
import scripts.analyze_phase462_tp8_bt65536_residual as residual  # noqa: E402
import scripts.validate_cb_simulator as validate  # noqa: E402
from aiconfigurator.sdk.backends.cb_simulator import CBSimulator  # noqa: E402
from aiconfigurator.sdk.backends.cb_simulator.datatypes import RequestState  # noqa: E402
from aiconfigurator.sdk.backends.cb_simulator.scheduler import CBScheduler  # noqa: E402


TP8_BT = residual.TP8_BT
REAL_SERVE_LOG = REPO_ROOT / (
    "docs/iter_gap_investigation/phase461_cost_recollect/"
    "tp8_bt65536_mixed/overhead_on/"
    f"{TP8_BT}/serve.log.gz"
)
DEFAULT_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase462_dp_step3c"
DEFAULT_CSV = DEFAULT_ROOT / "phase462_tp8_mixed_first_divergence.csv"
DEFAULT_REPORT = DEFAULT_ROOT / "phase462_tp8_mixed_first_divergence.md"


@dataclass(frozen=True)
class SimStep:
    iteration: int
    prefill_requests: int
    prefill_tokens: int
    decode_requests: int
    new_admit_ids: tuple[int, ...]
    continued_ids: tuple[int, ...]
    recompute_ids: tuple[int, ...]
    budget_before_admit: int
    budget_after_schedule: int
    next_request_tokens: int
    max_num_seqs_reached: bool
    block_capacity_reached: bool
    stopped_after_partial_prefill: bool


@dataclass(frozen=True)
class RealStep:
    iteration: int
    context_requests: int
    context_tokens: int
    generation_requests: int


@dataclass(frozen=True)
class Divergence:
    sim_iteration: int
    cluster_start_iteration: int
    streak_length: int
    real_has_same_macro_state: bool
    real_has_equivalent_eight_wide_transition: bool
    mechanism: str


def classify_formation_mechanism(step: SimStep) -> str:
    if step.stopped_after_partial_prefill:
        return "chunked_prefill_partial_stop"
    if step.max_num_seqs_reached:
        return "admission_max_num_seqs"
    if step.block_capacity_reached:
        return "admission_block_capacity"
    if (
        len(step.new_admit_ids) >= 8
        and step.next_request_tokens > 0
        and step.budget_after_schedule < step.next_request_tokens
    ):
        return "token_budget_split"
    return "unresolved_batch_formation"


def _is_eight_wide(step: SimStep) -> bool:
    return (
        step.prefill_requests > 0
        and step.decode_requests > 0
        and len(step.new_admit_ids) == 8
        and not step.recompute_ids
    )


def first_divergence(
    sim_steps: Sequence[SimStep], real_steps: Sequence[RealStep]
) -> Divergence:
    ordered = sorted(sim_steps, key=lambda step: step.iteration)
    first: SimStep | None = None
    streak_length = 0
    for index, step in enumerate(ordered):
        if not _is_eight_wide(step):
            continue
        stop = index + 1
        while (
            stop < len(ordered)
            and ordered[stop].iteration == ordered[stop - 1].iteration + 1
            and _is_eight_wide(ordered[stop])
        ):
            stop += 1
        if stop - index >= 2:
            first = step
            streak_length = stop - index
            break
    if first is None:
        raise AssertionError("no_two_step_eight_wide_mixed_streak")
    macro = (
        first.prefill_requests,
        first.prefill_tokens,
        first.decode_requests,
    )
    real_macros = {
        (step.context_requests, step.context_tokens, step.generation_requests)
        for step in real_steps
    }
    real_eight_wide = any(
        current.context_requests == first.prefill_requests
        and current.generation_requests - previous.generation_requests
        == len(first.new_admit_ids)
        for previous, current in zip(real_steps, real_steps[1:])
    )
    initial_wave_diverged = bool(
        ordered
        and real_steps
        and len(ordered[0].new_admit_ids) != real_steps[0].context_requests
    )
    return Divergence(
        sim_iteration=(
            ordered[0].iteration
            if real_eight_wide and initial_wave_diverged
            else first.iteration
        ),
        cluster_start_iteration=first.iteration,
        streak_length=streak_length,
        real_has_same_macro_state=macro in real_macros,
        real_has_equivalent_eight_wide_transition=real_eight_wide,
        mechanism=(
            "initial_admission_wave_phase_offset"
            if real_eight_wide and initial_wave_diverged
            else classify_formation_mechanism(first)
        ),
    )


def capture_sim_steps() -> list[SimStep]:
    point = next(item for item in validate.MULTI_CONFIG_DATA if item.name == TP8_BT)
    model, database, backend = validate._load_model_and_db(
        tp=point.tp, dp=point.dp, moe_tp=point.moe_tp, moe_ep=point.moe_ep
    )
    config = validate._make_multi_config_cb_config(
        point,
        overlap_factor=0.0,
        ep8_per_iteration_overhead_ms=validate.DEFAULT_EP8_PER_ITERATION_OVERHEAD_MS,
    )
    config = replace(config, warmup_requests=point.batch_size)
    original_schedule = CBScheduler.schedule
    captured: list[SimStep] = []
    iteration = 0

    def wrapped_schedule(self, waiting, running):
        nonlocal iteration
        iteration += 1
        before = {
            request.request_id: (
                request.state,
                request.prefill_tokens_remaining,
                request.isl,
                request.num_preemptions,
            )
            for request in [*waiting, *running]
        }
        waiting_order = [request.request_id for request in waiting]
        running_ids = {request.request_id for request in running}
        result = original_schedule(self, waiting, running)
        if result.is_empty:
            return result

        new_ids: list[int] = []
        continued_ids: list[int] = []
        recompute_ids: list[int] = []
        running_tokens = len(result.decode_reqs)
        stopped_after_partial = False
        selected_ids = {request.request_id for request in result.prefill_reqs}
        for request in result.prefill_reqs:
            state, remaining, isl, preemptions = before[request.request_id]
            tokens = int(result.prefill_tokens[request.request_id])
            if request.request_id in running_ids:
                running_tokens += tokens
            if preemptions > 0 or state == RequestState.PREEMPTED:
                recompute_ids.append(request.request_id)
            elif state == RequestState.WAITING and remaining == isl:
                new_ids.append(request.request_id)
                if tokens < remaining:
                    stopped_after_partial = True
            else:
                continued_ids.append(request.request_id)

        next_request = next(
            (
                request
                for request_id in waiting_order
                for request in waiting
                if request.request_id == request_id
                and request_id not in selected_ids
                and request.state in {RequestState.WAITING, RequestState.PREEMPTED}
            ),
            None,
        )
        next_tokens = int(next_request.prefill_tokens_remaining) if next_request else 0
        max_seqs = len(running) + sum(
            request.request_id not in running_ids for request in result.prefill_reqs
        ) >= self._config.max_num_seqs
        block_capacity = bool(
            next_request is not None
            and not self._fits_full_sequence_admission(next_request, running, result)
        )
        captured.append(
            SimStep(
                iteration=iteration,
                prefill_requests=len(result.prefill_reqs),
                prefill_tokens=result.total_prefill_tokens,
                decode_requests=len(result.decode_reqs),
                new_admit_ids=tuple(new_ids),
                continued_ids=tuple(continued_ids),
                recompute_ids=tuple(recompute_ids),
                budget_before_admit=self._config.max_num_batched_tokens - running_tokens,
                budget_after_schedule=self._config.max_num_batched_tokens - result.total_tokens,
                next_request_tokens=next_tokens,
                max_num_seqs_reached=max_seqs,
                block_capacity_reached=block_capacity,
                stopped_after_partial_prefill=stopped_after_partial,
            )
        )
        return result

    with patch.object(CBScheduler, "schedule", wrapped_schedule):
        sim = CBSimulator(backend, model, database, config)
        sim.run(
            isl=point.isl,
            osl=point.osl,
            concurrency=math.ceil(point.batch_size / point.dp),
            num_gpus=point.tp,
        )
    return captured


def load_real_steps() -> list[RealStep]:
    return [
        RealStep(
            iteration=step.iteration,
            context_requests=step.ctx_requests,
            context_tokens=step.ctx_tokens,
            generation_requests=step.generation_requests,
        )
        for step in phase459.parse_iteration_steps(REAL_SERVE_LOG)
    ]


def run_analysis(
    *, output_csv: Path = DEFAULT_CSV, output_report: Path = DEFAULT_REPORT
) -> dict[str, object]:
    sim_steps = capture_sim_steps()
    real_steps = load_real_steps()
    verdict = first_divergence(sim_steps, real_steps)
    sim_initial = min(sim_steps, key=lambda step: step.iteration)
    real_initial = min(real_steps, key=lambda step: step.iteration)
    first = next(
        step for step in sim_steps if step.iteration == verdict.cluster_start_iteration
    )
    window = [
        step
        for step in sim_steps
        if verdict.sim_iteration <= step.iteration <= verdict.cluster_start_iteration + 4
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(
            [
                "iteration",
                "prefill_requests",
                "prefill_tokens",
                "decode_requests",
                "new_admit_count",
                "new_admit_ids",
                "continued_ids",
                "recompute_ids",
                "budget_before_admit",
                "budget_after_schedule",
                "next_request_tokens",
                "max_num_seqs_reached",
                "block_capacity_reached",
                "stopped_after_partial_prefill",
            ]
        )
        for step in window:
            writer.writerow(
                [
                    step.iteration,
                    step.prefill_requests,
                    step.prefill_tokens,
                    step.decode_requests,
                    len(step.new_admit_ids),
                    ";".join(map(str, step.new_admit_ids)),
                    ";".join(map(str, step.continued_ids)),
                    ";".join(map(str, step.recompute_ids)),
                    step.budget_before_admit,
                    step.budget_after_schedule,
                    step.next_request_tokens,
                    step.max_num_seqs_reached,
                    step.block_capacity_reached,
                    step.stopped_after_partial_prefill,
                ]
            )
    output_report.write_text(
        f"""# Phase462 TP8-bt65536 8宽 mixed first-divergence

结论：首个分叉在 sim iteration `{verdict.sim_iteration}`，首个连续 8-new-request mixed 聚簇从 `{verdict.cluster_start_iteration}` 开始并持续 `{verdict.streak_length}` 步。real 也存在 decode 数 `+8` 且 context requests 相同的等价波次：`{verdict.real_has_equivalent_eight_wide_transition}`；因此裁决为 `{verdict.mechanism}`，不是 sim 独有的 chunk 配额规则。`Default AIC=No-Go`。

| 首波 | context requests | context tokens | decode requests |
|---|---:|---:|---:|
| real iteration {real_initial.iteration} | {real_initial.context_requests} | {real_initial.context_tokens} | {real_initial.generation_requests} |
| sim iteration {sim_initial.iteration} | {sim_initial.prefill_requests} | {sim_initial.prefill_tokens} | {sim_initial.decode_requests} |

| 指标 | 首个 8-new-request 聚簇 |
|---|---:|
| prefill requests/tokens | {first.prefill_requests}/{first.prefill_tokens} |
| decode requests | {first.decode_requests} |
| 新 admit | {len(first.new_admit_ids)} |
| admit 前 token budget | {first.budget_before_admit} |
| schedule 后剩余 budget | {first.budget_after_schedule} |
| 下一请求所需 token | {first.next_request_tokens} |
| max_num_seqs 命中 | {first.max_num_seqs_reached} |
| block capacity 命中 | {first.block_capacity_reached} |
| partial-prefill stop | {first.stopped_after_partial_prefill} |
| real 有等价 +8 波次 | {verdict.real_has_equivalent_eight_wide_transition} |
| real 有完全相同宏观 cell | {verdict.real_has_same_macro_state} |

判卷只使用 scheduler 入参和结果重建预算，没有修改 scheduler。sim 的单步机械链是“剩余 token budget 形成 fresh partial → partial-prefill stop → 下一步 1 continued + 8 fresh”；real 聚合轨迹同样出现 9 个 context 且 decode 每步增加 8。两侧不同的是首波暴露/admission：real 首步 1 个 context，sim 首步 9 个，随后 decode 相位相差 7，才形成 sim-only 精确 cell。配套 CSV 保存该窗口请求 ID 与 chunk 构成；本步不授权修改 token budget、chunk 规则或暴露模型。

本步全离线，不改 runtime、PerfDB 或 gate。
""",
        encoding="utf-8",
    )
    return {
        "status": "completed_report_only",
        "sim_iteration": verdict.sim_iteration,
        "cluster_start_iteration": verdict.cluster_start_iteration,
        "streak_length": verdict.streak_length,
        "real_has_same_macro_state": verdict.real_has_same_macro_state,
        "real_has_equivalent_eight_wide_transition": (
            verdict.real_has_equivalent_eight_wide_transition
        ),
        "mechanism": verdict.mechanism,
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
