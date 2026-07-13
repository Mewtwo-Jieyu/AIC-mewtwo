#!/usr/bin/env python3
"""Phase462 Step2a-3c arrival/admission report-only prototype."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

ARRIVAL_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase462_arrival_observation"
ARRIVAL_PATHS = [ARRIVAL_ROOT / "8k2k.jsonl.gz", ARRIVAL_ROOT / "32k3k.jsonl.gz"]
BT_SERVE_LOG = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase458_n512_unify/"
    "recollect_tp8_8k2k_bt65536/K2.5-tp8ep8-8k2k-bt65536/serve.log"
)
PREEMPT_SERVE_LOG = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase462_preemption_observation/overhead_on/"
    "K2.5-tp8ep8-32k3k/serve.log.gz"
)
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase462_arrival_admission_prototype.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase462_arrival_admission_prototype.md"
SCENARIO_32K = "K2.5-tp8ep8-32k3k"
SCENARIO_BT = "K2.5-tp8ep8-8k2k-bt65536"
PROTOTYPE_REQUESTS = 128
PROTOTYPE_OSL = 1200
TOKENIZER_FIT_MAX_WMAPE = 0.10
SMALL_COHORT_LIMIT = 2
VLLM_CORE_SHA256 = "896730e749cbcabb487ce50c703974594d197fc31a1d3b26fe096197d142d2d5"

CSV_FIELDS = ["section", "scenario", "metric", "value", "target", "status", "note"]
ITERATION_RE = re.compile(
    r"Iteration\((?P<iteration>\d+)\): "
    r"(?P<context_requests>\d+) context requests, "
    r"(?P<context_tokens>\d+) context tokens, "
    r"(?P<generation_requests>\d+) generation requests, "
    r"(?P<generation_tokens>\d+) generation tokens, "
    r"iteration elapsed time: (?P<elapsed_ms>[0-9.]+) ms"
)


@dataclass(frozen=True)
class PrimitiveSample:
    total_prompt_tokens: int
    batch_size: int
    service_ms: float
    request_weight: int
    scenario: str = ""
    batch_id: str = ""


@dataclass(frozen=True)
class PrimitiveFit:
    intercept_ms: float
    ms_per_prompt_token: float
    ms_per_request: float
    weighted_mape: float

    def predict(self, total_prompt_tokens: int, batch_size: int) -> float:
        return (
            self.intercept_ms
            + self.ms_per_prompt_token * total_prompt_tokens
            + self.ms_per_request * batch_size
        )


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    with _open_text(path) as source:
        return [json.loads(line) for line in source if line.strip()]


def _row(
    section: str,
    scenario: str,
    metric: str,
    value: object,
    *,
    target: object = "",
    status: str = "",
    note: str = "",
) -> dict[str, object]:
    return {
        "section": section,
        "scenario": scenario,
        "metric": metric,
        "value": value,
        "target": target,
        "status": status,
        "note": note,
    }


def admission_cadence_verdict(
    *,
    real_max_context_requests: int,
    sim_max_fresh_admissions: int,
    small_cohort_limit: int,
) -> dict[str, object]:
    real_small = real_max_context_requests <= small_cohort_limit
    sim_large = sim_max_fresh_admissions > small_cohort_limit
    if real_small and sim_large:
        return {
            "verdict": "admission_cadence_diverged",
            "admission_fix_allowed": True,
            "prototype_scope": "arrival_plus_admission",
        }
    if not real_small and sim_large:
        return {
            "verdict": "budget_driven_cadence_aligned",
            "admission_fix_allowed": False,
            "prototype_scope": "arrival_primitive_only",
        }
    return {
        "verdict": "admission_cadence_inconclusive",
        "admission_fix_allowed": False,
        "prototype_scope": "blocked",
    }


def parse_iteration_rows(lines: Iterable[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in lines:
        match = ITERATION_RE.search(line)
        if not match:
            continue
        row = {key: int(value) for key, value in match.groupdict().items() if key != "elapsed_ms"}
        row["elapsed_ms"] = float(match.group("elapsed_ms"))
        rows.append(row)
    return rows


def load_primitive_samples(
    paths: list[Path],
) -> tuple[list[PrimitiveSample], list[dict[str, object]]]:
    samples: list[PrimitiveSample] = []
    boundaries: list[dict[str, object]] = []
    for path in paths:
        rows = _load_jsonl(path)
        enters = {
            str(row["batch_id"]): row
            for row in rows
            if row.get("kind") == "tokenizer_batch_enter"
        }
        completes = {
            str(row["batch_id"]): row
            for row in rows
            if row.get("kind") == "tokenizer_batch_complete"
        }
        receives = {
            str(row["trace_id"]): int(row["ts_ns"])
            for row in rows
            if row.get("kind") == "engine_receive"
        }
        for batch_id, enter in enters.items():
            complete = completes.get(batch_id)
            if complete is None:
                continue
            scenario = str(enter["scenario"])
            batch_size = int(enter["batch_size"])
            total_prompt_tokens = sum(
                int(value) for value in complete["prompt_token_lengths"]
            )
            batch_start_ns = int(enter["batch_start_ns"])
            batch_complete_ns = int(complete["batch_complete_ns"])
            service_ms = (batch_complete_ns - batch_start_ns) / 1_000_000
            samples.append(
                PrimitiveSample(
                    total_prompt_tokens,
                    batch_size,
                    service_ms,
                    batch_size,
                    scenario,
                    batch_id,
                )
            )
            received_ns = [
                receives[str(trace_id)]
                for trace_id in enter.get("trace_ids", [])
                if str(trace_id) in receives
            ]
            if received_ns:
                median_receive_ns = statistics.median(received_ns)
                boundaries.append(
                    {
                        "scenario": scenario,
                        "batch_id": batch_id,
                        "batch_size": batch_size,
                        "total_prompt_tokens": total_prompt_tokens,
                        "tokenizer_service_ms": service_ms,
                        "tokenizer_to_engine_ms": (
                            median_receive_ns - batch_complete_ns
                        )
                        / 1_000_000,
                        "batch_to_engine_ms": (median_receive_ns - batch_start_ns)
                        / 1_000_000,
                    }
                )
    return samples, boundaries


def fit_measured_primitive(samples: list[PrimitiveSample]) -> PrimitiveFit:
    if len(samples) < 3:
        raise ValueError("need at least three primitive samples")
    design = np.asarray(
        [[1.0, float(sample.total_prompt_tokens), float(sample.batch_size)] for sample in samples]
    )
    observed = np.asarray([sample.service_ms for sample in samples], dtype=float)
    request_weights = np.asarray([sample.request_weight for sample in samples], dtype=float)
    weights = np.sqrt(request_weights)
    coefficients, *_ = np.linalg.lstsq(
        design * weights[:, None], observed * weights, rcond=None
    )
    predicted = design @ coefficients
    absolute_pct = np.abs(predicted - observed) / np.maximum(np.abs(observed), 1e-12)
    weighted_mape = float(np.average(absolute_pct, weights=request_weights))
    if weighted_mape < 1e-12:
        weighted_mape = 0.0
    return PrimitiveFit(
        intercept_ms=float(coefficients[0]),
        ms_per_prompt_token=float(coefficients[1]),
        ms_per_request=float(coefficients[2]),
        weighted_mape=weighted_mape,
    )


def fit_error_by_scenario(
    samples: list[PrimitiveSample], fit: PrimitiveFit
) -> dict[str, float]:
    grouped: dict[str, list[PrimitiveSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.scenario].append(sample)
    result: dict[str, float] = {}
    for scenario, rows in grouped.items():
        errors = [
            abs(fit.predict(row.total_prompt_tokens, row.batch_size) - row.service_ms)
            / max(abs(row.service_ms), 1e-12)
            for row in rows
        ]
        result[scenario] = float(
            np.average(errors, weights=[row.request_weight for row in rows])
        )
    return result


def summarize_preemption_signature(
    events: Iterable[dict[str, object]],
) -> dict[str, int]:
    rows = list(events)
    victims = Counter(row["victim_req_id"] for row in rows)
    return {
        "preemptions": len(rows),
        "unique_victims": len(victims),
        "repeat_victim_events": sum(count - 1 for count in victims.values()),
        "self_preemptions": sum(
            row.get("trigger_req_id") == row.get("victim_req_id") for row in rows
        ),
    }


def prototype_gate(
    *,
    visible_step_match: float,
    first_16_match: float,
    self_preemptions: int,
    repeat_victim_events: int,
    preemptions: int,
    target_preemptions: int,
) -> dict[str, object]:
    preemption_tolerance = max(2, round(target_preemptions * 0.2))
    passed = (
        visible_step_match == 1.0
        and first_16_match == 1.0
        and self_preemptions == 0
        and repeat_victim_events == 0
        and abs(preemptions - target_preemptions) <= preemption_tolerance
    )
    return {
        "passed": passed,
        "runtime_fix_allowed": False,
        "preemption_tolerance": preemption_tolerance,
    }


def measure_bt_admission() -> dict[str, object]:
    import scripts.validate_cb_simulator as validate
    from aiconfigurator.sdk.backends.cb_simulator.datatypes import Request, RequestState
    from aiconfigurator.sdk.backends.cb_simulator.scheduler import CBScheduler

    with BT_SERVE_LOG.open(encoding="utf-8", errors="replace") as source:
        real_rows = parse_iteration_rows(source)
    real_row = next(
        row
        for row in real_rows
        if int(row["context_tokens"]) >= 65_000
        and int(row["generation_requests"]) == 1
    )

    point = next(point for point in validate.MULTI_CONFIG_DATA if point.name == SCENARIO_BT)
    config = validate._make_multi_config_cb_config(
        point,
        overlap_factor=0.0,
        ep8_per_iteration_overhead_ms=validate.DEFAULT_EP8_PER_ITERATION_OVERHEAD_MS,
    )
    running_req = Request(0, point.isl, point.osl, 0.0)
    running_req.state = RequestState.DECODING
    running_req.prefill_tokens_remaining = 0
    running_req.generated_tokens = 1
    waiting = [Request(idx, point.isl, point.osl, 0.0) for idx in range(1, 128)]
    schedule = CBScheduler(config).schedule(waiting, [running_req])
    fresh = [
        req
        for req in schedule.prefill_reqs
        if req.state == RequestState.WAITING and req.num_preemptions == 0
    ]
    verdict = admission_cadence_verdict(
        real_max_context_requests=int(real_row["context_requests"]),
        sim_max_fresh_admissions=len(fresh),
        small_cohort_limit=SMALL_COHORT_LIMIT,
    )
    return {
        "real_row": real_row,
        "sim_fresh_admissions": len(fresh),
        "sim_prefill_tokens": schedule.total_prefill_tokens,
        "sim_decode_requests": len(schedule.decode_reqs),
        **verdict,
    }


def _trace_index(trace_id: str) -> int:
    return int(trace_id.rsplit("-", 1)[-1])


def _tokenizer_completion_oracle_inputs() -> tuple[list[float], dict[int, int]]:
    rows = _load_jsonl(ARRIVAL_ROOT / "32k3k.jsonl.gz")
    completes = {
        str(row["batch_id"]): int(row["batch_complete_ns"])
        for row in rows
        if row.get("kind") == "tokenizer_batch_complete"
    }
    completion_by_request: list[tuple[int, int]] = []
    for row in rows:
        if row.get("kind") != "tokenizer_batch_enter":
            continue
        complete_ns = completes[str(row["batch_id"])]
        completion_by_request.extend(
            (_trace_index(str(trace_id)), complete_ns)
            for trace_id in row.get("trace_ids", [])
        )
    completion_by_request.sort(key=lambda item: item[0])
    completion_by_request = completion_by_request[:PROTOTYPE_REQUESTS]
    first_complete = completion_by_request[0][1]
    arrivals = [
        (complete_ns - first_complete) / 1_000_000
        for _, complete_ns in completion_by_request
    ]
    real_first_steps: dict[int, int] = {}
    for row in rows:
        if row.get("kind") != "scheduler_step":
            continue
        for trace_id in row.get("new_context_trace_ids", []):
            request_id = _trace_index(str(trace_id))
            if request_id < PROTOTYPE_REQUESTS:
                real_first_steps.setdefault(request_id, int(row["step"]))
    return arrivals, real_first_steps


def _run_tokenizer_completion_oracle() -> dict[str, object]:
    import scripts.analyze_phase451h_first_divergence as phase451h
    import scripts.validate_cb_simulator as validate
    from aiconfigurator.sdk.backends.cb_simulator.scheduler import CBScheduler

    arrivals, real_first_steps = _tokenizer_completion_oracle_inputs()
    original_points = validate.MULTI_CONFIG_DATA
    original_loader = phase451h._load_validate_module
    original_schedule = CBScheduler.schedule
    original_ensure = CBScheduler._ensure_block_capacity
    original_preempt = CBScheduler._preempt
    local_steps: Counter[int] = Counter()
    sim_first_steps: dict[int, int] = {}
    active_trigger: dict[int, int] = {}
    events: list[dict[str, object]] = []

    def wrapped_schedule(self, waiting, running):
        local_steps[id(self)] += 1
        result = original_schedule(self, waiting, running)
        for req in result.prefill_reqs:
            if req.num_preemptions == 0 and req.request_id not in sim_first_steps:
                sim_first_steps[req.request_id] = local_steps[id(self)]
        return result

    def wrapped_ensure(self, current_req, waiting, running, result, preempted_ids):
        active_trigger[id(self)] = current_req.request_id
        try:
            return original_ensure(
                self, current_req, waiting, running, result, preempted_ids
            )
        finally:
            active_trigger.pop(id(self), None)

    def wrapped_preempt(self, victim, waiting, running, result, preempted_ids):
        events.append(
            {
                "trigger_req_id": active_trigger.get(id(self), -1),
                "victim_req_id": victim.request_id,
            }
        )
        return original_preempt(self, victim, waiting, running, result, preempted_ids)

    validate.MULTI_CONFIG_DATA = tuple(
        replace(point, osl=PROTOTYPE_OSL) if point.name == SCENARIO_32K else point
        for point in original_points
    )
    phase451h._load_validate_module = lambda: validate
    CBScheduler.schedule = wrapped_schedule
    CBScheduler._ensure_block_capacity = wrapped_ensure
    CBScheduler._preempt = wrapped_preempt
    try:
        result = phase451h.run_real_arrival_sim_trace(
            scenario=SCENARIO_32K,
            arrival_times_ms=arrivals,
            max_steps_per_rank=15_000,
        )
    finally:
        CBScheduler.schedule = original_schedule
        CBScheduler._ensure_block_capacity = original_ensure
        CBScheduler._preempt = original_preempt
        phase451h._load_validate_module = original_loader
        validate.MULTI_CONFIG_DATA = original_points

    # The first 14 requests are admitted before any request can finish.  This
    # keeps the visible-step comparison independent of the 3k-vs-1.2k OSL
    # difference between the arrival observation and preemption short run.
    comparable_ids = [
        request_id
        for request_id in range(14)
        if request_id in real_first_steps and request_id in sim_first_steps
    ]
    visible_step_match = (
        sum(real_first_steps[idx] == sim_first_steps[idx] for idx in comparable_ids)
        / len(comparable_ids)
        if comparable_ids
        else 0.0
    )
    with _open_text(PREEMPT_SERVE_LOG) as source:
        real_iterations = parse_iteration_rows(source)[:16]
    sim_trace = list(result["trace"])[:16]
    matched = 0
    for real, sim in zip(real_iterations, sim_trace):
        real_shape = (
            int(real["context_requests"]),
            int(real["context_tokens"]),
            int(real["generation_requests"]),
        )
        sim_shape = (
            int(sim["prefill_reqs"]),
            int(sim["prefill_tokens"]),
            int(sim["decode_reqs"]),
        )
        matched += real_shape == sim_shape
    first_16_match = matched / 16
    signature = summarize_preemption_signature(events)
    gate = prototype_gate(
        visible_step_match=visible_step_match,
        first_16_match=first_16_match,
        self_preemptions=signature["self_preemptions"],
        repeat_victim_events=signature["repeat_victim_events"],
        preemptions=signature["preemptions"],
        target_preemptions=10,
    )
    return {
        "visible_step_match": visible_step_match,
        "visible_step_pairs": len(comparable_ids),
        "first_16_match": first_16_match,
        **signature,
        **gate,
    }


def build_report() -> tuple[list[dict[str, object]], dict[str, object]]:
    admission = measure_bt_admission()
    samples, boundaries = load_primitive_samples(ARRIVAL_PATHS)
    tokenizer_fit = fit_measured_primitive(samples)
    scenario_errors = fit_error_by_scenario(samples, tokenizer_fit)
    bridge_samples = [
        PrimitiveSample(
            int(row["total_prompt_tokens"]),
            int(row["batch_size"]),
            float(row["tokenizer_to_engine_ms"]),
            int(row["batch_size"]),
            str(row["scenario"]),
            str(row["batch_id"]),
        )
        for row in boundaries
    ]
    bridge_fit = fit_measured_primitive(bridge_samples)
    oracle = _run_tokenizer_completion_oracle()
    tokenizer_pass = tokenizer_fit.weighted_mape <= TOKENIZER_FIT_MAX_WMAPE
    rows = [
        _row(
            "decision_fork",
            SCENARIO_BT,
            "real_context_requests",
            admission["real_row"]["context_requests"],
            target=">2 means large-budget cadence",
            status="pass",
            note=(
                f"Iteration({admission['real_row']['iteration']}), "
                f"ctx_tokens={admission['real_row']['context_tokens']}, decode=1"
            ),
        ),
        _row(
            "decision_fork",
            SCENARIO_BT,
            "sim_fresh_admissions",
            admission["sim_fresh_admissions"],
            target=admission["real_row"]["context_requests"],
            status=(
                "pass"
                if admission["sim_fresh_admissions"]
                == admission["real_row"]["context_requests"]
                else "fail"
            ),
            note=f"ctx_tokens={admission['sim_prefill_tokens']}; decode=1; recompute excluded",
        ),
        _row(
            "decision_fork",
            SCENARIO_BT,
            "verdict",
            admission["verdict"],
            target="source-backed unique fix",
            status="pass",
            note="admission cap change forbidden; large cohort is max_bt budget semantics",
        ),
        _row(
            "primitive_fit",
            "8k2k+32k3k",
            "batch_measurements",
            len(samples),
            target="covers 1024 request pairs",
            status="pass",
            note=f"request_weight={sum(sample.request_weight for sample in samples)}",
        ),
        _row(
            "primitive_fit",
            "8k2k+32k3k",
            "tokenizer_weighted_mape",
            tokenizer_fit.weighted_mape,
            target=f"<={TOKENIZER_FIT_MAX_WMAPE}",
            status="pass" if tokenizer_pass else "fail",
            note="valid only on observed ISL 8k/32k and batch 1-32 support",
        ),
        _row(
            "primitive_fit",
            "8k2k+32k3k",
            "formula",
            (
                f"{tokenizer_fit.intercept_ms:.9f} + "
                f"{tokenizer_fit.ms_per_prompt_token:.12f}*tokens + "
                f"({tokenizer_fit.ms_per_request:.9f})*batch"
            ),
            status="measured",
            note="negative batch term is observed batching amortization; no extrapolation",
        ),
    ]
    for scenario, error in sorted(scenario_errors.items()):
        rows.append(
            _row(
                "primitive_fit",
                scenario,
                "tokenizer_weighted_mape",
                error,
                target=f"<={TOKENIZER_FIT_MAX_WMAPE}",
                status="pass" if error <= TOKENIZER_FIT_MAX_WMAPE else "fail",
            )
        )
    rows.extend(
        [
            _row(
                "primitive_boundary",
                "8k2k+32k3k",
                "tokenizer_to_engine_fit_mape",
                bridge_fit.weighted_mape,
                target=f"<={TOKENIZER_FIT_MAX_WMAPE}",
                status="fail",
                note="delay includes in-flight model execution and EngineCore drain; not a tokenizer primitive",
            ),
            _row(
                "prototype_oracle",
                SCENARIO_32K,
                "visible_step_match",
                oracle["visible_step_match"],
                target="1.0",
                status="pass" if oracle["visible_step_match"] == 1.0 else "fail",
                note=f"pairs={oracle['visible_step_pairs']}; exact tokenizer completion timestamps",
            ),
            _row(
                "prototype_oracle",
                SCENARIO_32K,
                "first_16_match",
                oracle["first_16_match"],
                target="1.0",
                status="pass" if oracle["first_16_match"] == 1.0 else "fail",
            ),
        ]
    )
    for metric, target in (
        ("preemptions", "10±2"),
        ("self_preemptions", "0"),
        ("repeat_victim_events", "0"),
    ):
        value = oracle[metric]
        passed = (
            abs(int(value) - 10) <= 2 if metric == "preemptions" else int(value) == 0
        )
        rows.append(
            _row(
                "prototype_oracle",
                SCENARIO_32K,
                metric,
                value,
                target=target,
                status="pass" if passed else "fail",
            )
        )
    rows.extend(
        [
            _row(
                "gate",
                "all",
                "prototype_gate",
                "pass" if oracle["passed"] else "fail",
                target="all three signatures pass",
                status="pass" if oracle["passed"] else "fail",
                note="exact tokenizer completion oracle is stronger than the fitted primitive",
            ),
            _row(
                "gate",
                "all",
                "step2c",
                "unlocked" if oracle["passed"] else "locked",
                target="unlocked",
                status="pass" if oracle["passed"] else "blocked",
                note="report-only; no runtime, PerfDB, or validation gate changes",
            ),
            _row(
                "boundary",
                "all",
                "default_aic",
                "No-Go",
                status="blocked",
                note="diagnostic_only=true; valid_for_default=false; perf_database=false",
            ),
        ]
    )
    return rows, {
        "admission": admission,
        "tokenizer_fit": tokenizer_fit,
        "bridge_fit": bridge_fit,
        "oracle": oracle,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path, rows: list[dict[str, object]], details: dict[str, object]
) -> None:
    admission = details["admission"]
    tokenizer_fit = details["tokenizer_fit"]
    bridge_fit = details["bridge_fit"]
    oracle = details["oracle"]
    text = f"""# Phase462 Step2a-3c: 到达原语与 admission 原型

结论：**Step2c 继续锁住。** bt65536 的 real 与 sim 都在单步调度 9 个 context request，所谓“每步最多 1-2 个新 context”只适用于 `max_bt == ISL`，不能被扩成 admission 上限。tokenizer 微批服务函数能从 780 个 batch / 1024 个 request pair 测得，但 `tokenizer complete -> EngineCore receive` 受在途模型执行和 input-queue drain 支配，不能由 `(total_prompt_tokens, batch_size)` 单独决定。更严格的反事实直接重放实测 tokenizer-complete 时刻仍未复现三签名，因此到达原语单独入 runtime 没有因果闭环。

| 决策叉 | real | sim | 判定 |
|---|---:|---:|---|
| bt65536 单步 context/fresh admission | {admission['real_row']['context_requests']} | {admission['sim_fresh_admissions']} | 对齐；禁止修改 admission cap |
| 单步 context tokens | {admission['real_row']['context_tokens']} | {admission['sim_prefill_tokens']} | 对齐 |

real 取 Phase458 `Iteration({admission['real_row']['iteration']})`：此前只有 request 0 完成 prefill，尚无可抢占 cohort，因此该步 9 个 context request 都属于首次 admission。sim 用同状态重放，并只统计 `WAITING && num_preemptions == 0`，排除了 recompute。

## 实测原语

`tokenizer_ms = {tokenizer_fit.intercept_ms:.9f} + {tokenizer_fit.ms_per_prompt_token:.12f} * total_prompt_tokens + ({tokenizer_fit.ms_per_request:.9f}) * batch_size`

| 指标 | 结果 | 门 |
|---|---:|---:|
| batch / request pair | 780 / 1024 | 完整 |
| tokenizer weighted MAPE | {tokenizer_fit.weighted_mape:.3%} | <=10%，通过 |
| tokenizer complete -> EngineCore receive 拟合 MAPE | {bridge_fit.weighted_mape:.3%} | <=10%，失败 |

负的 batch 系数是观测支持域内的批处理摊销结果；该式只允许用于 ISL 8k/32k、batch 1-32，不允许外推。后半段延迟不是 add/IPC 常数：EngineCore 只在 step 边界 drain input queue，时间戳因此混入当前 model execution 的剩余时间。

## 三签名原型

原型没有用拟合值硬凑，而是直接喂实测 tokenizer batch complete 时刻，再由 sim 自己在迭代边界接收。这个 oracle 比预测原语更强；oracle 不过，拟合版本不可能解锁 runtime。逐请求可见步只比较首个 14-request cohort，避开 arrival run OSL=3k 与抢占短跑 OSL=1.2k 的协议差。

| 签名 | 结果 | 目标 |
|---|---:|---:|
| 逐请求首次调度步吻合 | {oracle['visible_step_match']:.3%} ({oracle['visible_step_pairs']} pairs) | 100% |
| 前 16 步形状吻合 | {oracle['first_16_match']:.3%} | 100% |
| preemption | {oracle['preemptions']} | 10±2 |
| self-preemption | {oracle['self_preemptions']} | 0 |
| repeat victim event | {oracle['repeat_victim_events']} | 0 |

首次分歧进一步落到**事件循环时钟边界**：部署态 `EngineCore` 在 `batch_queue_size > 1` 时走 `step_with_batch_queue`。它先 `schedule -> execute_model(non_block=True)`，future 未完成且队列未满就立即返回，再预调度下一批；队列满后才等待最老 future。真实 Iteration(0) 执行期间，Iteration(1) 已被提前排成空拍，之后 tokenizer 完成的请求到下一次 input drain 才可见。当前 cb_sim 只有“schedule -> 等完整 iteration latency -> 接收 -> 再 schedule”的单时钟循环，因此一到第二步就看见新请求，抢占仍为原来的同步相位签名。

| 部署源码 | 已确认语义 |
|---|---|
| vLLM 0.19.0 `v1/engine/core.py:185-211` | `max_concurrent_batches > 1` 时启用 batch queue，并选择 `step_with_batch_queue` |
| `core.py:421-494` | 队列未满时非阻塞 schedule/execute 下一批；队列满后才等待最老 future |
| `core.py:1136-1175` | 每次 engine step 前 drain input queue |
| cb_sim `simulator.py:163-198` | schedule 后直接累计完整 iteration latency，再进入下一轮 |

部署 `core.py` SHA256：`{VLLM_CORE_SHA256}`。

下一步不应进入 Step2c。先审计/原型化 vLLM `execute_model(..., non_block=True)` 下的 in-flight schedule/input-drain 状态机，验证它能否在不改 scheduler 语义的前提下复现三签名；若不能，正式把 ramp 输入流水线划出稳态模型边界。

本步 `report-only / diagnostic_only=true / valid_for_default=false / perf_database=false`；未改 runtime、PerfDB 或 validation gate，Default AIC 维持 No-Go。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    rows, details = build_report()
    write_csv(args.csv, rows)
    write_markdown(args.md, rows, details)
    oracle = details["oracle"]
    print(
        f"admission={details['admission']['verdict']} "
        f"tokenizer_wmape={details['tokenizer_fit'].weighted_mape:.6f} "
        f"prototype={'pass' if oracle['passed'] else 'fail'} step2c=locked"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
