#!/usr/bin/env python3
"""Phase462 Step2c-10 report-only engine-loop regression triage."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MethodType
from typing import Iterable, Mapping, Sequence
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import scripts.analyze_phase462_step2c9_six_point_ab as step2c9  # noqa: E402
import scripts.validate_cb_simulator as validate  # noqa: E402
from aiconfigurator.sdk.backends.cb_simulator import CBSimulator  # noqa: E402
from aiconfigurator.sdk.backends.cb_simulator import simulator as simulator_module  # noqa: E402


VERSION_PROFILE = "version_semantic_profile"
PHYSICAL_SCALING = "physical_scaling"
SCOPED_MEASUREMENT = "scoped_measurement"
REMOVE_FROM_DEFAULT = "remove_from_default_path"
CLASSIFICATIONS = frozenset(
    {VERSION_PROFILE, PHYSICAL_SCALING, SCOPED_MEASUREMENT, REMOVE_FROM_DEFAULT}
)

PHASE461_BASELINE_COMMIT = "01b88c72"
ENGINE_LOOP_COMMIT = "67007e10"
NULL_BLOCK_COMMIT = "55073d9d"
STEP2C9_CANDIDATE = REPO_ROOT / (
    "docs/iter_gap_investigation/phase462_step2c9_six_point_candidate.csv"
)
PANORAMA_CSV = REPO_ROOT / (
    "docs/iter_gap_investigation/phase462_dynamics_triage.csv"
)
DEFAULT_REPORT = REPO_ROOT / (
    "docs/iter_gap_investigation/phase462_engine_loop_regression_triage.md"
)
DEFAULT_PROVENANCE = REPO_ROOT / (
    "docs/iter_gap_investigation/phase462_engine_loop_constant_provenance.csv"
)
DEFAULT_PROFILE_APPENDIX = REPO_ROOT / (
    "docs/iter_gap_investigation/phase462_version_semantic_profile_design.md"
)
DEFAULT_NULL_SCOREBOARD = REPO_ROOT / (
    "docs/iter_gap_investigation/phase462_null_only_scoreboard.csv"
)


@dataclass(frozen=True)
class ConstantProvenance:
    constant: str
    current_value: str
    source: str
    measured_scope: str
    scaling_rule: str
    classification: str
    destination: str
    rationale: str


@dataclass(frozen=True)
class SplitAssessment:
    baseline_error: float
    core_error: float
    full_error: float
    full_improvement: float
    core_improvement: float
    core_share_of_full_improvement: float
    split_supported: bool


@dataclass(frozen=True)
class ParityAudit:
    status: str
    only_engine_loop_difference: bool
    confounds: tuple[str, ...]


@dataclass(frozen=True)
class DynamicSignature:
    iterations: int
    mixed_steps: int | None
    mixed_step_share: float | None
    preemptions: int | None
    self_preemptions: int | None
    repeat_victim_events: int | None
    recompute_tokens: int | None
    avg_decode_batch: float
    peak_decode_batch: int
    avg_prefill_tokens: float
    peak_prefill_tokens: int
    avg_total_tokens: float
    peak_total_tokens: int


@dataclass(frozen=True)
class VariantResult:
    scenario: str
    variant: str
    real_output_tok_s_gpu: float
    sim_output_tok_s_gpu: float
    error_ratio: float
    signature: DynamicSignature


class ImmediateArrivalLayer:
    """Expose submitted requests at the next engine-loop drain boundary."""

    def __init__(self, _primitive: object) -> None:
        self._ready: list[object] = []

    def submit_many(
        self,
        requests: Iterable[tuple[object, int]],
        *,
        now_ms: float,
    ) -> None:
        del now_ms
        self._ready.extend(request_id for request_id, _prompt_tokens in requests)

    def drain(self, now_ms: float) -> list[object]:
        del now_ms
        ready = self._ready
        self._ready = []
        return ready

    def next_event_ms(self) -> None:
        return None


def constant_provenance_rows() -> list[ConstantProvenance]:
    """Return the locked, one-class-per-constant provenance table."""
    rows = [
        ConstantProvenance(
            "backend_version_dispatch", "vllm/0.19.0",
            "arrival.py:resolve_tokenizer_primitive",
            "backend version", "exact version key", VERSION_PROFILE,
            "backend semantic profile resolver",
            "The state machine is source-version semantics, not workload timing.",
        ),
        ConstantProvenance(
            "null_block_reserve", "1 block per BlockPool",
            "datatypes.py:VLLM_NULL_BLOCKS_PER_POOL",
            "vLLM 0.19.0 BlockPool", "none", VERSION_PROFILE,
            "profile.null_blocks_per_pool",
            "Source-defined allocatable-capacity semantics.",
        ),
        ConstantProvenance(
            "batch_queue_depth", "2",
            "tokenizer_primitives.json:engine_loop.queue_depth",
            "vLLM 0.19.0, PP=1", "none", VERSION_PROFILE,
            "profile.batch_queue_depth",
            "EngineCore in-flight structure is version semantics; it is misplaced in the measured tokenizer row.",
        ),
        ConstantProvenance(
            "async_scheduling", "true",
            "tokenizer_primitives.json:engine_loop.async_scheduling",
            "vLLM 0.19.0, PP=1", "none", VERSION_PROFILE,
            "profile.async_scheduling",
            "Selects the source-defined non-blocking EngineCore path.",
        ),
        ConstantProvenance(
            "engine_loop_pp_applicability", "pipeline_parallel_size=1",
            "tokenizer_primitives.json:engine_loop.pipeline_parallel_size",
            "vLLM 0.19.0", "exact topology predicate", VERSION_PROFILE,
            "profile.applicability.pipeline_parallel_sizes",
            "Applicability is a version/topology contract, not a measured delay.",
        ),
        ConstantProvenance(
            "sampled_computed_placeholder_lifecycle", "source-aligned three-state counters",
            "engine_loop.py:AsyncCBScheduler; simulator.py:_run_single_engine_loop",
            "vLLM 0.19.0", "none", VERSION_PROFILE,
            "profile.request_progress_semantics",
            "Determines scheduler-visible progress and completion ordering.",
        ),
        ConstantProvenance(
            "preemption_victim_policy", "running tail",
            "scheduler.py:_ensure_block_capacity",
            "vLLM 0.19.0", "none", VERSION_PROFILE,
            "profile.preemption_victim_policy",
            "Source-defined scheduler policy.",
        ),
        ConstantProvenance(
            "admission_after_preemption", "stop new admission in the same step",
            "engine_loop.py:AsyncCBScheduler._next_waiting_candidate",
            "vLLM 0.19.0", "none", VERSION_PROFILE,
            "profile.admission.stop_after_preemption",
            "Source-defined same-step admission rule.",
        ),
        ConstantProvenance(
            "running_then_waiting_order", "running first, waiting second",
            "scheduler.py:schedule",
            "vLLM 0.19.0", "none", VERSION_PROFILE,
            "profile.admission.queue_order",
            "Scheduler ordering changes capacity and victim outcomes.",
        ),
        ConstantProvenance(
            "partial_prefill_admission", "stop after one partial prefill",
            "scheduler.py:schedule",
            "vLLM 0.19.0", "none", VERSION_PROFILE,
            "profile.chunking.stop_after_partial_prefill",
            "Source-defined chunking/admission interaction.",
        ),
        ConstantProvenance(
            "completion_release_order", "future completion then release/replacement submission",
            "simulator.py:_run_single_engine_loop.on_complete",
            "vLLM 0.19.0", "none", VERSION_PROFILE,
            "profile.completion_release_order",
            "Step2c-7 verified the source-aligned two-boundary ordering.",
        ),
        ConstantProvenance(
            "tokenizer_ms_per_prompt_token", "0.001003614837 ms/token",
            "tokenizer_primitives.json:formula.ms_per_prompt_token",
            "Kimi-K2.5, H200, vLLM 0.19.0, 8k/32k",
            "multiply by measured prompt tokens", PHYSICAL_SCALING,
            "optional tokenizer model parameter",
            "Token work has a physical token-count axis; the coefficient remains measurement-derived.",
        ),
        ConstantProvenance(
            "tokenizer_intercept_ms", "0.029270374 ms",
            "tokenizer_primitives.json:formula.intercept_ms",
            "Kimi-K2.5, H200, vLLM 0.19.0, N128 short runs",
            "constant only inside the measured scope", SCOPED_MEASUREMENT,
            "scoped tokenizer measurement row",
            "No source-defined cross-deployment scaling law.",
        ),
        ConstantProvenance(
            "tokenizer_ms_per_request", "-0.604326979 ms/request",
            "tokenizer_primitives.json:formula.ms_per_request",
            "Kimi-K2.5, H200, vLLM 0.19.0, batch 1-32",
            "linear only inside the measured scope", SCOPED_MEASUREMENT,
            "scoped tokenizer measurement row",
            "The fitted negative coefficient is not a portable physical constant.",
        ),
        ConstantProvenance(
            "tokenizer_support_scope", "prompt={8000,32000}, batch=1..32",
            "tokenizer_primitives.json:support",
            "two ISLs and one deployment", "exact scope match", SCOPED_MEASUREMENT,
            "scoped tokenizer measurement row key",
            "This is measured coverage, not engine semantics.",
        ),
        ConstantProvenance(
            "tokenizer_fit_error", "WMAPE=0.0591528581",
            "tokenizer_primitives.json:weighted_mape",
            "780 measured batches", "none", SCOPED_MEASUREMENT,
            "scoped tokenizer measurement metadata",
            "Quality metadata belongs with the measured row.",
        ),
        ConstantProvenance(
            "tokenizer_max_batch_size", "32 requests",
            "tokenizer_primitives.json:runtime.max_batch_size",
            "api-server tokenizer deployment", "none", REMOVE_FROM_DEFAULT,
            "optional diagnostic arrival component",
            "Client/api-server batching is deployment-specific and outside EngineCore semantics.",
        ),
        ConstantProvenance(
            "tokenizer_wait_timeout", "2.0 ms",
            "tokenizer_primitives.json:runtime.wait_timeout_ms",
            "api-server tokenizer deployment", "none", REMOVE_FROM_DEFAULT,
            "optional diagnostic arrival component",
            "The aggregation window is deployment cadence, not backend-core semantics.",
        ),
        ConstantProvenance(
            "tokenizer_workers", "1",
            "tokenizer_primitives.json:runtime.workers",
            "api-server tokenizer deployment", "none", REMOVE_FROM_DEFAULT,
            "optional diagnostic arrival component",
            "Worker count is deployment topology.",
        ),
        ConstantProvenance(
            "initial_workload_exposure", "closed-loop initial requests submitted at t=0",
            "simulator.py:_run_single_engine_loop",
            "benchmark harness", "none", REMOVE_FROM_DEFAULT,
            "optional diagnostic arrival component",
            "Exposure cadence is a client/HTTP/api-server boundary.",
        ),
        ConstantProvenance(
            "replacement_exposure", "submit replacement at prior future completion",
            "simulator.py:_run_single_engine_loop.on_complete",
            "closed-loop benchmark harness", "none", REMOVE_FROM_DEFAULT,
            "optional diagnostic arrival component",
            "Replacement submission timing is harness behavior, not scheduler semantics.",
        ),
    ]
    if len({row.constant for row in rows}) != len(rows):
        raise AssertionError("duplicate constant provenance row")
    if any(row.classification not in CLASSIFICATIONS for row in rows):
        raise AssertionError("unclassified engine-loop constant")
    return rows


def assess_core_split(
    *,
    baseline_error: float,
    core_error: float,
    full_error: float,
) -> SplitAssessment:
    """Require the core half to explain a strict majority of full improvement."""
    full_improvement = baseline_error - full_error
    core_improvement = baseline_error - core_error
    share = (
        core_improvement / full_improvement
        if full_improvement > 0
        else float("-inf")
    )
    return SplitAssessment(
        baseline_error=baseline_error,
        core_error=core_error,
        full_error=full_error,
        full_improvement=full_improvement,
        core_improvement=core_improvement,
        core_share_of_full_improvement=share,
        split_supported=(full_improvement > 0 and share > 0.5),
    )


def assess_step2c9_parity(
    *,
    baseline_has_null_block: bool,
    candidate_has_null_block: bool,
    tp_engine_loop_enabled: bool,
    dp_engine_loop_enabled: bool,
) -> ParityAudit:
    confounds = []
    if baseline_has_null_block != candidate_has_null_block:
        confounds.append("null_block_semantics")
    only_engine_loop = (
        not confounds and tp_engine_loop_enabled and not dp_engine_loop_enabled
    )
    return ParityAudit(
        status="pass_only_engine_loop" if only_engine_loop else "fail_confounded",
        only_engine_loop_difference=only_engine_loop,
        confounds=tuple(confounds),
    )


def summarize_dynamic_signature(
    *,
    trace: Sequence[Mapping[str, object]],
    preemption_events: Sequence[Mapping[str, object]],
    total_iterations: int | None = None,
    avg_decode_batch: float | None = None,
    peak_decode_batch: int | None = None,
    avg_total_tokens: float | None = None,
    peak_total_tokens: int | None = None,
    trace_observed: bool = True,
    preemption_events_observed: bool = True,
) -> DynamicSignature:
    iterations = total_iterations if total_iterations is not None else len(trace)
    mixed_steps = sum(
        bool(int(row.get("prefill_reqs", 0)) and int(row.get("decode_reqs", 0)))
        for row in trace
    )
    victims = Counter(int(event["victim_request_id"]) for event in preemption_events)
    preemptions = len(preemption_events)
    return DynamicSignature(
        iterations=iterations,
        mixed_steps=mixed_steps if trace_observed else None,
        mixed_step_share=(
            mixed_steps / iterations if trace_observed and iterations else None
        ),
        preemptions=preemptions if preemption_events_observed else None,
        self_preemptions=(
            sum(
                int(event["trigger_request_id"]) == int(event["victim_request_id"])
                for event in preemption_events
            )
            if preemption_events_observed
            else None
        ),
        repeat_victim_events=(
            sum(count - 1 for count in victims.values())
            if preemption_events_observed
            else None
        ),
        recompute_tokens=(
            sum(int(event.get("recompute_tokens", 0)) for event in preemption_events)
            if preemption_events_observed
            else None
        ),
        avg_decode_batch=(
            avg_decode_batch
            if avg_decode_batch is not None
            else sum(int(row.get("decode_reqs", 0)) for row in trace) / max(len(trace), 1)
        ),
        peak_decode_batch=(
            peak_decode_batch
            if peak_decode_batch is not None
            else max((int(row.get("decode_reqs", 0)) for row in trace), default=0)
        ),
        avg_prefill_tokens=(
            sum(int(row.get("prefill_tokens", 0)) for row in trace) / max(len(trace), 1)
        ),
        peak_prefill_tokens=max(
            (int(row.get("prefill_tokens", 0)) for row in trace), default=0
        ),
        avg_total_tokens=(
            avg_total_tokens
            if avg_total_tokens is not None
            else sum(int(row.get("total_tokens", 0)) for row in trace) / max(len(trace), 1)
        ),
        peak_total_tokens=(
            peak_total_tokens
            if peak_total_tokens is not None
            else max((int(row.get("total_tokens", 0)) for row in trace), default=0)
        ),
    )


def _git_contains(ancestor: str, descendant: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def _attach_legacy_probe(sim: CBSimulator) -> tuple[list[dict[str, int]], list[dict[str, int]]]:
    trace: list[dict[str, int]] = []
    events: list[dict[str, int]] = []
    scheduler = sim._scheduler
    active_trigger = {"request_id": -1}
    original_schedule = scheduler.schedule
    original_ensure = scheduler._ensure_block_capacity
    original_preempt = scheduler._preempt

    def schedule_wrapper(_self: object, waiting: list[object], running: list[object]):
        result = original_schedule(waiting, running)
        trace.append(
            {
                "prefill_reqs": len(result.prefill_reqs),
                "prefill_tokens": result.total_prefill_tokens,
                "decode_reqs": len(result.decode_reqs),
                "total_tokens": result.total_tokens,
            }
        )
        return result

    def ensure_wrapper(
        _self: object,
        current_req: object,
        waiting: list[object],
        running: list[object],
        result: object,
        preempted_ids: set[int],
    ):
        active_trigger["request_id"] = int(current_req.request_id)
        try:
            return original_ensure(current_req, waiting, running, result, preempted_ids)
        finally:
            active_trigger["request_id"] = -1

    def preempt_wrapper(
        _self: object,
        victim: object,
        waiting: list[object],
        running: list[object],
        result: object,
        preempted_ids: set[int],
    ):
        events.append(
            {
                "trigger_request_id": active_trigger["request_id"],
                "victim_request_id": int(victim.request_id),
                "recompute_tokens": int(victim.isl + victim.sampled_output_tokens),
            }
        )
        return original_preempt(victim, waiting, running, result, preempted_ids)

    scheduler.schedule = MethodType(schedule_wrapper, scheduler)
    scheduler._ensure_block_capacity = MethodType(ensure_wrapper, scheduler)
    scheduler._preempt = MethodType(preempt_wrapper, scheduler)
    return trace, events


def _run_variant(point: validate.MultiConfigPoint, variant: str) -> VariantResult:
    if variant not in {"null_only", "core_only", "full"}:
        raise ValueError(f"unknown variant: {variant}")
    if variant != "null_only" and point.dp != 1:
        raise ValueError("engine-loop variants are deferred for DP")

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
    config = replace(config, engine_loop_enabled=variant != "null_only")
    sim = CBSimulator(backend, model, database, config)
    probe_trace: list[dict[str, int]] = []
    probe_events: list[dict[str, int]] = []
    if variant == "null_only" and point.dp == 1:
        probe_trace, probe_events = _attach_legacy_probe(sim)

    def run_simulation():
        if point.dp > 1 and point.max_num_batched_tokens == point.isl:
            return sim.run_multi_replica(
                isl=point.isl,
                osl=point.osl,
                concurrency=point.batch_size,
                data_parallel_size=point.dp,
                num_gpus=point.tp * point.dp,
                lockstep=True,
            )
        return sim.run(
            isl=point.isl,
            osl=point.osl,
            concurrency=math.ceil(point.batch_size / point.dp),
            num_gpus=point.tp,
        )

    if variant == "core_only":
        with patch.object(simulator_module, "TokenizerArrivalLayer", ImmediateArrivalLayer):
            result = run_simulation()
    else:
        result = run_simulation()

    trace = sim.get_last_schedule_trace() if variant != "null_only" else probe_trace
    events = sim.get_last_preemption_events() if variant != "null_only" else probe_events
    signature = summarize_dynamic_signature(
        trace=trace,
        preemption_events=events,
        total_iterations=result.total_iterations,
        avg_decode_batch=result.avg_decode_reqs_per_iter,
        peak_decode_batch=result.peak_decode_reqs_per_iter,
        avg_total_tokens=result.avg_tokens_per_iter,
        peak_total_tokens=result.peak_tokens_per_iter,
        trace_observed=not (variant == "null_only" and point.dp > 1),
        preemption_events_observed=not (
            variant == "null_only" and point.dp > 1
        ),
    )
    return VariantResult(
        scenario=point.name,
        variant=variant,
        real_output_tok_s_gpu=point.real_output_tok_s_gpu,
        sim_output_tok_s_gpu=result.throughput_tok_s_gpu,
        error_ratio=validate._abs_error(
            result.throughput_tok_s_gpu,
            point.real_output_tok_s_gpu,
        ),
        signature=signature,
    )


def _load_step2c9_candidate() -> dict[str, float]:
    with STEP2C9_CANDIDATE.open(newline="", encoding="utf-8") as source:
        return {
            str(row["name"]): float(row["sim_output_tok_s_gpu"])
            for row in csv.DictReader(source)
        }


def _load_panorama() -> dict[str, tuple[int, int]]:
    values: dict[str, dict[str, int]] = {}
    with PANORAMA_CSV.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if row["section"] != "preemption_panorama":
                continue
            if row["metric"] not in {"real_preemptions", "sim_preemptions"}:
                continue
            values.setdefault(row["scenario"], {})[row["metric"]] = int(float(row["value"]))
    return {
        scenario: (metrics["real_preemptions"], metrics["sim_preemptions"])
        for scenario, metrics in values.items()
        if {"real_preemptions", "sim_preemptions"}.issubset(metrics)
    }


def _fmt(value: float | None, digits: int = 6) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def _fmt_count(value: int | None) -> str:
    return "-" if value is None else str(value)


def _write_provenance(path: Path, rows: Sequence[ConstantProvenance]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=list(asdict(rows[0])),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _write_null_scoreboard(
    path: Path,
    variants: Mapping[tuple[str, str], VariantResult],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario",
        "tp",
        "dp",
        "max_num_batched_tokens",
        "real_output_tok_s_gpu",
        "sim_output_tok_s_gpu",
        "error_ratio",
        "gate_pass",
        "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        for point in validate.MULTI_CONFIG_DATA:
            result = variants[(point.name, "null_only")]
            gate_pass = result.error_ratio <= 1.15
            writer.writerow(
                {
                    "scenario": point.name,
                    "tp": point.tp,
                    "dp": point.dp,
                    "max_num_batched_tokens": point.max_num_batched_tokens,
                    "real_output_tok_s_gpu": result.real_output_tok_s_gpu,
                    "sim_output_tok_s_gpu": result.sim_output_tok_s_gpu,
                    "error_ratio": result.error_ratio,
                    "gate_pass": gate_pass,
                    "status": "pass" if gate_pass else "fail",
                }
            )


def _write_profile_appendix(path: Path) -> None:
    path.write_text(
        """# Phase462 版本语义 profile 设计附录

结论：版本 profile 只承载 EngineCore 可由源码核验的结构语义；tokenizer、HTTP、client 暴露节奏不得进入默认 profile。本步只设计，不实施。

## 字段

| 字段 | 类型 | 来源 | 未知版本行为 |
|---|---|---|---|
| `backend` / `backend_version` | exact key | PerfDB deployment metadata | fail closed |
| `applicability.pipeline_parallel_sizes` | list[int] | source path audit | reject |
| `batch_queue_depth` | int | `step_with_batch_queue` source/config | reject |
| `async_scheduling` | bool | source/config | reject |
| `null_blocks_per_pool` | int | `BlockPool` source | reject |
| `request_progress_semantics` | enum | sampled/computed/placeholder source order | reject |
| `completion_release_order` | enum | future callback source order | reject |
| `preemption_victim_policy` | enum | scheduler source | reject |
| `admission.queue_order` | enum | scheduler source | reject |
| `admission.stop_after_preemption` | bool | scheduler source | reject |
| `chunking.stop_after_partial_prefill` | bool | scheduler source | reject |
| `router_policy` | enum | Step 3 source/runtime evidence | unresolved; DP reject |

## 分发

| 位置 | 当前问题 | 设计动作 |
|---|---|---|
| `datatypes.py` | null block 常数硬编码 | 由不可变 profile 注入 `CBSimConfig` |
| `arrival.py` / tokenizer resource | queue depth 与部署测量混放 | queue depth 移入 profile；arrival row 保持 scoped diagnostic |
| `engine_loop.py` | 生命周期与 admission 语义散落在类实现 | resolver 选择明确的语义枚举，运行时不猜 |
| `simulator.py` | engine loop 同时绑定 arrival 与 core | core input 只接 workload 可见请求；arrival adapter 独立可选 |
| `vllm_backend.py` | TP/DP 分发规则分散 | `backend_version -> profile` 一次解析；DP 在 router profile 未完成前显式拒绝 |

建议资源形态：`resources/backend_semantic_profiles.json` + 单一 `resolve_backend_semantic_profile(backend, version, topology)`。profile 只做精确分发，不含默认值、模糊匹配、版本回退或运行时启发式。

边界：`router_policy` 必须等 DP Step 3；本附录不改 runtime、gate 或 PerfDB。
""",
        encoding="utf-8",
    )


def _mechanism_label(
    point: validate.MultiConfigPoint,
    baseline_error: float,
    null_result: VariantResult,
    core_result: VariantResult | None,
    full_result: VariantResult | None,
) -> str:
    if point.dp > 1:
        return (
            f"null block delta={null_result.error_ratio - baseline_error:+.6f}; "
            "Step2c-9 did not run the engine loop on DP"
        )
    assert core_result is not None and full_result is not None
    return (
        f"null={null_result.error_ratio - baseline_error:+.6f}; "
        f"core={core_result.error_ratio - null_result.error_ratio:+.6f}; "
        f"arrival={full_result.error_ratio - core_result.error_ratio:+.6f}"
    )


def _write_report(
    path: Path,
    *,
    parity: ParityAudit,
    baseline: Mapping[str, Mapping[str, float | str]],
    variants: Mapping[tuple[str, str], VariantResult],
    split: SplitAssessment,
    panorama: Mapping[str, tuple[int, int]],
    provenance_rows: Sequence[ConstantProvenance],
) -> None:
    scenario_rows = []
    dynamic_rows = []
    for point in validate.MULTI_CONFIG_DATA:
        baseline_error = float(baseline[point.name]["error_ratio"])
        null_result = variants[(point.name, "null_only")]
        core_result = variants.get((point.name, "core_only"))
        full_result = variants.get((point.name, "full"))
        scenario_rows.append(
            "| " + " | ".join(
                [
                    point.name,
                    _fmt(baseline_error),
                    _fmt(null_result.error_ratio),
                    _fmt(core_result.error_ratio if core_result else None),
                    _fmt(full_result.error_ratio if full_result else None),
                    _mechanism_label(
                        point, baseline_error, null_result, core_result, full_result
                    ),
                ]
            ) + " |"
        )
        for variant in ("null_only", "core_only", "full"):
            item = variants.get((point.name, variant))
            if item is None:
                continue
            sig = item.signature
            preemption_shape = "/".join(
                _fmt_count(value)
                for value in (
                    sig.preemptions,
                    sig.self_preemptions,
                    sig.repeat_victim_events,
                )
            )
            mixed_shape = (
                "-"
                if sig.mixed_steps is None or sig.mixed_step_share is None
                else f"{sig.mixed_steps} ({sig.mixed_step_share:.4%})"
            )
            dynamic_rows.append(
                f"| {point.name} | {variant} | {sig.iterations} | "
                f"{preemption_shape} | {_fmt_count(sig.recompute_tokens)} | {mixed_shape} | "
                f"{sig.avg_decode_batch:.3f}/{sig.peak_decode_batch} | "
                f"{sig.avg_total_tokens:.3f}/{sig.peak_total_tokens} |"
            )

    classification_counts = Counter(row.classification for row in provenance_rows)
    class_table = "\n".join(
        f"| {name} | {classification_counts[name]} |"
        for name in (
            VERSION_PROFILE,
            PHYSICAL_SCALING,
            SCOPED_MEASUREMENT,
            REMOVE_FROM_DEFAULT,
        )
    )
    panorama_table = "\n".join(
        f"| {point.name} | {panorama.get(point.name, ('-', '-'))[0]} | "
        f"{panorama.get(point.name, ('-', '-'))[1]} |"
        for point in validate.MULTI_CONFIG_DATA
    )
    null_passed = sum(
        variants[(point.name, "null_only")].error_ratio <= 1.15
        for point in validate.MULTI_CONFIG_DATA
    )
    core_passed = sum(
        variants[
            (
                point.name,
                "core_only" if point.dp == 1 else "null_only",
            )
        ].error_ratio <= 1.15
        for point in validate.MULTI_CONFIG_DATA
    )
    target_null_error = variants[("K2.5-tp8ep8-32k3k", "null_only")].error_ratio
    arrival_zero = all(
        math.isclose(
            variants[(point.name, "core_only")].error_ratio,
            variants[(point.name, "full")].error_ratio,
            rel_tol=0,
            abs_tol=1e-12,
        )
        for point in validate.MULTI_CONFIG_DATA
        if point.dp == 1
    )
    if split.split_supported:
        verdict = (
            "因果拆分成立，但默认采用条件不成立：三个 TP 场景 core-only 与 full 完全一致，"
            "arrival/exposure 贡献为零；核心状态机仍把组合计分板从 "
            f"{null_passed}/6 拉到 {core_passed}/6。引擎环继续关闭，只保留 null block。"
        )
        next_step = (
            "版本 profile 可作为不启用行为的框架小步单独评审；动态主线回到 DP Step 3 与 bt65536 分解。"
        )
    else:
        verdict = (
            "拆分不成立：TP8-32k3k 改善依赖 arrival/exposure；引擎环整体只保留为诊断工具，"
            "默认路径仅保留 null block。"
        )
        next_step = "按 null-block-only 基线继续 DP Step 3 与 bt65536 分解。"

    path.write_text(
        f"""# Phase462 Step 2c-10 引擎环倒退分诊

结论：{verdict}

Step 2c-9 的“A/B 唯一差异是引擎环”审计失败：Phase461 baseline 产物来自 `{PHASE461_BASELINE_COMMIT}`，早于 null block `{NULL_BLOCK_COMMIT}`；candidate TP 是 `null block + engine loop`，candidate DP 是 `null block only`。因此五个倒退不能统一归因到 tokenizer 原语。

## 对等审计

| 项目 | 结果 |
|---|---|
| only engine-loop difference | `{str(parity.only_engine_loop_difference).lower()}` |
| status | `{parity.status}` |
| confounds | `{json.dumps(parity.confounds)}` |
| 同 HEAD 重建 | `null_only / core_only / full`；core-only 用即时暴露，无时间常数、无拟合参数 |

## 拆分裁决

预注册定义：“保留大部分改善”=`core improvement / full improvement > 0.5`。核心半边包含 null block 与 EngineCore 状态机；这是多数定义，不按输出调阈值。

| baseline | null-only | core-only | full | null 改善 | core 增量 | arrival 增量 | 核心占比 | 裁决 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| {split.baseline_error:.6f} | {target_null_error:.6f} | {split.core_error:.6f} | {split.full_error:.6f} | {split.baseline_error - target_null_error:+.6f} | {target_null_error - split.core_error:+.6f} | {split.core_error - split.full_error:+.6f} | {split.core_share_of_full_improvement:.2%} | {'split_supported' if split.split_supported else 'split_rejected'} |

三个 TP 场景 `core-only == full`：`{str(arrival_zero).lower()}`。这直接证伪 Step 2c-9 的“测量原语未迁移”解释；TP8-8k2k 与 TP8-bt65536 的额外倒退来自核心状态机，三个 DP 变化来自 null block。

## 六点分解

`null_only`、`core_only`、`full` 均在当前 HEAD 上离线重跑；DP 引擎环仍显式拒绝，所以 DP 只有 `null_only`。

| 场景 | Phase461 error | null-only | core-only | full | 机制判定 |
|---|---:|---:|---:|---:|---|
{chr(10).join(scenario_rows)}

## 动态签名

| 场景 | 变体 | iterations | preempt/self/repeat | recompute tokens | mixed steps | avg/peak decode | avg/peak tokens |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(dynamic_rows)}

历史 real/sim 抢占计数来自 N512 panorama，仅作方向对照；本表三层重跑是 N384 验证协议，不能直接比较绝对值。

| 场景 | real N512 | legacy sim N512 |
|---|---:|---:|
{panorama_table}

## 常数归置

每个常数只落一类；完整逐行证据见 `phase462_engine_loop_constant_provenance.csv`。

| 分类 | 数量 |
|---|---:|
{class_table}

`null-only` 计分板 `{null_passed}/6`；`core-only TP + null-only DP` 计分板 `{core_passed}/6`。版本 profile 字段与分发点位见 `phase462_version_semantic_profile_design.md`。{next_step}

边界：report-only；未改 runtime、门或 PerfDB；未跑 GPU；`Default AIC=No-Go`；归档行重审与 DP Step 3 顺延。
""",
        encoding="utf-8",
    )


def run_triage(
    *,
    report: Path = DEFAULT_REPORT,
    provenance: Path = DEFAULT_PROVENANCE,
    profile_appendix: Path = DEFAULT_PROFILE_APPENDIX,
    null_scoreboard: Path = DEFAULT_NULL_SCOREBOARD,
) -> dict[str, object]:
    baseline = step2c9.load_phase461_baseline()
    parity = assess_step2c9_parity(
        baseline_has_null_block=_git_contains(NULL_BLOCK_COMMIT, PHASE461_BASELINE_COMMIT),
        candidate_has_null_block=_git_contains(NULL_BLOCK_COMMIT, "HEAD"),
        tp_engine_loop_enabled=True,
        dp_engine_loop_enabled=False,
    )
    if parity.only_engine_loop_difference:
        raise AssertionError("Step2c-9 parity unexpectedly passed")

    variants: dict[tuple[str, str], VariantResult] = {}
    for point in validate.MULTI_CONFIG_DATA:
        variants[(point.name, "null_only")] = _run_variant(point, "null_only")
        if point.dp == 1:
            variants[(point.name, "core_only")] = _run_variant(point, "core_only")
            variants[(point.name, "full")] = _run_variant(point, "full")

    step2c9_candidate = _load_step2c9_candidate()
    for point in validate.MULTI_CONFIG_DATA:
        expected_variant = "full" if point.dp == 1 else "null_only"
        actual = variants[(point.name, expected_variant)].sim_output_tok_s_gpu
        if not math.isclose(actual, step2c9_candidate[point.name], rel_tol=0, abs_tol=1e-9):
            raise AssertionError(f"{point.name}: Step2c-9 candidate reproduction mismatch")

    target = "K2.5-tp8ep8-32k3k"
    split = assess_core_split(
        baseline_error=float(baseline[target]["error_ratio"]),
        core_error=variants[(target, "core_only")].error_ratio,
        full_error=variants[(target, "full")].error_ratio,
    )
    provenance_rows = constant_provenance_rows()
    _write_provenance(provenance, provenance_rows)
    _write_profile_appendix(profile_appendix)
    _write_null_scoreboard(null_scoreboard, variants)
    _write_report(
        report,
        parity=parity,
        baseline=baseline,
        variants=variants,
        split=split,
        panorama=_load_panorama(),
        provenance_rows=provenance_rows,
    )
    return {
        "status": "completed_report_only",
        "parity": asdict(parity),
        "split": asdict(split),
        "decision": (
            "causal_split_supported_default_blocked"
            if split.split_supported
            else "diagnostic_only"
        ),
        "default_aic": "No-Go",
        "runtime_changed": False,
        "gate_changed": False,
        "perfdb_changed": False,
    }


def main() -> int:
    print(json.dumps(run_triage(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
