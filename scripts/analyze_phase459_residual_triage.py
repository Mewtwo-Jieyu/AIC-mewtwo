#!/usr/bin/env python3
"""Phase459: triage residuals exposed by the Phase458 N=512 references."""

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
from typing import Iterable, TextIO


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import validate_cb_simulator as validate  # noqa: E402
from aiconfigurator.sdk.backends.cb_simulator import CBSimulator  # noqa: E402
from aiconfigurator.sdk.backends.cb_simulator.iteration_latency import (  # noqa: E402
    IterationLatencyCalculator,
)
from aiconfigurator.sdk.backends.cb_simulator.scheduler import CBScheduler  # noqa: E402


DEFAULT_AB_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase458_validate_ab.csv"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase459_residual_triage.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase459_residual_triage.md"
SERVING_STATE_PERF = (
    REPO_ROOT
    / "src/aiconfigurator/systems/data/h200_sxm/vllm/0.19.0/vllm_serving_state_perf.txt"
)
PHASE454_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase454_gpu_batch"
PHASE458_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase458_n512_unify"

TP8_32K = "K2.5-tp8ep8-32k3k"
TP8_BT = "K2.5-tp8ep8-8k2k-bt65536"
DP2_BT = "K2.5-tp4ep8dp2-8k2k-bt65536"
DP2_8K = "K2.5-tp4ep8dp2-8k2k"
DP2_32K = "K2.5-tp4ep8dp2-32k3k"
NUM_REQUESTS = 512
TARGET_RATIO = 1.15

CSV_FIELDS = [
    "section",
    "scenario",
    "side",
    "metric",
    "value",
    "target",
    "status",
    "source",
    "note",
]

ITER_RE = re.compile(
    r"EngineCore(?:_DP(?P<engine>\d+))?.*?"
    r"Iteration\((?P<iteration>\d+)\):\s+"
    r"(?P<ctx_req>\d+)\s+context requests,\s+"
    r"(?P<ctx_tok>\d+)\s+context tokens,\s+"
    r"(?P<gen_req>\d+)\s+generation requests,\s+"
    r"(?P<gen_tok>\d+)\s+generation tokens,\s+"
    r"iteration elapsed time:\s+(?P<elapsed>[0-9.]+)\s+ms"
)
PROM_RE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[-+0-9.eE]+)$"
)
LABEL_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="([^"]*)"')
KV_RE = re.compile(r"GPU KV cache size:\s+(?P<tokens>[\d,]+)\s+tokens")


@dataclass(frozen=True)
class IterationStep:
    engine: str
    iteration: int
    ctx_requests: int
    ctx_tokens: int
    generation_requests: int
    generation_tokens: int
    elapsed_ms: float

    @property
    def is_mixed(self) -> bool:
        return self.ctx_tokens > 0 and self.generation_requests > 0

    @property
    def is_decode(self) -> bool:
        return self.ctx_tokens == 0 and self.generation_requests > 0


@dataclass(frozen=True)
class MetricsProfile:
    preemptions: float
    recomputed_prompt_tokens: float
    successful_requests: float
    prompt_tokens: float
    running_p50: float
    running_p90: float
    waiting_p50: float
    kv_usage_p90: float
    samples: int


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    artifact_dir: Path
    max_bt: int

    @property
    def serve_log(self) -> Path:
        return self.artifact_dir / "serve.log"

    @property
    def metrics(self) -> Path:
        for name in ("metrics.jsonl.gz", "metrics.jsonl"):
            path = self.artifact_dir / name
            if path.exists():
                return path
        return self.artifact_dir / "metrics.jsonl.gz"


SCENARIOS = {
    TP8_32K: ScenarioSpec(
        TP8_32K,
        PHASE458_ROOT / "recollect_tp8_32k3k" / TP8_32K,
        32_000,
    ),
    TP8_BT: ScenarioSpec(
        TP8_BT,
        PHASE458_ROOT / "recollect_tp8_8k2k_bt65536" / TP8_BT,
        65_536,
    ),
    DP2_BT: ScenarioSpec(
        DP2_BT,
        PHASE458_ROOT / "recollect_dp2_8k2k_bt65536" / DP2_BT,
        65_536,
    ),
}


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


def _fmt(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _percentile(values: Iterable[float], pct: float) -> float:
    data = sorted(values)
    if not data:
        return math.nan
    pos = (len(data) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return data[lo]
    return data[lo] * (hi - pos) + data[hi] * (pos - lo)


def parse_iteration_steps(path: Path) -> list[IterationStep]:
    steps: list[IterationStep] = []
    with _open_text(path) as f:
        for line in f:
            match = ITER_RE.search(line)
            if not match:
                continue
            steps.append(
                IterationStep(
                    engine=match.group("engine") or "0",
                    iteration=int(match.group("iteration")),
                    ctx_requests=int(match.group("ctx_req")),
                    ctx_tokens=int(match.group("ctx_tok")),
                    generation_requests=int(match.group("gen_req")),
                    generation_tokens=int(match.group("gen_tok")),
                    elapsed_ms=float(match.group("elapsed")),
                )
            )
    if not steps:
        raise ValueError(f"missing iteration rows: {path}")
    return steps


def _parse_prometheus(line: str) -> tuple[str, dict[str, str], float] | None:
    match = PROM_RE.match(line.strip())
    if not match:
        return None
    labels = {key: value for key, value in LABEL_RE.findall(match.group("labels") or "")}
    return match.group("name"), labels, float(match.group("value"))


def parse_metrics_profile(path: Path) -> MetricsProfile:
    counters: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    running: list[float] = []
    waiting: list[float] = []
    kv_usage: list[float] = []
    samples = 0
    with _open_text(path) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if int(record.get("status", 0)) != 200:
                continue
            gauges: dict[str, dict[str, float]] = defaultdict(dict)
            for raw in str(record.get("body", "")).splitlines():
                parsed = _parse_prometheus(raw)
                if parsed is None:
                    continue
                name, labels, value = parsed
                engine = labels.get("engine", "0")
                if name in {
                    "vllm:num_preemptions_total",
                    "vllm:prompt_tokens_recomputed_total",
                    "vllm:prompt_tokens_total",
                }:
                    counters[name][engine].append(value)
                elif name == "vllm:request_success_total" and labels.get("finished_reason") == "length":
                    counters[name][engine].append(value)
                elif name == "vllm:num_requests_running":
                    gauges["running"][engine] = value
                elif name == "vllm:num_requests_waiting":
                    gauges["waiting"][engine] = value
                elif "cache_usage" in name or "gpu_cache" in name:
                    gauges["kv_usage"][engine] = value
            if gauges["running"] or gauges["waiting"]:
                samples += 1
                running.append(sum(gauges["running"].values()))
                waiting.append(sum(gauges["waiting"].values()))
                if gauges["kv_usage"]:
                    kv_usage.append(max(gauges["kv_usage"].values()))

    def delta(name: str) -> float:
        return sum(values[-1] - values[0] for values in counters[name].values() if values)

    return MetricsProfile(
        preemptions=delta("vllm:num_preemptions_total"),
        recomputed_prompt_tokens=delta("vllm:prompt_tokens_recomputed_total"),
        successful_requests=delta("vllm:request_success_total"),
        prompt_tokens=delta("vllm:prompt_tokens_total"),
        running_p50=_percentile(running, 50),
        running_p90=_percentile(running, 90),
        waiting_p50=_percentile(waiting, 50),
        kv_usage_p90=_percentile(kv_usage, 90),
        samples=samples,
    )


def parse_kv_cache_tokens(path: Path) -> int:
    with _open_text(path) as f:
        values = {int(match.group("tokens").replace(",", "")) for line in f if (match := KV_RE.search(line))}
    if len(values) != 1:
        raise ValueError(f"expected one KV capacity in {path}, got {sorted(values)}")
    return values.pop()


def load_serving_state_regimes(path: Path, topology: str) -> set[int]:
    with path.open(newline="", encoding="utf-8") as f:
        return {
            int(row["max_num_batched_tokens"])
            for row in csv.DictReader(f)
            if row["topology"] == topology
            and row["phase"] == "mixed_prefill"
            and row["row_kind"] == "forward_total"
        }


def summarize_iterations(steps: list[IterationStep]) -> dict[str, float]:
    mixed = [step for step in steps if step.is_mixed]
    decode = [step for step in steps if step.is_decode]
    prefill = [step for step in steps if step.ctx_tokens > 0]
    return {
        "total_steps": float(len(steps)),
        "mixed_steps": float(len(mixed)),
        "mixed_share": len(mixed) / len(steps),
        "mixed_ctx_p50": _percentile((step.ctx_tokens for step in mixed), 50),
        "mixed_batch_p50": _percentile((step.generation_requests for step in mixed), 50),
        "mixed_elapsed_p50": _percentile((step.elapsed_ms for step in mixed), 50),
        "decode_batch_p50": _percentile((step.generation_requests for step in decode), 50),
        "decode_elapsed_p50": _percentile((step.elapsed_ms for step in decode), 50),
        "prefill_elapsed_p50": _percentile((step.elapsed_ms for step in prefill), 50),
    }


def representative_large_mixed_step(steps: list[IterationStep], max_bt: int) -> IterationStep:
    candidates = [step for step in steps if step.is_mixed and step.ctx_tokens >= max_bt * 0.75]
    if not candidates:
        raise ValueError(f"missing large mixed step for max_bt={max_bt}")
    groups: dict[tuple[int, int], list[IterationStep]] = defaultdict(list)
    for step in candidates:
        groups[(step.ctx_requests, step.generation_requests)].append(step)
    _, selected = max(groups.items(), key=lambda item: (len(item[1]), item[0]))
    return IterationStep(
        engine=selected[0].engine,
        iteration=selected[0].iteration,
        ctx_requests=selected[0].ctx_requests,
        ctx_tokens=round(statistics.median(step.ctx_tokens for step in selected)),
        generation_requests=selected[0].generation_requests,
        generation_tokens=selected[0].generation_requests,
        elapsed_ms=statistics.median(step.elapsed_ms for step in selected),
    )


def representative_step(steps: list[IterationStep], step_class: str) -> IterationStep:
    if step_class == "mixed":
        candidates = [step for step in steps if step.is_mixed]
    elif step_class == "prefill":
        candidates = [step for step in steps if step.ctx_tokens > 0 and not step.is_mixed]
    elif step_class == "decode":
        candidates = [step for step in steps if step.is_decode]
    else:
        raise ValueError(f"unknown step class: {step_class}")
    if not candidates:
        raise ValueError(f"missing {step_class} steps")
    groups: dict[tuple[int, int], list[IterationStep]] = defaultdict(list)
    for step in candidates:
        groups[(step.ctx_requests, step.generation_requests)].append(step)
    _, selected = max(groups.items(), key=lambda item: (len(item[1]), item[0]))
    return IterationStep(
        engine=selected[0].engine,
        iteration=selected[0].iteration,
        ctx_requests=selected[0].ctx_requests,
        ctx_tokens=round(statistics.median(step.ctx_tokens for step in selected)),
        generation_requests=selected[0].generation_requests,
        generation_tokens=selected[0].generation_requests,
        elapsed_ms=statistics.median(step.elapsed_ms for step in selected),
    )


def decompose_tp8_32k_gap(
    *,
    real_tput: float,
    sim_tput: float,
    cost_replay_factor: float,
    useful_prompt_tokens: float,
    real_recompute_tokens: float,
    sim_recompute_tokens: float,
    real_preemptions: float,
    sim_preemptions: float,
    mixed_step_real_over_sim: float,
    decode_step_sim_over_real: float,
) -> dict[str, float | str]:
    total_gap = real_tput / sim_tput
    real_work = useful_prompt_tokens + real_recompute_tokens
    sim_work = useful_prompt_tokens + sim_recompute_tokens
    prompt_work_factor = sim_work / real_work if real_work > 0 else math.inf
    cost_residual = total_gap / cost_replay_factor if cost_replay_factor > 0 else math.inf
    preemption_ratio = sim_preemptions / real_preemptions if real_preemptions > 0 else math.inf
    verdict = (
        "decode_cost_overcharge_dominates"
        if cost_replay_factor >= 1.15 and abs(cost_residual - 1.0) <= 0.10
        else "mixed_dynamics_and_cost_gap"
    )
    preemption_role = (
        "secondary_unresolved_composition_error"
        if verdict == "decode_cost_overcharge_dominates"
        and preemption_ratio >= 2.0
        else "primary_or_unresolved"
    )
    return {
        "total_gap": total_gap,
        "cost_replay_factor": cost_replay_factor,
        "cost_replay_residual": cost_residual,
        "preemption_ratio": preemption_ratio,
        "prompt_work_factor": prompt_work_factor,
        "mixed_step_real_over_sim": mixed_step_real_over_sim,
        "decode_step_sim_over_real": decode_step_sim_over_real,
        "preemption_role": preemption_role,
        "verdict": verdict,
    }


def classify_bt_residual(
    *,
    step: IterationStep,
    sim_step_ms: float,
    audit_counts: dict[str, int],
) -> dict[str, float | int | str]:
    step_ratio = step.elapsed_ms / sim_step_ms if sim_step_ms > 0 else math.inf
    miss_count = sum(count for reason, count in audit_counts.items() if reason != "hit")
    verdict = (
        "large_mixed_step_undercharged_with_coverage_miss"
        if step_ratio >= 1.10 and miss_count > 0
        else "large_mixed_step_undercharged"
        if step_ratio >= 1.10
        else "large_mixed_step_cost_matched"
    )
    return {
        "real_over_sim_step_ratio": step_ratio,
        "coverage_miss_count": miss_count,
        "coverage_hit_count": audit_counts.get("hit", 0),
        "verdict": verdict,
    }


def classify_bt_wall_replay(
    *,
    observed_sim_wall_ratio: float,
    cost_replay_factor: float,
    modal_mixed_sim_over_real: float,
    modal_audit_counts: dict[str, int],
) -> dict[str, float | str]:
    residual = (
        observed_sim_wall_ratio / cost_replay_factor
        if cost_replay_factor > 0
        else math.inf
    )
    undercharged = modal_mixed_sim_over_real < 0.90 and cost_replay_factor < 0.95
    misses = sum(count for reason, count in modal_audit_counts.items() if reason != "hit")
    hits = modal_audit_counts.get("hit", 0)
    if undercharged and abs(residual - 1.0) <= 0.15 and misses:
        verdict = "mixed_step_undercharge_table_missing"
    elif undercharged and abs(residual - 1.0) <= 0.15 and hits:
        verdict = "mixed_step_undercharge_regime_hit_mismatch"
    else:
        verdict = "bt65536_residual_mixed_or_unresolved"
    return {
        "observed_sim_wall_ratio": observed_sim_wall_ratio,
        "cost_replay_factor": cost_replay_factor,
        "residual_after_cost_replay": residual,
        "modal_mixed_sim_over_real": modal_mixed_sim_over_real,
        "verdict": verdict,
    }


def _point(name: str):
    return next(point for point in validate.MULTI_CONFIG_DATA if point.name == name)


def _latency_calc(point) -> IterationLatencyCalculator:
    model, database, backend = validate._load_model_and_db(
        tp=point.tp,
        dp=point.dp,
        moe_tp=point.moe_tp,
        moe_ep=point.moe_ep,
    )
    return IterationLatencyCalculator(
        backend,
        model,
        database,
        overlap_factor=0.0,
        per_iteration_overhead_ms=validate.DEFAULT_EP8_PER_ITERATION_OVERHEAD_MS,
        serving_state_max_num_batched_tokens=point.max_num_batched_tokens,
    )


def charge_step(point, step: IterationStep) -> dict[str, object]:
    calc = _latency_calc(point)
    sim_ms = calc.compute(
        prefill_tokens=step.ctx_tokens,
        prefill_batch_size=step.ctx_requests,
        prefill_seq_len=point.isl,
        decode_batch_size=step.generation_requests,
        decode_avg_kv_len=point.isl + point.osl // 2 if step.generation_requests else 0,
    )
    breakdown = calc.get_last_breakdown()
    audit = calc.get_serving_state_query_audit()
    audit_counts = Counter("hit" if row.hit else row.miss_reason for row in audit)
    audit_bounds = [
        (
            row.row_kind,
            row.category,
            row.bucket_min,
            row.bucket_max,
            row.decode_batch_min,
            row.decode_batch_max,
        )
        for row in audit
    ]
    return {
        "sim_ms": sim_ms,
        "breakdown": breakdown.as_dict() if breakdown else {},
        "audit_counts": dict(audit_counts),
        "audit_bounds": audit_bounds,
    }


def wall_cost_replay(point, steps: list[IterationStep]) -> dict[str, object]:
    classes = {
        "mixed": [step for step in steps if step.is_mixed],
        "prefill": [step for step in steps if step.ctx_tokens > 0 and not step.is_mixed],
        "decode": [step for step in steps if step.is_decode],
    }
    total_real_wall = sum(step.elapsed_ms for step in steps)
    predicted_sim_wall = 0.0
    class_rows: dict[str, dict[str, object]] = {}
    for step_class, selected in classes.items():
        if not selected:
            continue
        representative = representative_step(steps, step_class)
        charge = charge_step(point, representative)
        real_wall = sum(step.elapsed_ms for step in selected)
        sim_over_real = float(charge["sim_ms"]) / representative.elapsed_ms
        predicted_sim_wall += real_wall * sim_over_real
        class_rows[step_class] = {
            "count": len(selected),
            "real_wall_ms": real_wall,
            "real_wall_share": real_wall / total_real_wall,
            "representative": representative,
            "sim_ms": float(charge["sim_ms"]),
            "sim_over_real": sim_over_real,
            "breakdown": charge["breakdown"],
            "audit_counts": charge["audit_counts"],
            "audit_bounds": charge["audit_bounds"],
        }
    return {
        "real_wall_ms": total_real_wall,
        "predicted_sim_wall_ms": predicted_sim_wall,
        "cost_replay_factor": predicted_sim_wall / total_real_wall,
        "classes": class_rows,
    }


def run_sim_profile(name: str) -> dict[str, object]:
    point = _point(name)
    model, database, backend = validate._load_model_and_db(
        tp=point.tp,
        dp=point.dp,
        moe_tp=point.moe_tp,
        moe_ep=point.moe_ep,
    )
    config = replace(
        validate._make_multi_config_cb_config(
            point,
            overlap_factor=0.0,
            ep8_per_iteration_overhead_ms=validate.DEFAULT_EP8_PER_ITERATION_OVERHEAD_MS,
        ),
        num_requests=NUM_REQUESTS,
        warmup_requests=point.batch_size,
    )
    stats: Counter[str] = Counter()
    mixed_batches: list[float] = []
    running_levels: list[float] = []
    original_preempt = CBScheduler._preempt
    original_schedule = CBScheduler.schedule

    def wrapped_preempt(self, victim, waiting, running, result, preempted_ids):
        stats["preemptions"] += 1
        stats["recompute_tokens"] += victim.isl + victim.generated_tokens
        return original_preempt(self, victim, waiting, running, result, preempted_ids)

    def wrapped_schedule(self, waiting, running):
        running_levels.append(float(len(running)))
        result = original_schedule(self, waiting, running)
        if result.prefill_reqs and result.decode_reqs:
            mixed_batches.append(float(len(result.decode_reqs)))
        return result

    CBScheduler._preempt = wrapped_preempt
    CBScheduler.schedule = wrapped_schedule
    try:
        sim = CBSimulator(backend, model, database, config)
        if point.dp > 1 and point.max_num_batched_tokens == point.isl:
            result = sim.run_multi_replica(
                isl=point.isl,
                osl=point.osl,
                concurrency=point.batch_size,
                data_parallel_size=point.dp,
                num_gpus=point.tp * point.dp,
                lockstep=True,
            )
        else:
            result = sim.run(
                isl=point.isl,
                osl=point.osl,
                concurrency=math.ceil(point.batch_size / point.dp),
                num_gpus=point.tp,
            )
    finally:
        CBScheduler._preempt = original_preempt
        CBScheduler.schedule = original_schedule

    return {
        "throughput_tok_s_gpu": result.throughput_tok_s_gpu,
        "preemptions": float(stats["preemptions"]),
        "recompute_tokens": float(stats["recompute_tokens"]),
        "mixed_batch_p50": _percentile(mixed_batches, 50),
        "running_p50": _percentile(running_levels, 50),
        "avg_decode_reqs": result.avg_decode_reqs_per_iter,
        "peak_decode_reqs": float(result.peak_decode_reqs_per_iter),
    }


def _load_ab_rows(path: Path = DEFAULT_AB_CSV) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return {row["name"]: row for row in csv.DictReader(f)}


def _add(
    rows: list[dict[str, object]],
    section: str,
    scenario: str,
    side: str,
    metric: str,
    value: object,
    *,
    target: str = "",
    status: str = "",
    source: str = "",
    note: str = "",
) -> None:
    rows.append(
        {
            "section": section,
            "scenario": scenario,
            "side": side,
            "metric": metric,
            "value": value,
            "target": target,
            "status": status,
            "source": source,
            "note": note,
        }
    )


def build_kv_audit_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    sources = {
        DP2_8K: PHASE454_ROOT / "recollect_dp2_8k2k" / DP2_8K / "serve.log",
        DP2_32K: PHASE454_ROOT / "recollect_dp2_32k3k" / DP2_32K / "serve.log",
    }
    for name, source in sources.items():
        real = parse_kv_cache_tokens(source)
        wired = validate._multi_config_kv_capacity(_point(name))
        matched = real == wired.kv_cache_tokens
        _add(
            rows,
            "kv_audit",
            name,
            "real_vs_validate",
            "kv_cache_tokens",
            real,
            target=str(wired.kv_cache_tokens),
            status="pass" if matched else "fail",
            source=_display_path(source),
            note=f"validate_source={wired.serve_log}; numeric_drift={not matched}",
        )
    return rows


def build_track_a_rows(ab: dict[str, dict[str, str]]) -> list[dict[str, object]]:
    spec = SCENARIOS[TP8_32K]
    steps = parse_iteration_steps(spec.serve_log)
    real_summary = summarize_iterations(steps)
    metrics = parse_metrics_profile(spec.metrics)
    sim = run_sim_profile(TP8_32K)
    replay = wall_cost_replay(_point(TP8_32K), steps)
    classes = replay["classes"]
    mixed = classes["mixed"]
    decode = classes["decode"]
    real_tput = float(ab[TP8_32K]["current_real_output_tok_s_gpu"])
    validate_sim_tput = float(ab[TP8_32K]["current_sim_output_tok_s_gpu"])
    decomposition = decompose_tp8_32k_gap(
        real_tput=real_tput,
        sim_tput=validate_sim_tput,
        cost_replay_factor=float(replay["cost_replay_factor"]),
        useful_prompt_tokens=metrics.prompt_tokens,
        real_recompute_tokens=metrics.recomputed_prompt_tokens,
        sim_recompute_tokens=float(sim["recompute_tokens"]),
        real_preemptions=metrics.preemptions,
        sim_preemptions=float(sim["preemptions"]),
        mixed_step_real_over_sim=1.0 / float(mixed["sim_over_real"]),
        decode_step_sim_over_real=float(decode["sim_over_real"]),
    )
    rows: list[dict[str, object]] = []
    source = _display_path(spec.serve_log)
    for metric in (
        "mixed_share",
        "mixed_ctx_p50",
        "mixed_batch_p50",
        "decode_batch_p50",
        "mixed_elapsed_p50",
        "decode_elapsed_p50",
    ):
        _add(rows, "track_a_fingerprint", TP8_32K, "real", metric, real_summary[metric], source=source)
    for metric, value in (
        ("running_p50", metrics.running_p50),
        ("running_p90", metrics.running_p90),
        ("kv_usage_p90", metrics.kv_usage_p90),
        ("preemptions", metrics.preemptions),
        ("recomputed_prompt_tokens", metrics.recomputed_prompt_tokens),
        ("successful_requests_scrape_delta", metrics.successful_requests),
    ):
        _add(rows, "track_a_fingerprint", TP8_32K, "real", metric, value, source=_display_path(spec.metrics))
    for metric in (
        "throughput_tok_s_gpu",
        "preemptions",
        "recompute_tokens",
        "mixed_batch_p50",
        "running_p50",
        "avg_decode_reqs",
        "peak_decode_reqs",
    ):
        _add(rows, "track_a_fingerprint", TP8_32K, "sim", metric, sim[metric], source="current CBSimulator N=512 diagnostic")
    for step_class, data in classes.items():
        representative = data["representative"]
        assert isinstance(representative, IterationStep)
        _add(
            rows,
            "track_a_cost_replay",
            TP8_32K,
            "real_trace",
            f"{step_class}_wall_share",
            data["real_wall_share"],
            source=source,
            note=f"count={data['count']}",
        )
        _add(
            rows,
            "track_a_cost_replay",
            TP8_32K,
            "sim_vs_real",
            f"{step_class}_representative_sim_over_real",
            data["sim_over_real"],
            source="current IterationLatencyCalculator",
            note=(
                f"ctx={representative.ctx_tokens}; ctx_reqs={representative.ctx_requests}; "
                f"decode_batch={representative.generation_requests}; real_ms={representative.elapsed_ms:.3f}; "
                f"sim_ms={float(data['sim_ms']):.3f}; breakdown={data['breakdown']}; audit={data['audit_counts']}"
            ),
        )
    decode_representative = decode["representative"]
    assert isinstance(decode_representative, IterationStep)
    decode_breakdown = decode["breakdown"]
    assert isinstance(decode_breakdown, dict)
    _add(
        rows,
        "track_a_attribution",
        TP8_32K,
        "real",
        "decode_real_total_ms",
        decode_representative.elapsed_ms,
        source=source,
    )
    _add(
        rows,
        "track_a_attribution",
        TP8_32K,
        "sim",
        "decode_generation_attention_ms",
        decode_breakdown["generation_attention_ms"],
        source="current IterationLatencyCalculator",
        note="attention term alone exceeds the real whole decode step",
    )
    for metric, value in decomposition.items():
        _add(
            rows,
            "track_a_attribution",
            TP8_32K,
            "real_vs_sim",
            metric,
            value,
            target="cost replay residual within 10%" if metric == "verdict" else "",
            status="pass" if metric == "verdict" and value == "decode_cost_overcharge_dominates" else "",
            note="three-class representative wall-share replay; preemption work factor is reported separately",
        )
    return rows


def build_track_b_rows(ab: dict[str, dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    regimes = load_serving_state_regimes(SERVING_STATE_PERF, "tp4dp2ep8")
    for name in (TP8_BT, DP2_BT):
        spec = SCENARIOS[name]
        steps = parse_iteration_steps(spec.serve_log)
        representative = representative_large_mixed_step(steps, spec.max_bt)
        charge = charge_step(_point(name), representative)
        large_verdict = classify_bt_residual(
            step=representative,
            sim_step_ms=float(charge["sim_ms"]),
            audit_counts=charge["audit_counts"],
        )
        replay = wall_cost_replay(_point(name), steps)
        modal = replay["classes"]["mixed"]
        real_tput = float(ab[name]["current_real_output_tok_s_gpu"])
        sim_tput = float(ab[name]["current_sim_output_tok_s_gpu"])
        wall_verdict = classify_bt_wall_replay(
            observed_sim_wall_ratio=real_tput / sim_tput,
            cost_replay_factor=float(replay["cost_replay_factor"]),
            modal_mixed_sim_over_real=float(modal["sim_over_real"]),
            modal_audit_counts=modal["audit_counts"],
        )
        _add(
            rows,
            "track_b_step_cost",
            name,
            "real",
            "representative_large_mixed_step_ms",
            representative.elapsed_ms,
            source=_display_path(spec.serve_log),
            note=f"ctx={representative.ctx_tokens}; ctx_reqs={representative.ctx_requests}; decode_batch={representative.generation_requests}",
        )
        _add(
            rows,
            "track_b_step_cost",
            name,
            "sim",
            "representative_large_mixed_step_ms",
            charge["sim_ms"],
            source="current IterationLatencyCalculator",
            note=f"breakdown={charge['breakdown']}",
        )
        _add(
            rows,
            "track_b_coverage",
            name,
            "sim",
            "serving_state_audit_counts",
            json.dumps(charge["audit_counts"], sort_keys=True),
            target="representative shape covered",
            status="fail" if int(large_verdict["coverage_miss_count"]) else "pass",
            source="vllm_serving_state_perf.txt query audit",
            note=f"bounds={charge['audit_bounds']}",
        )
        for step_class, data in replay["classes"].items():
            modal_step = data["representative"]
            assert isinstance(modal_step, IterationStep)
            _add(
                rows,
                "track_b_cost_replay",
                name,
                "real_trace",
                f"{step_class}_wall_share",
                data["real_wall_share"],
                source=_display_path(spec.serve_log),
                note=f"count={data['count']}",
            )
            _add(
                rows,
                "track_b_cost_replay",
                name,
                "sim_vs_real",
                f"{step_class}_representative_sim_over_real",
                data["sim_over_real"],
                source="current IterationLatencyCalculator",
                note=(
                    f"ctx={modal_step.ctx_tokens}; ctx_reqs={modal_step.ctx_requests}; "
                    f"decode_batch={modal_step.generation_requests}; audit={data['audit_counts']}"
                ),
            )
        for metric, value in large_verdict.items():
            _add(
                rows,
                "track_b_large_step",
                name,
                "real_vs_sim",
                f"large_{metric}",
                value,
            )
        for metric, value in wall_verdict.items():
            _add(
                rows,
                "track_b_attribution",
                name,
                "real_vs_sim",
                metric,
                value,
                status="fail" if metric == "verdict" else "",
            )
        if name == DP2_BT:
            _add(
                rows,
                "track_b_coverage",
                name,
                "sim",
                "forward_total_max_bt_regimes_in_shared_table",
                ",".join(str(value) for value in sorted(regimes)),
                target="query keyed by max_bt",
                status="fail",
                source=_display_path(SERVING_STATE_PERF),
                note="DP2 query currently omits max_num_batched_tokens, so these regimes share one interpolation table",
            )
        _add(
            rows,
            "scoreboard",
            name,
            "validate",
            "current_error_ratio",
            float(ab[name]["current_error_ratio"]),
            target=f"<={TARGET_RATIO}",
            status="fail",
            source=_display_path(DEFAULT_AB_CSV),
            note=f"sim={float(ab[name]['current_sim_output_tok_s_gpu']):.3f}; real={float(ab[name]['current_real_output_tok_s_gpu']):.3f}",
        )
    return rows


def build_decision_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    track_a = next(row for row in rows if row["metric"] == "verdict" and row["scenario"] == TP8_32K)
    bt_verdicts = [
        row for row in rows if row["metric"] == "verdict" and row["scenario"] in {TP8_BT, DP2_BT}
    ]
    return [
        {
            "section": "decision",
            "scenario": TP8_32K,
            "side": "proposal",
            "metric": "phase460_target",
            "value": "tp8_32k_mla_decode_cost_audit",
            "target": "red-to-green in a separate phase",
            "status": "proposed",
            "source": "Phase459 Track A",
            "note": f"root={track_a['value']}; no runtime change in Phase459",
        },
        {
            "section": "decision",
            "scenario": "bt65536",
            "side": "proposal",
            "metric": "phase461_target",
            "value": "bt65536_regime_scoped_mixed_step_rows",
            "target": "red-to-green in a separate phase",
            "status": "proposed",
            "source": "Phase459 Track B",
            "note": "; ".join(f"{row['scenario']}={row['value']}" for row in bt_verdicts),
        },
        {
            "section": "decision",
            "scenario": "MULTI_CONFIG",
            "side": "gate",
            "metric": "phase459_verdict",
            "value": "residuals_triaged_gate_unchanged",
            "target": "6/6 <=1.15 before tightening",
            "status": "fail",
            "source": _display_path(DEFAULT_AB_CSV),
            "note": "diagnostic_only=true; perf_database=false; Default AIC remains No-Go",
        },
    ]


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field, "")) for field in CSV_FIELDS})


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    def section(name: str) -> list[dict[str, object]]:
        return [row for row in rows if row["section"] == name]

    kv = section("kv_audit")
    track_a = section("track_a_attribution")
    track_a_cost = section("track_a_cost_replay")
    track_b = section("track_b_attribution")
    track_b_large = section("track_b_large_step")
    track_b_cost = section("track_b_cost_replay")
    decisions = section("decision")
    lines = [
        "# Phase459 residual triage",
        "",
        "结论：`dp2` KV 数值接线无漂移。`tp8-32k3k` 主因是 decode 单步高计费；抢占过发是并存但尚未独立分离的次级构成误差。两个 `bt65536` 点都由 mixed step 低计费主导，但 TP8 是表缺失，DP2 是跨 regime 命中旧行。Phase459 不改 runtime/PerfDB/gate。",
        "",
        "## KV 接线审计",
        "",
        "| 场景 | Phase454 真值 | validate | 结果 | 备注 |",
        "|---|---:|---:|---|---|",
    ]
    for row in kv:
        lines.append(f"| {row['scenario']} | {row['value']} | {row['target']} | {row['status']} | {row['note']} |")
    lines.extend(
        [
            "",
            "## Track A：tp8-32k3k",
            "",
            "| 指标 | 数值 | 含义 |",
            "|---|---:|---|",
        ]
    )
    for metric in (
        "total_gap",
        "cost_replay_factor",
        "cost_replay_residual",
        "decode_real_total_ms",
        "decode_generation_attention_ms",
        "preemption_ratio",
        "prompt_work_factor",
        "mixed_step_real_over_sim",
        "decode_step_sim_over_real",
        "preemption_role",
        "verdict",
    ):
        row = next(item for item in track_a if item["metric"] == metric)
        lines.append(f"| {metric} | {row['value']} | {row['note']} |")
    lines.extend(
        [
            "",
            "| 步型 | real 墙钟占比 | sim/real 代表步 |",
            "|---|---:|---:|",
        ]
    )
    track_a_cost_values = {row["metric"]: row["value"] for row in track_a_cost}
    for step_class in ("mixed", "prefill", "decode"):
        lines.append(
            f"| {step_class} | {track_a_cost_values[f'{step_class}_wall_share']} | "
            f"{track_a_cost_values[f'{step_class}_representative_sim_over_real']} |"
        )
    lines.extend(
        [
            "",
            "`tp8-32k3k` 中 decode 占 67% real 墙钟，sim 单步却高计费约 1.86x；其中 generation attention 一项约 31ms，已经高于 real decode 整步 21ms。mixed 单步低计费约 1.89x，形成明显误差抵消。三类代表步按 real 墙钟份额重放得到 1.42x，闭合 1.397x 总 gap。抢占 194 vs 42 仍是模型错误，但不是本点吞吐残差的第一修复靶。",
            "",
            "## Track B：bt65536",
            "",
            "| 场景 | observed sim wall/real | cost replay | replay 残差 | modal mixed sim/real | 结论 |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    by_scenario = {name: [row for row in track_b if row["scenario"] == name] for name in (TP8_BT, DP2_BT)}
    for name, items in by_scenario.items():
        values = {row["metric"]: row["value"] for row in items}
        lines.append(
            f"| {name} | {values['observed_sim_wall_ratio']} | {values['cost_replay_factor']} | "
            f"{values['residual_after_cost_replay']} | {values['modal_mixed_sim_over_real']} | {values['verdict']} |"
        )
    large_by_scenario = {
        name: {row["metric"]: row["value"] for row in track_b_large if row["scenario"] == name}
        for name in (TP8_BT, DP2_BT)
    }
    cost_by_scenario = {
        name: {row["metric"]: row for row in track_b_cost if row["scenario"] == name}
        for name in (TP8_BT, DP2_BT)
    }
    dp2_regime_row = next(
        row
        for row in section("track_b_coverage")
        if row["scenario"] == DP2_BT
        and row["metric"] == "forward_total_max_bt_regimes_in_shared_table"
    )
    lines.extend(
        [
            "",
            f"TP8 大 mixed step real/sim={large_by_scenario[TP8_BT]['large_real_over_sim_step_ratio']}，查询为 table missing。DP2 大 mixed step已命中且基本匹配，但占墙钟更多的 modal mixed 步命中后仍只有 real 的 {float(cost_by_scenario[DP2_BT]['mixed_representative_sim_over_real']['value']):.1%}。当前共享表混有 max_bt={dp2_regime_row['value']} 三套行，而 DP2 查询未带 max_bt 键，属于跨 regime 错误命中，不是简单的网格外 miss。",
            "",
            "Phase458 iteration 日志只能定位整步，不能拆 kernel 类别。`dp2-bt65536` 已有 Phase454 B2b 原始数据，下一相可先做 max_bt regime 键控重放；`tp8-bt65536` 没有同 regime B2b 证据，不能把 bt8000 行直接迁移。",
            "",
            "## 后续分相",
            "",
            "| 相位 | 靶点 | 状态 |",
            "|---|---|---|",
        ]
    )
    for row in decisions:
        lines.append(f"| {row['metric']} | {row['value']} | {row['status']} |")
    lines.extend(
        [
            "",
            "6/6 达标前不收紧 gate，不进入 fork 交接。`diagnostic_only=true valid_for_default=false perf_database=false`；Default AIC 维持 No-Go。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(csv_out: Path = DEFAULT_CSV, md_out: Path = DEFAULT_MD) -> list[dict[str, object]]:
    ab = _load_ab_rows()
    rows = build_kv_audit_rows()
    rows.extend(build_track_a_rows(ab))
    rows.extend(build_track_b_rows(ab))
    rows.extend(build_decision_rows(rows))
    write_csv(csv_out, rows)
    write_markdown(md_out, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    rows = run_analysis(args.csv_out, args.md_out)
    verdict = next(row["value"] for row in rows if row["metric"] == "phase459_verdict")
    print(f"wrote {args.csv_out}")
    print(f"wrote {args.md_out}")
    print(f"verdict={verdict}")


if __name__ == "__main__":
    main()
