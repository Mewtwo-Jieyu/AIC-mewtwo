#!/usr/bin/env python3
"""Phase462 Step2a-3f report-only workload-exposure sensitivity."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

PRIMITIVE_CSV = (
    REPO_ROOT / "docs/iter_gap_investigation/phase462_arrival_admission_prototype.csv"
)
CELL_TRIAGE_CSV = (
    REPO_ROOT / "docs/iter_gap_investigation/phase461_step4_cell_triage.csv"
)
SIGN_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase462_first_divergence.csv"
DEFAULT_CSV = (
    REPO_ROOT / "docs/iter_gap_investigation/phase462_exposure_sensitivity.csv"
)
DEFAULT_MD = (
    REPO_ROOT / "docs/iter_gap_investigation/phase462_exposure_sensitivity.md"
)
CSV_FIELDS = ["section", "scenario", "metric", "value", "target", "status", "note"]

NORMAL_95_Z = 1.959963984540054
REQUEST_LIMIT = 128
QUEUE_DEPTH = 2
BLOCK_SIZE = 16

EXPOSURE_BOUNDARIES = {
    "t0_source_model": "delivery_candidate",
    "oracle_target_timestamps": "experiment_only",
    "target_timestamps_in_runtime": False,
}
REPORT_BOUNDARIES = {
    "diagnostic_only": True,
    "valid_for_default": False,
    "perf_database": False,
    "default_aic": "No-Go",
}


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    scenario: str
    prompt_tokens: int
    output_tokens: int
    arrival_jsonl: Path
    overhead_gate_json: Path
    sim_only_cell_scenario: str | None


@dataclass(frozen=True)
class SteadyWindow:
    start_step: int
    end_step: int
    mode: str


@dataclass(frozen=True)
class MetricComparison:
    delta: float
    noise_band: float
    equivalent: bool


EXPERIMENTS = (
    ExperimentSpec(
        name="tp8_32k3k_short",
        scenario="K2.5-tp8ep8-32k3k",
        prompt_tokens=32_000,
        output_tokens=1_200,
        arrival_jsonl=(
            REPO_ROOT
            / "docs/iter_gap_investigation/phase462_arrival_observation/32k3k.jsonl.gz"
        ),
        overhead_gate_json=(
            REPO_ROOT
            / "docs/iter_gap_investigation/phase462_preemption_observation/overhead_gate.json"
        ),
        sim_only_cell_scenario=None,
    ),
    ExperimentSpec(
        name="tp8_bt65536",
        scenario="K2.5-tp8ep8-8k2k-bt65536",
        prompt_tokens=8_000,
        output_tokens=2_000,
        arrival_jsonl=(
            REPO_ROOT
            / "docs/iter_gap_investigation/phase462_arrival_observation/8k2k.jsonl.gz"
        ),
        overhead_gate_json=(
            REPO_ROOT
            / "docs/iter_gap_investigation/phase461_cost_recollect/"
            "tp8_bt65536_mixed/overhead_gate.json"
        ),
        sim_only_cell_scenario="K2.5-tp8ep8-8k2k-bt65536",
    ),
)


def derive_steady_window(
    *,
    first_prefill_completion_steps: dict[int, int],
    request_completion_steps: dict[int, int],
) -> SteadyWindow:
    if not first_prefill_completion_steps or not request_completion_steps:
        raise ValueError("cannot derive steady window without lifecycle boundaries")
    final_first_prefill = max(first_prefill_completion_steps.values())
    first_completion = min(request_completion_steps.values())
    if final_first_prefill < first_completion:
        start = final_first_prefill + 1
        end = first_completion
        mode = "fully_admitted_before_drain"
    else:
        start = first_completion + 1
        end = final_first_prefill + 1
        mode = "replacement_plateau_while_waiting_nonempty"
    if start >= end:
        raise ValueError(f"empty steady window: start={start}, end={end}")
    return SteadyWindow(start_step=start, end_step=end, mode=mode)


def paired_variation_noise_band(
    values_a: Iterable[float], values_b: Iterable[float]
) -> MetricComparison:
    a = [float(value) for value in values_a]
    b = [float(value) for value in values_b]
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("paired variation band requires equal samples per variant")
    differences = [left - right for left, right in zip(a, b)]
    delta = abs(statistics.mean(differences))
    noise_band = NORMAL_95_Z * statistics.stdev(differences)
    return MetricComparison(delta, noise_band, delta <= noise_band)


def phase_distribution_comparison(
    counts_a: dict[int, int], counts_b: dict[int, int]
) -> MetricComparison:
    total_a = sum(counts_a.values())
    total_b = sum(counts_b.values())
    if total_a <= 0 or total_b <= 0:
        raise ValueError("phase comparison requires decode samples")
    phases = set(counts_a) | set(counts_b)
    delta = 0.5 * sum(
        abs(counts_a.get(phase, 0) / total_a - counts_b.get(phase, 0) / total_b)
        for phase in phases
    )
    noise_band = 0.5 * NORMAL_95_Z * sum(
        math.sqrt(
            max(
                0.0,
                ((counts_a.get(phase, 0) + counts_b.get(phase, 0)) / (total_a + total_b))
                * (
                    1
                    - (counts_a.get(phase, 0) + counts_b.get(phase, 0))
                    / (total_a + total_b)
                )
                * (1 / total_a + 1 / total_b),
            )
        )
        for phase in phases
    )
    return MetricComparison(delta, noise_band, delta <= noise_band)


def weighted_share_comparison(
    samples_a: Iterable[tuple[float, bool]],
    samples_b: Iterable[tuple[float, bool]],
) -> MetricComparison:
    a = [(float(weight), bool(hit)) for weight, hit in samples_a]
    b = [(float(weight), bool(hit)) for weight, hit in samples_b]
    if not a or not b or any(weight <= 0 for weight, _ in [*a, *b]):
        raise ValueError("weighted comparison requires positive samples")

    def stats(samples: list[tuple[float, bool]]) -> tuple[float, float]:
        total = sum(weight for weight, _ in samples)
        share = sum(weight for weight, hit in samples if hit) / total
        effective_n = total * total / sum(weight * weight for weight, _ in samples)
        return share, effective_n

    share_a, effective_a = stats(a)
    share_b, effective_b = stats(b)
    pooled = (share_a * effective_a + share_b * effective_b) / (
        effective_a + effective_b
    )
    noise_band = NORMAL_95_Z * math.sqrt(
        max(0.0, pooled * (1 - pooled) * (1 / effective_a + 1 / effective_b))
    )
    delta = abs(share_a - share_b)
    return MetricComparison(delta, noise_band, delta <= noise_band)


def relative_noise_comparison(
    value_a: float, value_b: float, *, noise_fraction: float
) -> MetricComparison:
    midpoint = (value_a + value_b) / 2
    if midpoint <= 0 or noise_fraction < 0:
        raise ValueError("relative comparison requires positive values and noise")
    delta = abs(value_a - value_b) / midpoint
    return MetricComparison(delta, noise_fraction, delta <= noise_fraction)


def block_throughput_samples(
    rows: Iterable[dict[str, object]], *, block_size: int
) -> list[float]:
    items = list(rows)
    if block_size < 1:
        raise ValueError("block size must be positive")
    samples: list[float] = []
    for offset in range(0, len(items) - block_size + 1, block_size):
        block = items[offset : offset + block_size]
        wall_ms = sum(float(row["latency_ms"]) for row in block)
        if wall_ms <= 0:
            raise ValueError("throughput block has non-positive wall time")
        samples.append(
            sum(int(row["decode_reqs"]) for row in block) * 1000 / wall_ms
        )
    if len(samples) < 2:
        raise ValueError("steady window has fewer than two source-period blocks")
    return samples


def scope_verdict(experiment_equivalence: dict[str, bool]) -> dict[str, object]:
    if experiment_equivalence and all(experiment_equivalence.values()):
        return {
            "step2c": "candidate",
            "scope": "engine_loop_with_t0_workload_exposure",
            "ramp_boundary": "first_schedule_trajectory_not_modeled",
            "runtime_change_allowed": False,
        }
    return {
        "step2c": "locked",
        "scope": "one_client_source_audit_or_narrow_engine_loop_scope",
        "ramp_boundary": "steady_state_is_exposure_sensitive",
        "runtime_change_allowed": False,
    }


def _load_sim_only_cells(path: Path, scenario: str) -> set[tuple[int, int]]:
    with path.open(newline="", encoding="utf-8") as source:
        cells = {
            (int(row["bucket_tokens"]), int(row["decode_batch"]))
            for row in csv.DictReader(source)
            if row["section"] == "cell"
            and row["scenario"] == scenario
            and row["classification"] == "sim_only_for_phase458_reference"
        }
    if not cells:
        raise ValueError(f"no sim-only cells for {scenario}")
    return cells


def _load_noise_fraction(path: Path) -> float:
    data = json.loads(path.read_text(encoding="utf-8"))
    pct = data.get("absolute_delta_pct", data.get("overhead_pct"))
    if pct is None or not data.get("passed"):
        raise ValueError(f"invalid observed run-pair envelope: {path}")
    return float(pct) / 100


def _request_samples(
    events: Iterable[dict[str, object]], *, request_count: int, field: str
) -> list[float]:
    totals: Counter[int] = Counter()
    for event in events:
        request_id = int(event["victim_req_id"])
        totals[request_id] += float(event[field]) if field in event else 1.0
    return [float(totals[request_id]) for request_id in range(request_count)]


def _summarize_variant(
    result: dict[str, object], *, sim_only_cells: set[tuple[int, int]]
) -> dict[str, object]:
    window = derive_steady_window(
        first_prefill_completion_steps=result["first_prefill_completion_steps"],
        request_completion_steps=result["request_completion_steps"],
    )
    steady = [
        row
        for row in result["trace"]
        if window.start_step <= int(row["step"]) < window.end_step
    ]
    if not steady:
        raise ValueError("steady window contains no scheduled steps")
    phase_counts: Counter[int] = Counter()
    for row in steady:
        phase_counts.update(
            {int(phase): int(count) for phase, count in row["decode_phase_counts"].items()}
        )
    steady_wall_ms = sum(float(row["latency_ms"]) for row in steady)
    steady_decode_tokens = sum(int(row["decode_reqs"]) for row in steady)
    giant_samples = [
        (
            float(row["latency_ms"]),
            (int(row["prefill_tokens"]), int(row["decode_reqs"])) in sim_only_cells,
        )
        for row in steady
    ]
    events = list(result["preemption_events"])
    request_count = int(result["request_count"])
    giant_wall = sum(weight for weight, hit in giant_samples if hit)
    return {
        "window": window,
        "steady_steps": len(steady),
        "steady_wall_ms": steady_wall_ms,
        "phase_counts": dict(phase_counts),
        "giant_samples": giant_samples,
        "giant_wall_share": giant_wall / steady_wall_ms,
        "steady_throughput_proxy": steady_decode_tokens * 1000 / steady_wall_ms,
        "steady_throughput_samples": block_throughput_samples(
            steady, block_size=BLOCK_SIZE
        ),
        "full_throughput_proxy": request_count * int(result["prototype_osl"]) * 1000 / float(result["wall_ms"]),
        "preemption_per_request": _request_samples(
            events, request_count=request_count, field="preemption_count"
        ),
        "recompute_tokens_per_request": _request_samples(
            events, request_count=request_count, field="recompute_tokens"
        ),
        "preemption": result["preemption"],
        "recompute_tokens": sum(int(event["recompute_tokens"]) for event in events),
        "self_preemption_steps": [
            int(event["local_step"])
            for event in events
            if event["trigger_req_id"] == event["victim_req_id"]
        ],
    }


def _compare_variants(
    a: dict[str, object],
    b: dict[str, object],
    *,
    include_giant_cells: bool,
) -> dict[str, MetricComparison]:
    comparisons = {
        "steady_phase_tv": phase_distribution_comparison(
            a["phase_counts"], b["phase_counts"]
        ),
        "full_preemptions_per_request": paired_variation_noise_band(
            a["preemption_per_request"], b["preemption_per_request"]
        ),
        "full_recompute_tokens_per_request": paired_variation_noise_band(
            a["recompute_tokens_per_request"], b["recompute_tokens_per_request"]
        ),
        "steady_throughput_proxy": paired_variation_noise_band(
            a["steady_throughput_samples"], b["steady_throughput_samples"]
        ),
    }
    if include_giant_cells:
        comparisons["steady_sim_only_giant_wall_share"] = weighted_share_comparison(
            a["giant_samples"], b["giant_samples"]
        )
    return comparisons


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


def _sign_predictions(enabled: bool) -> list[dict[str, object]]:
    with SIGN_CSV.open(newline="", encoding="utf-8") as source:
        source_rows = [
            row for row in csv.DictReader(source) if row["section"] == "sign_prediction"
        ]
    return [
        _row(
            "six_point_sign_prediction",
            row["scenario"],
            "engine_loop_reduces_preemption_sim_throughput",
            row["value"] if enabled else "not_identifiable",
            target=row["target"],
            status="pending_dynamic_ab" if enabled else "blocked",
            note=(row["note"] if enabled else "exposure sensitivity gate failed"),
        )
        for row in source_rows
    ]


def build_report() -> tuple[list[dict[str, object]], dict[str, object]]:
    import scripts.analyze_phase462_engine_loop_state_machine as engine
    import scripts.analyze_phase462_predictive_engine_loop as predictive

    primitive = predictive.load_approved_tokenizer_primitive(PRIMITIVE_CSV)
    tokenizer = predictive.TokenizerRuntimeSpec(32, 2.0, 1)
    experiment_results: dict[str, dict[str, object]] = {}
    rows = [
        _row(
            "preregistration",
            "all",
            "steady_window",
            "interval between first completion and final first-prefill completion",
            target="state order selects full-admission or replacement plateau; no fixed ramp",
            status="locked_before_run",
            note=(
                "initial all-active-only definition failed closed before A/B: "
                "KV-constrained queue reverses the boundaries"
            ),
        ),
        _row(
            "preregistration",
            "all",
            "metric_noise",
            "95% paired-unit variation bands; throughput uses source 16-token phase blocks",
            target="A/B delta <= metric noise band",
            status="locked_before_run",
            note="same-scenario on/off envelope is retained as full-run ramp diagnostic only",
        ),
    ]

    for spec in EXPERIMENTS:
        protocol = predictive.BenchProtocol(
            num_requests=REQUEST_LIMIT,
            concurrency=REQUEST_LIMIT,
            prompt_tokens=spec.prompt_tokens,
        )
        prediction = predictive.build_predictive_tokenizer_arrivals(
            protocol=protocol,
            tokenizer=tokenizer,
            primitive=primitive,
        )
        judge_rows = predictive._load_jsonl(spec.arrival_jsonl)
        oracle_arrivals = engine.build_actual_tokenizer_inputs(
            judge_rows, request_limit=REQUEST_LIMIT
        )
        sim_only_cells = (
            _load_sim_only_cells(CELL_TRIAGE_CSV, spec.sim_only_cell_scenario)
            if spec.sim_only_cell_scenario
            else set()
        )
        common_args = {
            "measured_iteration_latencies_ms": None,
            "queue_depth": QUEUE_DEPTH,
            "scenario": spec.scenario,
            "request_limit": REQUEST_LIMIT,
            "prototype_osl": spec.output_tokens,
        }
        result_a = engine._run_cb_queue_prototype(
            arrivals=list(prediction.arrivals), **common_args
        )
        result_b = engine._run_cb_queue_prototype(
            arrivals=oracle_arrivals, **common_args
        )
        summary_a = _summarize_variant(result_a, sim_only_cells=sim_only_cells)
        summary_b = _summarize_variant(result_b, sim_only_cells=sim_only_cells)
        throughput_noise = _load_noise_fraction(spec.overhead_gate_json)
        comparisons = _compare_variants(
            summary_a,
            summary_b,
            include_giant_cells=bool(spec.sim_only_cell_scenario),
        )
        run_pair_check = relative_noise_comparison(
            float(summary_a["full_throughput_proxy"]),
            float(summary_b["full_throughput_proxy"]),
            noise_fraction=throughput_noise,
        )
        equivalent = all(item.equivalent for item in comparisons.values())
        source_batches = [len(batch.request_ids) for batch in prediction.batches]
        oracle_batches = predictive._initial_target_batch_sizes(
            judge_rows, request_limit=REQUEST_LIMIT
        )
        rows.extend(
            [
                _row(
                    "exposure",
                    spec.name,
                    "t0_source_batches",
                    json.dumps(source_batches),
                    target="bench t=0 + runtime tokenizer primitive",
                    status="delivery_candidate",
                ),
                _row(
                    "exposure",
                    spec.name,
                    "oracle_batches",
                    json.dumps(oracle_batches),
                    target="experiment control only",
                    status="judge_only",
                ),
                _row(
                    "noise",
                    spec.name,
                    "throughput_run_pair_fraction",
                    throughput_noise,
                    target="same-scenario on/off observed delta",
                    status="measured",
                    note=str(spec.overhead_gate_json.relative_to(REPO_ROOT)),
                ),
            ]
        )
        for mode, summary in (("t0", summary_a), ("oracle", summary_b)):
            signature = summary["preemption"]
            for metric, value in (
                ("steady_start_step", summary["window"].start_step),
                ("steady_end_step", summary["window"].end_step),
                ("steady_window_mode", summary["window"].mode),
                ("steady_steps", summary["steady_steps"]),
                ("steady_throughput_proxy", summary["steady_throughput_proxy"]),
                ("full_throughput_proxy", summary["full_throughput_proxy"]),
                ("giant_wall_share", summary["giant_wall_share"]),
                ("recompute_tokens", summary["recompute_tokens"]),
                ("preemptions", signature["preemptions"]),
                ("self_preemptions", signature["self_preemptions"]),
                ("repeat_victim_events", signature["repeat_victim_events"]),
            ):
                rows.append(_row("variant_metric", f"{spec.name}:{mode}", metric, value))
        for metric, comparison in comparisons.items():
            rows.append(
                _row(
                    "steady_comparison",
                    spec.name,
                    metric,
                    comparison.delta,
                    target=f"<={comparison.noise_band}",
                    status="equivalent" if comparison.equivalent else "sensitive",
                )
            )
        rows.append(
            _row(
                "ramp_diagnostic",
                spec.name,
                "full_run_throughput_vs_onoff_envelope",
                run_pair_check.delta,
                target=f"<={run_pair_check.noise_band}",
                status="inside" if run_pair_check.equivalent else "outside",
                note="diagnostic only; full run includes exposure ramp and does not gate steady equivalence",
            )
        )
        self_a = int(summary_a["preemption"]["self_preemptions"])
        self_b = int(summary_b["preemption"]["self_preemptions"])
        def regions(summary: dict[str, object]) -> dict[str, int]:
            window = summary["window"]
            result = Counter()
            for step in summary["self_preemption_steps"]:
                if step < window.start_step:
                    result["ramp"] += 1
                elif step < window.end_step:
                    result["steady"] += 1
                else:
                    result["drain"] += 1
            return dict(result)
        regions_a = regions(summary_a)
        regions_b = regions(summary_b)
        rows.append(
            _row(
                "self_preemption_attribution",
                spec.name,
                "t0_vs_oracle",
                f"{self_a}/{self_b}",
                target="0/0",
                status=(
                    "exposure_sensitive" if self_a > self_b else "engine_loop_residual"
                    if self_a == self_b and self_a > 0
                    else "closed"
                ),
                note=f"t0_regions={regions_a}; oracle_regions={regions_b}",
            )
        )
        experiment_results[spec.name] = {
            "equivalent": equivalent,
            "t0": summary_a,
            "oracle": summary_b,
            "comparisons": comparisons,
            "source_batches": source_batches,
            "oracle_batches": oracle_batches,
            "throughput_noise": throughput_noise,
            "run_pair_check": run_pair_check,
            "self_regions_t0": regions_a,
            "self_regions_oracle": regions_b,
        }

    scope = scope_verdict(
        {name: bool(result["equivalent"]) for name, result in experiment_results.items()}
    )
    rows.append(
        _row(
            "decision",
            "all",
            "step2c_scope",
            scope["scope"],
            target="engine_loop_with_t0_workload_exposure",
            status="candidate" if scope["step2c"] == "candidate" else "blocked",
            note=scope["ramp_boundary"],
        )
    )
    sign_predictions = _sign_predictions(scope["step2c"] == "candidate")
    rows.extend(sign_predictions)
    for key, value in EXPOSURE_BOUNDARIES.items():
        rows.append(_row("exposure_boundary", "all", key, value, status="pass"))
    for key, value in REPORT_BOUNDARIES.items():
        rows.append(
            _row(
                "report_boundary",
                "all",
                key,
                value,
                status="blocked" if key == "default_aic" else "pass",
            )
        )
    return rows, {
        "experiments": experiment_results,
        "scope": scope,
        "sign_predictions": sign_predictions,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, details: dict[str, object]) -> None:
    scope = details["scope"]
    equivalent = scope["step2c"] == "candidate"
    lines = [
        "# Phase462 Step 2a-3f 暴露时序止损评估",
        "",
        (
            "稳态对暴露爬坡不敏感；2c 可进入‘引擎环 + t=0 workload 暴露’方案评审。"
            if equivalent
            else (
                "稳态仍受暴露爬坡影响；2c 继续锁定，只允许一次 bench 客户端"
                "源码审计或收窄范围。"
            )
        ),
        "",
        "| 场景 | 相位 TV | 抢占率 | 重算量 | 巨 bucket | 稳态吞吐 | 判决 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, result in details["experiments"].items():
        comparisons = result["comparisons"]
        show = lambda key: (
            f"{comparisons[key].delta:.4g} / {comparisons[key].noise_band:.4g}"
            if key in comparisons
            else "不适用"
        )
        lines.append(
            f"| {name} | {show('steady_phase_tv')} | "
            f"{show('full_preemptions_per_request')} | "
            f"{show('full_recompute_tokens_per_request')} | "
            f"{show('steady_sim_only_giant_wall_share')} | "
            f"{show('steady_throughput_proxy')} | "
            f"{'A≈B' if result['equivalent'] else 'A≉B'} |"
        )
    lines.extend(
        [
            "",
            (
                "表中每格为 `A/B 差值 / 预注册噪声带`。稳态窗口取第一次请求完成与"
                "最后一次首次 prefill 完成之间的状态区间：容量充足时是全 admission "
                "平台，KV 受限时是 waiting 非空的替换平台；没有固定毫秒或固定步数。"
                "原先只接受前一种顺序，首次执行在产生 A/B 结果前 fail closed，本报告"
                "保留该纠错。相位使用有限样本带；抢占和重算使用同 request 配对差的 "
                "95% 单元波动带；巨 bucket 使用墙钟加权有效样本数；吞吐按源码 "
                "16-token KV 周期配对分块后计算 95% 块间波动。这里比较的是实际波动"
                "尺度，不是随样本数缩小的均值置信区间。采集器 on/off 差只作包含 "
                "ramp 的全程吞吐诊断，不冒充重复性噪声。"
            ),
            (
                "巨 bucket 的 `0/0` 只证明两种暴露输入之间不敏感；本原型没有命中 "
                "Phase458 的目标 cell，不能替代 N=512 动态验收。bt65536 的全程吞吐"
                "差超出 on/off 包络，但该读数包含 ramp，只作为已声明边界记录。"
            ),
            "",
            "## 两个暴露变体",
            "",
            "| 场景 | A: t=0 workload + 源码 tokenizer | B: oracle 暴露 | 自抢占 A/B |",
            "|---|---|---|---:|",
        ]
    )
    for name, result in details["experiments"].items():
        self_a = result["t0"]["preemption"]["self_preemptions"]
        self_b = result["oracle"]["preemption"]["self_preemptions"]
        lines.append(
            (
                f"| {name} | `{result['source_batches']}` | `{result['oracle_batches']}` | "
                f"{self_a}/{self_b}（{result['self_regions_t0']} / "
                f"{result['self_regions_oracle']}） |"
            )
        )
    lines.extend(
        [
            "",
            (
                "Oracle 时间戳只用于本敏感度实验，不得进入 runtime、PerfDB 或交付路径。"
                "暴露过程正式归入 workload 规格；本步不再继续拟合 API 客户端到 "
                "tokenizer queue 的过程。"
            ),
            "",
            "## 六点符号预测",
            "",
            "| 场景 | 预测方向 | 状态 |",
            "|---|---|---|",
        ]
    )
    for prediction in details["sign_predictions"]:
        lines.append(
            f"| {prediction['scenario']} | `{prediction['value']}` | "
            f"`{prediction['status']}` |"
        )
    lines.extend(
        [
            "",
            (
                "方向只来自已登记的‘引擎环减少抢占、sim 吞吐上移’假设，"
                "不是验收结果；必须由 2c 动态 `--ab` 逐点证伪。"
            ),
            "",
            "## 2c 范围",
            "",
            f"| 项目 | 结论 |",
            "|---|---|",
            f"| Step 2c | `{scope['step2c']}` |",
            f"| 范围 | `{scope['scope']}` |",
            f"| ramp 边界 | `{scope['ramp_boundary']}` |",
            "| 自抢占验收 | 2c 仍须按稳态分区判到 0；本步只裁决 workload 暴露敏感度 |",
            "| runtime / PerfDB / validate / gate | 本步均未改 |",
            "| Default AIC | No-Go |",
            "",
            "边界：`diagnostic_only=true`、`valid_for_default=false`、`perf_database=false`。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    rows, details = build_report()
    write_csv(args.csv, rows)
    write_markdown(args.md, details)
    print(
        "phase462_exposure_sensitivity "
        f"step2c={details['scope']['step2c']} "
        + " ".join(
            f"{name}={'equivalent' if result['equivalent'] else 'sensitive'}"
            for name, result in details["experiments"].items()
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
