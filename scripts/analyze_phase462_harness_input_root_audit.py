#!/usr/bin/env python3
"""Audit Phase462 DP Step 3d harness inputs without changing runtime behavior."""

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

import scripts.analyze_phase451g_wave_groundtruth as wave_groundtruth  # noqa: E402
import scripts.analyze_phase462_tp8_mixed_first_divergence as tp8_divergence  # noqa: E402
import scripts.validate_cb_simulator as validate  # noqa: E402
from aiconfigurator.sdk.backends.cb_simulator import CBSimulator  # noqa: E402
from aiconfigurator.sdk.backends.cb_simulator.scheduler import CBScheduler  # noqa: E402


DP2_BT = "K2.5-tp4ep8dp2-8k2k-bt65536"
TP8_BT = "K2.5-tp8ep8-8k2k-bt65536"
DEFAULT_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase462_dp_step3d"
DEFAULT_CSV = DEFAULT_ROOT / "phase462_harness_input_root_audit.csv"
DEFAULT_REPORT = DEFAULT_ROOT / "phase462_harness_input_root_audit.md"
DEFAULT_SUMMARY = REPO_ROOT / "docs/iter_gap_investigation/phase462_dp_step3d.md"
DP_RECORDS = REPO_ROOT / (
    "docs/iter_gap_investigation/phase462_dp_route_observation/capture/"
    f"{DP2_BT}-capture/bench_records.jsonl"
)
DP_BENCH = DP_RECORDS.with_name("bench_result.json")
TP8_RECORDS = REPO_ROOT / (
    "docs/iter_gap_investigation/phase461_cost_recollect/"
    "tp8_bt65536_mixed/overhead_on/"
    f"{TP8_BT}/bench_records.jsonl"
)
TP8_BENCH = TP8_RECORDS.with_name("bench_result.json")


@dataclass(frozen=True)
class InitialWaveShape:
    full_requests: int
    partial_tokens: int
    context_requests: int


@dataclass(frozen=True)
class GapRow:
    segment: str
    parent: str
    child: str
    delta_error: float


@dataclass(frozen=True)
class GapDecomposition:
    raw_gap: float
    rows: tuple[GapRow, ...]
    horizon_fraction: float
    topology_fraction: float
    matched_residual: float
    verdict: str


@dataclass(frozen=True)
class DPControl:
    name: str
    topology: str
    num_requests: int
    warmup_requests: int
    sim_tok_s_gpu: float
    error_ratio: float
    total_iterations: int
    steady_iterations: int
    steady_time_ms: float
    serving_hits: int
    serving_queries: int


@dataclass(frozen=True)
class TP8Control:
    name: str
    sim_tok_s_gpu: float
    error_ratio: float
    total_iterations: int
    steady_iterations: int
    serving_hits: int
    serving_queries: int
    first_window: tuple[tuple[int, int, int], ...]


def initial_wave_shape(*, max_num_batched_tokens: int, isl: int) -> InitialWaveShape:
    if max_num_batched_tokens <= 0 or isl <= 0:
        raise ValueError("max_num_batched_tokens and isl must be positive")
    full, partial = divmod(max_num_batched_tokens, isl)
    return InitialWaveShape(full, partial, full + int(partial > 0))


def decompose_dp_gap(
    *,
    official_error: float,
    short_single_error: float,
    o0_error: float,
    matched_multi_error: float,
) -> GapDecomposition:
    raw_gap = o0_error - official_error
    rows = (
        GapRow(
            "per_replica_observation_horizon",
            "single_N384_W128",
            "single_N192_W64",
            short_single_error - official_error,
        ),
        GapRow(
            "multi_replica_execution_and_assembly",
            "single_N192_W64",
            "multi_N384_W128",
            o0_error - short_single_error,
        ),
    )
    if not math.isclose(
        sum(row.delta_error for row in rows), raw_gap, rel_tol=0.0, abs_tol=1e-12
    ):
        raise AssertionError("official_to_o0_decomposition_does_not_close")
    denominator = abs(raw_gap)
    horizon_fraction = abs(rows[0].delta_error) / denominator if denominator else 0.0
    topology_fraction = abs(rows[1].delta_error) / denominator if denominator else 0.0
    return GapDecomposition(
        raw_gap=raw_gap,
        rows=rows,
        horizon_fraction=horizon_fraction,
        topology_fraction=topology_fraction,
        matched_residual=matched_multi_error - official_error,
        verdict="observation_horizon_and_topology_not_arrival",
    )


def assess_input_contract(
    *,
    explicit_start_timestamp: bool,
    reconstructed_initial_zero_count: int,
    expected_concurrency: int,
    real_first_context_requests: int,
    controlled_error: float,
    acceptance_limit: float,
) -> str:
    if (
        not explicit_start_timestamp
        and reconstructed_initial_zero_count == expected_concurrency
        and real_first_context_requests != reconstructed_initial_zero_count
    ):
        return "engine_visibility_boundary_not_observed_keep_official_protocol"
    if controlled_error > acceptance_limit:
        return "input_control_does_not_close_error_keep_official_protocol"
    return "ready_for_separate_protocol_review"


def first_window_exact_fraction(
    reference: Sequence[tuple[int, int, int]],
    candidate: Sequence[tuple[int, int, int]],
) -> float:
    if not reference:
        return 0.0
    return sum(left == right for left, right in zip(reference, candidate)) / len(
        reference
    )


def _point(name: str):
    return next(item for item in validate.MULTI_CONFIG_DATA if item.name == name)


def _config(point):
    return validate._make_multi_config_cb_config(
        point,
        overlap_factor=0.0,
        ep8_per_iteration_overhead_ms=validate.DEFAULT_EP8_PER_ITERATION_OVERHEAD_MS,
    )


def _serving_counts(sim: CBSimulator) -> tuple[int, int]:
    audit = sim.get_last_serving_state_query_audit()
    return sum(row.miss_reason == "hit" for row in audit), len(audit)


def _run_dp_control(
    *,
    name: str,
    topology: str,
    point,
    model,
    database,
    backend,
    config,
) -> DPControl:
    sim = CBSimulator(backend, model, database, config)
    if topology == "single":
        result = sim.run(
            isl=point.isl,
            osl=point.osl,
            concurrency=math.ceil(point.batch_size / point.dp),
            num_gpus=point.tp,
        )
    elif topology == "multi":
        result = sim.run_multi_replica(
            isl=point.isl,
            osl=point.osl,
            concurrency=point.batch_size,
            data_parallel_size=point.dp,
            num_gpus=point.tp * point.dp,
            lockstep=False,
        )
    else:
        raise ValueError(f"unknown topology:{topology}")
    hits, queries = _serving_counts(sim)
    return DPControl(
        name=name,
        topology=topology,
        num_requests=config.num_requests,
        warmup_requests=config.warmup_requests,
        sim_tok_s_gpu=result.throughput_tok_s_gpu,
        error_ratio=validate._abs_error(
            result.throughput_tok_s_gpu, point.real_output_tok_s_gpu
        ),
        total_iterations=result.total_iterations,
        steady_iterations=result.steady_state_iterations,
        steady_time_ms=result.steady_state_time_ms,
        serving_hits=hits,
        serving_queries=queries,
    )


def run_dp_controls() -> list[DPControl]:
    point = _point(DP2_BT)
    model, database, backend = validate._load_model_and_db(
        tp=point.tp, dp=point.dp, moe_tp=point.moe_tp, moe_ep=point.moe_ep
    )
    base = _config(point)
    specs = (
        ("single_N384_W128", "single", base),
        (
            "single_N192_W64",
            "single",
            replace(
                base,
                num_requests=base.num_requests // point.dp,
                warmup_requests=base.warmup_requests // point.dp,
            ),
        ),
        ("multi_N384_W128_O0", "multi", base),
        (
            "multi_N768_W128",
            "multi",
            replace(base, num_requests=base.num_requests * point.dp),
        ),
        (
            "multi_N768_W256_matched",
            "multi",
            replace(
                base,
                num_requests=base.num_requests * point.dp,
                warmup_requests=base.warmup_requests * point.dp,
            ),
        ),
    )
    return [
        _run_dp_control(
            name=name,
            topology=topology,
            point=point,
            model=model,
            database=database,
            backend=backend,
            config=config,
        )
        for name, topology, config in specs
    ]


def _tp8_visibility_rule(name: str, call: int) -> int | None:
    if name == "baseline":
        return None
    if name == "first_context_only" and call == 1:
        return 1
    if name == "real_initial_ramp":
        if call == 1:
            return 1
        if call == 2:
            return 0
    return None


def _run_tp8_control(
    *, name: str, point, model, database, backend, config
) -> TP8Control:
    original_schedule = CBScheduler.schedule
    calls = 0
    first_window: list[tuple[int, int, int]] = []

    def wrapped_schedule(self, waiting, running):
        nonlocal calls
        calls += 1
        limit = _tp8_visibility_rule(name, calls)
        hidden = [] if limit is None else list(waiting[limit:])
        if limit is not None:
            del waiting[limit:]
        result = original_schedule(self, waiting, running)
        if hidden:
            waiting.extend(hidden)
        if len(first_window) < 12:
            first_window.append(
                (
                    len(result.prefill_reqs),
                    result.total_prefill_tokens,
                    len(result.decode_reqs),
                )
            )
        return result

    with patch.object(CBScheduler, "schedule", wrapped_schedule):
        sim = CBSimulator(backend, model, database, config)
        result = sim.run(
            isl=point.isl,
            osl=point.osl,
            concurrency=point.batch_size,
            num_gpus=point.tp,
        )
    hits, queries = _serving_counts(sim)
    return TP8Control(
        name=name,
        sim_tok_s_gpu=result.throughput_tok_s_gpu,
        error_ratio=validate._abs_error(
            result.throughput_tok_s_gpu, point.real_output_tok_s_gpu
        ),
        total_iterations=result.total_iterations,
        steady_iterations=result.steady_state_iterations,
        serving_hits=hits,
        serving_queries=queries,
        first_window=tuple(first_window),
    )


def run_tp8_controls() -> tuple[list[TP8Control], list[tuple[int, int, int]]]:
    point = _point(TP8_BT)
    model, database, backend = validate._load_model_and_db(
        tp=point.tp, dp=point.dp, moe_tp=point.moe_tp, moe_ep=point.moe_ep
    )
    config = _config(point)
    controls = [
        _run_tp8_control(
            name=name,
            point=point,
            model=model,
            database=database,
            backend=backend,
            config=config,
        )
        for name in ("baseline", "first_context_only", "real_initial_ramp")
    ]
    real = [
        (step.context_requests, step.context_tokens, step.generation_requests)
        for step in tp8_divergence.load_real_steps()[:12]
    ]
    return controls, real


def _bench_contract(records_path: Path, bench_path: Path) -> dict[str, object]:
    records = wave_groundtruth.read_jsonl(records_path)
    bench = json.loads(bench_path.read_text(encoding="utf-8"))
    concurrency = int(bench["max_concurrency"])
    timeline = wave_groundtruth.reconstruct_closed_loop_timeline(
        records, max_concurrency=concurrency
    )
    explicit_keys = {
        "start_ms",
        "start_time_ms",
        "arrival_time_ms",
        "started_at",
        "timestamp",
    }
    explicit = any(explicit_keys.intersection(row) for row in records)
    wall_s = max(float(row["finish_ms"]) for row in timeline) / 1000.0
    expected_wall_s = float(bench["wall_s"])
    return {
        "records": len(records),
        "concurrency": concurrency,
        "explicit_start_timestamp": explicit,
        "initial_zero_count": sum(float(row["start_ms"]) == 0.0 for row in timeline),
        "reconstructed_wall_error": abs(wall_s - expected_wall_s) / expected_wall_s,
        "request_128_start_ms": float(timeline[concurrency]["start_ms"]),
    }


def _fmt(value: object) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.9f}"
    return str(value)


def _write_csv(
    path: Path,
    *,
    dp_controls: Sequence[DPControl],
    gap: GapDecomposition,
    tp8_controls: Sequence[TP8Control],
    real_window: Sequence[tuple[int, int, int]],
    bench_contracts: dict[str, dict[str, object]],
    contract_verdict: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    def add(section: str, name: str, metric: str, value: object, note: str = "") -> None:
        rows.append(
            {
                "section": section,
                "name": name,
                "metric": metric,
                "value": _fmt(value),
                "note": note,
            }
        )

    for item in dp_controls:
        for metric in (
            "topology",
            "num_requests",
            "warmup_requests",
            "sim_tok_s_gpu",
            "error_ratio",
            "total_iterations",
            "steady_iterations",
            "steady_time_ms",
            "serving_hits",
            "serving_queries",
        ):
            add("dp_control", item.name, metric, getattr(item, metric))
    for item in gap.rows:
        add("dp_attribution", item.segment, "delta_error", item.delta_error, f"{item.parent}->{item.child}")
    add("dp_attribution", "official_to_o0", "raw_gap", gap.raw_gap)
    add("dp_attribution", "official_to_o0", "horizon_fraction", gap.horizon_fraction)
    add("dp_attribution", "official_to_o0", "topology_fraction", gap.topology_fraction)
    add("dp_attribution", "matched_multi", "residual_vs_official", gap.matched_residual)
    add("dp_attribution", "verdict", "value", gap.verdict)

    reference = list(real_window[:8])
    for item in tp8_controls:
        add("tp8_control", item.name, "sim_tok_s_gpu", item.sim_tok_s_gpu)
        add("tp8_control", item.name, "error_ratio", item.error_ratio)
        add(
            "tp8_control",
            item.name,
            "first8_exact_fraction",
            first_window_exact_fraction(reference, item.first_window[:8]),
        )
        add("tp8_control", item.name, "serving_hits", item.serving_hits)
        add("tp8_control", item.name, "serving_queries", item.serving_queries)
        for index, macro in enumerate(item.first_window[:8]):
            add("tp8_window", item.name, f"step_{index}", "/".join(map(str, macro)))
    for index, macro in enumerate(reference):
        add("tp8_window", "real", f"step_{index}", "/".join(map(str, macro)))

    for name, contract in bench_contracts.items():
        for metric, value in contract.items():
            add("bench_contract", name, metric, value)
    add("input_contract", "decision", "value", contract_verdict)
    add("boundary", "runtime", "changed", False)
    add("boundary", "perfdb", "changed", False)
    add("boundary", "gate", "changed", False)
    add("boundary", "gpu", "used", False)
    add("boundary", "default_aic", "status", "No-Go")

    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=["section", "name", "metric", "value", "note"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_reports(
    report: Path,
    summary: Path,
    *,
    dp_controls: Sequence[DPControl],
    gap: GapDecomposition,
    tp8_controls: Sequence[TP8Control],
    real_window: Sequence[tuple[int, int, int]],
    bench_contracts: dict[str, dict[str, object]],
    contract_verdict: str,
) -> None:
    dp = {item.name: item for item in dp_controls}
    tp8 = {item.name: item for item in tp8_controls}
    wave = initial_wave_shape(max_num_batched_tokens=65536, isl=8000)
    first_context_gap = abs(real_window[2][2] - tp8["first_context_only"].first_window[1][2])
    baseline_gap = abs(real_window[2][2] - tp8["baseline"].first_window[1][2])
    reverse_recovery = (
        dp["multi_N768_W128"].error_ratio - dp["multi_N384_W128_O0"].error_ratio
    ) / abs(gap.raw_gap)
    detailed = f"""# Phase462 DP Step 3d harness input root audit

结论：预注册的“到达合成主导 0.11”被证伪。官方与 O0 都是 `t=0` 合成闭环；原始差值由每副本观测窗和执行拓扑共同产生。TP8 首拍 9-context 确由 `65536 = {wave.full_requests} x 8000 + {wave.partial_tokens}` 机械生成；把首拍可见输入改为 1 后，decode cell 差从 {baseline_gap} 降到 {first_context_gap}，但误差只从 {tp8['baseline'].error_ratio:.6f} 到 {tp8['first_context_only'].error_ratio:.6f}，仍未过 1.15。`Default AIC=No-Go`。

## DP 0.11 控制矩阵

| 控制 | 拓扑 | N/W | tok/s/GPU | error | serving-state hit/query |
|---|---|---:|---:|---:|---:|
"""
    lines = [detailed.rstrip()]
    for item in dp_controls:
        lines.append(
            f"| {item.name} | {item.topology} | {item.num_requests}/{item.warmup_requests} | "
            f"{item.sim_tok_s_gpu:.6f} | {item.error_ratio:.6f} | "
            f"{item.serving_hits}/{item.serving_queries} |"
        )
    lines.extend(
        [
            "",
            "| official -> O0 固定切换路径 | delta error | 路径内占比 |",
            "|---|---:|---:|",
            f"| 每副本观测窗 384/128 -> 192/64 | {gap.rows[0].delta_error:+.6f} | {gap.horizon_fraction:.2%} |",
            f"| 单副本 -> 双副本全局组装 | {gap.rows[1].delta_error:+.6f} | {gap.topology_fraction:.2%} |",
            f"| 合计 | {gap.raw_gap:+.6f} | 100.00% |",
            "",
            f"反向核验：O0 只把 N 从 384 归一到 768，已追回原始差值的 {reverse_recovery:.2%}；N/W 都按双副本归一后 error={dp['multi_N768_W256_matched'].error_ratio:.6f}，相对官方残差 {gap.matched_residual:+.6f}。该非线性说明 O0 的 1.0895 是观测窗口径产物，不能拿来证明 route/drain 改善。上表 65.02%/34.98% 只对当前切换顺序成立，不是顺序无关的唯一因果比例。",
            "",
            "到达、闭环、seed、KV 和 PerfDB 输入审计：",
            "",
            "| 维度 | 官方 | O0 | 判定 |",
            "|---|---|---|---|",
            "| 到达 | t=0 注入 C64，完成即补位 | t=0 全局注入 C128，完成即补位 | 同为合成闭环，O0 未读取实测到达 |",
            "| seed | 无随机分支 | 无随机分支 | 无可切换 seed |",
            f"| KV | physical={_config(_point(DP2_BT)).num_gpu_blocks}，每 pool 保留 null block | 相同 | 输入相同 |",
            "| serving-state 数据 | 同一 backend/version/DB | 相同 | hit/query 差是批构成输出，不是输入切换 |",
            "| 观测窗 | 单副本 N384/W128 | 双副本全局 N384/W128 | 每副本仅约 N192/W64，是实际混入项 |",
            "",
            "## TP8 首波控制",
            "",
            "| 控制 | 首 8 步逐位吻合 real | decode cell 差 | tok/s/GPU | error |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in tp8_controls:
        exact = first_window_exact_fraction(real_window[:8], item.first_window[:8])
        cell_gap = (
            baseline_gap if item.name == "baseline" else first_context_gap
        )
        lines.append(
            f"| {item.name} | {exact:.2%} | {cell_gap} | "
            f"{item.sim_tok_s_gpu:.6f} | {item.error_ratio:.6f} |"
        )
    lines.extend(
        [
            "",
            "`first_context_only` 只在第一次 schedule 暂时暴露 1 个请求；`real_initial_ramp` 再复现下一拍无新 context。两者都不改 chunk、token budget、KV、成本模型或默认 runtime。前 8 步构成可恢复，但吞吐几乎不动，因此首波错位只解释 sim-only 精确 cell，不解释 1.257 的主体。",
            "",
            "## 输入契约评审",
            "",
            "| artifact | 显式 start timestamp | 重建首拍 t=0 请求 | wall 重建误差 |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, contract in bench_contracts.items():
        lines.append(
            f"| {name} | {contract['explicit_start_timestamp']} | "
            f"{contract['initial_zero_count']} | "
            f"{float(contract['reconstructed_wall_error']):.6%} |"
        )
    lines.extend(
        [
            "",
            f"裁决：`{contract_verdict}`。`bench_records` 只有 latency；按冻结的 128-worker 闭环语义可重建客户端提交时刻，且 wall 误差很小，但首批仍是 128 个 t=0。TP8 的 EngineCore 首拍 1-context 发生在 client 提交之后，现有 bt65536 artifact 没有逐请求 receive/visible timestamp；拿 scheduler 聚合输出反灌会形成 target leakage。",
            "",
            "因此本步不改计分板协议：继续并列保存官方合成口径与诊断输入控制，暂不把 `bench_records` 宣称为 EngineCore 到达 oracle。生产合成器保真度仍是独立工程项。DP real 8.54x spread 挂起，Step 4 顺延。",
            "",
            "本步全离线、report-only；未改 runtime、PerfDB、gate，未使用 GPU。",
            "",
        ]
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    summary.write_text(
        "\n".join(
            [
                "# Phase462 DP Step 3d",
                "",
                "| 项 | 结果 |",
                "|---|---|",
                f"| DP 0.11 | 每副本观测窗 {gap.horizon_fraction:.2%} + 双副本执行/组装 {gap.topology_fraction:.2%}；到达主导证伪 |",
                f"| TP8 首波 | phase gap {baseline_gap} -> {first_context_gap}，error {tp8['baseline'].error_ratio:.6f} -> {tp8['first_context_only'].error_ratio:.6f}，仍 fail |",
                f"| 输入契约 | `{contract_verdict}`；官方计分板口径不变 |",
                "| 边界 | report-only；runtime/PerfDB/gate/GPU 均未动；Default AIC No-Go |",
                "",
                f"详见 [{report.name}](phase462_dp_step3d/{report.name})。",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_analysis(
    *,
    output_csv: Path = DEFAULT_CSV,
    output_report: Path = DEFAULT_REPORT,
    output_summary: Path = DEFAULT_SUMMARY,
) -> dict[str, object]:
    dp_controls = run_dp_controls()
    dp = {item.name: item for item in dp_controls}
    scoreboard_path = (
        REPO_ROOT / "docs/iter_gap_investigation/phase462_null_only_scoreboard.csv"
    )
    with scoreboard_path.open(encoding="utf-8") as source:
        scoreboard = list(csv.DictReader(source))
    official_score = next(row for row in scoreboard if row["scenario"] == DP2_BT)
    if not math.isclose(
        dp["single_N384_W128"].error_ratio,
        float(official_score["error_ratio"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise AssertionError("official_scoreboard_parity_failed")
    gap = decompose_dp_gap(
        official_error=dp["single_N384_W128"].error_ratio,
        short_single_error=dp["single_N192_W64"].error_ratio,
        o0_error=dp["multi_N384_W128_O0"].error_ratio,
        matched_multi_error=dp["multi_N768_W256_matched"].error_ratio,
    )
    tp8_controls, real_window = run_tp8_controls()
    tp8 = {item.name: item for item in tp8_controls}
    tp8_score = next(row for row in scoreboard if row["scenario"] == TP8_BT)
    if not math.isclose(
        tp8["baseline"].error_ratio,
        float(tp8_score["error_ratio"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise AssertionError("tp8_scoreboard_parity_failed")
    contracts = {
        "dp2_bt65536": _bench_contract(DP_RECORDS, DP_BENCH),
        "tp8_bt65536": _bench_contract(TP8_RECORDS, TP8_BENCH),
    }
    contract_verdict = assess_input_contract(
        explicit_start_timestamp=bool(
            contracts["tp8_bt65536"]["explicit_start_timestamp"]
        ),
        reconstructed_initial_zero_count=int(
            contracts["tp8_bt65536"]["initial_zero_count"]
        ),
        expected_concurrency=int(contracts["tp8_bt65536"]["concurrency"]),
        real_first_context_requests=real_window[0][0],
        controlled_error=tp8["first_context_only"].error_ratio,
        acceptance_limit=1.15,
    )
    _write_csv(
        output_csv,
        dp_controls=dp_controls,
        gap=gap,
        tp8_controls=tp8_controls,
        real_window=real_window,
        bench_contracts=contracts,
        contract_verdict=contract_verdict,
    )
    _write_reports(
        output_report,
        output_summary,
        dp_controls=dp_controls,
        gap=gap,
        tp8_controls=tp8_controls,
        real_window=real_window,
        bench_contracts=contracts,
        contract_verdict=contract_verdict,
    )
    return {
        "status": "completed_report_only",
        "dp_verdict": gap.verdict,
        "tp8_baseline_error": tp8["baseline"].error_ratio,
        "tp8_first_context_error": tp8["first_context_only"].error_ratio,
        "input_contract": contract_verdict,
        "runtime_changed": False,
        "perfdb_changed": False,
        "gate_changed": False,
        "gpu_used": False,
        "default_aic": "No-Go",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    print(
        json.dumps(
            run_analysis(
                output_csv=args.output_csv,
                output_report=args.output_report,
                output_summary=args.output_summary,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
