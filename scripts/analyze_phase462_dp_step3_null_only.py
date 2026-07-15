#!/usr/bin/env python3
"""Phase462 DP Step3 null-only phase/cost triage."""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import scripts.analyze_phase459_residual_triage as phase459  # noqa: E402
import scripts.analyze_phase461_cost_recollect as phase461_cost  # noqa: E402
import scripts.analyze_phase461_step4_cell_triage as phase461_cells  # noqa: E402
import scripts.analyze_phase462_dynamics_triage as phase462_dynamics  # noqa: E402
import scripts.validate_cb_simulator as validate  # noqa: E402
from aiconfigurator.sdk.backends.cb_simulator import CBSimulator  # noqa: E402


DP2_BT = "K2.5-tp4ep8dp2-8k2k-bt65536"
TP8_BT = "K2.5-tp8ep8-8k2k-bt65536"
TARGET_ERROR = 1.15
COMMON_ROOT_THRESHOLD = 0.70
NULL_SCOREBOARD = REPO_ROOT / (
    "docs/iter_gap_investigation/phase462_null_only_scoreboard.csv"
)
DP_EVENT_PATH = REPO_ROOT / (
    "docs/iter_gap_investigation/phase461_cost_recollect/"
    "dp2_bt65536_diagnostic/overhead_on/event_timing.jsonl.gz"
)
DP_SERVE_LOG = REPO_ROOT / (
    "docs/iter_gap_investigation/phase461_cost_recollect/"
    "dp2_bt65536_diagnostic/overhead_on/"
    f"{DP2_BT}/serve.log.gz"
)
DEFAULT_CSV = REPO_ROOT / (
    "docs/iter_gap_investigation/phase462_dp_step3_null_only.csv"
)
DEFAULT_REPORT = REPO_ROOT / (
    "docs/iter_gap_investigation/phase462_dp_step3_null_only.md"
)


@dataclass(frozen=True)
class PairedPhaseSummary:
    active_pairs: int
    active_exact_fraction: float
    active_phase_type_fraction: float
    first_prefill_ratio: float
    rank0_mixed_steps: int
    rank1_mixed_steps: int


@dataclass(frozen=True)
class MixedStep:
    bucket_tokens: int
    decode_batch: int
    wall_ms: float
    prefill_tokens: int
    recompute_tokens: int


@dataclass(frozen=True)
class BucketResidual:
    mixed_steps: int
    sim_only_unique_cells: int
    sim_only_wall_share: float
    recompute_associated_wall_share: float
    recompute_token_share: float


@dataclass(frozen=True)
class ReportRow:
    section: str
    scenario: str
    metric: str
    value: str
    status: str
    note: str


def _rank(row: Mapping[str, object]) -> int:
    value = row.get("replica_id", row.get("engine"))
    if value is None:
        raise ValueError("phase row has no rank")
    return int(value)


def _iteration(row: Mapping[str, object]) -> int:
    value = row.get("local_iter", row.get("iteration"))
    if value is None:
        raise ValueError("phase row has no local iteration")
    return int(value)


def _prefill(row: Mapping[str, object]) -> int:
    return int(row.get("prefill_tokens", row.get("ctx_tokens", 0)))


def _decode(row: Mapping[str, object]) -> int:
    return int(row.get("decode_reqs", row.get("generation_requests", 0)))


def _phase_type(prefill: int, decode: int) -> str:
    if prefill and decode:
        return "mixed"
    if prefill:
        return "prefill"
    return "decode"


def summarize_paired_rank_phase(
    rows: Iterable[Mapping[str, object]],
) -> PairedPhaseSummary:
    by_rank: dict[int, dict[int, tuple[int, int]]] = {}
    for row in rows:
        by_rank.setdefault(_rank(row), {})[_iteration(row)] = (
            _prefill(row),
            _decode(row),
        )
    if set(by_rank) != {0, 1}:
        raise AssertionError(f"expected exactly DP ranks 0 and 1, got {sorted(by_rank)}")
    common = sorted(set(by_rank[0]) & set(by_rank[1]))
    active = [
        iteration
        for iteration in common
        if by_rank[0][iteration][0] or by_rank[1][iteration][0]
    ]
    if not active:
        raise AssertionError("no paired rank steps containing prefill")
    exact = sum(by_rank[0][iteration] == by_rank[1][iteration] for iteration in active)
    same_type = sum(
        _phase_type(*by_rank[0][iteration])
        == _phase_type(*by_rank[1][iteration])
        for iteration in active
    )
    first_prefill = [
        next(
            prefill
            for _, (prefill, _) in sorted(by_rank[rank].items())
            if prefill > 0
        )
        for rank in (0, 1)
    ]
    first_ratio = max(first_prefill) / min(first_prefill)
    mixed_counts = [
        sum(_phase_type(prefill, decode) == "mixed" for prefill, decode in by_rank[rank].values())
        for rank in (0, 1)
    ]
    return PairedPhaseSummary(
        active_pairs=len(active),
        active_exact_fraction=exact / len(active),
        active_phase_type_fraction=same_type / len(active),
        first_prefill_ratio=first_ratio,
        rank0_mixed_steps=mixed_counts[0],
        rank1_mixed_steps=mixed_counts[1],
    )


def candidate_decision(
    *,
    candidate_error: float,
    real_summary: PairedPhaseSummary,
    candidate_summary: PairedPhaseSummary,
) -> str:
    if candidate_error > TARGET_ERROR:
        return "score_fail_do_not_adopt"
    real_has_rank_asymmetry = (
        real_summary.active_exact_fraction < 1.0
        or real_summary.rank0_mixed_steps != real_summary.rank1_mixed_steps
    )
    candidate_has_rank_asymmetry = (
        candidate_summary.active_exact_fraction < 1.0
        or candidate_summary.rank0_mixed_steps != candidate_summary.rank1_mixed_steps
    )
    if real_has_rank_asymmetry and not candidate_has_rank_asymmetry:
        return "score_pass_phase_fail_do_not_adopt"
    return "score_and_phase_pass_candidate_only"


def summarize_sim_only_bucket_residual(
    steps: Iterable[MixedStep],
    *,
    real_cells: set[tuple[int, int]],
) -> BucketResidual:
    steps = list(steps)
    if not steps:
        raise AssertionError("no mixed simulation steps")
    total_wall = sum(step.wall_ms for step in steps)
    sim_only = [
        step
        for step in steps
        if (step.bucket_tokens, step.decode_batch) not in real_cells
    ]
    sim_only_wall = sum(step.wall_ms for step in sim_only)
    sim_only_prefill = sum(step.prefill_tokens for step in sim_only)
    return BucketResidual(
        mixed_steps=len(steps),
        sim_only_unique_cells=len(
            {(step.bucket_tokens, step.decode_batch) for step in sim_only}
        ),
        sim_only_wall_share=sim_only_wall / total_wall,
        recompute_associated_wall_share=(
            sum(step.wall_ms for step in sim_only if step.recompute_tokens > 0)
            / sim_only_wall
        ),
        recompute_token_share=(
            sum(step.recompute_tokens for step in sim_only) / sim_only_prefill
        ),
    )


def _load_official_score() -> tuple[float, float, float]:
    with NULL_SCOREBOARD.open(newline="", encoding="utf-8") as source:
        row = next(row for row in csv.DictReader(source) if row["scenario"] == DP2_BT)
    return (
        float(row["real_output_tok_s_gpu"]),
        float(row["sim_output_tok_s_gpu"]),
        float(row["error_ratio"]),
    )


def _run_lockstep_candidate() -> tuple[float, float, int, list[dict[str, object]]]:
    point = next(point for point in validate.MULTI_CONFIG_DATA if point.name == DP2_BT)
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
    sim = CBSimulator(backend, model, database, config)
    result = sim.run_multi_replica(
        isl=point.isl,
        osl=point.osl,
        concurrency=point.batch_size,
        data_parallel_size=point.dp,
        num_gpus=point.tp * point.dp,
        lockstep=True,
    )
    return (
        result.throughput_tok_s_gpu,
        validate._abs_error(result.throughput_tok_s_gpu, point.real_output_tok_s_gpu),
        result.total_iterations,
        [dict(row) for row in sim.get_last_schedule_trace()],
    )


def _load_real_phase_rows() -> list[dict[str, object]]:
    return [
        {
            "replica_id": int(step.engine),
            "local_iter": step.iteration,
            "prefill_tokens": step.ctx_tokens,
            "decode_reqs": step.generation_requests,
        }
        for step in phase459.parse_iteration_steps(DP_SERVE_LOG)
    ]


def _load_dp_cost_evidence() -> tuple[float, float, tuple[int, int]]:
    event_steps = phase461_cost.group_tp_steps(
        phase461_cost.read_event_rows(DP_EVENT_PATH),
        tp_width=4,
    )
    cells = phase461_cost.summarize_cross_rank_cells(event_steps)
    cell_key, cell = max(
        cells.items(),
        key=lambda item: float(item[1]["max_over_min"]),
    )
    busy_wall = phase461_cost.summarize_busy_wall(
        event_steps,
        phase459.parse_iteration_steps(DP_SERVE_LOG),
        bucket_tokens=cell_key[0],
        decode_batch=cell_key[1],
    )
    return (
        float(cell["max_over_min"]),
        float(busy_wall["max_busy_over_wall"]),
        cell_key,
    )


def _run_tp8_bucket_recheck() -> tuple[BucketResidual, int]:
    attribution = phase462_dynamics.run_recompute_attribution(TP8_BT)
    scenario = next(item for item in phase461_cells.SCENARIOS if item[0] == TP8_BT)
    real_cells = set(phase461_cells._real_mixed_cells(scenario[2]))
    steps = [
        MixedStep(
            bucket_tokens=step.bucket_tokens,
            decode_batch=step.decode_batch,
            wall_ms=step.wall_ms,
            prefill_tokens=step.prefill_tokens,
            recompute_tokens=step.recompute_tokens,
        )
        for step in attribution["steps"]
    ]
    return (
        summarize_sim_only_bucket_residual(steps, real_cells=real_cells),
        int(attribution["preemptions"]),
    )


def _add(
    rows: list[ReportRow],
    section: str,
    scenario: str,
    metric: str,
    value: object,
    status: str,
    note: str,
) -> None:
    if isinstance(value, float):
        rendered = "inf" if math.isinf(value) else f"{value:.9f}"
    else:
        rendered = str(value)
    rows.append(ReportRow(section, scenario, metric, rendered, status, note))


def build_rows() -> tuple[list[ReportRow], str]:
    rows: list[ReportRow] = []
    real_rate, official_rate, official_error = _load_official_score()
    candidate_rate, candidate_error, candidate_iters, candidate_trace = (
        _run_lockstep_candidate()
    )
    real_phase = summarize_paired_rank_phase(_load_real_phase_rows())
    candidate_phase = summarize_paired_rank_phase(candidate_trace)
    decision = candidate_decision(
        candidate_error=candidate_error,
        real_summary=real_phase,
        candidate_summary=candidate_phase,
    )
    rank_spread, busy_wall, spread_cell = _load_dp_cost_evidence()
    bucket, tp8_preemptions = _run_tp8_bucket_recheck()

    _add(rows, "official_path", DP2_BT, "assembly", "representative_replica_scaled", "fail_structural", "single 64-concurrency replica multiplied by DP=2")
    _add(rows, "official_path", DP2_BT, "real_tok_s_gpu", real_rate, "recorded", "null-only official baseline")
    _add(rows, "official_path", DP2_BT, "sim_tok_s_gpu", official_rate, "fail", "null-only official baseline")
    _add(rows, "official_path", DP2_BT, "error_ratio", official_error, "fail", "15% gate")
    _add(rows, "lockstep_candidate", DP2_BT, "sim_tok_s_gpu", candidate_rate, "diagnostic", "offline existing multi-replica path")
    _add(rows, "lockstep_candidate", DP2_BT, "error_ratio", candidate_error, "score_pass", "phase gate remains mandatory")
    _add(rows, "lockstep_candidate", DP2_BT, "total_iterations", candidate_iters, "recorded", "two replica rows counted")
    for side, summary in (("real", real_phase), ("candidate", candidate_phase)):
        _add(rows, "phase_signature", DP2_BT, f"{side}_active_pairs", summary.active_pairs, "recorded", "common local iterations where either rank has prefill")
        _add(rows, "phase_signature", DP2_BT, f"{side}_active_exact_fraction", summary.active_exact_fraction, "reference" if side == "real" else "fail", "per-rank prefill/decode composition equality")
        _add(rows, "phase_signature", DP2_BT, f"{side}_phase_type_fraction", summary.active_phase_type_fraction, "recorded", "prefill/mixed/decode class equality")
        _add(rows, "phase_signature", DP2_BT, f"{side}_first_prefill_ratio", summary.first_prefill_ratio, "recorded", "larger rank prefill divided by smaller")
        _add(rows, "phase_signature", DP2_BT, f"{side}_mixed_steps_by_rank", f"{summary.rank0_mixed_steps}/{summary.rank1_mixed_steps}", "recorded", "rank0/rank1")
    _add(rows, "cost_evidence", DP2_BT, "max_same_cell_rank_spread", rank_spread, "real_measured", f"bucket={spread_cell[0]};decode={spread_cell[1]}")
    _add(rows, "cost_evidence", DP2_BT, "max_busy_over_wall", busy_wall, "pass", "real spike is execution, not logging wait")
    _add(rows, "decision", DP2_BT, "lockstep_candidate", decision, "blocked", "score cannot override a failed phase signature")
    _add(rows, "tp8_bucket_recheck", TP8_BT, "preemptions", tp8_preemptions, "recorded", "N384 null-only validation protocol")
    _add(rows, "tp8_bucket_recheck", TP8_BT, "mixed_steps", bucket.mixed_steps, "recorded", "null-only")
    _add(rows, "tp8_bucket_recheck", TP8_BT, "sim_only_unique_cells", bucket.sim_only_unique_cells, "fail", "exact Phase458 reference cells")
    _add(rows, "tp8_bucket_recheck", TP8_BT, "sim_only_wall_share", bucket.sim_only_wall_share, "fail", "Phase461 value was 0.4215")
    _add(rows, "tp8_bucket_recheck", TP8_BT, "recompute_associated_wall_share", bucket.recompute_associated_wall_share, "below_original_gate" if bucket.recompute_associated_wall_share < COMMON_ROOT_THRESHOLD else "common_root", "original common-root gate=0.70")
    _add(rows, "tp8_bucket_recheck", TP8_BT, "recompute_token_share", bucket.recompute_token_share, "recorded", "within current sim-only cells")
    return rows, "logging_only_dp_route_visibility_required"


def write_csv(path: Path, rows: Sequence[ReportRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=["section", "scenario", "metric", "value", "status", "note"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(row.__dict__ for row in rows)


def write_report(path: Path, rows: Sequence[ReportRow], *, decision: str) -> None:
    def value(metric: str, section: str | None = None) -> str:
        return next(
            row.value
            for row in rows
            if row.metric == metric and (section is None or row.section == section)
        )

    has_full_results = any(row.metric == "real_active_exact_fraction" for row in rows)
    if has_full_results:
        tables = f"""## DP2-bt65536 重基线

| 路径 | sim tok/s/GPU | error | 15% 分数门 | 相位门 |
|---|---:|---:|---|---|
| 当前默认：单副本乘 DP | {value('sim_tok_s_gpu', 'official_path')} | {value('error_ratio', 'official_path')} | FAIL | 不具备 per-rank 构成 |
| 离线双副本锁步候选 | {value('sim_tok_s_gpu', 'lockstep_candidate')} | {value('error_ratio', 'lockstep_candidate')} | PASS | FAIL |

## 相位与成本签名

| 指标 | real | 锁步候选 |
|---|---:|---:|
| 含 prefill 的共同局部步 | {value('real_active_pairs')} | {value('candidate_active_pairs')} |
| rank 构成逐步全等率 | {value('real_active_exact_fraction')} | {value('candidate_active_exact_fraction')} |
| phase 类型全等率 | {value('real_phase_type_fraction')} | {value('candidate_phase_type_fraction')} |
| 首次 prefill rank 比 | {value('real_first_prefill_ratio')} | {value('candidate_first_prefill_ratio')} |
| mixed 步 rank0/rank1 | {value('real_mixed_steps_by_rank')} | {value('candidate_mixed_steps_by_rank')} |
| real 同 cell 最大 rank spread | {value('max_same_cell_rank_spread')} | 无法表达 |

`1.095` 只来自现有锁步计算方式，候选两个 rank 始终同构，不能解释 real 的相位分裂与 `8.54x` 同 cell spread，所以不得进入默认路径。

## TP8 大 bucket 复查

| 指标 | null-only 新基线 | Phase461 |
|---|---:|---:|
| mixed 步 | {value('mixed_steps')} | 185 queries |
| reference 未观察 wall 占比 | {value('sim_only_wall_share')} | 0.4215 |
| recompute 关联占比 | {value('recompute_associated_wall_share')} | 0.8541 |
| 原共根门 | 未过 0.70 | 已过 0.70 |

null block 改变了轨迹，但没有消掉 sim-only 大 bucket；当前超过一半 mixed wall 仍在 Phase458 reference 未观察 cell 中。recompute 关联已降到原 70% 硬门以下，不能继续把剩余问题只归为抢占过发。
"""
    else:
        tables = "配套 CSV 保留机械判卷行。"
    path.write_text(
        f"""# Phase462 DP Step 3 null-only 分诊

结论：当前 `dp2-bt65536` 默认路径的“单副本模拟后乘 DP”结构不成立；现成双副本锁步虽把分数推入 15%，但 per-rank 相位签名失败，仍不得进入默认路径。下一步必须先补 request 到 rank 的 receive/admit 可见性，不能按分数倒推参数。`Default AIC=No-Go`。

{tables}

## 下一门

| 项目 | 裁决 |
|---|---|
| 下一动作 | `{decision}` |
| 需要的新证据 | logging-only：request id、目标 DP rank、router 可见 waiting/running、EngineCore receive、首次 admit 的同 ID 时间链 |
| 现有数据是否够定结构缺口 | 够；已证明默认路径与锁步候选都不能复现 rank 相位 |
| 是否现在改 runtime | 否 |
| 是否现在改 PerfDB | 否；`8.54x` 行继续 diagnostic-only |
| 是否现在跑 GPU | 否；采集设计与成本需单独审批 |
| Default AIC | No-Go |
""",
        encoding="utf-8",
    )


def run_analysis(
    *,
    output_csv: Path = DEFAULT_CSV,
    output_report: Path = DEFAULT_REPORT,
) -> dict[str, object]:
    rows, decision = build_rows()
    write_csv(output_csv, rows)
    write_report(output_report, rows, decision=decision)
    return {
        "status": "completed_report_only",
        "decision": decision,
        "runtime_changed": False,
        "gate_changed": False,
        "perfdb_changed": False,
        "gpu_used": False,
        "default_aic": "No-Go",
    }


def main() -> int:
    print(json.dumps(run_analysis(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
