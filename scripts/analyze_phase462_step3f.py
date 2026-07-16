#!/usr/bin/env python3
"""Rebaseline Phase462 scoring and audit existing iteration-log fields."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from aiconfigurator.sdk import common  # noqa: E402
from aiconfigurator.sdk.backends.vllm_backend import VLLMBackend  # noqa: E402
from aiconfigurator.sdk.config import RuntimeConfig  # noqa: E402
from scripts import validate_cb_simulator as validate  # noqa: E402


BASELINE_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase462_null_only_scoreboard.csv"
TP8_LOG = REPO_ROOT / (
    "docs/iter_gap_investigation/phase458_n512_unify/"
    "recollect_tp8_8k2k_bt65536/K2.5-tp8ep8-8k2k-bt65536/serve.log"
)
DP2_LOG = REPO_ROOT / (
    "docs/iter_gap_investigation/phase458_n512_unify/"
    "recollect_dp2_8k2k_bt65536/K2.5-tp4ep8dp2-8k2k-bt65536/serve.log"
)
OUTPUT_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase462_step3f"
OUTPUT_SCOREBOARD = OUTPUT_ROOT / "phase462_measurement_fidelity_scoreboard.csv"
OUTPUT_AUDIT = OUTPUT_ROOT / "phase462_existing_log_field_audit.csv"
OUTPUT_REPORT = OUTPUT_ROOT / "phase462_step3f.md"
GATE = 1.15
TARGET_BUCKET = 6486
TARGET_DECODE_BATCH = 15


ITERATION_MARKER = "[core.py:359] Iteration("
ITERATION_RE = re.compile(
    r"\(EngineCore(?:_DP(?P<dp_rank>\d+))? pid=\d+\).*?"
    r"\[core\.py:359\] Iteration\((?P<iteration>\d+)\): "
    r"(?P<context_requests>\d+) context requests, "
    r"(?P<context_tokens>\d+) context tokens, "
    r"(?P<generation_requests>\d+) generation requests, "
    r"(?P<generation_tokens>\d+) generation tokens, "
    r"iteration elapsed time: (?P<elapsed_ms>[0-9.]+) ms"
)


@dataclass(frozen=True)
class IterationRecord:
    dp_rank: int
    iteration: int
    context_requests: int
    context_tokens: int
    generation_requests: int
    generation_tokens: int
    elapsed_ms: float

    @property
    def bucket_tokens(self) -> int:
        return self.context_tokens + self.generation_requests

    @property
    def is_mixed(self) -> bool:
        return self.context_requests > 0 and self.generation_requests > 0


@dataclass(frozen=True)
class SpreadSummary:
    max_over_min: float
    rank_medians: dict[int, float]
    rank_samples: dict[int, int]


@dataclass(frozen=True)
class ScoreRow:
    scenario: str
    tp: int
    dp: int
    max_num_batched_tokens: int
    real_output_tok_s_gpu: float
    before_sim_output_tok_s_gpu: float
    before_error_ratio: float
    after_sim_output_tok_s_gpu: float
    after_error_ratio: float


def _record(match: re.Match[str]) -> IterationRecord:
    values = match.groupdict()
    return IterationRecord(
        dp_rank=int(values["dp_rank"] or 0),
        iteration=int(values["iteration"]),
        context_requests=int(values["context_requests"]),
        context_tokens=int(values["context_tokens"]),
        generation_requests=int(values["generation_requests"]),
        generation_tokens=int(values["generation_tokens"]),
        elapsed_ms=float(values["elapsed_ms"]),
    )


def parse_iteration_line(line: str) -> IterationRecord | None:
    match = ITERATION_RE.search(line)
    if match is not None:
        return _record(match)
    if ITERATION_MARKER in line:
        raise AssertionError("iteration_log_schema_drift")
    return None


def read_iteration_records(path: Path) -> list[IterationRecord]:
    records: list[IterationRecord] = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            matches = list(ITERATION_RE.finditer(line))
            if line.count(ITERATION_MARKER) != len(matches):
                raise AssertionError(f"iteration_log_schema_drift:{path}")
            records.extend(_record(match) for match in matches)
    if not records:
        raise AssertionError(f"iteration_records_missing:{path}")
    return records


def same_cell_spread(
    records: list[IterationRecord],
    *,
    bucket_tokens: int,
    decode_batch: int,
) -> SpreadSummary:
    by_rank: dict[int, list[float]] = {}
    for record in records:
        if (
            record.bucket_tokens == bucket_tokens
            and record.generation_requests == decode_batch
        ):
            by_rank.setdefault(record.dp_rank, []).append(record.elapsed_ms)
    if set(by_rank) != {0, 1}:
        raise AssertionError(f"target_cell_rank_coverage:{sorted(by_rank)}")
    medians = {
        rank: statistics.median(values) for rank, values in sorted(by_rank.items())
    }
    if min(medians.values()) <= 0:
        raise AssertionError("target_cell_non_positive_latency")
    return SpreadSummary(
        max_over_min=max(medians.values()) / min(medians.values()),
        rank_medians=medians,
        rank_samples={rank: len(values) for rank, values in sorted(by_rank.items())},
    )


def audit_existing_fields() -> dict[str, object]:
    return {
        "available_fields": (
            "dp_rank",
            "iteration",
            "context_requests",
            "context_tokens",
            "generation_requests",
            "generation_tokens",
            "elapsed_ms",
        ),
        "missing_fields": (
            "context_chunk_tokens_multiset",
            "context_state_token_counts",
            "decode_kv_token_sum",
            "cudagraph_mode",
        ),
        "coarse_spread_reproducible": True,
        "composition_attribution_executable": False,
        "exact_undercharge_judgement_executable": False,
    }


def lightweight_v2_design(
    *,
    total_steps: int,
    mixed_steps: int,
    minimum_target_rank_samples: int,
) -> dict[str, object]:
    if total_steps <= 0 or not 0 < mixed_steps <= total_steps:
        raise ValueError("invalid_step_counts")
    if minimum_target_rank_samples <= 0:
        raise ValueError("invalid_target_sample_count")
    return {
        "periodic_k": None,
        "periodic_sampling_rejected": minimum_target_rank_samples == 1,
        "trigger": "mixed_steps_only",
        "emitted_rows": mixed_steps,
        "emitted_fraction": mixed_steps / total_steps,
    }


def _load_baseline(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 6:
        raise AssertionError(f"baseline_six_point_count:{len(rows)}")
    return {row["scenario"]: row for row in rows}


def measure_official_scoreboard(
    baseline_path: Path = BASELINE_CSV,
) -> list[ScoreRow]:
    baseline = _load_baseline(baseline_path)
    backend = VLLMBackend()
    loaded: dict[tuple[int, int, int, int], tuple[object, object]] = {}
    rows: list[ScoreRow] = []
    for point in validate.MULTI_CONFIG_DATA:
        topology = (point.tp, point.dp, point.moe_tp, point.moe_ep)
        if topology not in loaded:
            model, database, _ = validate._load_model_and_db(
                tp=point.tp,
                dp=point.dp,
                moe_tp=point.moe_tp,
                moe_ep=point.moe_ep,
            )
            loaded[topology] = (model, database)
        model, database = loaded[topology]
        config = validate._make_official_validation_cb_config(
            point,
            overlap_factor=0.0,
            ep8_per_iteration_overhead_ms=validate.DEFAULT_EP8_PER_ITERATION_OVERHEAD_MS,
        )
        with validate._official_validation_metric_scope(point):
            summary = backend.run_agg(
                model,
                database,
                RuntimeConfig(
                    batch_size=point.batch_size,
                    isl=point.isl,
                    osl=point.osl,
                ),
                ctx_tokens=point.max_num_batched_tokens,
                database_mode=common.DatabaseMode.HYBRID,
                method="cb_sim",
                cb_config=config,
            )
        result = summary.get_result_dict()
        if result is None:
            raise AssertionError(f"missing_cb_result:{point.name}")
        after_sim = float(result["tokens/s/gpu"])
        before = baseline[point.name]
        before_sim = float(before["sim_output_tok_s_gpu"])
        if point.dp == 1 and after_sim != before_sim:
            raise AssertionError(
                f"tp8_anchor_moved:{point.name}:{before_sim}:{after_sim}"
            )
        rows.append(
            ScoreRow(
                scenario=point.name,
                tp=point.tp,
                dp=point.dp,
                max_num_batched_tokens=point.max_num_batched_tokens,
                real_output_tok_s_gpu=point.real_output_tok_s_gpu,
                before_sim_output_tok_s_gpu=before_sim,
                before_error_ratio=float(before["error_ratio"]),
                after_sim_output_tok_s_gpu=after_sim,
                after_error_ratio=validate._abs_error(
                    after_sim,
                    point.real_output_tok_s_gpu,
                ),
            )
        )
    return rows


def _write_scoreboard(path: Path, rows: list[ScoreRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=[
                "scenario",
                "tp",
                "dp",
                "max_num_batched_tokens",
                "real_output_tok_s_gpu",
                "before_sim_output_tok_s_gpu",
                "before_error_ratio",
                "after_sim_output_tok_s_gpu",
                "after_error_ratio",
                "delta_error_ratio",
                "gate_pass",
                "status",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            passed = row.after_error_ratio < GATE
            writer.writerow(
                {
                    **row.__dict__,
                    "delta_error_ratio": (
                        row.after_error_ratio - row.before_error_ratio
                    ),
                    "gate_pass": passed,
                    "status": "pass" if passed else "fail",
                }
            )


def _write_audit(path: Path, audit: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(("field", "availability", "purpose"))
        for field in audit["available_fields"]:
            writer.writerow((field, "available", "coarse_iteration_cell"))
        purposes = {
            "context_chunk_tokens_multiset": "exact_context_composition",
            "context_state_token_counts": "fresh_vs_recompute_vs_resume",
            "decode_kv_token_sum": "sim_cost_query",
            "cudagraph_mode": "execution_regime",
        }
        for field in audit["missing_fields"]:
            writer.writerow((field, "missing", purposes[field]))


def _write_report(
    path: Path,
    *,
    scores: list[ScoreRow],
    tp8_records: list[IterationRecord],
    dp2_records: list[IterationRecord],
    spread: SpreadSummary,
    audit: dict[str, object],
    design: dict[str, object],
) -> None:
    score_lines = []
    for row in scores:
        score_lines.append(
            f"| {row.scenario} | {row.before_error_ratio:.6f} | "
            f"{row.after_error_ratio:.6f} | "
            f"{row.after_error_ratio - row.before_error_ratio:+.6f} | "
            f"{'pass' if row.after_error_ratio < GATE else 'fail'} |"
        )
    pass_count = sum(row.after_error_ratio < GATE for row in scores)
    score_by_name = {row.scenario: row for row in scores}
    dp2_bt = score_by_name["K2.5-tp4ep8dp2-8k2k-bt65536"]
    available = ", ".join(f"`{field}`" for field in audit["available_fields"])
    missing = ", ".join(f"`{field}`" for field in audit["missing_fields"])
    lines = [
        "# Phase462 Step 3f 度量保真修复与日志字段审计",
        "",
        f"结论：度量修复后计分板仍为 `{pass_count}/6`。TP8 三点逐字节不动；DP2-bt65536 从 `1.200038` 收敛到 `{dp2_bt.after_error_ratio:.6f}`，但仍未过 `1.15`。Phase458 日志可复现 coarse cell spread，不能完成 exact composition / undercharge 判卷。",
        "",
        "| 场景 | 修复前 error | 修复后 error | delta | 1.15 门 |",
        "|---|---:|---:|---:|---|",
        *score_lines,
        "",
        "DP2 采用与 real bench 相同的全局 N512、warmup=0、首个 arrival 到最后 completion 的完整墙钟；TP8 不进入新组装路径。历史 `65.02%/34.98%` 只对 Step 3d 的固定切换顺序成立，不可交换，也不是唯一因果比例。",
        "",
        "## 现有字段",
        "",
        "| 项 | 结果 |",
        "|---|---|",
        f"| TP8 iteration rows | {len(tp8_records)} |",
        f"| DP2 iteration rows | {len(dp2_records)}；rank0={sum(row.dp_rank == 0 for row in dp2_records)}；rank1={sum(row.dp_rank == 1 for row in dp2_records)} |",
        f"| DP2 mixed rows | {sum(row.is_mixed for row in dp2_records)} |",
        f"| `(6486,15)` samples | rank0={spread.rank_samples[0]}；rank1={spread.rank_samples[1]} |",
        f"| `(6486,15)` median elapsed | rank0={spread.rank_medians[0]:.3f} ms；rank1={spread.rank_medians[1]:.3f} ms；spread={spread.max_over_min:.6f}x |",
        f"| 已有 | {available} |",
        f"| 缺失 | {missing} |",
        "",
        "相同聚合 cell 内仍有 8.53x 差异，所以已有字段只能复现问题，不能把差异归给 chunk 构成或其他隐藏维。当前 sim 成本查询还需要 decode KV 长度，日志也没有按 fresh/recompute/resume 分开的 context 状态；用常数补齐会造成现场拟合，禁止执行。",
        "",
        "## 轻量采集 v2",
        "",
        "| 项 | 设计 |",
        "|---|---|",
        "| 触发 | 仅 mixed step；不做周期 K 采样 |",
        "| 原因 | 目标 cell 在 rank0 只有 1 次，任何 K>1 都可能漏样 |",
        f"| 输出规模 | {design['emitted_rows']}/{len(dp2_records)} rows（{design['emitted_fraction']:.4%}）；字节数须由确定的序列化格式再算，不伪造上界 |",
        "| 新字段 | context chunk token multiset；fresh/recompute/resume token counts；decode KV token sum；cudagraph mode |",
        "| 明确不采 | request id、arrival 链、逐请求生命周期、每步同步写文件 |",
        "| 开销结论 | 离线只能证明输出量很小，不能证明 <=2%；下一次仍须独立 off/on GPU 硬门 |",
        "",
        "本步未改 simulator runtime、PerfDB 或 gate；Default AIC 维持 No-Go，Step 4 继续顺延。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_analysis(
    *,
    baseline_path: Path = BASELINE_CSV,
    tp8_log: Path = TP8_LOG,
    dp2_log: Path = DP2_LOG,
    output_scoreboard: Path = OUTPUT_SCOREBOARD,
    output_audit: Path = OUTPUT_AUDIT,
    output_report: Path = OUTPUT_REPORT,
) -> dict[str, object]:
    scores = measure_official_scoreboard(baseline_path)
    tp8_records = read_iteration_records(tp8_log)
    dp2_records = read_iteration_records(dp2_log)
    spread = same_cell_spread(
        dp2_records,
        bucket_tokens=TARGET_BUCKET,
        decode_batch=TARGET_DECODE_BATCH,
    )
    audit = audit_existing_fields()
    design = lightweight_v2_design(
        total_steps=len(dp2_records),
        mixed_steps=sum(record.is_mixed for record in dp2_records),
        minimum_target_rank_samples=min(spread.rank_samples.values()),
    )
    _write_scoreboard(output_scoreboard, scores)
    _write_audit(output_audit, audit)
    _write_report(
        output_report,
        scores=scores,
        tp8_records=tp8_records,
        dp2_records=dp2_records,
        spread=spread,
        audit=audit,
        design=design,
    )
    return {
        "status": "completed_report_only",
        "scoreboard_passed": sum(row.after_error_ratio < GATE for row in scores),
        "scoreboard_total": len(scores),
        "dp2_target_cell_spread": spread.max_over_min,
        "existing_logs_sufficient": False,
        "lightweight_v2_required": True,
        "runtime_changed": False,
        "perfdb_changed": False,
        "gate_changed": False,
        "default_aic": "No-Go",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=BASELINE_CSV)
    parser.add_argument("--tp8-log", type=Path, default=TP8_LOG)
    parser.add_argument("--dp2-log", type=Path, default=DP2_LOG)
    parser.add_argument("--output-scoreboard", type=Path, default=OUTPUT_SCOREBOARD)
    parser.add_argument("--output-audit", type=Path, default=OUTPUT_AUDIT)
    parser.add_argument("--output-report", type=Path, default=OUTPUT_REPORT)
    args = parser.parse_args()
    print(
        json.dumps(
            run_analysis(
                baseline_path=args.baseline,
                tp8_log=args.tp8_log,
                dp2_log=args.dp2_log,
                output_scoreboard=args.output_scoreboard,
                output_audit=args.output_audit,
                output_report=args.output_report,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
