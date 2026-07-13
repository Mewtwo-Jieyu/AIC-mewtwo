#!/usr/bin/env python3
"""Phase462 Step2a: offline first-preemption observability audit."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import scripts.analyze_phase451_preemption_ledger as phase451
import scripts.analyze_phase451d_preemption_forensics as phase451d


SCENARIO = "K2.5-tp8ep8-32k3k"
SCENARIO_DIR = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase458_n512_unify/recollect_tp8_32k3k"
    / SCENARIO
)
DEFAULT_METRICS = SCENARIO_DIR / "metrics.jsonl.gz"
DEFAULT_SERVE_LOG = SCENARIO_DIR / "serve.log"
DEFAULT_AB = REPO_ROOT / "docs/iter_gap_investigation/phase461_step4b_final_ab.csv"
DEFAULT_PANORAMA = REPO_ROOT / "docs/iter_gap_investigation/phase462_dynamics_triage.csv"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase462_first_divergence.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase462_first_divergence.md"
NUM_REQUESTS = 512
ERROR_THRESHOLD = 1.15

METRIC_NAMES = {
    "vllm:num_preemptions_total": "num_preemptions_total",
    "vllm:num_requests_running": "num_requests_running",
    "vllm:num_requests_waiting": "num_requests_waiting",
    "vllm:kv_cache_usage_perc": "kv_cache_usage_perc",
}
REQUIRED_DECISION_FIELDS = {
    "decision_timestamp",
    "free_blocks_before",
    "requested_blocks",
    "trigger_request_id",
    "victim_request_id",
    "victim_queue_position",
}
REAL_AVAILABLE_FIELDS = {
    "counter_bracket",
    "running_gauge",
    "waiting_gauge",
    "kv_usage_gauge",
    "iteration_composition",
}

ITER_RE = re.compile(
    r"INFO (?P<month>\d{2})-(?P<day>\d{2}) (?P<clock>\d{2}:\d{2}:\d{2}).*?"
    r"Iteration\((?P<iteration>\d+)\): "
    r"(?P<ctx_req>\d+) context requests, (?P<ctx_tok>\d+) context tokens, "
    r"(?P<gen_req>\d+) generation requests, (?P<gen_tok>\d+) generation tokens, "
    r"iteration elapsed time: (?P<elapsed>[0-9.]+) ms"
)

CSV_FIELDS = [
    "section",
    "scenario",
    "metric",
    "value",
    "target",
    "status",
    "source",
    "note",
]


@dataclass(frozen=True)
class MetricSnapshot:
    timestamp_s: float
    values: dict[str, float]


@dataclass(frozen=True)
class CounterBracket:
    start_s: float
    end_s: float
    start_value: float
    end_value: float
    delta: float
    precision: str


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def _row(
    section: str,
    metric: str,
    value: object,
    *,
    target: object = "",
    status: str = "",
    source: str = "",
    note: str = "",
    scenario: str = SCENARIO,
) -> dict[str, object]:
    return {
        "section": section,
        "scenario": scenario,
        "metric": metric,
        "value": value,
        "target": target,
        "status": status,
        "source": source,
        "note": note,
    }


def read_metric_snapshots(path: Path, *, engine: str = "0") -> list[MetricSnapshot]:
    snapshots: list[MetricSnapshot] = []
    with _open_text(path) as source:
        for line in source:
            if not line.strip():
                continue
            payload = json.loads(line)
            source_ts = str(payload["ts"])
            timestamp_s = datetime.fromisoformat(source_ts).timestamp()
            values: dict[str, float] = {}
            for metric_line in str(payload.get("body", "")).splitlines():
                parsed = phase451.parse_prometheus_line(metric_line)
                if parsed is None:
                    continue
                name, labels, value = parsed
                output_name = METRIC_NAMES.get(name)
                if output_name is not None and labels.get("engine") == engine:
                    values[output_name] = value
            if values:
                snapshots.append(MetricSnapshot(timestamp_s, values))
    if not snapshots:
        raise ValueError(f"no target metrics found in {path}")
    return snapshots


def first_counter_increase(
    snapshots: Iterable[MetricSnapshot], metric: str
) -> CounterBracket:
    rows = [row for row in snapshots if metric in row.values]
    for previous, current in zip(rows, rows[1:]):
        start = previous.values[metric]
        end = current.values[metric]
        if end > start:
            return CounterBracket(
                previous.timestamp_s,
                current.timestamp_s,
                start,
                end,
                end - start,
                "scrape_interval_only",
            )
    raise ValueError(f"counter {metric} never increases")


def decision_observability_gate(available_fields: set[str]) -> dict[str, object]:
    missing = sorted(REQUIRED_DECISION_FIELDS - available_fields)
    return {
        "status": "pass" if not missing else "blocked",
        "missing_fields": missing,
        "available_fields": sorted(available_fields),
    }


def predict_preemption_fix_direction(
    *, real_throughput: float, sim_throughput: float, threshold: float
) -> dict[str, float | str]:
    current_ratio = max(real_throughput, sim_throughput) / min(
        real_throughput, sim_throughput
    )
    maximum = real_throughput * threshold
    if sim_throughput >= real_throughput:
        direction = "worsen"
        risk = (
            "current_pass_at_risk"
            if current_ratio <= threshold
            else "current_fail_worsens"
        )
    else:
        direction = "improve_until_crossing"
        risk = "current_pass_not_immediately_at_risk"
    return {
        "current_error_ratio": current_ratio,
        "direction": direction,
        "threshold_risk": risk,
        "max_sim_increase_before_threshold": maximum / sim_throughput - 1.0,
    }


def select_step2b_scenario(
    panorama: Iterable[dict[str, object]],
) -> dict[str, object]:
    rows = list(panorama)
    if not rows:
        raise ValueError("empty preemption panorama")
    return max(rows, key=lambda row: float(row["sim_over_real"]))


def load_preemption_panorama(path: Path) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if row["section"] != "preemption_panorama":
                continue
            scenario = row["scenario"]
            grouped.setdefault(scenario, {"scenario": scenario})[row["metric"]] = float(
                row["value"]
            )
    return [row for row in grouped.values() if "sim_over_real" in row]


def load_ab_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as source:
        return [dict(row) for row in csv.DictReader(source)]


def iteration_rows_in_bracket(
    path: Path, bracket: CounterBracket, *, year: int = 2026
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with _open_text(path) as source:
        for line in source:
            match = ITER_RE.search(line)
            if match is None:
                continue
            timestamp_s = datetime.fromisoformat(
                f"{year}-{match.group('month')}-{match.group('day')}T"
                f"{match.group('clock')}+00:00"
            ).timestamp()
            if bracket.start_s <= timestamp_s <= bracket.end_s:
                rows.append(
                    {
                        "iteration": int(match.group("iteration")),
                        "context_requests": int(match.group("ctx_req")),
                        "context_tokens": int(match.group("ctx_tok")),
                        "generation_requests": int(match.group("gen_req")),
                        "generation_tokens": int(match.group("gen_tok")),
                        "elapsed_ms": float(match.group("elapsed")),
                    }
                )
    return rows


def build_report_rows(
    *,
    metrics_path: Path = DEFAULT_METRICS,
    serve_log: Path = DEFAULT_SERVE_LOG,
    ab_path: Path = DEFAULT_AB,
    panorama_path: Path = DEFAULT_PANORAMA,
) -> list[dict[str, object]]:
    snapshots = read_metric_snapshots(metrics_path)
    bracket = first_counter_increase(snapshots, "num_preemptions_total")
    bracket_end = next(
        snapshot
        for snapshot in snapshots
        if snapshot.timestamp_s == bracket.end_s
    )
    bracket_iterations = iteration_rows_in_bracket(serve_log, bracket)
    gate = decision_observability_gate(REAL_AVAILABLE_FIELDS)
    panorama = load_preemption_panorama(panorama_path)
    selected = select_step2b_scenario(panorama)
    sim = phase451d.run_sim_preemption_forensics(
        scenario=SCENARIO, num_requests=NUM_REQUESTS
    )
    first_event = dict(sim["events"][0])
    metrics_source = str(metrics_path.relative_to(REPO_ROOT))
    rows: list[dict[str, object]] = [
        _row(
            "sim_first_preemption",
            "local_iter",
            first_event["local_iter"],
            status="observed_exact",
            source="current N=512 sim decision hook",
            note=f"replica={first_event['replica_id']}; trigger={first_event['trigger_state']}",
        ),
        _row(
            "sim_first_preemption",
            "scheduled_demand_minus_capacity_blocks",
            first_event["over_blocks_before"],
            target=">0 enters capacity handling",
            status="preempt_triggered",
            source="current N=512 sim decision hook",
            note=(
                f"total={first_event['total_blocks_before']}; "
                f"capacity={first_event['num_gpu_blocks']}; "
                f"trigger_req={first_event['trigger_req_id']}; "
                f"victim={first_event['victim_req_id']}"
            ),
        ),
        _row(
            "sim_first_preemption",
            "state",
            f"running={first_event['running_before']},waiting={first_event['waiting_before']}",
            status="observed_exact",
            source="current N=512 sim decision hook",
            note=(
                f"generated={first_event['trigger_generated_tokens']}; "
                f"kv={first_event['trigger_kv_cache_len']}"
            ),
        ),
        _row(
            "real_first_preemption",
            "counter_bracket_seconds",
            bracket.end_s - bracket.start_s,
            target="decision-point timestamp",
            status="insufficient_precision",
            source=metrics_source,
            note=(
                f"counter {bracket.start_value:g}->{bracket.end_value:g}; "
                f"delta={bracket.delta:g}; precision={bracket.precision}"
            ),
        ),
        _row(
            "real_first_preemption",
            "iteration_rows_in_bracket",
            len(bracket_iterations),
            target="1 decision-aligned row",
            status="ambiguous",
            source=str(serve_log.relative_to(REPO_ROOT)),
            note="INFO iteration rows have composition but no victim or block-allocation state",
        ),
        _row(
            "real_first_preemption",
            "scrape_end_state",
            (
                f"running={bracket_end.values['num_requests_running']},"
                f"waiting={bracket_end.values['num_requests_waiting']},"
                f"kv_usage={bracket_end.values['kv_cache_usage_perc']}"
            ),
            target="decision-point block ledger",
            status="aggregate_only",
            source=metrics_source,
            note="state is sampled after one or more decisions in the 2s interval",
        ),
        _row(
            "real_first_preemption",
            "prefix_caching",
            False,
            status="excluded_candidate",
            source=str(serve_log.relative_to(REPO_ROOT)),
            note="startup args explicitly set enable_prefix_caching=False",
        ),
        _row(
            "observability_gate",
            "status",
            gate["status"],
            target="all decision fields observed",
            status=str(gate["status"]),
            source="Phase458 INFO log + 2s metrics",
            note="missing=" + ",".join(gate["missing_fields"]),
        ),
    ]

    for raw in load_ab_rows(ab_path):
        real = float(raw["current_real_output_tok_s_gpu"])
        sim_throughput = float(raw["current_sim_output_tok_s_gpu"])
        prediction = predict_preemption_fix_direction(
            real_throughput=real,
            sim_throughput=sim_throughput,
            threshold=ERROR_THRESHOLD,
        )
        rows.append(
            _row(
                "sign_prediction",
                "preemption_fix_sim_throughput_increase",
                prediction["direction"],
                target=f"error_ratio<={ERROR_THRESHOLD}",
                status=str(prediction["threshold_risk"]),
                source=str(ab_path.relative_to(REPO_ROOT)),
                note=(
                    f"real={real:.6f}; sim={sim_throughput:.6f}; "
                    f"current_ratio={prediction['current_error_ratio']:.6f}; "
                    f"max_increase_before_fail={prediction['max_sim_increase_before_threshold']:.6f}"
                ),
                scenario=str(raw["name"]),
            )
        )

    rows.extend(
        [
            _row(
                "decision",
                "step2a_verdict",
                "real_decision_state_insufficient",
                target="unique source-backed semantic difference",
                status="blocked",
                source="observability gate",
                note="2s counter bracket cannot distinguish block ledger, victim choice, or admission state",
            ),
            _row(
                "decision",
                "step2b_required",
                True,
                target="only when Step2a real state is insufficient",
                status="triggered",
                source="observability gate",
                note=(
                    f"selected={selected['scenario']}; "
                    f"sim_over_real={float(selected['sim_over_real']):.6f}"
                ),
            ),
            _row(
                "decision",
                "runtime_fix_allowed",
                False,
                target="unique semantic verdict after real decision logging",
                status="blocked",
                source="Phase462 discipline",
                note="Step2c remains locked; no runtime, PerfDB, or gate change",
            ),
            _row(
                "decision",
                "default_aic",
                "No-Go",
                target="6/6 <= 1.15",
                status="No-Go",
                source="Phase461 scorecard",
            ),
        ]
    )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    by_section: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_section.setdefault(str(row["section"]), []).append(row)
    predictions = by_section["sign_prediction"]
    step2b = next(row for row in by_section["decision"] if row["metric"] == "step2b_required")
    sim_block = next(
        row
        for row in by_section["sim_first_preemption"]
        if row["metric"] == "scheduled_demand_minus_capacity_blocks"
    )
    sim_iter = next(row for row in by_section["sim_first_preemption"] if row["metric"] == "local_iter")
    real_bracket = next(row for row in by_section["real_first_preemption"] if row["metric"] == "counter_bracket_seconds")
    lines = [
        "# Phase462 Step2a: 抢占首次分歧离线审计",
        "",
        "结论：离线数据不足以裁定唯一抢占语义差，Step2c 继续锁住。sim 首次抢占可精确定位到 decode 增长使块账本超容量 1 块；真实侧只能把首次计数增长夹在约 2 秒 metrics 区间，缺 free blocks、申请块数、trigger/victim id 和 victim 队列位置。因此触发 Step2b logging-only 短跑，不允许从 KV usage 百分比反推决策点。Default AIC 维持 No-Go。",
        "",
        "## 首次分歧",
        "",
        "| 侧 | 可定位结果 | 精度 |",
        "|---|---|---|",
        f"| sim | local_iter={sim_iter['value']}，decode growth，超容量 {sim_block['value']} 块 | 决策点精确 hook |",
        f"| real | preemption counter 首次跃迁区间 {float(real_bracket['value']):.3f}s | scrape 区间，非决策时刻 |",
        "| 对比结论 | 只能确认两边都发生抢占，不能判断为何 real 少抢 | blocked |",
        "",
        "Phase458 已能排除 prefix reuse：该 session 明确 `enable_prefix_caching=False`；容量真值与 sim 同为 461,200 tokens / 28,825 blocks。剩余候选仍包括块增长账本、申请块口径、victim 选择和 waiting/running 时序，现有日志无法互相区分。",
        "",
        "## 符号预测",
        "",
        "抢占减少会减少 recompute 工作，静态一阶符号是 sim 吞吐上升。这里只判方向，不预测幅度。",
        "",
        "| 场景 | 当前 ratio | 方向 | 15% 风险 |",
        "|---|---:|---|---|",
    ]
    for row in predictions:
        note = dict(part.split("=", 1) for part in str(row["note"]).split("; "))
        lines.append(
            f"| {row['scenario']} | {float(note['current_ratio']):.3f} | {row['value']} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "关键反证：当前三个 fail 点的 sim 吞吐都已高于 real，单独减少抢占会让它们继续变差；抢占修复是语义正确性修复和步构成修复，不是这三个点的直接数值补丁。`tp8-8k2k` 目前虽达标，但 sim 只剩约 1.54% 上升空间就会越过 1.15，Step2c 必须按动态 `--ab` 判卷。",
            "",
            "## Step2b 最小观测",
            "",
            f"选择 `{str(step2b['note']).split(';')[0].split('=', 1)[1]}`：它的 sim/real 抢占比最高。只在 allocate 失败/抢占决策点记录以下字段：decision timestamp、free blocks before、requested blocks、trigger request id、victim request id、victim queue position。logging-only、diagnostic-only、运行行为不变，开销门仍为 2%。",
            "",
            "本步未使用 GPU，未改 runtime、PerfDB 或 gate。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--serve-log", type=Path, default=DEFAULT_SERVE_LOG)
    parser.add_argument("--ab", type=Path, default=DEFAULT_AB)
    parser.add_argument("--panorama", type=Path, default=DEFAULT_PANORAMA)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    rows = build_report_rows(
        metrics_path=args.metrics,
        serve_log=args.serve_log,
        ab_path=args.ab,
        panorama_path=args.panorama,
    )
    write_csv(args.csv, rows)
    write_markdown(args.md, rows)
    print(f"wrote {args.csv}")
    print(f"wrote {args.md}")


if __name__ == "__main__":
    main()
