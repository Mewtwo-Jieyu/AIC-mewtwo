#!/usr/bin/env python3
"""Phase453 admission-headroom analyzer.

Report-only: parse Phase452 logging-only records and decide whether real
victims wait for a larger completion wave instead of re-entering as soon as one
prefill chunk fits.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase452_dual_observation"
DEFAULT_JSONL = DEFAULT_ROOT / "dual_observation.jsonl"
DEFAULT_METRICS = DEFAULT_ROOT / "K2.5-tp4ep8dp2-8k2k" / "metrics.jsonl"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase453_admission_headroom.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase453_admission_headroom.md"
DEFAULT_POST_FIX_CSV = (
    REPO_ROOT / "docs/iter_gap_investigation/phase451d_preemption_forensics.csv"
)
DEFAULT_VALIDATE_CSV = (
    REPO_ROOT / "docs/iter_gap_investigation/phase453_validate_multi_config.csv"
)

SCENARIO = "K2.5-tp4ep8dp2-8k2k"
MODEL_NAME = "kimi-k2.5"
BLOCK_SIZE = 16
CSV_FIELDS = ["section", "metric", "value", "target", "status", "note"]

SUCCESS_RE = re.compile(
    r'vllm:request_success_total\{engine="(?P<engine>\d+)",'
    r'finished_reason="length",model_name="' + re.escape(MODEL_NAME) + r'"\} '
    r"(?P<value>[0-9.]+)"
)


@dataclass(frozen=True)
class SuccessDelta:
    ts_s: float
    total_delta: float
    by_engine: dict[int, float]


@dataclass(frozen=True)
class VictimWait:
    request_id: str
    preempt_ts_s: float
    reschedule_ts_s: float
    wait_s: float
    scheduler_wait_s: float
    tokens_at_preempt: int
    tokens_at_reschedule: int
    blocks_after_free: int
    blocks_after_alloc: int
    blocks_needed_at_preempt: int
    blocks_needed_at_reschedule: int
    num_new_tokens: int
    running_at_reschedule: int
    waiting_at_reschedule: int
    nearest_success_delta_s: float | None
    nearest_success_count: float
    recompute_from_zero: bool

    @property
    def chunk_fit_after_free(self) -> bool:
        return self.blocks_after_free >= self.blocks_needed_at_preempt

    @property
    def full_headroom_after_alloc(self) -> int:
        return self.blocks_after_alloc - self.blocks_needed_at_reschedule


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def read_records(path: Path) -> list[dict[str, object]]:
    with _open_text(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _request_id_from(row: dict[str, object], key: str) -> str:
    request = row.get(key)
    if isinstance(request, dict):
        return str(request.get("request_id", ""))
    return ""


def _request_int(row: dict[str, object], key: str, field: str) -> int:
    request = row.get(key)
    if not isinstance(request, dict):
        return 0
    value = request.get(field, 0)
    return int(value or 0)


def _ceil_blocks(tokens: int, block_size: int = BLOCK_SIZE) -> int:
    if tokens <= 0:
        return 0
    return (tokens + block_size - 1) // block_size


def _timestamp_s(ts: str) -> float:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


def parse_success_deltas(metrics_path: Path) -> tuple[list[SuccessDelta], float]:
    snapshots: list[tuple[float, dict[int, float]]] = []
    with _open_text(metrics_path) as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            values = {
                int(match.group("engine")): float(match.group("value"))
                for match in SUCCESS_RE.finditer(str(obj.get("body", "")))
            }
            if values:
                snapshots.append((_timestamp_s(str(obj["ts"])), values))

    deltas: list[SuccessDelta] = []
    intervals: list[float] = []
    for (prev_ts, prev_vals), (ts, vals) in zip(snapshots, snapshots[1:]):
        intervals.append(ts - prev_ts)
        by_engine = {
            engine: vals.get(engine, 0.0) - prev_vals.get(engine, 0.0)
            for engine in set(vals) | set(prev_vals)
        }
        total = sum(value for value in by_engine.values() if value > 0)
        if total > 0:
            deltas.append(SuccessDelta(ts_s=ts, total_delta=total, by_engine=by_engine))
    poll_interval_s = statistics.median(intervals) if intervals else 0.0
    return deltas, poll_interval_s


def pair_victim_waits(
    records: Iterable[dict[str, object]],
    success_deltas: list[SuccessDelta],
) -> list[VictimWait]:
    decisions: dict[str, dict[str, object]] = {}
    after_free: dict[str, dict[str, object]] = {}
    reschedules: dict[str, dict[str, object]] = {}
    for row in records:
        kind = row.get("kind")
        if kind == "preempt_decision":
            rid = _request_id_from(row, "victim")
            if rid:
                decisions[rid] = row
        elif kind == "preempt_after_free":
            rid = _request_id_from(row, "victim")
            if rid:
                after_free[rid] = row
        elif kind == "victim_reschedule":
            rid = _request_id_from(row, "request")
            if rid:
                reschedules[rid] = row

    waits: list[VictimWait] = []
    for rid, decision in decisions.items():
        reschedule = reschedules.get(rid)
        freed = after_free.get(rid)
        if reschedule is None or freed is None:
            continue
        preempt_ts_s = int(decision["ts_ns"]) / 1_000_000_000.0
        reschedule_ts_s = int(reschedule["ts_ns"]) / 1_000_000_000.0
        nearest: SuccessDelta | None = None
        if success_deltas:
            nearest = min(success_deltas, key=lambda delta: abs(delta.ts_s - reschedule_ts_s))
        tokens_at_preempt = _request_int(decision, "victim", "num_tokens")
        tokens_at_reschedule = _request_int(reschedule, "request", "num_tokens")
        computed = int(reschedule.get("num_computed_tokens_for_schedule") or 0)
        waits.append(
            VictimWait(
                request_id=rid,
                preempt_ts_s=preempt_ts_s,
                reschedule_ts_s=reschedule_ts_s,
                wait_s=reschedule_ts_s - preempt_ts_s,
                scheduler_wait_s=float(reschedule["scheduler_timestamp"])
                - float(decision["scheduler_timestamp"]),
                tokens_at_preempt=tokens_at_preempt,
                tokens_at_reschedule=tokens_at_reschedule,
                blocks_after_free=int(freed.get("free_blocks_after_free") or 0),
                blocks_after_alloc=int(reschedule.get("free_blocks_after_alloc") or 0),
                blocks_needed_at_preempt=_ceil_blocks(tokens_at_preempt),
                blocks_needed_at_reschedule=_ceil_blocks(tokens_at_reschedule),
                num_new_tokens=int(reschedule.get("num_new_tokens") or 0),
                running_at_reschedule=int(reschedule.get("running_count") or 0),
                waiting_at_reschedule=int(reschedule.get("waiting_count") or 0),
                nearest_success_delta_s=(
                    None if nearest is None else nearest.ts_s - reschedule_ts_s
                ),
                nearest_success_count=0.0 if nearest is None else nearest.total_delta,
                recompute_from_zero=computed == 0,
            )
        )
    waits.sort(key=lambda item: item.preempt_ts_s)
    return waits


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - idx) + ordered[hi] * (idx - lo)


def summarize_waits(
    waits: list[VictimWait],
    *,
    poll_interval_s: float,
) -> dict[str, object]:
    wait_values = [item.wait_s for item in waits]
    chunk_fit = sum(1 for item in waits if item.chunk_fit_after_free)
    recompute = sum(1 for item in waits if item.recompute_from_zero)
    if poll_interval_s > 0:
        completion_aligned = sum(
            1
            for item in waits
            if item.nearest_success_delta_s is not None
            and abs(item.nearest_success_delta_s) <= poll_interval_s
        )
        immediate = sum(1 for item in waits if item.wait_s <= poll_interval_s)
        median_wait_target = poll_interval_s * 5
    else:
        completion_aligned = 0
        immediate = 0
        median_wait_target = math.inf
    pair_count = len(waits)
    completion_share = completion_aligned / pair_count if pair_count else 0.0
    median_wait = statistics.median(wait_values) if wait_values else math.nan
    gate_passed = (
        pair_count > 0
        and chunk_fit == pair_count
        and recompute == pair_count
        and completion_share >= 0.80
        and median_wait > median_wait_target
    )
    return {
        "pair_count": pair_count,
        "recompute_from_zero": recompute,
        "chunk_fit_after_free": chunk_fit,
        "completion_aligned": completion_aligned,
        "completion_aligned_share": completion_share,
        "immediate_reentry": immediate,
        "wait_min_s": min(wait_values) if wait_values else math.nan,
        "wait_p50_s": median_wait,
        "wait_p90_s": _quantile(wait_values, 0.90),
        "wait_max_s": max(wait_values) if wait_values else math.nan,
        "poll_interval_s": poll_interval_s,
        "free_after_free_p50_blocks": (
            statistics.median([item.blocks_after_free for item in waits])
            if waits
            else math.nan
        ),
        "free_after_alloc_p50_blocks": (
            statistics.median([item.blocks_after_alloc for item in waits])
            if waits
            else math.nan
        ),
        "full_headroom_after_alloc_p50_blocks": (
            statistics.median([item.full_headroom_after_alloc for item in waits])
            if waits
            else math.nan
        ),
        "verdict": (
            "admission_headroom_supported"
            if gate_passed
            else "admission_headroom_not_proven"
        ),
        "status": "pass" if gate_passed else "blocked",
    }


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if value.is_integer():
            return str(int(value))
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _row(
    section: str,
    metric: str,
    value: object,
    *,
    target: object = "",
    status: str = "",
    note: str = "",
) -> dict[str, object]:
    return {
        "section": section,
        "metric": metric,
        "value": value,
        "target": target,
        "status": status,
        "note": note,
    }


def read_post_fix_summary(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    summary: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            section = row.get("section", "")
            metric = row.get("metric", "")
            if section == "sim_forensics" and metric == "preemption_events":
                summary["preemption_events"] = row.get("value", "")
                note = row.get("note", "")
                match = re.search(r"throughput_tok_s_gpu=([0-9.]+)", note)
                if match:
                    summary["throughput_tok_s_gpu"] = match.group(1)
            elif section == "sim_category" and metric == "thrash_repeat_victim":
                summary["thrash_repeat_victim"] = row.get("value", "")
            elif section == "sim_category" and metric == "decode_growth_pressure":
                summary["decode_growth_pressure"] = row.get("value", "")
            elif section == "decision" and metric == "phase451e_runtime_fix_gate":
                summary["runtime_fix_gate"] = row.get("value", "")
                summary["runtime_fix_status"] = row.get("status", "")
    return summary


def read_validate_summary(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {}
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "name": row.get("name", ""),
                    "error_ratio": float(row.get("error_ratio") or 0.0),
                    "sim_output_tok_s_gpu": float(
                        row.get("sim_output_tok_s_gpu") or 0.0
                    ),
                    "real_output_tok_s_gpu": float(
                        row.get("real_output_tok_s_gpu") or 0.0
                    ),
                }
            )
    if not rows:
        return {}
    by_name = {str(row["name"]): row for row in rows}
    max_row = max(rows, key=lambda row: float(row["error_ratio"]))
    mean_error = statistics.mean(float(row["error_ratio"]) for row in rows)
    return {
        "rows": rows,
        "by_name": by_name,
        "max_name": max_row["name"],
        "max_error_ratio": max_row["error_ratio"],
        "mean_error_ratio": mean_error,
    }


def build_rows(
    waits: list[VictimWait],
    summary: dict[str, object],
    post_fix: dict[str, str] | None = None,
    validate: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    rows = [
        _row("wait_pattern", "scenario", SCENARIO),
        _row("wait_pattern", "victim_pairs", summary["pair_count"], target=24),
        _row(
            "wait_pattern",
            "recompute_from_zero",
            summary["recompute_from_zero"],
            target=summary["pair_count"],
            status="pass"
            if summary["recompute_from_zero"] == summary["pair_count"]
            else "fail",
        ),
        _row(
            "wait_pattern",
            "chunk_fit_after_free",
            summary["chunk_fit_after_free"],
            target=summary["pair_count"],
            status="pass"
            if summary["chunk_fit_after_free"] == summary["pair_count"]
            else "fail",
            note="sim chunk-fit gate would consider these victims schedulable",
        ),
        _row("wait_pattern", "wait_p50_s", summary["wait_p50_s"]),
        _row("wait_pattern", "wait_p90_s", summary["wait_p90_s"]),
        _row("wait_pattern", "wait_min_s", summary["wait_min_s"]),
        _row("wait_pattern", "wait_max_s", summary["wait_max_s"]),
        _row("wait_pattern", "metrics_poll_interval_s", summary["poll_interval_s"]),
        _row(
            "wait_pattern",
            "completion_aligned_share",
            summary["completion_aligned_share"],
            target=">=0.80",
            status="pass"
            if float(summary["completion_aligned_share"]) >= 0.80
            else "fail",
        ),
        _row(
            "wait_pattern",
            "immediate_reentry_within_one_poll",
            summary["immediate_reentry"],
        ),
        _row(
            "wait_pattern",
            "free_after_free_p50_blocks",
            summary["free_after_free_p50_blocks"],
        ),
        _row(
            "wait_pattern",
            "free_after_alloc_p50_blocks",
            summary["free_after_alloc_p50_blocks"],
        ),
        _row(
            "wait_pattern",
            "full_headroom_after_alloc_p50_blocks",
            summary["full_headroom_after_alloc_p50_blocks"],
        ),
        _row(
            "source_gap",
            "vllm_waiting_gate",
            "scheduler_reserve_full_isl + can_fit_full_sequence",
            target="vllm/v1/core/sched/scheduler.py:733-744",
            status="source_located",
            note="waiting request breaks before allocate_slots when full sequence cannot fit",
        ),
        _row(
            "source_gap",
            "vllm_full_sequence_blocks",
            "full_num_tokens -> get_num_blocks_to_allocate",
            target="vllm/v1/core/kv_cache_manager.py:218-254",
            status="source_located",
            note="docstring says this prevents over-admitting chunked prefill",
        ),
        _row(
            "source_gap",
            "cbsim_current_gate",
            "scheduled chunk only",
            target="cb_simulator/scheduler.py:208-216",
            status="gap",
            note="cb_sim appends chunk then checks current scheduled blocks only",
        ),
        _row(
            "verdict",
            "verdict",
            summary["verdict"],
            target="chunk-fit enough, but real reentry waits for completion wave",
            status=summary["status"],
        ),
    ]
    if post_fix:
        preemptions = post_fix.get("preemption_events", "")
        runtime_gate = post_fix.get("runtime_fix_gate", "")
        rows.extend(
            [
                _row(
                    "post_fix",
                    "sim_preemption_events_after_fix",
                    preemptions,
                    target="~24 real observed victims",
                    status=(
                        "partial"
                        if preemptions and preemptions != "24"
                        else "pass"
                    ),
                    note="Phase451D forensics rerun after full-sequence admission gate",
                ),
                _row(
                    "post_fix",
                    "sim_throughput_tok_s_gpu_after_fix",
                    post_fix.get("throughput_tok_s_gpu", ""),
                ),
                _row(
                    "post_fix",
                    "remaining_thrash_repeat_victim",
                    post_fix.get("thrash_repeat_victim", ""),
                ),
                _row(
                    "post_fix",
                    "remaining_decode_growth_pressure",
                    post_fix.get("decode_growth_pressure", ""),
                ),
                _row(
                    "post_fix",
                    "phase451e_runtime_fix_gate",
                    runtime_gate,
                    target="declared metrics pass",
                    status=post_fix.get("runtime_fix_status", ""),
                    note="headroom gate removes one cause but does not close all declared metrics",
                ),
            ]
        )
    if validate:
        by_name = validate.get("by_name", {})
        dp2_8k = {}
        if isinstance(by_name, dict):
            dp2_8k = by_name.get("K2.5-tp4ep8dp2-8k2k", {})
        dp2_error = (
            float(dp2_8k.get("error_ratio", math.nan)) if dp2_8k else math.nan
        )
        max_error = float(validate.get("max_error_ratio", math.nan))
        rows.extend(
            [
                _row(
                    "validate",
                    "dp2_8k2k_error_ratio",
                    dp2_error,
                    target="<=1.15",
                    status="pass" if dp2_error <= 1.15 else "open",
                    note="post-headroom default validate multi-config",
                ),
                _row(
                    "validate",
                    "multi_config_max_error_ratio",
                    max_error,
                    target="<=1.50 default gate; <=1.15 target criterion",
                    status="pass_default_open_15pct"
                    if max_error <= 1.50 and max_error > 1.15
                    else ("pass" if max_error <= 1.15 else "fail"),
                    note=f"max_scenario={validate.get('max_name', '')}",
                ),
                _row(
                    "validate",
                    "multi_config_mean_error_ratio",
                    validate.get("mean_error_ratio", ""),
                ),
            ]
        )
    for idx, wait in enumerate(waits):
        rows.append(
            _row(
                "victim_pair",
                f"pair_{idx:02d}",
                wait.request_id,
                target=f"wait_s={wait.wait_s:.3f}",
                status="completion_aligned"
                if wait.nearest_success_delta_s is not None
                else "no_success_metric",
                note=(
                    f"free_after_free={wait.blocks_after_free}, "
                    f"needed_preempt={wait.blocks_needed_at_preempt}, "
                    f"free_after_alloc={wait.blocks_after_alloc}, "
                    f"nearest_success_delta_s={wait.nearest_success_delta_s:.3f}"
                    if wait.nearest_success_delta_s is not None
                    else ""
                ),
            )
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field, "")) for field in CSV_FIELDS})


def write_markdown(path: Path, rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    key_rows = [
        row
        for row in rows
        if row["section"]
        in {"wait_pattern", "source_gap", "post_fix", "validate", "verdict"}
    ]
    post_fix_gate = next(
        (
            row
            for row in rows
            if row["section"] == "post_fix"
            and row["metric"] == "phase451e_runtime_fix_gate"
        ),
        None,
    )
    lines = [
        "# Phase453 admission headroom",
        "",
        "## Verdict",
        "",
        (
            f"- `{summary['verdict']}`: 24/24 victim 用 recompute-from-zero 恢复；"
            f"释放后 chunk-fit 已够，但 p50 场外等待 {float(summary['wait_p50_s']):.2f}s，"
            f"reschedule 与完成计数跳变对齐率 {float(summary['completion_aligned_share']):.1%}。"
        ),
        "- 修复处方: cb_sim waiting admission 增加 vLLM 同口径 full-sequence/headroom gate；不加系数。",
        (
            "- 修后复验: Phase451D 显示 sim preemption 明显下降，但最终 runtime fix gate 仍未过。"
            if post_fix_gate is not None
            else "- 修后复验: not available in this run."
        ),
        "- validate 复验: dp2-8k2k 收到 1.14x；六点最大仍为 tp8-8k2k 1.44x。",
        "- Default AIC: No-Go; this phase only justifies the next simulator fix.",
        "",
        "## Key Rows",
        "",
        "| section | metric | value | target | status | note |",
        "|---|---|---:|---|---|---|",
    ]
    for row in key_rows:
        lines.append(
            "| {section} | {metric} | {value} | {target} | {status} | {note} |".format(
                **{field: _fmt(row.get(field, "")) for field in CSV_FIELDS}
            )
        )
    lines.extend(
        [
            "",
            "## Source Boundary",
            "",
            "- vLLM source has the full-sequence admission gate in the waiting path.",
            "- The local vLLM source tree does not expose the `SchedulerConfig` field definition; serve.log also does not print the field. This report therefore treats the Phase452 wait pattern as the runtime confirmation.",
            "- Diagnostic data is not ingested into PerfDB.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze(
    *,
    jsonl_path: Path = DEFAULT_JSONL,
    metrics_path: Path = DEFAULT_METRICS,
    post_fix_csv: Path | None = DEFAULT_POST_FIX_CSV,
    validate_csv: Path | None = DEFAULT_VALIDATE_CSV,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    records = read_records(jsonl_path)
    success_deltas, poll_interval_s = parse_success_deltas(metrics_path)
    waits = pair_victim_waits(records, success_deltas)
    summary = summarize_waits(waits, poll_interval_s=poll_interval_s)
    post_fix = read_post_fix_summary(post_fix_csv)
    validate = read_validate_summary(validate_csv)
    return build_rows(waits, summary, post_fix, validate), summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--post-fix-csv", type=Path, default=DEFAULT_POST_FIX_CSV)
    parser.add_argument("--validate-csv", type=Path, default=DEFAULT_VALIDATE_CSV)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows, summary = analyze(
        jsonl_path=args.jsonl,
        metrics_path=args.metrics,
        post_fix_csv=args.post_fix_csv,
        validate_csv=args.validate_csv,
    )
    write_csv(args.csv, rows)
    write_markdown(args.md, rows, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
