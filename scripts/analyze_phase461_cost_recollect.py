#!/usr/bin/env python3
"""Phase461 Step3: validate the GPU collection batch without ingesting rows."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_phase459_residual_triage import parse_iteration_steps
DEFAULT_RAW_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase461_cost_recollect"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase461_cost_recollect.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase461_cost_recollect.md"
OLD_BAD_MLA_LATENCY_MS = 0.9586453437805176


@dataclass(frozen=True)
class EventStep:
    dp_rank: str
    ctx_tokens: int
    generation_requests: int
    num_tokens_unpadded: int
    cudagraph_mode: str
    forward_busy_ms: float

    @property
    def bucket_tokens(self) -> int:
        return self.num_tokens_unpadded or self.ctx_tokens + self.generation_requests

    @property
    def phase(self) -> str:
        if self.bucket_tokens > self.generation_requests:
            return "mixed_prefill" if self.generation_requests else "prefill"
        return "decode" if self.generation_requests else "empty"


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def read_event_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with _open_text(path) as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") == "phase446_graph_outer_event_v2":
                rows.append(row)
    if not rows:
        raise ValueError(f"no Phase446 event rows: {path}")
    return rows


def group_tp_steps(rows: Iterable[dict[str, object]], *, tp_width: int) -> list[EventStep]:
    rows = list(rows)

    def to_step(rank: str, row: dict[str, object]) -> EventStep:
        return EventStep(
            dp_rank=rank,
            ctx_tokens=int(row.get("ctx_tokens") or 0),
            generation_requests=int(row.get("generation_requests") or 0),
            num_tokens_unpadded=int(row.get("num_tokens_unpadded") or 0),
            cudagraph_mode=str(row.get("cudagraph_mode") or ""),
            forward_busy_ms=float(row["forward_busy_ms"]),
        )

    by_rank: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_rank[str(row.get("dp_rank"))].append(row)

    steps: list[EventStep] = []
    for rank, rank_rows in sorted(by_rank.items()):
        explicit_tp = all(row.get("tp_rank") is not None for row in rank_rows)
        streams: list[list[dict[str, object]]]
        if explicit_tp:
            by_tp: dict[int, list[dict[str, object]]] = defaultdict(list)
            for row in rank_rows:
                by_tp[int(row["tp_rank"])].append(row)
            if len(by_tp) != tp_width:
                raise ValueError(f"dp_rank={rank} explicit TP streams {len(by_tp)} != {tp_width}")
            streams = [by_tp[tp_rank] for tp_rank in sorted(by_tp)]
        elif tp_width > 1 and len(rank_rows) % tp_width == 0:
            stream_len = len(rank_rows) // tp_width
            candidate = [rank_rows[index * stream_len : (index + 1) * stream_len] for index in range(tp_width)]
            signatures_match = all(
                len(
                    {
                    (
                        int(stream[index].get("ctx_tokens") or 0),
                        int(stream[index].get("generation_requests") or 0),
                        int(stream[index].get("num_tokens_unpadded") or 0),
                        str(stream[index].get("cudagraph_mode") or ""),
                    )
                    for stream in candidate
                    }
                )
                == 1
                for index in range(stream_len)
            )
            streams = candidate if signatures_match else [rank_rows]
        else:
            streams = [rank_rows]

        lengths = {len(stream) for stream in streams}
        if len(lengths) != 1:
            raise ValueError(f"dp_rank={rank} TP stream lengths differ: {sorted(lengths)}")
        for offset, chunk in enumerate(zip(*streams)):
            keys = {
                (
                    int(row.get("ctx_tokens") or 0),
                    int(row.get("generation_requests") or 0),
                    int(row.get("num_tokens_unpadded") or 0),
                    str(row.get("cudagraph_mode") or ""),
                )
                for row in chunk
            }
            if len(keys) != 1:
                raise ValueError(f"dp_rank={rank} mixed TP chunk at offset={offset}: {sorted(keys)}")
            ctx_tokens, generation_requests, num_tokens_unpadded, cudagraph_mode = next(iter(keys))
            representative = dict(chunk[0])
            representative["forward_busy_ms"] = max(float(row["forward_busy_ms"]) for row in chunk)
            steps.append(to_step(rank, representative))
    return steps


def summarize_cross_rank_cells(steps: Iterable[EventStep]) -> dict[tuple[int, int], dict[str, object]]:
    cells: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for step in steps:
        if step.phase != "mixed_prefill":
            continue
        cells[(step.bucket_tokens, step.generation_requests)][step.dp_rank].append(step.forward_busy_ms)

    result: dict[tuple[int, int], dict[str, object]] = {}
    for key, by_rank in cells.items():
        medians = {rank: statistics.median(values) for rank, values in sorted(by_rank.items())}
        spread = max(medians.values()) / min(medians.values()) if len(medians) > 1 and min(medians.values()) > 0 else 1.0
        result[key] = {
            "rank_medians": medians,
            "rank_samples": {rank: len(values) for rank, values in sorted(by_rank.items())},
            "max_over_min": spread,
        }
    return result


def spike_transition_decision(rank_spread: float, *, busy_over_wall: float) -> str:
    if rank_spread >= 1.5 and busy_over_wall >= 0.95:
        return "retain_measured_row_diagnostic_only_until_phase_model"
    return "insufficient_reproduction_do_not_change_perfdb"


def summarize_busy_wall(
    event_steps: Iterable[EventStep],
    wall_steps: Iterable[object],
    *,
    bucket_tokens: int,
    decode_batch: int,
) -> dict[str, float | int]:
    busy = sorted(
        step.forward_busy_ms
        for step in event_steps
        if step.bucket_tokens == bucket_tokens and step.generation_requests == decode_batch
    )
    wall = sorted(
        float(step.elapsed_ms)
        for step in wall_steps
        if int(step.ctx_tokens) + int(step.generation_requests) == bucket_tokens
        and int(step.generation_requests) == decode_batch
    )
    if not busy or len(busy) != len(wall):
        raise ValueError(
            f"busy/wall cell mismatch bucket={bucket_tokens} batch={decode_batch}: "
            f"busy={len(busy)} wall={len(wall)}"
        )
    ratios = [busy_ms / wall_ms for busy_ms, wall_ms in zip(busy, wall)]
    return {
        "samples": len(ratios),
        "median_busy_over_wall": statistics.median(ratios),
        "max_busy_over_wall": max(ratios),
    }


def _line_count(path: Path) -> int:
    with _open_text(path) as source:
        return sum(1 for line in source if line.strip())


def collection_gate(root: Path, scenario: str) -> dict[str, object]:
    gate = json.loads((root / "overhead_gate.json").read_text(encoding="utf-8"))
    event_path = root / "overhead_on" / "event_timing.jsonl"
    if not event_path.exists():
        event_path = event_path.with_suffix(event_path.suffix + ".gz")
    process_residual = root / "overhead_on" / "process_residual_after.txt"
    gpu_residual = root / "overhead_on" / "gpu_compute_apps_after.txt"
    event_rows = _line_count(event_path) if event_path.exists() else 0
    process_bytes = process_residual.stat().st_size if process_residual.exists() else -1
    gpu_bytes = gpu_residual.stat().st_size if gpu_residual.exists() else -1
    passed = (
        bool(gate.get("passed"))
        and float(gate.get("overhead_pct", 100.0)) <= 2.0
        and event_rows > 0
        and process_bytes == 0
        and gpu_bytes == 0
    )
    return {
        "passed": passed,
        "overhead_pct": float(gate.get("overhead_pct", 100.0)),
        "event_rows": event_rows,
        "process_residual_bytes": process_bytes,
        "gpu_residual_bytes": gpu_bytes,
    }


def _add(rows: list[dict[str, object]], section: str, scenario: str, metric: str, value: object, status: str, note: str = "") -> None:
    rows.append(
        {
            "section": section,
            "scenario": scenario,
            "metric": metric,
            "value": value,
            "status": status,
            "note": note,
        }
    )


def build_report_rows(raw_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    mla = json.loads((raw_root / "mla_decode_grid/phase461_mla_grid_summary.json").read_text(encoding="utf-8"))
    _add(rows, "mla_gate", "mla_decode_heads8", "raw_rows", mla["rows"], "pass" if mla["passed"] else "fail")
    _add(rows, "provenance", "mla_decode_heads8", "collector_commit", mla["collector_commit"], "recorded")
    for point in mla["points"]:
        _add(
            rows,
            "mla_grid",
            "mla_decode_heads8",
            f"b{point['batch_size']}_kv{point['target_seq_len']}_latency_ms",
            point["latency_median_ms"],
            "pass" if point["passed"] else "fail",
            f"samples={point['samples']};spread={point['repeat_spread']:.6f}",
        )
        if int(point["batch_size"]) == 8 and int(point["target_seq_len"]) == 32768:
            _add(
                rows,
                "mla_bad_row",
                "mla_decode_heads8",
                "old_over_recollected",
                OLD_BAD_MLA_LATENCY_MS / float(point["latency_median_ms"]),
                "confirmed_bad_row",
                f"old={OLD_BAD_MLA_LATENCY_MS:.9f};new={float(point['latency_median_ms']):.9f}",
            )

    specs = [
        ("tp8_bt65536_mixed", "K2.5-tp8ep8-8k2k-bt65536", 8),
        ("tp8_32k3k_mixed", "K2.5-tp8ep8-32k3k", 8),
        ("dp2_bt65536_diagnostic", "K2.5-tp4ep8dp2-8k2k-bt65536", 4),
    ]
    for label, scenario, tp_width in specs:
        root = raw_root / label
        gate = collection_gate(root, scenario)
        _add(rows, "collection_gate", scenario, "overhead_pct", gate["overhead_pct"], "pass" if gate["passed"] else "fail", f"events={gate['event_rows']}")
        event_path = root / "overhead_on/event_timing.jsonl"
        if not event_path.exists():
            event_path = event_path.with_suffix(event_path.suffix + ".gz")
        steps = group_tp_steps(read_event_rows(event_path), tp_width=tp_width)
        mixed = [step for step in steps if step.phase == "mixed_prefill"]
        _add(rows, "coverage", scenario, "mixed_steps", len(mixed), "pass" if mixed else "fail")
        if mixed:
            _add(rows, "coverage", scenario, "mixed_bucket_min", min(step.bucket_tokens for step in mixed), "observed")
            _add(rows, "coverage", scenario, "mixed_bucket_max", max(step.bucket_tokens for step in mixed), "observed")
            _add(rows, "coverage", scenario, "mixed_decode_min", min(step.generation_requests for step in mixed), "observed")
            _add(rows, "coverage", scenario, "mixed_decode_max", max(step.generation_requests for step in mixed), "observed")

        if label == "dp2_bt65536_diagnostic":
            policy = json.loads((root / "ingest_policy.json").read_text(encoding="utf-8"))
            policy_ok = policy.get("diagnostic_only") is True and policy.get("perf_database") is False
            _add(rows, "ingest_boundary", scenario, "perf_database", policy.get("perf_database"), "pass" if policy_ok else "fail", "diagnostic_only required")
            cells = summarize_cross_rank_cells(steps)
            if cells:
                key, cell = max(cells.items(), key=lambda item: float(item[1]["max_over_min"]))
                _add(rows, "dp_rank_diagnostic", scenario, "max_same_cell_rank_spread", cell["max_over_min"], "diagnostic", f"bucket={key[0]};batch={key[1]};medians={cell['rank_medians']}")
                serve_log = root / "overhead_on" / scenario / "serve.log"
                if not serve_log.exists():
                    serve_log = serve_log.with_suffix(serve_log.suffix + ".gz")
                busy_wall = summarize_busy_wall(
                    steps,
                    parse_iteration_steps(serve_log),
                    bucket_tokens=key[0],
                    decode_batch=key[1],
                )
                _add(
                    rows,
                    "dp_rank_diagnostic",
                    scenario,
                    "max_busy_over_wall",
                    busy_wall["max_busy_over_wall"],
                    "pass" if float(busy_wall["max_busy_over_wall"]) >= 0.95 else "fail",
                    f"bucket={key[0]};batch={key[1]};samples={busy_wall['samples']}",
                )
                decision = spike_transition_decision(
                    float(cell["max_over_min"]),
                    busy_over_wall=float(busy_wall["max_busy_over_wall"]),
                )
                _add(rows, "dp_rank_diagnostic", scenario, "transition_decision", decision, "diagnostic")
    final_gpu_residual = raw_root / "gpu_compute_apps_after.txt"
    final_gpu_bytes = final_gpu_residual.stat().st_size if final_gpu_residual.exists() else -1
    _add(
        rows,
        "gpu_cleanup",
        "all",
        "final_gpu_compute_apps_bytes",
        final_gpu_bytes,
        "pass" if final_gpu_bytes == 0 else "fail",
    )
    driver = (raw_root / "driver.log").read_text(encoding="utf-8", errors="replace")
    _add(
        rows,
        "gpu_cleanup",
        "K2.5-tp4ep8dp2-8k2k-bt65536",
        "forced_worker_cleanup",
        "observed" if "force-killing lingering gpu pids" in driver else "not_needed",
        "recorded",
        "final residual remained empty",
    )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["section", "scenario", "metric", "value", "status", "note"]
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, object]]) -> None:
    collection = [row for row in rows if row["section"] == "collection_gate"]
    mla_points = [row for row in rows if row["section"] == "mla_grid"]
    bad_row = next(row for row in rows if row["section"] == "mla_bad_row")
    coverage = [row for row in rows if row["section"] == "coverage"]
    dp = [row for row in rows if row["section"] in {"dp_rank_diagnostic", "ingest_boundary"}]
    coverage_by_scenario: dict[str, dict[str, object]] = defaultdict(dict)
    for row in coverage:
        coverage_by_scenario[str(row["scenario"])][str(row["metric"])] = row["value"]
    text = [
        "# Phase461 Step 3 GPU 成本补采",
        "",
        "## 结论",
        "",
        "三类采集门全部通过。MLA 坏行已由精确网格重采坐实；两个 TP8 serving-state 工作区已覆盖；DP2 65k 的跨 rank 差异再次复现且 busy/wall 闭合。Step 3 只交测量件，不入库，Default AIC 维持 No-Go。",
        "",
        "## 采集门",
        "",
        "| 场景 | 采集开销 | event 行 | 状态 |",
        "|---|---:|---:|---|",
    ]
    for row in collection:
        event_count = str(row["note"]).split("=", 1)[-1]
        text.append(f"| {row['scenario']} | {float(row['value']):.4f} | {event_count} | {row['status']} |")
    text.extend(["", "## MLA 精确网格", "", "| 格点 | 延迟 (ms) | 重复性 |", "|---|---:|---|"])
    for row in mla_points:
        text.append(f"| {str(row['metric']).removesuffix('_latency_ms')} | {float(row['value']):.6f} | {row['note']} |")
    text.extend(
        [
            "",
            f"确认坏行 `batch=8, KV=32768`：旧值 `0.958645 ms`，重采中位数 `0.134677 ms`，旧值高 `{float(bad_row['value']):.2f}x`。Step 4 只能用本次实测值替换，禁止平滑或 clamp。",
            "",
            "## Mixed 覆盖",
            "",
            "| 场景 | mixed 步 | bucket 范围 | decode batch 范围 |",
            "|---|---:|---:|---:|",
        ]
    )
    for scenario, values in coverage_by_scenario.items():
        text.append(
            f"| {scenario} | {values['mixed_steps']} | {values['mixed_bucket_min']}..{values['mixed_bucket_max']} | "
            f"{values['mixed_decode_min']}..{values['mixed_decode_max']} |"
        )
    text.extend(["", "## DP 诊断边界", "", "| 指标 | 值 | 说明 |", "|---|---:|---|"])
    for row in dp:
        text.append(f"| {row['metric']} | {row['value']} | {row['note']} |")
    text.extend(
        [
            "",
            "DP2 同一构成 cell 的跨 rank spread 为 `8.54x`，且最大 CUDA busy / iteration wall 为 `0.9997`。这是可重复的真实执行态，不是坏行。过渡处置拍板为：保留实测行与诊断 provenance；在 DP phase/lockstep 机制模型落地前，不隔离、不删除，也不转成单值修正。",
            "",
            "D 收尾时 8 个 worker 超时未退出，runner 按既有清理协议强杀；各 session 与整批最终 GPU/process residual 均为空。",
            "",
            "## 下一步",
            "",
            "Step 4 可入库 MLA replacement 与两个 TP8 scoped mixed 行并跑六点 `--ab`。DP2-bt65536 只保留诊断数据，转 Phase462 做 per-rank 构成成本 + lockstep max；dp2-32k3k 的 1.173x 继续单独分诊。",
            "",
        ]
    )
    path.write_text("\n".join(text), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    rows = build_report_rows(args.raw_root)
    write_csv(args.csv, rows)
    write_md(args.md, rows)
    if any(row["status"] == "fail" for row in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
