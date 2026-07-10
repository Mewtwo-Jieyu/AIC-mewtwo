#!/usr/bin/env python3
"""Phase461 Step 0: report-only H200 vLLM 0.19 PerfDB anomaly scan."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import NamedTuple


REPO_ROOT = Path(__file__).resolve().parents[1]
PERFDB_ROOT = REPO_ROOT / "src/aiconfigurator/systems/data/h200_sxm/vllm/0.19.0"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase461_perfdb_anomaly_scan.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase461_perfdb_anomaly_scan.md"
THRESHOLD = 3.0

TABLE_AXES = {
    "context_attention_perf.txt": ("batch_size", "isl"),
    "context_mla_perf.txt": ("batch_size", "isl"),
    "custom_allreduce_perf.txt": ("message_size",),
    "gemm_perf.txt": ("m", "n", "k"),
    "generation_attention_perf.txt": ("batch_size", "effective_seq"),
    "generation_mla_perf.txt": ("batch_size", "effective_seq"),
    "moe_perf.txt": ("num_tokens",),
    "vllm_ep8_a2a_decode_perf.txt": ("bucket_tokens",),
    "vllm_module_perf.txt": ("bucket_tokens",),
    "vllm_serving_state_perf.txt": ("bucket_tokens", "decode_batch"),
}

CSV_FIELDS = [
    "file",
    "line",
    "axis",
    "axis_count",
    "direction",
    "axis_value",
    "left_axis",
    "left_latency_ms",
    "actual_latency_ms",
    "right_axis",
    "right_latency_ms",
    "expected_latency_ms",
    "deviation_ratio",
    "active_six_scope",
    "gpu_disposition",
    "shape",
    "semantic_key",
]


class Anomaly(NamedTuple):
    line: int
    axis: str
    direction: str
    axis_value: int
    left_axis: int
    left_latency_ms: float
    actual_latency_ms: float
    right_axis: int
    right_latency_ms: float
    expected_latency_ms: float
    deviation_ratio: float
    semantic_key: tuple[tuple[str, str], ...]


def add_effective_sequence(rows: list[dict[str, object]]) -> None:
    for row in rows:
        row["effective_seq"] = int(row["isl"]) + int(row["step"])


def _linear(x0: int, y0: float, x1: int, y1: float, x: int) -> float:
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def _group_key(row: dict[str, object], axis: str) -> tuple[tuple[str, str], ...]:
    excluded = {"latency", axis, "__line__"}
    if axis == "effective_seq":
        excluded.update(("isl", "step"))
    return tuple(
        (key, str(value))
        for key, value in sorted(row.items())
        if key not in excluded
    )


def scan_axis(
    rows: list[dict[str, object]], *, axis: str, threshold: float = THRESHOLD
) -> list[Anomaly]:
    groups: dict[
        tuple[tuple[str, str], ...], dict[int, list[tuple[float, int]]]
    ] = defaultdict(lambda: defaultdict(list))
    for offset, row in enumerate(rows, start=2):
        try:
            axis_value = int(row[axis])
            latency = float(row["latency"])
        except (KeyError, TypeError, ValueError):
            continue
        line = int(row.get("__line__", offset))
        groups[_group_key(row, axis)][axis_value].append((latency, line))

    anomalies: list[Anomaly] = []
    for semantic_key, points in groups.items():
        medians = sorted(
            (axis_value, statistics.median(value for value, _ in samples), samples)
            for axis_value, samples in points.items()
        )
        for left, current, right in zip(medians, medians[1:], medians[2:]):
            left_axis, left_latency, _ = left
            axis_value, _, samples = current
            right_axis, right_latency, _ = right
            expected = _linear(
                left_axis, left_latency, right_axis, right_latency, axis_value
            )
            if expected <= 0:
                continue
            for actual, line in samples:
                if actual <= 0:
                    continue
                deviation = max(actual / expected, expected / actual)
                spike = actual > threshold * max(left_latency, right_latency)
                valley = actual * threshold < min(left_latency, right_latency)
                if deviation <= threshold or not (spike or valley):
                    continue
                anomalies.append(
                    Anomaly(
                        line=line,
                        axis=axis,
                        direction="spike" if spike else "valley",
                        axis_value=axis_value,
                        left_axis=left_axis,
                        left_latency_ms=left_latency,
                        actual_latency_ms=actual,
                        right_axis=right_axis,
                        right_latency_ms=right_latency,
                        expected_latency_ms=expected,
                        deviation_ratio=deviation,
                        semantic_key=semantic_key,
                    )
                )
    return sorted(anomalies, key=lambda item: (item.line, item.axis))


def load_rows(path: Path, axes: tuple[str, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line, row in enumerate(csv.DictReader(handle), start=2):
            row["__line__"] = line
            rows.append(row)
    if "effective_seq" in axes:
        add_effective_sequence(rows)
    return rows


def _is_active_six_scope(filename: str, row: dict[str, object]) -> bool:
    if filename in {"context_mla_perf.txt", "generation_mla_perf.txt"}:
        return (
            row.get("mla_dtype") == "float16"
            and row.get("kv_cache_dtype") == "float16"
            and int(row["num_heads"]) in {8, 16}
        )
    if filename == "moe_perf.txt":
        return (
            row.get("moe_dtype") == "int4_wo"
            and int(row["hidden_size"]) == 7168
            and int(row["inter_size"]) == 2048
            and int(row["topk"]) == 8
            and int(row["num_experts"]) == 384
            and int(row["moe_ep_size"]) == 8
        )
    if filename == "vllm_serving_state_perf.txt":
        return row.get("model") == "kimi-k2.5" and row.get("topology") in {
            "tp8dp1ep8",
            "tp4dp2ep8",
        }
    if filename == "vllm_module_perf.txt":
        return row.get("model") == "kimi-k2.5"
    return False


def _is_confirmed_phase460_row(filename: str, row: dict[str, object]) -> bool:
    return (
        filename == "generation_mla_perf.txt"
        and row.get("mla_dtype") == "float16"
        and row.get("kv_cache_dtype") == "float16"
        and int(row["num_heads"]) == 8
        and int(row["batch_size"]) == 8
        and int(row["isl"]) + int(row["step"]) == 32768
        and abs(float(row["latency"]) - 0.9586453437805176) < 1e-12
    )


def _shape_summary(filename: str, row: dict[str, object]) -> str:
    if filename in {"context_mla_perf.txt", "generation_mla_perf.txt"}:
        seq = int(row["isl"]) + int(row["step"])
        return f"heads={row['num_heads']};batch={row['batch_size']};seq={seq}"
    if filename == "vllm_serving_state_perf.txt":
        return (
            f"topology={row['topology']};max_bt={row['max_num_batched_tokens']};"
            f"phase={row['phase']};kind={row['row_kind']};category={row['category']};"
            f"bucket={row['bucket_tokens']};decode={row['decode_batch']}"
        )
    if filename == "moe_perf.txt":
        return (
            f"dtype={row['moe_dtype']};tokens={row['num_tokens']};hidden={row['hidden_size']};"
            f"inter={row['inter_size']};topk={row['topk']};ep={row['moe_ep_size']}"
        )
    return f"op={row.get('op_name', row.get('module_boundary', 'unknown'))}"


def scan_all_tables() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    output: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for filename, axes in TABLE_AXES.items():
        path = PERFDB_ROOT / filename
        rows = load_rows(path, axes)
        by_line = {int(row["__line__"]): row for row in rows}
        detections: list[Anomaly] = []
        for axis in axes:
            detections.extend(scan_axis(rows, axis=axis))
        axis_counts = Counter(item.line for item in detections)
        unique_lines = sorted(axis_counts)
        active_lines = {
            line for line in unique_lines if _is_active_six_scope(filename, by_line[line])
        }
        summary.append(
            {
                "file": filename,
                "rows": len(rows),
                "axes": ",".join(axes),
                "detections": len(detections),
                "candidate_rows": len(unique_lines),
                "multi_axis_rows": sum(axis_counts[line] >= 2 for line in unique_lines),
                "active_six_rows": len(active_lines),
            }
        )
        for anomaly in detections:
            raw = by_line[anomaly.line]
            active = anomaly.line in active_lines
            if _is_confirmed_phase460_row(filename, raw):
                disposition = "mandatory_recollect"
            elif active:
                disposition = "gpu_batch_candidate_review"
            else:
                disposition = "out_of_active_six_scope"
            output.append(
                {
                    "file": filename,
                    "line": anomaly.line,
                    "axis": anomaly.axis,
                    "axis_count": axis_counts[anomaly.line],
                    "direction": anomaly.direction,
                    "axis_value": anomaly.axis_value,
                    "left_axis": anomaly.left_axis,
                    "left_latency_ms": anomaly.left_latency_ms,
                    "actual_latency_ms": anomaly.actual_latency_ms,
                    "right_axis": anomaly.right_axis,
                    "right_latency_ms": anomaly.right_latency_ms,
                    "expected_latency_ms": anomaly.expected_latency_ms,
                    "deviation_ratio": anomaly.deviation_ratio,
                    "active_six_scope": str(active).lower(),
                    "gpu_disposition": disposition,
                    "shape": _shape_summary(filename, raw),
                    "semantic_key": json.dumps(
                        dict(anomaly.semantic_key), sort_keys=True, separators=(",", ":")
                    ),
                }
            )
    output.sort(key=lambda row: (-float(row["deviation_ratio"]), row["file"], int(row["line"])))
    return output, summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _unique_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    best: dict[tuple[str, int], dict[str, object]] = {}
    axes: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in rows:
        key = (str(row["file"]), int(row["line"]))
        axes[key].add(str(row["axis"]))
        if key not in best or float(row["deviation_ratio"]) > float(best[key]["deviation_ratio"]):
            best[key] = dict(row)
    for key, row in best.items():
        row["axes"] = ",".join(sorted(axes[key]))
    return sorted(best.values(), key=lambda row: -float(row["deviation_ratio"]))


def write_markdown(
    path: Path,
    rows: list[dict[str, object]],
    summary: list[dict[str, object]],
) -> None:
    unique = _unique_rows(rows)
    active = [row for row in unique if row["active_six_scope"] == "true"]
    mandatory = [row for row in unique if row["gpu_disposition"] == "mandatory_recollect"]
    multi_axis = [row for row in unique if int(row["axis_count"]) >= 2]
    total_table_rows = sum(int(row["rows"]) for row in summary)
    lines = [
        "# Phase461 Step 0: H200 vLLM 0.19 PerfDB anomaly scan",
        "",
        f"结论：扫描 {len(summary)} 张表、{total_table_rows:,} 行，按严格局部极值规则得到 {len(unique)} 个可疑行，其中 {len(multi_axis)} 行被两个运行轴同时命中，{len(active)} 行落在当前 K2.5 六点语义范围。Phase460 已确认的 MLA 坏行被正确检出并列为 mandatory。其余结果仍是 candidate，不得自动删除、平滑或替换。",
        "",
        "## 扫描规则",
        "",
        "| 规则 | 约束 |",
        "|---|---|",
        "| 语义隔离 | 除被扫描运行轴外，其余列必须完全相同 |",
        "| 邻域 | 目标点必须有左右两个实测格点 |",
        f"| 偏离门 | 实测/邻点线性趋势的对称比 > {THRESHOLD:.1f}x |",
        f"| 极值门 | 同时高于两邻点 {THRESHOLD:.1f}x，或低于两邻点 {THRESHOLD:.1f}x |",
        "| 输出语义 | candidate only；kernel 切换等合法 cliff 需 provenance 或重采确认 |",
        "",
        "## 表级汇总",
        "",
        "| 表 | 行数 | 扫描轴 | 检出记录 | 可疑行 | 双轴命中 | 六点范围 |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['file']} | {row['rows']} | {row['axes']} | {row['detections']} | "
        f"{row['candidate_rows']} | {row['multi_axis_rows']} | {row['active_six_rows']} |"
        for row in summary
    )
    lines.extend(
        [
            "",
            "## 当前六点范围候选",
            "",
            "| 表:行 | shape | 轴 | 方向 | 实测 ms | 邻点趋势 ms | 偏离 | GPU 处置 |",
            "|---|---|---|---|---:|---:|---:|---|",
        ]
    )
    lines.extend(
        f"| {row['file']}:{row['line']} | {row['shape']} | {row['axes']} | {row['direction']} | "
        f"{float(row['actual_latency_ms']):.6f} | {float(row['expected_latency_ms']):.6f} | "
        f"{float(row['deviation_ratio']):.2f}x | {row['gpu_disposition']} |"
        for row in active
    )
    root = mandatory[0]
    lines.extend(
        [
            "",
            "Phase460 根因行在 `generation_mla_perf.txt:2521`，同时被 batch 与 effective-sequence 两轴检出；scanner 对已知故障的召回门通过。",
            "",
            "## GPU 批次影响",
            "",
            "| 类型 | 数量 | 处理 |",
            "|---|---:|---|",
            f"| 已确认坏行 | {len(mandatory)} | 必采；沿用 Phase460 的 heads=8 × batch 8/16 × KV 16k/32k/64k 网格 |",
            f"| 六点范围其他候选 | {len(active) - len(mandatory)} | 先做 provenance/运行命中审核；确认可达后并入对应 microbench 或 serving window |",
            f"| 六点范围外候选 | {len(unique) - len(active)} | 不塞入本轮 GPU 批；归档为全库数据维护清单 |",
            "",
            "不能把全部 candidate 无差别塞进一次 GPU：context/generation MLA 是 kernel microbench，serving-state 是在位 workload 窗口，测量原语不同。Step 3 清单应在 Step 1/2 的实际查询命中审计后按原语拆分，但仍可共用同一次 GPU 分配窗口。",
            "",
            "## 后续门",
            "",
            "- Step 1 才允许修改 serving-state schema/runtime；本阶段没有代码路径或数据表改动。",
            "- Step 1 必须带 `max_num_batched_tokens` 精确键，miss 走解析模型，禁止跨 regime 静默复用。",
            "- 抢占残差、路由竞态、tp8-bt65536 replay 残差 1.118、LOO 清单继续存档。",
            "- Default AIC 维持 No-Go；6/6 进入 1.15 前不收紧门限。",
            "",
            f"完整逐轴邻域证据见 `{DEFAULT_CSV.relative_to(REPO_ROOT)}`。",
        ]
    )
    if root["file"] != "generation_mla_perf.txt" or int(root["line"]) != 2521:
        raise ValueError("confirmed Phase460 row was not retained as mandatory")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    rows, summary = scan_all_tables()
    if not rows:
        raise ValueError("anomaly scan produced no candidates")
    write_csv(args.csv, rows)
    write_markdown(args.md, rows, summary)
    print(f"wrote {args.csv}")
    print(f"wrote {args.md}")
    print(f"candidate_rows={len(_unique_rows(rows))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
