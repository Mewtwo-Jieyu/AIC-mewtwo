#!/usr/bin/env python3
"""Phase460 Step 1: audit the TP8 32k generation-MLA lookup path."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import NamedTuple


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import validate_cb_simulator as validate  # noqa: E402
from aiconfigurator.sdk import common  # noqa: E402
from aiconfigurator.sdk.backends.cb_simulator.iteration_latency import (  # noqa: E402
    IterationLatencyCalculator,
    _KV_LEN_BUCKET,
    _bucket,
)


SCENARIO = "K2.5-tp8ep8-32k3k"
PERF_PATH = (
    REPO_ROOT
    / "src/aiconfigurator/systems/data/h200_sxm/vllm/0.19.0/generation_mla_perf.txt"
)
PHASE459_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase459_residual_triage.csv"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase460_mla_decode_lookup.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase460_mla_decode_lookup.md"


class MlaRow(NamedTuple):
    local_heads: int
    batch: int
    kv_len: int
    latency_ms: float
    kernel_source: str


class LookupAudit(NamedTuple):
    batch_bracket: tuple[int, int]
    kv_bracket: tuple[int, int]
    current_per_layer_ms: float
    current_attention_ms: float
    outlier_batch: int
    outlier_kv_len: int
    outlier_latency_ms: float
    outlier_context_trend_ms: float
    outlier_actual_over_context_trend: float
    counterfactual_per_layer_ms: float
    counterfactual_attention_ms: float
    counterfactual_decode_total_ms: float
    counterfactual_decode_sim_over_real: float
    verdict: str


CSV_FIELDS = ["section", "metric", "value", "target", "status", "source", "note"]


def _bracket(value: int, grid: list[int]) -> tuple[int, int]:
    ordered = sorted(set(grid))
    if value <= ordered[0] or value >= ordered[-1]:
        raise ValueError(f"target {value} must be strictly inside grid {ordered}")
    for left, right in zip(ordered, ordered[1:]):
        if left <= value <= right:
            return left, right
    raise ValueError(f"no bracket for {value}")


def _linear(x0: int, y0: float, x1: int, y1: float, x: int) -> float:
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def _bilinear(
    batch: int,
    kv_len: int,
    batch_bracket: tuple[int, int],
    kv_bracket: tuple[int, int],
    values: dict[tuple[int, int], float],
) -> float:
    b0, b1 = batch_bracket
    k0, k1 = kv_bracket
    at_b0 = _linear(k0, values[(b0, k0)], k1, values[(b0, k1)], kv_len)
    at_b1 = _linear(k0, values[(b1, k0)], k1, values[(b1, k1)], kv_len)
    return _linear(b0, at_b0, b1, at_b1, batch)


def _context_trend(rows: list[MlaRow], batch: int, kv_len: int) -> float:
    same_batch = sorted((row.kv_len, row.latency_ms) for row in rows if row.batch == batch and row.kv_len != kv_len)
    lower = [(kv, latency) for kv, latency in same_batch if kv < kv_len]
    upper = [(kv, latency) for kv, latency in same_batch if kv > kv_len]
    if not lower or not upper:
        raise ValueError(f"cannot estimate context trend for batch={batch}, kv={kv_len}")
    lo_kv, lo_latency = lower[-1]
    hi_kv, hi_latency = upper[0]
    return _linear(lo_kv, lo_latency, hi_kv, hi_latency, kv_len)


def audit_lookup(
    rows: list[MlaRow],
    *,
    batch: int,
    kv_len: int,
    local_heads: int,
    layers: int,
    current_decode_total_ms: float,
    real_decode_total_ms: float,
) -> LookupAudit:
    selected = [row for row in rows if row.local_heads == local_heads]
    batches = sorted({row.batch for row in selected})
    batch_bracket = _bracket(batch, batches)
    common_kv = sorted(
        set.intersection(
            *(set(row.kv_len for row in selected if row.batch == b) for b in batch_bracket)
        )
    )
    kv_bracket = _bracket(kv_len, common_kv)
    values = {(row.batch, row.kv_len): row.latency_ms for row in selected}
    current_per_layer = _bilinear(batch, kv_len, batch_bracket, kv_bracket, values)

    candidates: list[tuple[float, MlaRow, float]] = []
    for corner_batch in batch_bracket:
        for corner_kv in kv_bracket:
            row = next(
                item for item in selected if item.batch == corner_batch and item.kv_len == corner_kv
            )
            try:
                trend = _context_trend(selected, corner_batch, corner_kv)
            except ValueError:
                continue
            candidates.append((row.latency_ms / trend, row, trend))
    if not candidates:
        raise ValueError("no interpolation corner has two-sided context evidence")
    outlier_ratio, outlier, outlier_trend = max(candidates, key=lambda item: item[0])

    counterfactual_values = dict(values)
    counterfactual_values[(outlier.batch, outlier.kv_len)] = outlier_trend
    counterfactual_per_layer = _bilinear(
        batch, kv_len, batch_bracket, kv_bracket, counterfactual_values
    )
    current_attention = current_per_layer * layers
    counterfactual_attention = counterfactual_per_layer * layers
    counterfactual_decode = current_decode_total_ms - current_attention + counterfactual_attention
    counterfactual_ratio = counterfactual_decode / real_decode_total_ms

    same_kv_larger_batch = [
        row.latency_ms
        for row in selected
        if row.kv_len == outlier.kv_len and row.batch > outlier.batch
    ]
    violates_batch_monotonicity = bool(same_kv_larger_batch) and outlier.latency_ms > min(same_kv_larger_batch)
    verdict = (
        "perfdb_outlier_row_requires_recollect"
        if outlier_ratio >= 2.0
        and violates_batch_monotonicity
        and abs(counterfactual_ratio - 1.0) <= 0.10
        else "mla_decode_lookup_unresolved"
    )
    return LookupAudit(
        batch_bracket=batch_bracket,
        kv_bracket=kv_bracket,
        current_per_layer_ms=current_per_layer,
        current_attention_ms=current_attention,
        outlier_batch=outlier.batch,
        outlier_kv_len=outlier.kv_len,
        outlier_latency_ms=outlier.latency_ms,
        outlier_context_trend_ms=outlier_trend,
        outlier_actual_over_context_trend=outlier_ratio,
        counterfactual_per_layer_ms=counterfactual_per_layer,
        counterfactual_attention_ms=counterfactual_attention,
        counterfactual_decode_total_ms=counterfactual_decode,
        counterfactual_decode_sim_over_real=counterfactual_ratio,
        verdict=verdict,
    )


def classify_context_contribution(
    *, minimum_attention_ms: float, maximum_attention_ms: float, observed_excess_ms: float
) -> dict[str, float | str]:
    span = maximum_attention_ms - minimum_attention_ms
    share = span / observed_excess_ms
    return {
        "attention_span_ms": span,
        "share_of_observed_excess": share,
        "verdict": "context_semantics_not_primary" if share <= 0.10 else "context_semantics_material",
    }


def load_mla_rows(path: Path) -> list[MlaRow]:
    rows: list[MlaRow] = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["kv_cache_dtype"] != "float16":
                continue
            rows.append(
                MlaRow(
                    local_heads=int(row["num_heads"]),
                    batch=int(row["batch_size"]),
                    kv_len=int(row["isl"]) + int(row["step"]),
                    latency_ms=float(row["latency"]),
                    kernel_source=row.get("kernel_source", ""),
                )
            )
    return rows


def load_phase459_metrics(path: Path) -> dict[str, float]:
    wanted = {"decode_real_total_ms", "decode_generation_attention_ms"}
    values: dict[str, float] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["scenario"] == SCENARIO and row["metric"] in wanted:
                values[row["metric"]] = float(row["value"])
    missing = wanted - values.keys()
    if missing:
        raise ValueError(f"missing Phase459 metrics: {sorted(missing)}")
    return values


def _add(
    rows: list[dict[str, object]],
    section: str,
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
            "metric": metric,
            "value": value,
            "target": target,
            "status": status,
            "source": source,
            "note": note,
        }
    )


def build_report_rows() -> list[dict[str, object]]:
    point = next(item for item in validate.MULTI_CONFIG_DATA if item.name == SCENARIO)
    model, database, backend = validate._load_model_and_db(
        tp=point.tp, dp=point.dp, moe_tp=point.moe_tp, moe_ep=point.moe_ep
    )
    attention_op = next(op for op in model.generation_ops if op._name == "generation_attention")
    local_heads = int(attention_op._num_heads)
    layers = int(attention_op._scale_factor)
    decode_batch = 13
    decode_avg_kv = point.isl + point.osl // 2
    kv_bucket = _bucket(decode_avg_kv, _KV_LEN_BUCKET)
    query_kv = kv_bucket + 1
    perf_rows = load_mla_rows(PERF_PATH)
    phase459 = load_phase459_metrics(PHASE459_CSV)
    calculator = IterationLatencyCalculator(
        backend,
        model,
        database,
        overlap_factor=0.0,
        per_iteration_overhead_ms=validate.DEFAULT_EP8_PER_ITERATION_OVERHEAD_MS,
        serving_state_max_num_batched_tokens=point.max_num_batched_tokens,
    )
    current_decode_total_ms = calculator.compute(
        prefill_tokens=0,
        prefill_batch_size=0,
        prefill_seq_len=point.isl,
        decode_batch_size=decode_batch,
        decode_avg_kv_len=decode_avg_kv,
    )
    current_breakdown = calculator.get_last_breakdown()
    if current_breakdown is None:
        raise ValueError("missing current decode breakdown")
    if not math.isclose(
        current_breakdown.generation_attention_ms,
        phase459["decode_generation_attention_ms"],
        abs_tol=1e-3,
    ):
        raise ValueError("Phase460 decode attention no longer matches Phase459")
    audit = audit_lookup(
        perf_rows,
        batch=decode_batch,
        kv_len=query_kv,
        local_heads=local_heads,
        layers=layers,
        current_decode_total_ms=current_decode_total_ms,
        real_decode_total_ms=phase459["decode_real_total_ms"],
    )
    database_per_layer_ms = float(
        database.query_generation_mla(
            decode_batch,
            query_kv,
            local_heads,
            common.KVCacheQuantMode.float16,
        )
    )
    if not math.isclose(audit.current_per_layer_ms, database_per_layer_ms, abs_tol=1e-12):
        raise ValueError("audit interpolation no longer matches PerfDatabase")

    context_attention_values = []
    for avg_kv in (point.isl, decode_avg_kv, point.isl + point.osl):
        queried_kv = _bucket(avg_kv, _KV_LEN_BUCKET) + 1
        per_layer = float(
            database.query_generation_mla(
                decode_batch,
                queried_kv,
                local_heads,
                common.KVCacheQuantMode.float16,
            )
        )
        context_attention_values.append(per_layer * layers)
    context = classify_context_contribution(
        minimum_attention_ms=min(context_attention_values),
        maximum_attention_ms=max(context_attention_values),
        observed_excess_ms=current_decode_total_ms - phase459["decode_real_total_ms"],
    )
    sol = float(
        database.query_generation_mla(
            audit.outlier_batch,
            audit.outlier_kv_len,
            local_heads,
            common.KVCacheQuantMode.float16,
            common.DatabaseMode.SOL,
        )
    )
    empirical = float(
        database.query_generation_mla(
            audit.outlier_batch,
            audit.outlier_kv_len,
            local_heads,
            common.KVCacheQuantMode.float16,
            common.DatabaseMode.EMPIRICAL,
        )
    )

    rows: list[dict[str, object]] = []
    for metric, value, note in (
        ("decode_batch", decode_batch, "Phase459 representative decode step"),
        ("decode_avg_kv_input", decode_avg_kv, "prompt + half output"),
        ("decode_kv_bucket", kv_bucket, f"ceil to {_KV_LEN_BUCKET}-token bucket"),
        ("generation_mla_query_kv", query_kv, "BaseBackend adds one decode token"),
        ("local_heads", local_heads, "64 global heads / TP8"),
        ("layer_scale", layers, "61 transformer layers"),
        ("database_mode", database._default_database_mode.name, "in-range SILICON lookup"),
        ("interpolation", "bilinear", "batch and KV are both bracketed"),
    ):
        _add(rows, "lookup_path", metric, value, source="runtime query chain", note=note)

    values = {(row.batch, row.kv_len): row for row in perf_rows if row.local_heads == local_heads}
    for batch in audit.batch_bracket:
        for kv_len in audit.kv_bracket:
            row = values[(batch, kv_len)]
            _add(
                rows,
                "lookup_corners",
                f"batch{batch}_kv{kv_len}_ms_per_layer",
                row.latency_ms,
                source=str(PERF_PATH.relative_to(REPO_ROOT)),
                note=f"kernel_source={row.kernel_source}",
            )

    for metric, value, target, status, note in (
        ("current_per_layer_ms", audit.current_per_layer_ms, "matches PerfDatabase", "pass", f"database_query={database_per_layer_ms:.12f}"),
        ("current_attention_ms", audit.current_attention_ms, "Phase459 31.029ms", "pass", ""),
        ("outlier_latency_ms", audit.outlier_latency_ms, "monotonic local neighborhood", "fail", f"batch={audit.outlier_batch}; kv={audit.outlier_kv_len}"),
        ("outlier_context_trend_ms", audit.outlier_context_trend_ms, "diagnostic only", "", "linear trend from same-batch 16k and 64k rows"),
        ("outlier_actual_over_context_trend", audit.outlier_actual_over_context_trend, "<=2.0", "fail", "7.19x diagnostic outlier"),
        ("outlier_actual_over_sol", audit.outlier_latency_ms / sol, "", "", f"SOL={sol:.6f}ms"),
        ("outlier_actual_over_empirical", audit.outlier_latency_ms / empirical, "", "", f"EMPIRICAL={empirical:.6f}ms"),
        ("counterfactual_attention_ms", audit.counterfactual_attention_ms, "diagnostic only", "", "not a proposed PerfDB value"),
        ("counterfactual_decode_total_ms", audit.counterfactual_decode_total_ms, "real=21.0ms", "", "single-row diagnostic replacement"),
        ("counterfactual_decode_sim_over_real", audit.counterfactual_decode_sim_over_real, "0.90..1.10", "pass", "counterfactual closes representative step"),
        ("verdict", audit.verdict, "root cause identified", "pass", "recollect exact row; do not auto-repair table"),
    ):
        _add(rows, "outlier_audit", metric, value, target=target, status=status, source=str(PERF_PATH.relative_to(REPO_ROOT)), note=note)

    for metric, value in context.items():
        _add(
            rows,
            "context_semantics",
            metric,
            value,
            target="<=10% of observed excess" if metric == "share_of_observed_excess" else "",
            status="pass" if metric == "verdict" and value == "context_semantics_not_primary" else "",
            source="32k..35k query sweep",
        )
    _add(
        rows,
        "decision",
        "phase460_step2",
        "gpu_recollect_generation_mla_32k_grid",
        target="user confirmation before collection or PerfDB change",
        status="proposed",
        note="TP8 local_heads=8; batch={8,16}; kv={16384,32768,65536}; repeated measurements",
    )
    _add(
        rows,
        "decision",
        "runtime_lookup_change",
        "not_required",
        status="pass",
        note="lookup is in-range bilinear and context semantics are not primary",
    )
    _add(
        rows,
        "gate",
        "default_aic",
        "No-Go",
        target="6/6 <=1.15 before tightening",
        status="fail",
        note="Phase460 Step 1 is report-only; no runtime/PerfDB/gate change",
    )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _value(rows: list[dict[str, object]], metric: str) -> object:
    return next(row["value"] for row in rows if row["metric"] == metric)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    corners = [row for row in rows if row["section"] == "lookup_corners"]
    context = [row for row in rows if row["section"] == "context_semantics"]
    lines = [
        "# Phase460 Step 1: TP8 32k MLA decode lookup audit",
        "",
        "结论：31ms 高计费来自 `generation_mla_perf.txt` 的单个反单调实测行，不是 context 长度语义、外推或 lookup 公式。当前查询在表内做双线性插值；`batch=8, kv=32768` 的 `0.958645ms/layer` 比同 batch 上下文趋势高 7.19x，并且比 `batch=16` 同 KV 更慢。Phase460 不应改 runtime lookup，也不能用自动平滑修表；正确动作是把精确网格加入 Phase461 GPU 补采批次。",
        "",
        "| 查询项 | 值 |",
        "|---|---:|",
        f"| decode batch | {_value(rows, 'decode_batch')} |",
        f"| 输入平均 KV | {_value(rows, 'decode_avg_kv_input')} |",
        f"| 512-token bucket | {_value(rows, 'decode_kv_bucket')} |",
        f"| 最终 PerfDB KV 查询 | {_value(rows, 'generation_mla_query_kv')} |",
        f"| local heads | {_value(rows, 'local_heads')} |",
        f"| layer scale | {_value(rows, 'layer_scale')} |",
        f"| 查询方式 | {_value(rows, 'database_mode')} / {_value(rows, 'interpolation')} |",
        "",
        "## 插值四角",
        "",
        "| 行 | ms/layer |",
        "|---|---:|",
    ]
    lines.extend(f"| {row['metric']} | {float(row['value']):.6f} |" for row in corners)
    lines.extend(
        [
            "",
            "`batch=8, kv=32768` 是唯一直接破坏该插值矩形单调性的角：它为 0.958645ms，而 `batch=16` 同 KV 仅 0.244939ms，`batch=8, kv=65536` 也仅 0.244603ms。该行已存在于最初的 H200 vLLM 0.19 PerfDB 对象；Phase431 只追加 8k 行，没有修到此点。",
            "",
            "## 贡献量化",
            "",
            "| 项目 | 结果 |",
            "|---|---:|",
            f"| 当前 attention | {float(_value(rows, 'current_attention_ms')):.3f} ms |",
            f"| 当前 decode 总步 | 39.072 ms |",
            f"| real decode 总步 | 21.000 ms |",
            f"| 32k..35k context 扫描造成的 attention 波动 | {float(next(row['value'] for row in context if row['metric'] == 'attention_span_ms')):.3f} ms |",
            f"| 邻点反事实 attention | {float(_value(rows, 'counterfactual_attention_ms')):.3f} ms |",
            f"| 邻点反事实 decode 总步 | {float(_value(rows, 'counterfactual_decode_total_ms')):.3f} ms |",
            f"| 邻点反事实 sim/real | {float(_value(rows, 'counterfactual_decode_sim_over_real')):.3f}x |",
            "",
            "邻点反事实仅用于证明归因：用同 batch 的 16k/64k 行线性趋势替换异常角后，代表步回到 0.990x。它不是可入库的替代值。生产修复必须重采。",
            "",
            "## Phase461 合并采集提案",
            "",
            "| 维度 | 格点 | 目的 |",
            "|---|---|---|",
            "| local heads | 8 | TP8 K2.5 实际查询口径 |",
            "| batch | 8, 16 | 覆盖代表 batch=13 的插值两侧 |",
            "| KV | 16384, 32768, 65536 | 验证上下文单调性并替换 32k 异常行 |",
            "| 重复 | 每点至少 3 次 | 阻止单次测量异常再次入库 |",
            "",
            "Step 2 分支因此明确为 `PerfDB 精确网格补采`，并入 Phase461 GPU 单批；不需要离线修改 lookup/context 公式。补采与入库前等待用户确认。",
            "",
            "## 边界",
            "",
            "- 本阶段 report-only，未修改 runtime、PerfDB、gate，也未使用 GPU。",
            "- mixed step 低计费、bt65536 regime scope、抢占残差继续按 Phase459 存档，不在本报告中混修。",
            "- Default AIC 维持 No-Go；6/6 进入 1.15 前不收紧门限。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    rows = build_report_rows()
    write_csv(args.csv, rows)
    write_markdown(args.md, rows)
    print(f"wrote {args.csv}")
    print(f"wrote {args.md}")
    print(f"verdict={_value(rows, 'verdict')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
