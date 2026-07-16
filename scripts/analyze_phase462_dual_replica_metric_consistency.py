#!/usr/bin/env python3
"""Audit Phase462 real/sim throughput definitions without changing runtime."""

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

import scripts.validate_cb_simulator as validate  # noqa: E402
from aiconfigurator.sdk.backends.cb_simulator import CBSimulator  # noqa: E402


SCENARIO = "K2.5-tp4ep8dp2-8k2k-bt65536"
BENCH_PATH = REPO_ROOT / (
    "docs/iter_gap_investigation/phase458_n512_unify/"
    f"recollect_dp2_8k2k_bt65536/{SCENARIO}/bench_result.json"
)
SCOREBOARD_PATH = REPO_ROOT / (
    "docs/iter_gap_investigation/phase462_null_only_scoreboard.csv"
)
DEFAULT_ROOT = REPO_ROOT / (
    "docs/iter_gap_investigation/phase462_dual_replica_metric_consistency"
)
DEFAULT_CSV = DEFAULT_ROOT / "phase462_dual_replica_metric_consistency.csv"
DEFAULT_REPORT = DEFAULT_ROOT / "phase462_dual_replica_metric_consistency.md"


@dataclass(frozen=True)
class FullWindowMetric:
    output_tokens: int
    wall_ms: float
    throughput_tok_s_gpu: float


@dataclass(frozen=True)
class LegacyDPAssembly:
    global_throughput: float
    total_gpus: int
    throughput_tok_s_gpu: float


@dataclass(frozen=True)
class SimMeasurement:
    name: str
    num_requests_per_replica: int
    warmup_requests_per_replica: int
    steady_output_tok_s_gpu: float
    steady_error_ratio: float
    full_output_tok_s_gpu: float
    full_error_ratio: float
    decode_only_full_output_tok_s_gpu: float
    completed_requests: int
    full_wall_ms: float


def full_window_metric(
    completed: Sequence[object],
    *,
    num_gpus: int,
    include_prefill_sample: bool = True,
) -> FullWindowMetric:
    if not completed:
        raise AssertionError("no_completed_requests")
    if num_gpus <= 0:
        raise ValueError("num_gpus_must_be_positive")
    wall_ms = max(float(getattr(request, "finish_ms")) for request in completed)
    if wall_ms <= 0:
        raise AssertionError("non_positive_full_window")
    output_tokens = sum(
        int(getattr(request, "osl")) - int(not include_prefill_sample)
        for request in completed
    )
    return FullWindowMetric(
        output_tokens=output_tokens,
        wall_ms=wall_ms,
        throughput_tok_s_gpu=output_tokens / (wall_ms / 1000.0) / num_gpus,
    )


def legacy_dp_assembly(
    *,
    tp_group_throughput: float,
    tensor_parallel_size: int,
    data_parallel_size: int,
) -> LegacyDPAssembly:
    if tensor_parallel_size <= 0 or data_parallel_size <= 0:
        raise ValueError("parallel_sizes_must_be_positive")
    global_throughput = tp_group_throughput * data_parallel_size
    total_gpus = tensor_parallel_size * data_parallel_size
    return LegacyDPAssembly(
        global_throughput=global_throughput,
        total_gpus=total_gpus,
        throughput_tok_s_gpu=global_throughput / total_gpus,
    )


def definition_verdict(
    *,
    real_window: str,
    sim_window: str,
    real_token_definition: str,
    sim_token_definition: str,
) -> str:
    if real_window != sim_window or real_token_definition != sim_token_definition:
        return "measurement_fidelity_bug"
    return "definitions_consistent_model_error"


def order_dependency_note(
    *, forward: tuple[float, float], reverse: tuple[float, float]
) -> str:
    if all(math.isclose(left, right, abs_tol=1e-12) for left, right in zip(forward, reverse)):
        return "order_independent"
    return "path_dependent_not_unique_causal_fraction"


def _point():
    return next(point for point in validate.MULTI_CONFIG_DATA if point.name == SCENARIO)


def _base_config(point):
    return validate._make_multi_config_cb_config(
        point,
        overlap_factor=0.0,
        ep8_per_iteration_overhead_ms=validate.DEFAULT_EP8_PER_ITERATION_OVERHEAD_MS,
    )


def _run_sim_measurement(*, name: str, config) -> SimMeasurement:
    point = _point()
    model, database, backend = validate._load_model_and_db(
        tp=point.tp,
        dp=point.dp,
        moe_tp=point.moe_tp,
        moe_ep=point.moe_ep,
    )
    captured: list[object] = []
    original_collect = CBSimulator._collect_metrics

    def capture_collect(self, completed, *args, **kwargs):
        captured.extend(completed)
        return original_collect(self, completed, *args, **kwargs)

    with patch.object(CBSimulator, "_collect_metrics", capture_collect):
        simulator = CBSimulator(backend, model, database, config)
        result = simulator.run(
            isl=point.isl,
            osl=point.osl,
            concurrency=math.ceil(point.batch_size / point.dp),
            num_gpus=point.tp,
        )
    full = full_window_metric(captured, num_gpus=point.tp)
    decode_only = full_window_metric(
        captured,
        num_gpus=point.tp,
        include_prefill_sample=False,
    )
    return SimMeasurement(
        name=name,
        num_requests_per_replica=config.num_requests,
        warmup_requests_per_replica=config.warmup_requests,
        steady_output_tok_s_gpu=result.throughput_tok_s_gpu,
        steady_error_ratio=validate._abs_error(
            result.throughput_tok_s_gpu, point.real_output_tok_s_gpu
        ),
        full_output_tok_s_gpu=full.throughput_tok_s_gpu,
        full_error_ratio=validate._abs_error(
            full.throughput_tok_s_gpu, point.real_output_tok_s_gpu
        ),
        decode_only_full_output_tok_s_gpu=decode_only.throughput_tok_s_gpu,
        completed_requests=len(captured),
        full_wall_ms=full.wall_ms,
    )


def _load_real() -> dict[str, float | int]:
    bench = json.loads(BENCH_PATH.read_text(encoding="utf-8"))
    point = _point()
    return {
        "num_prompts_global": int(bench["num_prompts"]),
        "warmup_requests": int(bench["warmup_requests"]),
        "total_completion_tokens": int(bench["total_completion_tokens"]),
        "wall_ms": float(bench["wall_s"]) * 1000.0,
        "output_tok_s_global": float(bench["output_tok_s"]),
        "output_tok_s_gpu": float(bench["output_tok_s"]) / (point.tp * point.dp),
    }


def _official_score() -> dict[str, float]:
    with SCOREBOARD_PATH.open(newline="", encoding="utf-8") as source:
        row = next(row for row in csv.DictReader(source) if row["scenario"] == SCENARIO)
    return {
        "sim_output_tok_s_gpu": float(row["sim_output_tok_s_gpu"]),
        "real_output_tok_s_gpu": float(row["real_output_tok_s_gpu"]),
        "error_ratio": float(row["error_ratio"]),
    }


def _write_csv(
    path: Path,
    *,
    real: dict[str, float | int],
    measurements: Sequence[SimMeasurement],
    verdict: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for metric, value in real.items():
        rows.append({"section": "real", "name": SCENARIO, "metric": metric, "value": value})
    for item in measurements:
        for metric in item.__dataclass_fields__:
            if metric == "name":
                continue
            rows.append(
                {
                    "section": "sim",
                    "name": item.name,
                    "metric": metric,
                    "value": getattr(item, metric),
                }
            )
    rows.extend(
        [
            {"section": "verdict", "name": "definition", "metric": "value", "value": verdict},
            {"section": "boundary", "name": "runtime", "metric": "changed", "value": False},
            {"section": "boundary", "name": "perfdb", "metric": "changed", "value": False},
            {"section": "boundary", "name": "gate", "metric": "changed", "value": False},
        ]
    )
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=["section", "name", "metric", "value"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    path: Path,
    *,
    real: dict[str, float | int],
    measurements: Sequence[SimMeasurement],
    verdict: str,
) -> None:
    official, matched = measurements
    point = _point()
    assembly = legacy_dp_assembly(
        tp_group_throughput=official.steady_output_tok_s_gpu * point.tp,
        tensor_parallel_size=point.tp,
        data_parallel_size=point.dp,
    )
    first_token_delta = (
        official.full_output_tok_s_gpu
        / official.decode_only_full_output_tok_s_gpu
        - 1.0
    )
    report = f"""# Phase462 双副本度量一致性审计

结论：`{verdict}`。real 是 N512 完整客户端墙钟；官方 sim 是单副本 N384/W128 稳态窗，再把同一 TP-group 吞吐乘 DP2。DP 乘除本身守恒，但窗口、请求数和 output token 定义均不一致，不能把官方 1.200038 全部记为模型误差。

| 口径 | 请求数 | warmup | token 分子 | 时间分母 | tok/s/GPU | error |
|---|---:|---:|---|---|---:|---:|
| real bench | {real['num_prompts_global']} global | {real['warmup_requests']} | {real['total_completion_tokens']}，含首 token | 完整 wall {float(real['wall_ms']) / 1000.0:.6f}s | {float(real['output_tok_s_gpu']):.6f} | 1.000000 |
| official sim | {official.num_requests_per_replica} per replica | {official.warmup_requests_per_replica} | steady decode steps，不含 prefill sample | replacement plateau | {official.steady_output_tok_s_gpu:.6f} | {official.steady_error_ratio:.6f} |
| official trajectory/full wall | {official.num_requests_per_replica} per replica | 0 effective | 完成请求 OSL，含首 token | 0 到最后完成 | {official.full_output_tok_s_gpu:.6f} | {official.full_error_ratio:.6f} |
| real-size symmetric replica/full wall | {matched.num_requests_per_replica} per replica | 0 | 完成请求 OSL，含首 token | 0 到最后完成 | {matched.full_output_tok_s_gpu:.6f} | {matched.full_error_ratio:.6f} |

双副本组装恒等式：TP-group `{assembly.global_throughput / point.dp:.6f}` tok/s x DP2 = global `{assembly.global_throughput:.6f}` tok/s；再除 {assembly.total_gpus} GPU 得 `{assembly.throughput_tok_s_gpu:.6f}` tok/s/GPU。这里没有额外 2x bug，问题在组装前的单副本测量定义。

首 token 口径只造成 {first_token_delta:.4%}，不是主体。窗口从稳态改为同轨迹完整墙钟后的变化才是主项；N256 是 real N512 在“两个完全对称副本”假设下的控制值，不代表真实 DP 路由构成。

## 修复预注册

| 项 | 冻结方案 | 红绿门 |
|---|---|---|
| 验收度量 | 六场景统一增加显式 `full_closed_loop` 计分口径：总完成 OSL token / 首次提交到最后完成的墙钟 | 当前六点先复现旧值；新口径逐点与 bench 的 N/C/warmup 对等 |
| 生产默认 | 保留 steady-state 作为无有限请求窗的容量预测，不拿它直接和 full-wall bench 判误差 | 默认 AIC 输出在本修复前后逐字节不变 |
| DP 组装 | full-wall 验收必须按全局 N/C 跑真实多副本组装；bt65536 未过构成门前不得用单副本复制冒充实测 DP | DP 全局完成数、每 rank 完成数、全局 wall 三项闭合 |
| token 分子 | 验收统一计 OSL，包含 prefill 产生的首 token | OSL=1 与 OSL>1 定向测试 |

本步只预注册，不实施；A/B 合并判卷前不改 runtime、PerfDB 或 gate。

永久脚注：Step 3d 的 65.02%/34.98% 来自固定切换顺序。反向切换不交换，故只能写作 `path_dependent_not_unique_causal_fraction`，不能当成唯一因果比例。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def run_analysis(
    *, output_csv: Path = DEFAULT_CSV, output_report: Path = DEFAULT_REPORT
) -> dict[str, object]:
    point = _point()
    base = _base_config(point)
    official = _run_sim_measurement(name="official_N384_W128", config=base)
    real = _load_real()
    per_replica_requests = int(real["num_prompts_global"]) // point.dp
    if per_replica_requests * point.dp != int(real["num_prompts_global"]):
        raise AssertionError("real_requests_not_divisible_by_dp")
    matched = _run_sim_measurement(
        name="real_size_symmetric_N256_W0",
        config=replace(
            base,
            num_requests=per_replica_requests,
            warmup_requests=0,
        ),
    )
    score = _official_score()
    if not math.isclose(
        official.steady_output_tok_s_gpu,
        score["sim_output_tok_s_gpu"],
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise AssertionError("official_scoreboard_parity_failed")
    verdict = definition_verdict(
        real_window="full_client_wall",
        sim_window="warmup_trimmed_replacement_plateau",
        real_token_definition="completion_tokens_including_first_sample",
        sim_token_definition="decode_steps_excluding_prefill_sample",
    )
    measurements = [official, matched]
    _write_csv(output_csv, real=real, measurements=measurements, verdict=verdict)
    _write_report(output_report, real=real, measurements=measurements, verdict=verdict)
    return {
        "status": "completed_report_only",
        "verdict": verdict,
        "official_error": official.steady_error_ratio,
        "official_full_error": official.full_error_ratio,
        "matched_full_error": matched.full_error_ratio,
        "runtime_changed": False,
        "perfdb_changed": False,
        "gate_changed": False,
        "default_aic": "No-Go",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    print(
        json.dumps(
            run_analysis(output_csv=args.output_csv, output_report=args.output_report),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
