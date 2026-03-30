#!/usr/bin/env python3
"""
拟合 Continuous Batching 效率因子。

策略：
1. 从 AIC pareto.csv 读取 AIC 在各配置下的预测（包含多种 tp/dp 组合）
2. 对比 AIC 最优吞吐 vs vLLM 实测最优吞吐
3. 拟合 cb_factor = f(b, ISL, OSL) 应用于 AIC 的 output_throughput 计算

核心思路：
- AIC 的 batch 模型在每个 (b, tp, dp) 点的预测偏低
- CB factor 在吞吐量计算时施加，使 Pareto 搜索能正确评估高并发配置
- TTFT 不受影响（prefill 阶段与 CB 无关）

两级分析：
- Level 1: pareto-level (AIC 最优 vs 实测最优) — 决定总偏差
- Level 2: per-batch-level (AIC 在多个 b 下的预测 vs 实测) — 拟合 b 的影响
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aiconfigurator.sdk.backends.vllm_backend import VLLMBackend
from aiconfigurator.sdk.config import ModelConfig, RuntimeConfig
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.perf_database import PerfDatabase
from aiconfigurator.sdk import common

logging.basicConfig(level=logging.WARNING, format="%(name)s - %(levelname)s - %(message)s")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "benchmark_comparison"

# ── Benchmark 实测数据 (vLLM 0.17, H200x16, Kimi-K2.5, tp=16 dp=1) ──────────
BENCHMARK_DATA = [
    # (scenario_name, isl, osl, best_concurrency, best_tok_s_gpu)
    ("3k-3k", 3000, 3000, 128, 254.2),
    ("8k-2k", 8000, 2000, 256, 575.4),
    ("10k-2k", 10000, 2000, 32, 267.6),
    ("10k-3k", 10000, 3000, 128, 386.9),
    ("16k-2k", 16000, 2000, 32, 352.7),
    ("30k-3k", 30000, 3000, 8, 179.2),
    ("32k-1k", 32000, 1000, 16, 559.9),
    # 120k-4k excluded: AIC reports OOM
]


def read_pareto_data() -> dict:
    """从各场景 pareto.csv 读取 AIC 预测。"""
    results = {}
    for name, isl, osl, _, _ in BENCHMARK_DATA:
        pareto_dir = RESULTS_DIR / f"{name}_relaxed"
        csvs = list(pareto_dir.glob("**/pareto.csv"))
        if not csvs:
            print(f"  {name}: 找不到 pareto.csv")
            continue
        df = pd.read_csv(csvs[0])
        results[name] = df
    return results


def level1_analysis(pareto_data: dict) -> list[dict]:
    """Level 1: AIC 最优 vs 实测最优，计算总偏差。"""
    print("=" * 80)
    print("Level 1: Pareto-level 偏差分析")
    print("=" * 80)

    points = []
    for name, isl, osl, bench_conc, bench_tps_gpu in BENCHMARK_DATA:
        if name not in pareto_data:
            continue
        df = pareto_data[name]
        best_row = df.loc[df["tokens/s/gpu"].idxmax()]
        aic_tps_gpu = float(best_row["tokens/s/gpu"])
        aic_bs = int(best_row["bs"])
        aic_conc = int(best_row["concurrency"])
        aic_tp = int(best_row["tp"])
        aic_dp = int(best_row["dp"])

        ratio = bench_tps_gpu / aic_tps_gpu

        print(f"\n  {name}: ISL={isl}, OSL={osl}, ISL/OSL={isl/osl:.1f}")
        print(f"    AIC best: b={aic_bs} conc={aic_conc} tp={aic_tp} dp={aic_dp} "
              f"→ {aic_tps_gpu:.1f} tok/s/GPU")
        print(f"    Bench:    conc={bench_conc} tp=16 dp=1 → {bench_tps_gpu:.1f} tok/s/GPU")
        print(f"    Ratio: {ratio:.1f}x")

        points.append({
            "name": name, "isl": isl, "osl": osl,
            "aic_bs": aic_bs, "aic_conc": aic_conc,
            "aic_tp": aic_tp, "aic_dp": aic_dp,
            "aic_tps_gpu": aic_tps_gpu,
            "bench_conc": bench_conc, "measured_tps_gpu": bench_tps_gpu,
            "ratio": ratio,
        })

    return points

def level2_per_batch_analysis(pareto_data: dict):
    """Level 2: 观察 AIC 在多个 batch 下的预测，理解 b 对偏差的影响。"""
    print(f"\n\n{'='*80}")
    print("Level 2: Per-batch 预测分布（从 pareto.csv）")
    print("=" * 80)

    for name, isl, osl, bench_conc, bench_tps_gpu in BENCHMARK_DATA:
        if name not in pareto_data:
            continue
        df = pareto_data[name]
        print(f"\n  {name} (ISL={isl} OSL={osl}, 实测 {bench_tps_gpu} tok/s/GPU)")

        # 按 tp 分组看
        for tp_val in sorted(df["tp"].unique()):
            sub = df[df["tp"] == tp_val].sort_values("bs")
            dp_val = sub["dp"].iloc[0]
            print(f"    tp={tp_val} dp={dp_val}:")
            for _, row in sub.iterrows():
                bs = int(row["bs"])
                conc = int(row["concurrency"])
                tps = float(row["tokens/s/gpu"])
                tpot = float(row["tpot"])
                ratio = bench_tps_gpu / tps
                print(f"      b={bs:>3} (conc={conc:>3}): {tps:>7.1f} tok/s/GPU, "
                      f"TPOT={tpot:>6.1f}ms, ratio={ratio:>5.1f}x")


# ── 拟合函数 ─────────────────────────────────────────────────────────────────

def model_A(X, a, b, c):
    """ratio = a * (ISL/OSL)^b + c"""
    isl, osl = X
    return a * (isl / osl) ** b + c


def model_B(X, a, b):
    """ratio = a * ISL^b / OSL^(b/2)"""
    isl, osl = X
    return a * (isl ** b) / (osl ** (b / 2))


def model_C(X, a, b, c):
    """ratio = a * log(ISL)^b * log(OSL)^c"""
    isl, osl = X
    return a * np.log(isl) ** b * np.log(osl) ** c


def model_D(X, a, b, c):
    """ratio = a * ISL^b * OSL^c"""
    isl, osl = X
    return a * (isl ** b) * (osl ** c)


def model_E(X, a, b, c, d):
    """ratio = a * ISL^b * OSL^c * bs^d (4 params, includes AIC batch size)"""
    isl, osl, bs = X
    return a * (isl ** b) * (osl ** c) * (bs ** d)


def model_F(X, a, b, c, d):
    """ratio = a * (ISL/OSL)^b * bs^c + d"""
    isl, osl, bs = X
    return a * ((isl / osl) ** b) * (bs ** c) + d


def fit_level1(points: list[dict]):
    """拟合 Level 1 偏差 as f(ISL, OSL)。"""
    print(f"\n\n{'='*80}")
    print("Step 3: 拟合 CB 校正因子 = f(ISL, OSL)")
    print("=" * 80)

    isl_arr = np.array([p["isl"] for p in points], dtype=float)
    osl_arr = np.array([p["osl"] for p in points], dtype=float)
    bs_arr = np.array([p["aic_bs"] for p in points], dtype=float)
    y_arr = np.array([p["ratio"] for p in points], dtype=float)
    X2 = np.vstack([isl_arr, osl_arr])
    X3 = np.vstack([isl_arr, osl_arr, bs_arr])

    models = [
        (model_A, "A: a*(ISL/OSL)^b + c", [3.0, 1.0, 0.0], (-np.inf, np.inf), X2),
        (model_B, "B: a*ISL^b/OSL^(b/2)", [0.001, 1.0], (0, [1e6, 5]), X2),
        (model_C, "C: a*log(ISL)^b*log(OSL)^c", [0.01, 2.0, -1.0], (-np.inf, np.inf), X2),
        (model_D, "D: a*ISL^b*OSL^c", [0.001, 1.0, -0.5], (-np.inf, np.inf), X2),
        (model_E, "E: a*ISL^b*OSL^c*bs^d", [10.0, 1.0, -1.0, -0.5], (-np.inf, np.inf), X3),
        (model_F, "F: a*(ISL/OSL)^b*bs^c+d", [10.0, 1.0, -0.5, 0.0], (-np.inf, np.inf), X3),
    ]

    best = {"name": None, "params": None, "fn": None, "loocv_max": float("inf"), "X": None}

    for fn, name, p0, bounds, X in models:
        try:
            popt, _ = curve_fit(fn, X, y_arr, p0=p0, maxfev=50000, bounds=bounds)
        except Exception as e:
            print(f"\n  {name}: 拟合失败 - {e}")
            continue

        y_pred = fn(X, *popt)
        ratios = np.maximum(y_pred / y_arr, y_arr / y_pred)

        print(f"\n  模型 {name}")
        param_names = ['a', 'b', 'c', 'd'][:len(popt)]
        print(f"    参数: {dict(zip(param_names, ['%.6f' % p for p in popt]))}")
        print(f"    拟合偏比: max={ratios.max():.2f}x, mean={ratios.mean():.2f}x")

        for p, yp, r in zip(points, y_pred, ratios):
            corrected = p["aic_tps_gpu"] * yp
            final = max(corrected / p["measured_tps_gpu"],
                        p["measured_tps_gpu"] / corrected)
            print(f"      {p['name']:>6}: pred_factor={yp:>6.1f}, actual={p['ratio']:>6.1f}, "
                  f"fit={r:.2f}x | corrected={corrected:>6.1f} vs measured={p['measured_tps_gpu']:>6.1f} "
                  f"({final:.2f}x)")

        # LOOCV
        loocv_errors = []
        for i in range(len(points)):
            train_idx = [j for j in range(len(points)) if j != i]
            X_train = X[:, train_idx]
            y_train = y_arr[train_idx]
            try:
                popt_loo, _ = curve_fit(fn, X_train, y_train, p0=p0,
                                        maxfev=50000, bounds=bounds)
                y_test = fn(X[:, i:i+1], *popt_loo)[0]
                err = max(y_test / y_arr[i], y_arr[i] / y_test)
            except Exception:
                err = float("inf")
            loocv_errors.append(err)

        max_loocv = max(loocv_errors)
        print(f"    LOOCV: max={max_loocv:.2f}x, mean={np.mean(loocv_errors):.2f}x")
        for p, e in zip(points, loocv_errors):
            print(f"      {p['name']:>6}: {e:.2f}x")

        if max_loocv < best["loocv_max"]:
            best = {"name": name, "params": popt, "fn": fn, "loocv_max": max_loocv, "X": X}

    return best


def main():
    pareto_data = read_pareto_data()

    # Level 1: 总偏差
    points = level1_analysis(pareto_data)
    if len(points) < 3:
        print("数据不足")
        return

    # Level 2: 观察 per-batch 分布
    level2_per_batch_analysis(pareto_data)

    # 拟合
    best = fit_level1(points)

    # 输出最佳结果
    print(f"\n\n{'='*80}")
    print("最终结果")
    print("=" * 80)

    if best["params"] is not None:
        fn, popt, name = best["fn"], best["params"], best["name"]
        print(f"  最佳模型: {name}")
        print(f"  参数: {['%.6f' % p for p in popt]}")
        print(f"  LOOCV 最大误差: {best['loocv_max']:.2f}x")

        print(f"\n  校正后预测:")
        for p in points:
            if best["X"].shape[0] == 2:
                x = np.array([[p["isl"]], [p["osl"]]])
            else:
                x = np.array([[p["isl"]], [p["osl"]], [p["aic_bs"]]])
            factor = fn(x, *popt)[0]
            corrected = p["aic_tps_gpu"] * factor
            err = max(corrected / p["measured_tps_gpu"],
                      p["measured_tps_gpu"] / corrected)
            print(f"    {p['name']:>6}: {p['aic_tps_gpu']:>5.1f} x {factor:>5.1f} "
                  f"= {corrected:>6.1f} vs {p['measured_tps_gpu']:>6.1f} ({err:.2f}x)")

        print(f"\n  ── vllm_backend.py 代码 ──")
        print(f"  # CB efficiency correction: {name}")
        print(f"  # LOOCV max error: {best['loocv_max']:.2f}x")
        param_names = ['a', 'b', 'c'][:len(popt)]
        for pname, pval in zip(param_names, popt):
            print(f"  {pname} = {pval:.6f}")
    else:
        print("  拟合失败")


if __name__ == "__main__":
    main()
