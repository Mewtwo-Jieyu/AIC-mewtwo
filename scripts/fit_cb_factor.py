#!/usr/bin/env python3
"""
拟合 Continuous Batching 吞吐量校正因子（B1 v2）。

v2 改进：在同一并行配置（tp16dp1）下对齐拟合，消除 tp/dp 配置差异的混淆。

策略：
1. Monkey-patch 禁用现有 B1/B1b 因子，获取 AIC raw 预测
2. 在 tp=16 dp=1 下，对每个 benchmark 场景调用 AIC SDK
3. 计算 ratio = measured_tok_s_gpu / aic_raw_tok_s_gpu
4. 拟合 cb_factor = f(ISL, OSL, b) 的最佳函数形式
5. LOOCV 验证泛化能力
6. 输出可直接粘贴到 vllm_backend.py 的代码

数据来源：vllm h200 kimi 实测数据-整理版 0.17.md
配置：Kimi-K2.5, vLLM 0.17, H200 SXM x16, tp=16 dp=1
"""

import logging
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aiconfigurator.sdk.backends.vllm_backend import VLLMBackend
from aiconfigurator.sdk.config import ModelConfig, RuntimeConfig
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.perf_database import PerfDatabase
from aiconfigurator.sdk import common

logging.basicConfig(level=logging.WARNING, format="%(name)s - %(levelname)s - %(message)s")

# ── Benchmark 实测数据 (vLLM 0.17, H200x16, Kimi-K2.5, tp=16 dp=1) ──────────
# 每条：(场景名, ISL, OSL, 实测并发, 实测 tok/s/GPU)
# 注：实测并发不一定是帕累托最优点，是固定并发下的测量值。
# 后续可在此列表中追加更多 (场景, b) 点以提高拟合精度。
BENCHMARK_DATA = [
    ("3k-3k", 3000, 3000, 128, 254.2),
    ("8k-2k", 8000, 2000, 256, 575.4),
    ("10k-2k", 10000, 2000, 32, 267.6),
    ("10k-3k", 10000, 3000, 128, 386.9),
    ("16k-2k", 16000, 2000, 32, 352.7),
    ("30k-3k", 30000, 3000, 8, 179.2),
    ("32k-1k", 32000, 1000, 16, 559.9),
    # 120k-4k excluded: AIC reports OOM
    # ── 后续可补充的测试点（提高拟合精度）──
    # ("3k-3k",  3000, 3000, 32,  ???),
    # ("3k-3k",  3000, 3000, 64,  ???),
    # ("8k-2k",  8000, 2000, 32,  ???),
    # ("8k-2k",  8000, 2000, 64,  ???),
    # ("8k-2k",  8000, 2000, 128, ???),
    # ("16k-2k", 16000, 2000, 64, ???),
    # ("30k-3k", 30000, 3000, 16, ???),
]

MODEL_PATH = "moonshotai/Kimi-K2.5"
SYSTEM = "h200_sxm"
BACKEND = "vllm"
TP = 16


def collect_aic_predictions() -> list[dict]:
    """调用 AIC SDK 获取 raw 吞吐量预测（禁用 B1/B1b 因子）。"""
    # ── Monkey-patch: 禁用现有校正因子，获取 raw 预测 ──
    original_cb = VLLMBackend._get_cb_efficiency_factor
    original_ttft = VLLMBackend._get_ttft_cb_correction
    VLLMBackend._get_cb_efficiency_factor = staticmethod(lambda b, isl, osl: 1.0)
    VLLMBackend._get_ttft_cb_correction = staticmethod(lambda b, isl, osl: 1.0)

    try:
        model_config = ModelConfig(tp_size=TP, pp_size=1, moe_tp_size=TP, moe_ep_size=1)
        model = get_model(MODEL_PATH, model_config, backend_name=BACKEND)

        systems_root = str(
            Path(__file__).resolve().parent.parent / "src" / "aiconfigurator" / "systems"
        )
        db = PerfDatabase(
            system=SYSTEM,
            backend=BACKEND,
            version="0.12.0",
            systems_root=systems_root,
        )
        backend = VLLMBackend()

        print("收集 AIC raw 吞吐量预测（tp=16, dp=1, B1/B1b 已禁用）...")
        results = []
        for scenario, isl, osl, bench_b, measured_tps_gpu in BENCHMARK_DATA:
            ctx_tokens = isl
            try:
                summary = backend.run_agg(
                    model,
                    db,
                    RuntimeConfig(isl=isl, osl=osl, batch_size=bench_b),
                    database_mode=common.DatabaseMode.HYBRID,
                    ctx_tokens=ctx_tokens,
                )
                df = summary.get_summary_df()
                if df is None or df.empty:
                    print(f"  {scenario} b={bench_b:>4}: 无结果")
                    continue
                row = df.iloc[0]
                aic_tps_gpu = float(row.get("tokens/s/gpu", float("nan")))
                if np.isnan(aic_tps_gpu) or aic_tps_gpu <= 0:
                    print(f"  {scenario} b={bench_b:>4}: 吞吐无效 ({aic_tps_gpu})")
                    continue

                ratio = measured_tps_gpu / aic_tps_gpu
                results.append({
                    "scenario": scenario,
                    "isl": isl,
                    "osl": osl,
                    "b": bench_b,
                    "aic_tps_gpu": aic_tps_gpu,
                    "measured_tps_gpu": measured_tps_gpu,
                    "ratio": ratio,
                })
                print(
                    f"  {scenario:>6} b={bench_b:>4}: "
                    f"AIC_raw={aic_tps_gpu:>8.1f}  measured={measured_tps_gpu:>8.1f}  "
                    f"ratio={ratio:>5.2f}x"
                )
            except Exception as e:
                print(f"  {scenario} b={bench_b:>4}: 错误 - {e}")

        return results
    finally:
        # ── 恢复原始方法 ──
        VLLMBackend._get_cb_efficiency_factor = original_cb
        VLLMBackend._get_ttft_cb_correction = original_ttft


# ── 拟合函数（ratio = measured / aic_raw，ratio > 1 表示 AIC 低估）──────────────

def model_A(X, a, b_coeff, c):
    """ratio = a * (ISL/OSL)^b_coeff + c  (只依赖 ISL/OSL)"""
    isl_osl, = X
    return a * (isl_osl ** b_coeff) + c


def model_B(X, a, b_coeff, c, d):
    """ratio = a * (ISL/OSL)^b_coeff * bs^c + d  (ISL/OSL + batch size)"""
    isl_osl, bs = X
    return a * (isl_osl ** b_coeff) * (bs ** c) + d


def model_C(X, a, b_coeff, c):
    """ratio = a * ln(ISL/OSL) + b_coeff * ln(bs) + c  (log-linear)"""
    isl_osl, bs = X
    return a * np.log(np.maximum(isl_osl, 0.01)) + b_coeff * np.log(np.maximum(bs, 1)) + c


def model_D(X, a, b_coeff, c):
    """ratio = a * (ISL/OSL)^b_coeff / bs^c  (无偏移)"""
    isl_osl, bs = X
    return a * (isl_osl ** b_coeff) / (bs ** c)


def model_E(X, a, b_coeff, c, d):
    """ratio = a * ISL^b_coeff * OSL^c + d"""
    isl, osl = X
    return a * (isl ** b_coeff) * (osl ** c) + d


def model_F(X, a, b_coeff, c, d):
    """ratio = a * ln(ISL) + b_coeff * ln(OSL) + c * ln(bs) + d"""
    isl, osl, bs = X
    return a * np.log(isl) + b_coeff * np.log(osl) + c * np.log(np.maximum(bs, 1)) + d


def fit_cb_factor(points: list[dict]) -> dict:
    """拟合 CB 吞吐量校正因子，使用 LOOCV 评估泛化。"""
    print(f"\n{'='*80}")
    print(f"拟合数据集: {len(points)} 个点")
    print("=" * 80)

    if len(points) < 3:
        print("数据不足（需至少 3 个点），退出")
        return {}

    isl_arr = np.array([p["isl"] for p in points], dtype=float)
    osl_arr = np.array([p["osl"] for p in points], dtype=float)
    isl_osl_arr = isl_arr / osl_arr
    bs_arr = np.array([p["b"] for p in points], dtype=float)
    y_arr = np.array([p["ratio"] for p in points], dtype=float)

    X_isl_osl = isl_osl_arr.reshape(1, -1)
    X_isl_osl_bs = np.vstack([isl_osl_arr, bs_arr])
    X_isl_osl_separate = np.vstack([isl_arr, osl_arr])
    X_isl_osl_bs_separate = np.vstack([isl_arr, osl_arr, bs_arr])

    models = [
        (model_A, "A: a*(ISL/OSL)^b + c",             [1.0, 0.5, 1.0],       (-np.inf, np.inf), X_isl_osl),
        (model_B, "B: a*(ISL/OSL)^b*bs^c + d",        [1.0, 0.5, -0.3, 1.0], (-np.inf, np.inf), X_isl_osl_bs),
        (model_C, "C: a*ln(ISL/OSL)+b*ln(bs)+c",      [1.0, -0.5, 1.0],      (-np.inf, np.inf), X_isl_osl_bs),
        (model_D, "D: a*(ISL/OSL)^b/bs^c",            [1.0, 0.5, 0.3],       (0, np.inf),        X_isl_osl_bs),
        (model_E, "E: a*ISL^b*OSL^c + d",             [0.001, 0.5, -0.3, 0], (-np.inf, np.inf), X_isl_osl_separate),
        (model_F, "F: a*ln(ISL)+b*ln(OSL)+c*ln(bs)+d", [1.0, -1.0, -0.5, 0], (-np.inf, np.inf), X_isl_osl_bs_separate),
    ]

    best = {"name": None, "params": None, "fn": None, "loocv_max": float("inf"), "X": None, "X_dim": 0}

    for fn, name, p0, bounds, X in models:
        try:
            popt, _ = curve_fit(fn, X, y_arr, p0=p0, maxfev=100000, bounds=bounds)
        except Exception as e:
            print(f"\n  {name}: 拟合失败 - {e}")
            continue

        y_pred = fn(X, *popt)
        # 对称相对误差
        ratios = np.maximum(y_pred / y_arr, y_arr / y_pred)

        print(f"\n  模型 {name}")
        param_names = ["a", "b", "c", "d"][:len(popt)]
        print(f"    参数: {dict(zip(param_names, [f'{p:.6f}' for p in popt]))}")
        print(f"    拟合误差: max={ratios.max():.2f}x  mean={ratios.mean():.2f}x")

        for p, yp, r in zip(points, y_pred, ratios):
            corrected = p["aic_tps_gpu"] * yp
            final = max(corrected / p["measured_tps_gpu"],
                        p["measured_tps_gpu"] / corrected)
            print(f"      {p['scenario']:>8} b={p['b']:>4}: "
                  f"ratio_pred={yp:>6.2f}  actual={p['ratio']:>6.2f}  "
                  f"fit_err={r:.2f}x  |  corrected={corrected:>8.1f} "
                  f"vs measured={p['measured_tps_gpu']:>8.1f} ({final:.2f}x)")

        # LOOCV
        loocv_errors = []
        for i in range(len(points)):
            train_idx = [j for j in range(len(points)) if j != i]
            X_train = X[:, train_idx]
            y_train = y_arr[train_idx]
            try:
                popt_loo, _ = curve_fit(fn, X_train, y_train, p0=p0,
                                        maxfev=100000, bounds=bounds)
                y_test = fn(X[:, i:i+1], *popt_loo)[0]
                err = max(y_test / y_arr[i], y_arr[i] / y_test)
            except Exception:
                err = float("inf")
            loocv_errors.append(err)

        max_loocv = max(loocv_errors)
        print(f"    LOOCV: max={max_loocv:.2f}x  mean={np.mean(loocv_errors):.2f}x")
        for p, e in zip(points, loocv_errors):
            print(f"      {p['scenario']:>8} b={p['b']:>4}: {e:.2f}x")

        if max_loocv < best["loocv_max"]:
            best = {"name": name, "params": popt, "fn": fn,
                    "loocv_max": max_loocv, "X": X, "X_dim": X.shape[0]}

    return best


def print_final_result(best: dict, points: list[dict]) -> None:
    """输出最终结果和可粘贴代码。"""
    print(f"\n{'='*80}")
    print("最终结果")
    print("=" * 80)

    if best.get("params") is None:
        print("  拟合失败，无法输出结果")
        return

    fn, popt, name = best["fn"], best["params"], best["name"]
    x_dim = best["X_dim"]

    print(f"  最佳模型: {name}")
    param_names = ["a", "b", "c", "d"][:len(popt)]
    print(f"  参数: {dict(zip(param_names, [f'{p:.6f}' for p in popt]))}")
    print(f"  LOOCV 最大误差: {best['loocv_max']:.2f}x")

    print(f"\n  校正后吞吐量预测:")
    for p in points:
        isl_osl = p["isl"] / p["osl"]
        b = float(p["b"])
        if x_dim == 1:
            x = np.array([[isl_osl]])
        elif x_dim == 2:
            x = np.array([[isl_osl], [b]])
        else:
            x = np.array([[p["isl"]], [p["osl"]], [b]])
        factor = max(fn(x, *popt)[0], 1.0)
        corrected = p["aic_tps_gpu"] * factor
        err = max(corrected / p["measured_tps_gpu"],
                  p["measured_tps_gpu"] / corrected)
        print(f"    {p['scenario']:>8} b={p['b']:>4}: "
              f"{p['aic_tps_gpu']:>8.1f} x {factor:>5.2f} "
              f"= {corrected:>8.1f}  vs  {p['measured_tps_gpu']:>8.1f}  ({err:.2f}x)")

    # ── 生成可粘贴代码 ──
    print(f"\n{'─'*80}")
    print("  ── 可粘贴到 vllm_backend.py 的代码 ──")
    print(f"{'─'*80}")
    param_str = ", ".join(f"{v:.6f}" for v in popt)
    print(f"""
    @staticmethod
    def _get_cb_efficiency_factor(b: int, isl: int, osl: int) -> float:
        \"\"\"Continuous batching efficiency correction factor (v2).

        Fitted at tp=16 dp=1, same config as benchmark, to isolate CB
        efficiency from parallelism config differences.

        Fit: {name}
        LOOCV max error: {best['loocv_max']:.2f}x
        Data: Kimi-K2.5 + vLLM 0.17 + H200 SXM x16, tp=16 dp=1
        \"\"\"
        if b <= 1:
            return 1.0""")

    for pname, pval in zip(param_names, popt):
        print(f"        {pname} = {pval:.6f}")

    if "A:" in name:
        print(f"        isl_osl_ratio = isl / max(osl, 1)")
        print(f"        factor = a * (isl_osl_ratio ** b) + c")
    elif "B:" in name:
        print(f"        isl_osl_ratio = isl / max(osl, 1)")
        print(f"        factor = a * (isl_osl_ratio ** b) * (b ** c) + d")
    elif "C:" in name:
        print(f"        import math")
        print(f"        isl_osl_ratio = isl / max(osl, 1)")
        print(f"        factor = a * math.log(max(isl_osl_ratio, 0.01)) + b * math.log(max(b, 1)) + c")
    elif "D:" in name:
        print(f"        isl_osl_ratio = isl / max(osl, 1)")
        print(f"        factor = a * (isl_osl_ratio ** b) / (b ** c)")
    elif "E:" in name:
        print(f"        factor = a * (isl ** b) * (osl ** c) + d")
    elif "F:" in name:
        print(f"        import math")
        print(f"        factor = a * math.log(isl) + b * math.log(osl) + c * math.log(max(b, 1)) + d")

    print(f"        return max(factor, 1.0)")


def print_suggested_measurements(points: list[dict]) -> None:
    """输出建议补充的测试点。"""
    print(f"\n{'='*80}")
    print("建议补充的测试点（提高拟合精度）")
    print("=" * 80)

    # 找出数据稀疏的 ISL/OSL 和 b 区间
    existing = {(p["isl"], p["osl"], p["b"]) for p in points}
    suggestions = []

    # 每个场景补充 2-3 个不同 b 值
    for scenario, isl, osl, _, _ in BENCHMARK_DATA:
        for b_candidate in [4, 8, 16, 32, 64, 128, 256]:
            if (isl, osl, b_candidate) not in existing:
                suggestions.append((scenario, isl, osl, b_candidate))

    # 按 ISL/OSL 排序，每个场景最多显示 3 个
    from collections import Counter
    shown = Counter()
    for scenario, isl, osl, b in sorted(suggestions, key=lambda x: (x[1]/x[2], x[3])):
        if shown[scenario] < 3:
            print(f"  ({scenario!r:>10}, {isl:>5}, {osl:>5}, {b:>4}, ???),")
            shown[scenario] += 1


def main():
    print("B1 v2: CB 吞吐量校正因子拟合（同配置对齐）")
    print("=" * 80)
    print(f"配置: {MODEL_PATH}, {SYSTEM}, tp={TP} dp=1")
    print()

    # Step 1: 收集 AIC raw 预测
    points = collect_aic_predictions()

    if len(points) < 3:
        print(f"\n有效数据点不足（{len(points)} 个），请检查 AIC SDK 配置")
        return

    # Step 2: 展示 ratio 分布
    print(f"\n{'='*80}")
    print("吞吐量 ratio 分布（measured / aic_raw）")
    print("=" * 80)
    ratios = [p["ratio"] for p in points]
    print(f"  n={len(ratios)}, min={min(ratios):.2f}x, max={max(ratios):.2f}x, "
          f"mean={np.mean(ratios):.2f}x, median={np.median(ratios):.2f}x")

    # Step 3: 拟合
    best = fit_cb_factor(points)

    # Step 4: 输出最终结果
    print_final_result(best, points)

    # Step 5: 建议后续补充的测试点
    print_suggested_measurements(points)


if __name__ == "__main__":
    main()
