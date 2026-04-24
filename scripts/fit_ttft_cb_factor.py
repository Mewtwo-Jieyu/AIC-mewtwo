#!/usr/bin/env python3
"""
B1b：拟合 TTFT Continuous Batching 校正因子。

在 vLLM CB 模式下，TTFT 不随并发线性增长（每个请求立即开始 prefill）。
AIC 的批处理假设（TTFT ∝ b × prefill_time）导致大并发场景严重高估 TTFT。

本脚本：
1. Monkey-patch 禁用现有 TTFT 校正，调用 AIC SDK 获取各场景 raw AIC 预测 TTFT
2. 与实测 P95 TTFT 对比，计算 ttft_ratio = aic_ttft / real_ttft
3. 拟合 ttft_ratio = f(b, ISL, OSL) 的最佳函数形式
4. LOOCV 验证泛化能力
5. 输出可直接粘贴到 vllm_backend.py 的代码

校正应用：
    ttft_correction = _get_ttft_cb_correction(b, isl, osl)
    ttft /= ttft_correction  # AIC 的 batch 模型假设除掉 CB 效率

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

# ── 实测 P95 TTFT 数据（vLLM 0.17, H200x16, Kimi-K2.5, tp=16 dp=1）────────────
# 数据来源：vllm h200 kimi 实测数据-整理版 0.17.md
# quality: "clean"=可信，"noisy"=存疑（仅用于验证，不参与拟合）
REAL_TTFT_DATA = [
    # (scenario, isl, osl, b, p95_ttft_ms, quality)
    #
    # quality:
    #   "clean"  — 可信且在 CB 过高估 regime（AIC >> real，适合拟合 B1b）
    #   "noisy"  — 测量存疑，仅用于验证
    #   "exclude" — 在 queue-wait 主导 regime（AIC << real），B1b 不适用此区间
    #
    # 30k-3k: ISL/OSL=10，b 线性递增，高质量
    ("30k-3k", 30000, 3000,   4,  1231.0, "clean"),
    ("30k-3k", 30000, 3000,   6,  1679.0, "clean"),
    ("30k-3k", 30000, 3000,   8,  1823.0, "clean"),
    # 20k-5k: ISL/OSL=4
    ("20k-5k", 20000, 5000,   4,  1326.0, "clean"),
    ("20k-5k", 20000, 5000,   8,  1649.0, "clean"),
    # 16k-2k: ISL/OSL=8，中等并发
    ("16k-2k", 16000, 2000,  16,   782.0, "clean"),
    ("16k-2k", 16000, 2000,  32,   814.0, "clean"),
    # 32k-1k: AIC 已在目标精度内（ratio=1.44x < 2x），且 queue-wait ≈ batch-time
    # → 不需要 B1b 修正，不参与拟合（以免拉低 correction 估计）
    ("32k-1k", 32000, 1000,  16,  9155.0, "noisy"),
    # 8k-2k: b=128,256 在 queue-wait 主导 regime（AIC 严重低估 TTFT），B1b 无效
    # 这些 b 值下队列等待时间主导，AIC 完全不建模 queue wait → ratio < 1
    # B2（CB queue-aware 建模）才是正确解法
    ("8k-2k",   8000, 2000, 128, 14831.0, "exclude"),
    ("8k-2k",   8000, 2000, 256, 27372.0, "exclude"),
    # 3k-3k: ISL/OSL=1，两次测量差异大，取较大值（更保守）
    ("3k-3k",   3000, 3000,  64,   617.0, "noisy"),
    ("3k-3k",   3000, 3000, 100,  2773.0, "noisy"),
    ("3k-3k",   3000, 3000, 128,  4164.0, "noisy"),
    # 10k-2k: b=16 合理，b=8/32 异常（b=32 < b=8），只用 b=16
    ("10k-2k", 10000, 2000,  16,   855.0, "noisy"),
]

MODEL_PATH = "moonshotai/Kimi-K2.5"
SYSTEM = "h200_sxm"
BACKEND = "vllm"
TP = 16


def collect_aic_predictions() -> list[dict]:
    """调用 AIC SDK 获取每个数据点的 raw AIC 预测 TTFT（tp=16, dp=1）。"""
    original_ttft = VLLMBackend._get_ttft_cb_correction
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

        print("收集 AIC raw 预测 TTFT（tp=16, dp=1, B1b 已禁用）...")
        results = []
        for scenario, isl, osl, b, real_ttft_ms, quality in REAL_TTFT_DATA:
            # exclude 表示 queue-wait 主导 regime，B1b 不适用，但仍收集 raw 预测供参考

            ctx_tokens = isl  # 标准假设
            try:
                summary = backend.run_agg(
                    model,
                    db,
                    RuntimeConfig(isl=isl, osl=osl, batch_size=b),
                    database_mode=common.DatabaseMode.HYBRID,
                    ctx_tokens=ctx_tokens,
                )
                df = summary.get_summary_df()
                if df is None or df.empty:
                    print(f"  {scenario} b={b:>4}: 无结果（get_summary_df 为空）")
                    continue
                row = df.iloc[0]
                aic_ttft_ms = float(row.get("ttft", float("nan")))
                if np.isnan(aic_ttft_ms) or aic_ttft_ms <= 0:
                    print(f"  {scenario} b={b:>4}: TTFT 无效 ({aic_ttft_ms})")
                    continue
                results.append({
                    "scenario": scenario,
                    "isl": isl,
                    "osl": osl,
                    "b": b,
                    "aic_ttft_ms": aic_ttft_ms,
                    "real_ttft_ms": real_ttft_ms,
                    "ttft_ratio": aic_ttft_ms / real_ttft_ms,
                    "quality": quality,
                })
                print(
                    f"  {scenario} b={b:>4}: AIC_raw={aic_ttft_ms:>8.1f}ms  "
                    f"real={real_ttft_ms:>8.1f}ms  ratio={aic_ttft_ms/real_ttft_ms:>5.2f}x  "
                    f"[{quality}]"
                )
            except Exception as e:
                print(f"  {scenario} b={b:>4}: 错误 - {e}")

        return results
    finally:
        VLLMBackend._get_ttft_cb_correction = original_ttft


# ── 拟合函数（ratio = aic_ttft / real_ttft，ratio > 1 表示 AIC 高估）─────────────

def model_A(X, a, c, d):
    """ratio = a * b^c + d  (ISL/OSL 无关)"""
    b, = X
    return a * (b ** c) + d


def model_B(X, a, b_exp, c, d):
    """ratio = a * (ISL/OSL)^b_exp * b^c + d  (B1 形式)"""
    isl_osl, b = X
    return a * (isl_osl ** b_exp) * (b ** c) + d


def model_C(X, a, b_exp, c, d):
    """ratio = a * (ISL/OSL)^(-b_exp) * b^c + d  (ISL/OSL 负相关，更长 ISL → 更小比率)"""
    isl_osl, b = X
    return a * (isl_osl ** (-b_exp)) * (b ** c) + d


def model_D(X, a, b_exp, c):
    """ratio = a * b^c / (ISL/OSL)^b_exp  (无偏移项)"""
    isl_osl, b = X
    return a * (b ** c) / (isl_osl ** b_exp)


def model_E(X, a, c, d, e):
    """ratio = a * b^c * log(b)^e + d  (对数修正)"""
    b, = X
    return a * (b ** c) * (np.log(np.maximum(b, 1)) ** e) + d


def model_F(X, a, b_exp, c):
    """ratio = 1 + a * (b-1)^c / (ISL/OSL)^b_exp  (b=1 → ratio=1 物理约束)"""
    isl_osl, b = X
    return 1.0 + a * (np.maximum(b - 1, 0) ** c) / (isl_osl ** b_exp)


def model_G(X, a, b_coeff, c):
    """ratio = a * ln(ISL/OSL) + b_coeff * ln(b) + c  (log-linear)

    Model B 的解析等价式（通过 Taylor 展开）。参数稳定，物理可解释。
    """
    isl_osl, b = X
    return a * np.log(np.maximum(isl_osl, 0.01)) + b_coeff * np.log(np.maximum(b, 1)) + c


def fit_ttft_correction(points: list[dict]) -> dict:
    """拟合 TTFT CB 校正因子，使用 LOOCV 评估泛化。"""
    # 只用 clean 数据拟合
    clean = [p for p in points if p["quality"] == "clean"]
    print(f"\n{'='*80}")
    print(f"拟合数据集: {len(clean)} 个 clean 点（共 {len(points)} 个）")
    print("=" * 80)

    if len(clean) < 4:
        print("数据不足（需至少 4 个 clean 点），退出")
        return {}

    isl_osl_arr = np.array([p["isl"] / p["osl"] for p in clean], dtype=float)
    b_arr = np.array([p["b"] for p in clean], dtype=float)
    y_arr = np.array([p["ttft_ratio"] for p in clean], dtype=float)

    X1 = b_arr.reshape(1, -1)                       # 只用 b
    X2 = np.vstack([isl_osl_arr, b_arr])              # ISL/OSL + b

    models = [
        (model_A, "A: a*b^c + d",                   [5.0,  1.0, 1.0],       (-np.inf, np.inf), X1),
        (model_B, "B: a*(ISL/OSL)^e*b^c+d (正)",   [2.0,  0.3, 1.0, 1.0],  (-np.inf, np.inf), X2),
        (model_C, "C: a*(ISL/OSL)^(-e)*b^c+d (负)", [5.0,  0.3, 1.0, 1.0],  (0, np.inf),       X2),
        (model_D, "D: a*b^c/(ISL/OSL)^e",           [2.0,  0.3, 1.0],       (0, np.inf),       X2),
        (model_E, "E: a*b^c*log(b)^e + d",          [1.0,  0.5, 1.0, 0.5],  (-np.inf, np.inf), X1),
        (model_F, "F: 1+a*(b-1)^c/(ISL/OSL)^e",   [5.0,  0.5, 1.0],       (0, np.inf),       X2),
        (model_G, "G: a*ln(ISL/OSL)+b*ln(b)+c",  [2.0,  2.0, -2.0],      (-np.inf, np.inf), X2),
    ]

    best = {"name": None, "params": None, "fn": None, "loocv_max": float("inf"), "X": None}

    for fn, name, p0, bounds, X in models:
        try:
            popt, _ = curve_fit(fn, X, y_arr, p0=p0, maxfev=100000, bounds=bounds)
        except Exception as e:
            print(f"\n  {name}: 拟合失败 - {e}")
            continue

        y_pred = fn(X, *popt)
        # 对称相对误差：max(pred/actual, actual/pred)
        ratios = np.maximum(y_pred / y_arr, y_arr / y_pred)

        print(f"\n  模型 {name}")
        param_names = ["a", "b_exp", "c", "d", "e"][:len(popt)]
        print(f"    参数: {dict(zip(param_names, [f'{p:.6f}' for p in popt]))}")
        print(f"    拟合误差: max={ratios.max():.2f}x  mean={ratios.mean():.2f}x")

        for p, yp, r in zip(clean, y_pred, ratios):
            corrected_ttft = p["aic_ttft_ms"] / yp
            final_err = max(corrected_ttft / p["real_ttft_ms"],
                            p["real_ttft_ms"] / corrected_ttft)
            print(f"      {p['scenario']:>8} b={p['b']:>4}: "
                  f"ratio_pred={yp:>6.2f}  actual={p['ttft_ratio']:>6.2f}  "
                  f"fit_err={r:.2f}x  |  corrected={corrected_ttft:>8.1f}ms "
                  f"real={p['real_ttft_ms']:>8.1f}ms ({final_err:.2f}x)")

        # LOOCV
        loocv_errors = []
        for i in range(len(clean)):
            train_idx = [j for j in range(len(clean)) if j != i]
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
        for p, e in zip(clean, loocv_errors):
            print(f"      {p['scenario']:>8} b={p['b']:>4}: {e:.2f}x")

        if max_loocv < best["loocv_max"]:
            best = {"name": name, "params": popt, "fn": fn,
                    "loocv_max": max_loocv, "X": X, "X_dim": X.shape[0]}

    return best


def validate_noisy_points(points: list[dict], best: dict) -> None:
    """用最佳模型对 noisy/exclude 数据点做预测，供参考。"""
    noisy = [p for p in points if p["quality"] in ("noisy", "exclude")]
    if not noisy or best.get("params") is None:
        return
    print(f"\n{'='*80}")
    print("Noisy 数据点验证（仅用于参考，未参与拟合）")
    print("=" * 80)
    fn, popt = best["fn"], best["params"]
    x_dim = best["X_dim"]
    for p in noisy:
        isl_osl = p["isl"] / p["osl"]
        b = float(p["b"])
        if x_dim == 1:
            x = np.array([[b]])
        else:
            x = np.array([[isl_osl], [b]])
        try:
            ratio_pred = max(fn(x, *popt)[0], 1.0)
        except Exception:
            ratio_pred = 1.0
        corrected_ttft = p["aic_ttft_ms"] / ratio_pred
        final_err = max(corrected_ttft / p["real_ttft_ms"],
                        p["real_ttft_ms"] / corrected_ttft)
        print(f"  {p['scenario']:>8} b={p['b']:>4}: "
              f"AIC={p['aic_ttft_ms']:>8.1f}ms  ratio_pred={ratio_pred:>5.2f}  "
              f"corrected={corrected_ttft:>8.1f}ms  real={p['real_ttft_ms']:>8.1f}ms  "
              f"err={final_err:.2f}x")


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
    clean = [p for p in points if p["quality"] == "clean"]

    print(f"  最佳模型: {name}")
    print(f"  参数: {[f'{p:.6f}' for p in popt]}")
    print(f"  LOOCV 最大误差: {best['loocv_max']:.2f}x")

    print(f"\n  校正后 TTFT 预测:")
    for p in clean:
        isl_osl = p["isl"] / p["osl"]
        b = float(p["b"])
        if x_dim == 1:
            x = np.array([[b]])
        else:
            x = np.array([[isl_osl], [b]])
        ratio_pred = max(fn(x, *popt)[0], 1.0)
        corrected = p["aic_ttft_ms"] / ratio_pred
        err = max(corrected / p["real_ttft_ms"], p["real_ttft_ms"] / corrected)
        print(f"    {p['scenario']:>8} b={p['b']:>4}: "
              f"{p['aic_ttft_ms']:>8.1f}ms / {ratio_pred:.2f} "
              f"= {corrected:>8.1f}ms  vs  {p['real_ttft_ms']:>8.1f}ms  ({err:.2f}x)")

    print(f"\n{'─'*80}")
    print("  ── 可粘贴到 vllm_backend.py 的代码 ──")
    print(f"{'─'*80}")
    if "A:" in name:
        param_names = ["a", "c", "d"]
    elif "B:" in name or "C:" in name:
        param_names = ["a", "b_exp", "c", "d"]
    elif "D:" in name:
        param_names = ["a", "b_exp", "c"]
    elif "E:" in name:
        param_names = ["a", "c", "d", "e"]
    elif "F:" in name:
        param_names = ["a", "b_exp", "c"]
    elif "G:" in name:
        param_names = ["a", "b_coeff", "c"]
    else:
        param_names = [f"p{i}" for i in range(len(popt))]
    param_str = ", ".join(f"{v:.6f}" for v in popt)
    print(f"""
    @staticmethod
    def _get_ttft_cb_correction(b: int, isl: int, osl: int) -> float:
        \"\"\"TTFT correction factor for vLLM Continuous Batching mode.

        In CB mode TTFT does not scale linearly with concurrency — each request
        starts prefilling immediately on arrival. This factor corrects the
        batch-synchronous overestimate from AIC's prefill model.

        Fit: {name}
        LOOCV max error: {best['loocv_max']:.2f}x
        Data: Kimi-K2.5 + vLLM 0.17 + H200 SXM x16, tp=16 dp=1
        \"\"\"
        if b <= 1:
            return 1.0""")

    # Print parameter assignments
    for pname, pval in zip(param_names, popt):
        print(f"        {pname} = {pval:.6f}")

    # Print formula based on model type
    if x_dim == 1:
        if "A:" in name:
            print(f"        factor = a * (b ** c) + d")
        elif "E:" in name:
            print(f"        import math")
            print(f"        factor = a * (b ** c) * (math.log(max(b, 1)) ** e) + d")
        else:
            print("        factor = 1.0")
    else:
        print(f"        isl_osl_ratio = isl / max(osl, 1)")
        if "B:" in name:
            print(f"        factor = a * (isl_osl_ratio ** b_exp) * (b ** c) + d")
        elif "C:" in name:
            print(f"        factor = a * (isl_osl_ratio ** (-b_exp)) * (b ** c) + d")
        elif "D:" in name:
            print(f"        factor = a * (b ** c) / (isl_osl_ratio ** b_exp)")
        elif "G:" in name:
            print(f"        import math")
            print(f"        factor = a * math.log(max(isl_osl_ratio, 0.01)) + b_coeff * math.log(max(b, 1)) + c")

    print(f"        return max(factor, 1.0)")


def main():
    print("B1b TTFT CB 校正因子拟合")
    print("=" * 80)

    # Step 1: 收集 AIC 预测
    points = collect_aic_predictions()

    if len(points) < 4:
        print(f"\n有效数据点不足（{len(points)} 个），请检查 AIC SDK 配置")
        return

    # Step 2: 展示比率分布（包含所有质量级别）
    print(f"\n{'='*80}")
    print("TTFT 比率分布（aic_ttft / real_ttft）")
    print("=" * 80)
    for quality in ("clean", "noisy", "exclude"):
        pts = [p for p in points if p["quality"] == quality]
        if pts:
            ratios = [p["ttft_ratio"] for p in pts]
            over = [r for r in ratios if r > 1.0]
            under = [r for r in ratios if r <= 1.0]
            print(f"  {quality:8s}: n={len(ratios)}, "
                  f"min={min(ratios):.2f}x, max={max(ratios):.2f}x, "
                  f"mean={np.mean(ratios):.2f}x "
                  f"(高估={len(over)}, 低估={len(under)})")
    print()
    clean_ratios = [p["ttft_ratio"] for p in points if p["quality"] == "clean"]

    # Step 3: 拟合
    best = fit_ttft_correction(points)

    # Step 4: 验证 noisy 点
    validate_noisy_points(points, best)

    # Step 5: 输出最终结果
    print_final_result(best, points)

    # Step 6: 场景 A 关键验证提示
    print(f"\n{'='*80}")
    print("场景 A 验证（执行以下命令确认修复效果）")
    print("=" * 80)
    print("""
  cd .worktrees/feature-pr403
  conda run -n aic aiconfigurator cli default \\
    --model-path moonshotai/Kimi-K2.5 --system h200_sxm --backend vllm \\
    --total-gpus 16 --isl 3000 --osl 3000 --ttft 1500 --tpot 500

  期望：推荐最大并发 >> 2（修复前为 b=2，目标 b=32+，实测 b=64 满足 SLA）
""")


if __name__ == "__main__":
    main()
