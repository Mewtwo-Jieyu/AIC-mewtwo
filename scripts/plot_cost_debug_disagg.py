#!/usr/bin/env python3
"""
基于 aiconfigurator 生成的 disagg 场景 pareto.csv / best_config_topn.csv，
做“Prefill/Decode 硬件成本调试”分析：

- 以同一组 disagg 实验（例如 L40_P + A100_D）为基础
- 对 a100 / l40 的节点月租价格进行多组假设
- 在一张图里画出多条 tokens/s/10k vs tokens/s/user 曲线
- 同时在每条曲线下标出 best_config_topn 中的点位置

用法示例：

  python scripts/plot_cost_debug_disagg.py \\
    output_scale/scale_cost/Qwen/\\
    Qwen3-32B_l40s_trtllm_isl2048_osl512_ttft1000_tpot50_368870

会在该目录下生成：
  cost_debug_disagg.png

如果你想调不同的成本场景，可以直接修改本脚本中的 SCENARIOS 配置。
"""

import sys
import math
from pathlib import Path
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt
import pandas as pd

# 单机 8 卡节点的月租金（单位：元 / 月）
BASE_MONTHLY_NODE_RENT: Dict[str, float] = {
    "a100": 35000.0,  # 8×A100 SXM
    "l40": 15000.0,  # 8×L40S
}

# 真正参与计算的租金表（会在不同场景下被覆盖）
MONTHLY_NODE_RENT: Dict[str, float] = BASE_MONTHLY_NODE_RENT.copy()

# 单机 GPU 数（仅用于“多少 GPU 构成一个节点”的分组，成本以节点计）
GPUS_PER_NODE = 8.0

# 希望换算成 tokens/s/10k
TEN_K_CURRENCY = 10_000.0


def normalize_gpu_type_for_cost(raw_type: str) -> str:
    """
    将更细粒度的 GPU 标识（如 a100_sxm, l40s 等）映射到成本表使用的粗粒度类型。
    """
    if not raw_type:
        return "unknown"
    lower = str(raw_type).lower()
    if "a100" in lower:
        return "a100"
    if "l40" in lower:
        return "l40"
    return "unknown"


def safe_int(val):
    try:
        return int(float(val))
    except Exception:
        return None


def safe_get(row, key, default=None):
    """
    安全地从行数据中获取值，支持 pandas Series 和 dict。
    """
    if row is None:
        return default
    try:
        if key in row:
            return row[key]
    except (TypeError, KeyError):
        pass

    try:
        return row.get(key, default)
    except AttributeError:
        return default


def calculate_tokens_per_10k_rmb_disagg(
    tokens_per_gpu: float,
    num_total_gpus: float,
    p_gpu_type: str,
    d_gpu_type: str,
    p_workers: int,
    d_workers: int,
    p_gpus_worker: int,
    d_gpus_worker: int,
    replicas,
) -> float:
    """
    Disagg 场景下按“Prefill/Decode 分别计节点成本”的逻辑计算 tokens/s/10k。

    Step 1: 先算单套服务（replica=1）的 GPU 需求
      P_single_replica_gpus = p_workers × p_gpus_per_worker
      D_single_replica_gpus = d_workers × d_gpus_per_worker

    Step 2: 计算单套的节点数（单机 8 卡）
      P_single_replica_nodes = ceil(P_single_replica_gpus / 8)
      D_single_replica_nodes = ceil(D_single_replica_gpus / 8)

    Step 3: 乘以 replica 数量得到总节点数
      P_total_nodes = P_single_replica_nodes × replicas
      D_total_nodes = D_single_replica_nodes × replicas
      total_nodes = P_total_nodes + D_total_nodes

    成本:
      cost = P_total_nodes × MONTHLY_NODE_RENT[p_type]
           + D_total_nodes × MONTHLY_NODE_RENT[d_type]
    """
    p_type = normalize_gpu_type_for_cost(p_gpu_type)
    d_type = normalize_gpu_type_for_cost(d_gpu_type)

    try:
        tokens_per_gpu_f = float(tokens_per_gpu)
        num_total_gpus_f = float(num_total_gpus)
    except (TypeError, ValueError):
        return 0.0

    if tokens_per_gpu_f <= 0 or num_total_gpus_f <= 0:
        return 0.0

    try:
        r = float(replicas)
    except (TypeError, ValueError):
        r = 1.0
    replicas_int = max(1, int(round(r)))

    try:
        p_single_gpus = max(0, int(p_workers) * int(p_gpus_worker))
        d_single_gpus = max(0, int(d_workers) * int(d_gpus_worker))
    except (TypeError, ValueError):
        return 0.0

    def nodes_for_single_replica(single_gpus: int) -> int:
        if single_gpus <= 0:
            return 0
        return int(math.ceil(single_gpus / GPUS_PER_NODE))

    p_single_nodes = nodes_for_single_replica(p_single_gpus)
    d_single_nodes = nodes_for_single_replica(d_single_gpus)

    if p_single_nodes <= 0 and d_single_nodes <= 0:
        return 0.0

    p_total_nodes = p_single_nodes * replicas_int
    d_total_nodes = d_single_nodes * replicas_int

    if p_total_nodes <= 0 and d_total_nodes <= 0:
        return 0.0

    total_monthly_cost = 0.0
    if p_total_nodes > 0 and p_type in MONTHLY_NODE_RENT:
        total_monthly_cost += p_total_nodes * MONTHLY_NODE_RENT[p_type]
    if d_total_nodes > 0 and d_type in MONTHLY_NODE_RENT:
        total_monthly_cost += d_total_nodes * MONTHLY_NODE_RENT[d_type]

    if total_monthly_cost <= 0:
        return 0.0

    total_tokens_per_sec = tokens_per_gpu_f * num_total_gpus_f
    return (total_tokens_per_sec / total_monthly_cost) * TEN_K_CURRENCY


def has_disagg_cost_columns(df: pd.DataFrame) -> bool:
    """判断 DataFrame 是否包含 disagg 成本计算所需的关键列。"""
    cols = set(df.columns)
    if "(p)workers" not in cols or "(d)workers" not in cols:
        return False
    has_p_gpu_info = "p_gpus_worker" in cols or "(p)tp" in cols
    has_d_gpu_info = "d_gpus_worker" in cols or "(d)tp" in cols
    return has_p_gpu_info and has_d_gpu_info


def infer_gpu_type(exp_name: str, row=None) -> str:
    """
    推断 GPU 类型，优先从行数据的 (p)system/(d)system 列获取，
    否则根据实验目录名推断：
    - 以 "a100" 开头 => a100
    - 以 "l40" 开头  => l40
    其它情况返回 "unknown"。
    """
    if row is not None:
        try:
            p_sys = row["(p)system"] if "(p)system" in row else None
        except (KeyError, TypeError):
            try:
                p_sys = row.get("(p)system")
            except (AttributeError, TypeError):
                p_sys = None

        try:
            d_sys = row["(d)system"] if "(d)system" in row else None
        except (KeyError, TypeError):
            try:
                d_sys = row.get("(d)system")
            except (AttributeError, TypeError):
                d_sys = None

        if p_sys and str(p_sys).strip():
            normalized = normalize_gpu_type_for_cost(str(p_sys))
            if normalized != "unknown":
                return normalized

        if d_sys and str(d_sys).strip():
            normalized = normalize_gpu_type_for_cost(str(d_sys))
            if normalized != "unknown":
                return normalized

    lower = exp_name.lower()
    if lower.startswith("a100"):
        return "a100"
    if lower.startswith("l40"):
        return "l40"
    return "unknown"


def tokens_per_10k_currency_disagg_row(row, exp_name: str) -> float:
    """对单行 disagg 记录进行成本计算，返回 tokens/s/10k。"""
    tokens_per_gpu = safe_get(row, "tokens/s/gpu", 0) or 0
    num_total_gpus = safe_get(row, "num_total_gpus", 0) or 0

    try:
        if float(tokens_per_gpu) <= 0 or float(num_total_gpus) <= 0:
            return 0.0
    except (TypeError, ValueError):
        return 0.0

    p_gpu_type = safe_get(row, "(p)system")
    d_gpu_type = safe_get(row, "(d)system")

    if not p_gpu_type:
        p_gpu_type = infer_gpu_type(exp_name, row)
    if not d_gpu_type:
        d_gpu_type = infer_gpu_type(exp_name, row)

    p_workers = safe_int(safe_get(row, "(p)workers")) or 0
    d_workers = safe_int(safe_get(row, "(d)workers")) or 0

    replicas = safe_get(row, "replicas", 1)

    p_gpus_worker = safe_get(row, "p_gpus_worker")
    d_gpus_worker = safe_get(row, "d_gpus_worker")

    if p_gpus_worker is None or p_gpus_worker == "" or (hasattr(pd, "isna") and pd.isna(p_gpus_worker)):
        p_tp = safe_int(safe_get(row, "(p)tp")) or 1
        p_pp = safe_int(safe_get(row, "(p)pp")) or 1
        p_dp = safe_int(safe_get(row, "(p)dp")) or 1
        p_gpus_worker = p_tp * p_pp * p_dp

    if d_gpus_worker is None or d_gpus_worker == "" or (hasattr(pd, "isna") and pd.isna(d_gpus_worker)):
        d_tp = safe_int(safe_get(row, "(d)tp")) or 1
        d_pp = safe_int(safe_get(row, "(d)pp")) or 1
        d_dp = safe_int(safe_get(row, "(d)dp")) or 1
        d_gpus_worker = d_tp * d_pp * d_dp

    return calculate_tokens_per_10k_rmb_disagg(
        tokens_per_gpu,
        num_total_gpus,
        p_gpu_type,
        d_gpu_type,
        p_workers,
        d_workers,
        p_gpus_worker,
        d_gpus_worker,
        replicas,
    )


def infer_model_label(frames: Dict[str, pd.DataFrame], model_dir: Path) -> str:
    """
    标题只展示模型名。优先从 CSV 的 'model' 列推断（如 'Qwen/Qwen3-32B' -> 'Qwen3-32B'）。
    如果缺失，则退回从目录名里提取（取 '_' 分割后的第一段）。
    """
    for df in frames.values():
        if "model" not in df.columns or df.empty:
            continue
        val = df["model"].dropna()
        if val.empty:
            continue
        raw = str(val.iloc[0])
        if "/" in raw:
            raw = raw.split("/")[-1]
        raw = raw.strip()
        if raw:
            return raw

    return model_dir.name.split("_")[0]


def load_pareto_and_best_frames(
    model_dir: Path,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    从给定的 model_dir 下面收集所有子目录中的 pareto.csv 和 best_config_topn*.csv，
    返回：
      - frames: {experiment_name: pareto_df}
      - best_points: {experiment_name: best_df}
    """
    frames: Dict[str, pd.DataFrame] = {}
    best_points: Dict[str, pd.DataFrame] = {}

    for pareto_path in model_dir.rglob("pareto.csv"):
        exp_dir = pareto_path.parent
        exp_name = exp_dir.name

        try:
            pareto_df = pd.read_csv(pareto_path)
        except Exception as e:
            print(f"[WARN] 读取 {pareto_path} 失败: {e}", file=sys.stderr)
            continue

        if pareto_df.empty:
            continue

        # 对于 disagg 实验，优先使用带 replica 信息的 CSV（如果存在）
        is_disagg = has_disagg_cost_columns(pareto_df)
        if is_disagg:
            pareto_with_replica = exp_dir / "pareto_with_replica.csv"
            if pareto_with_replica.is_file():
                try:
                    pareto_df = pd.read_csv(pareto_with_replica)
                except Exception as e:
                    print(f"[WARN] 读取 {pareto_with_replica} 失败，使用原始文件: {e}", file=sys.stderr)

        required_cols = ["tokens/s/gpu", "tokens/s/user", "num_total_gpus"]
        missing = [c for c in required_cols if c not in pareto_df.columns]
        if missing:
            print(f"[WARN] {pareto_path} 缺少列: {missing}，跳过", file=sys.stderr)
            continue

        pareto_df = pareto_df.drop_duplicates()

        # 读取 best_config_topn*.csv（如果存在）
        best_path = None
        if is_disagg:
            best_with_replica = exp_dir / "best_config_topn_with_replica.csv"
            if best_with_replica.is_file():
                best_path = best_with_replica
        if best_path is None:
            candidate = exp_dir / "best_config_topn.csv"
            if candidate.is_file():
                best_path = candidate

        best_df_for_exp: pd.DataFrame | None = None
        if best_path is not None and best_path.is_file():
            try:
                best_df = pd.read_csv(best_path)
            except pd.errors.EmptyDataError:
                best_df = None
            except Exception as e:
                print(f"[WARN] 读取 {best_path} 失败: {e}", file=sys.stderr)
                best_df = None
            else:
                if best_df is not None and not best_df.empty:
                    best_missing = [c for c in required_cols if c not in best_df.columns]
                    if best_missing:
                        print(
                            f"[WARN] {best_path} 缺少列: {best_missing}，无法用于标记 best_config_topn",
                            file=sys.stderr,
                        )
                        best_df_for_exp = None
                    else:
                        best_df_for_exp = best_df

        frames[exp_name] = pareto_df
        if best_df_for_exp is not None:
            best_points[exp_name] = best_df_for_exp

    return frames, best_points


def _frame_contains_l40_or_a100(df: pd.DataFrame) -> bool:
    """简单判断某个实验是否是我们关心的 L40/A100 混合 disagg。"""
    if df is None or df.empty:
        return False
    for col in ("(p)system", "(d)system"):
        if col in df.columns:
            try:
                s = df[col].astype(str).str.lower()
                if (s.str.contains("l40").any() and s.str.contains("a100").any()) or (
                    s.str.contains("l40").any() or s.str.contains("a100").any()
                ):
                    return True
            except Exception:
                continue
    return False


# 成本调试场景配置：
# 你可以根据需要修改 / 新增场景，例如：
#   - 调整 L40/A100 的绝对价格
#   - 或者用 factor 对 baseline 做乘法
SCENARIOS: List[dict] = [
    {
        "name": "baseline_l40_1.5w_a100_3.5w",
        "overrides": {
            "l40": 15000.0,
            "a100": 35000.0,
        },
    },
    {
        "name": "l40_2.0w_vs_a100_3.5w",
        "overrides": {
            "l40": 20000.0,
        },
    },
    {
        "name": "l40_2.5w_vs_a100_3.5w",
        "overrides": {
            "l40": 25000.0,
        },
    },
    {
        "name": "l40_3.0w_vs_a100_3.5w",
        "overrides": {
            "l40": 30000.0,
        },
    },
    {
        "name": "l40_3.5w_vs_a100_3.5w",
        "overrides": {
            "l40": 35000.0,
        },
    },
]


def plot_cost_debug_disagg(model_dir: Path, output_name: str = "cost_debug_disagg.png") -> Path:
    """
    从 model_dir 下找到一个 disagg + L40/A100 的实验，
    在同一张图里对多组 (p/d) 硬件成本假设画多条曲线。
    """
    frames, best_frames = load_pareto_and_best_frames(model_dir)
    if not frames:
        raise RuntimeError(f"在 {model_dir} 下没有找到任何 pareto.csv")

    # 选择第一个包含 L40/A100 信息的 disagg 实验
    target_exp = None
    target_df: pd.DataFrame | None = None
    for exp_name, df in frames.items():
        if has_disagg_cost_columns(df) and _frame_contains_l40_or_a100(df):
            target_exp = exp_name
            target_df = df
            break

    if target_exp is None or target_df is None or target_df.empty:
        raise RuntimeError(f"在 {model_dir} 下没有找到包含 L40/A100 的 disagg 实验")

    target_best_df = best_frames.get(target_exp)
    target_df_sorted = target_df.sort_values(by="tokens/s/user")

    # 记录 baseline，绘制不同场景时会覆盖再恢复
    global MONTHLY_NODE_RENT
    baseline_rent = BASE_MONTHLY_NODE_RENT.copy()

    plt.figure(figsize=(10, 6), dpi=120)
    colors = plt.cm.tab10.colors

    for idx, scenario in enumerate(SCENARIOS):
        name = scenario.get("name", f"scenario_{idx}")
        overrides = scenario.get("overrides", {})

        # 应用场景覆盖（未覆盖的类型沿用 baseline）
        MONTHLY_NODE_RENT = baseline_rent.copy()
        for gpu_type, price in overrides.items():
            MONTHLY_NODE_RENT[gpu_type] = float(price)

        # 逐行计算在该成本假设下的 tokens/s/10k 曲线
        y_vals = target_df_sorted.apply(
            lambda r: tokens_per_10k_currency_disagg_row(r, target_exp), axis=1
        )

        color = colors[idx % len(colors)]
        plt.plot(
            target_df_sorted["tokens/s/user"],
            y_vals,
            label=f"{name}",
            color=color,
            linewidth=1.6,
            alpha=0.9,
        )

        # 在同一成本假设下标出 best_config_topn 中的点
        if target_best_df is not None and not target_best_df.empty:
            for _, row in target_best_df.iterrows():
                if (
                    "tokens/s/user" not in row
                    or "tokens/s/gpu" not in row
                    or "num_total_gpus" not in row
                ):
                    continue
                x_val = row["tokens/s/user"]
                y_val = tokens_per_10k_currency_disagg_row(row, target_exp)
                plt.scatter(
                    [x_val],
                    [y_val],
                    marker="x",
                    s=70,
                    color=color,
                    linewidths=2,
                    zorder=5,
                )

    # 恢复 baseline，避免对后续调用有副作用
    MONTHLY_NODE_RENT = baseline_rent.copy()

    model_label = infer_model_label({target_exp: target_df_sorted}, model_dir)

    plt.xlabel("tokens/s/user")
    plt.ylabel("tokens/s/10k")
    plt.title(f"{model_label} disagg cost debug (Prefill/Decode cost scenarios)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        fontsize=9,
    )
    plt.tight_layout()

    out_path = model_dir / output_name
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "用法: python scripts/plot_cost_debug_disagg.py <model_dir>\n"
            "示例:\n"
            "  python scripts/plot_cost_debug_disagg.py \\\n"
            "    output_scale/scale_cost/Qwen/\\\n"
            "    Qwen3-32B_l40s_trtllm_isl2048_osl512_ttft1000_tpot50_368870",
            file=sys.stderr,
        )
        sys.exit(1)

    model_dir = Path(sys.argv[1])
    if not model_dir.is_dir():
        print(f"错误: {model_dir} 不是有效目录", file=sys.stderr)
        sys.exit(1)

    out_path = plot_cost_debug_disagg(model_dir)
    print(f"已生成成本调试图: {out_path}")


if __name__ == "__main__":
    main()

