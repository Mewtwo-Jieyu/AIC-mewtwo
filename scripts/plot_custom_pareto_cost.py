#!/usr/bin/env python3
"""
v3: cost-aware pareto plot (with disagg support).

This script:
- Reads pareto.csv and optional best_config_topn*.csv under <model_dir>
- Plots tokens/s/10k RMB vs tokens/s/user
- Handles both aggregated and disaggregated (prefill/decode) deployments

Usage:
  python scripts/plot_custom_pareto_cost_v3.py <model_dir>

Output:
  <model_dir>/pareto_user_vs_cost_v3.png
"""

import math
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd


GPUS_PER_NODE = 8.0  # 8 GPUs per node
MONTHLY_NODE_RENT = {
    "a100": 35000.0,
    "l40": 15000.0,
}
TEN_K_CURRENCY = 10_000.0


def normalize_gpu_type_for_cost(raw_type: str) -> str:
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
    Disagg cost model:

      Step 1: per-replica GPU demand
        P_single_replica_gpus = p_workers * p_gpus_per_worker
        D_single_replica_gpus = d_workers * d_gpus_per_worker

      Step 2: per-replica nodes (8 GPUs per node, ceil)
        P_single_replica_nodes = ceil(P_single_replica_gpus / 8)
        D_single_replica_nodes = ceil(D_single_replica_gpus / 8)

      Step 3: total nodes across replicas
        P_total_nodes = P_single_replica_nodes * replicas
        D_total_nodes = D_single_replica_nodes * replicas

      Cost:
        cost = P_total_nodes * MONTHLY_NODE_RENT[p_type] +
               D_total_nodes * MONTHLY_NODE_RENT[d_type]
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

    def nodes_for(single_gpus: int) -> int:
        if single_gpus <= 0:
            return 0
        return int(math.ceil(single_gpus / GPUS_PER_NODE))

    p_single_nodes = nodes_for(p_single_gpus)
    d_single_nodes = nodes_for(d_single_gpus)
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
    cols = set(df.columns)
    if "(p)workers" not in cols or "(d)workers" not in cols:
        return False
    has_p_gpu_info = "p_gpus_worker" in cols or "(p)tp" in cols
    has_d_gpu_info = "d_gpus_worker" in cols or "(d)tp" in cols
    return has_p_gpu_info and has_d_gpu_info


def tokens_per_10k_currency_disagg_row(row, exp_name: str) -> float:
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


def infer_gpu_type(exp_name: str, row=None) -> str:
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
            t = normalize_gpu_type_for_cost(str(p_sys))
            if t != "unknown":
                return t
        if d_sys and str(d_sys).strip():
            t = normalize_gpu_type_for_cost(str(d_sys))
            if t != "unknown":
                return t

    lower = exp_name.lower()
    if lower.startswith("a100"):
        return "a100"
    if lower.startswith("l40"):
        return "l40"
    return "unknown"


def to_tokens_per_10k_currency(
    tokens_per_gpu: pd.Series | float,
    num_total_gpus: pd.Series | float,
    exp_name: str,
    row=None,
) -> pd.Series | float:
    gpu_type = infer_gpu_type(exp_name, row)
    if gpu_type not in MONTHLY_NODE_RENT:
        print(
            f"[WARN] 无法从实验名 {exp_name} 推断 GPU 类型，保持 tokens/s/gpu 单位",
            file=sys.stderr,
        )
        return tokens_per_gpu

    num_nodes_float = num_total_gpus / GPUS_PER_NODE
    if hasattr(num_nodes_float, "apply"):
        num_nodes = num_nodes_float.apply(math.ceil).astype(int)
    else:
        num_nodes = int(math.ceil(num_nodes_float))

    total_monthly_cost = num_nodes * MONTHLY_NODE_RENT[gpu_type]
    total_tokens_per_sec = tokens_per_gpu * num_total_gpus
    return (total_tokens_per_sec / total_monthly_cost) * TEN_K_CURRENCY


def infer_model_label(frames: Dict[str, pd.DataFrame], model_dir: Path) -> str:
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

        is_disagg = has_disagg_cost_columns(pareto_df)
        pareto_data_path = pareto_path
        if is_disagg:
            pareto_with_replica = exp_dir / "pareto_with_replica.csv"
            if pareto_with_replica.is_file():
                pareto_data_path = pareto_with_replica
                try:
                    pareto_df = pd.read_csv(pareto_data_path)
                except Exception as e:
                    print(f"[WARN] 读取 {pareto_with_replica} 失败，使用原始文件: {e}", file=sys.stderr)

        best_path = None
        if is_disagg:
            best_with_replica = exp_dir / "best_config_topn_with_replica.csv"
            if best_with_replica.is_file():
                best_path = best_with_replica
        if best_path is None:
            candidate = exp_dir / "best_config_topn.csv"
            if candidate.is_file():
                best_path = candidate

        required_cols = ["tokens/s/gpu", "tokens/s/user", "num_total_gpus"]
        missing = [c for c in required_cols if c not in pareto_df.columns]
        if missing:
            print(f"[WARN] {pareto_path} 缺少列: {missing}，跳过", file=sys.stderr)
            continue

        pareto_df = pareto_df.drop_duplicates()

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
                if not best_df.empty:
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


def nice_label(exp_name: str) -> str:
    return exp_name.replace("_", " ")


def plot_tokens_per_gpu_vs_user(
    model_dir: Path,
    output_name: str = "pareto_user_vs_cost_v3.png",
) -> Path:
    frames, best_frames = load_pareto_and_best_frames(model_dir)
    if not frames:
        raise RuntimeError(f"在 {model_dir} 下没有找到任何 pareto.csv")

    model_label = infer_model_label(frames, model_dir)
    colors = plt.cm.tab20.colors
    markers = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*"]

    plt.figure(figsize=(10, 6), dpi=120)

    sorted_items = sorted(frames.items(), key=lambda kv: kv[0])
    exp_color_map: Dict[str, tuple] = {}

    for idx, (exp_name, df) in enumerate(sorted_items):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        exp_color_map[exp_name] = color

        df_sorted = df.sort_values(by="tokens/s/user")

        if has_disagg_cost_columns(df_sorted):
            y_vals = df_sorted.apply(
                lambda r: tokens_per_10k_currency_disagg_row(r, exp_name), axis=1
            )
        else:
            first_row = df_sorted.iloc[0] if len(df_sorted) > 0 else None
            y_vals = to_tokens_per_10k_currency(
                df_sorted["tokens/s/gpu"],
                df_sorted["num_total_gpus"],
                exp_name,
                first_row,
            )

        plt.plot(
            df_sorted["tokens/s/user"],
            y_vals,
            label=nice_label(exp_name),
            color=color,
            marker=marker,
            markersize=4,
            linewidth=1.4,
            alpha=0.9,
        )

    from collections import Counter

    all_best = []
    for exp_name, best_df in best_frames.items():
        if best_df is None or best_df.empty:
            continue
        for _, row in best_df.iterrows():
            if (
                "tokens/s/user" not in row
                or "tokens/s/gpu" not in row
                or "num_total_gpus" not in row
            ):
                continue
            x_val = row["tokens/s/user"]
            if has_disagg_cost_columns(best_df):
                y_val = tokens_per_10k_currency_disagg_row(row, exp_name)
            else:
                y_val = to_tokens_per_10k_currency(
                    row["tokens/s/gpu"], row["num_total_gpus"], exp_name, row
                )
            color = exp_color_map.get(exp_name, "black")
            all_best.append((x_val, y_val, color))

    counts = Counter((x, y) for x, y, _ in all_best)
    positions = {}
    added_best_legend = False
    for x_val, y_val, color in all_best:
        label = "CLI best (SLA)" if not added_best_legend else None
        added_best_legend = True
        if counts[(x_val, y_val)] > 1:
            idx = positions.get((x_val, y_val), 0)
            span = 0.2
            offset = (idx - (counts[(x_val, y_val)] - 1) / 2) * span
            positions[(x_val, y_val)] = idx + 1
            x_plot = x_val + offset
        else:
            x_plot = x_val
        plt.scatter(
            [x_plot],
            [y_val],
            marker="x",
            s=80,
            color=color,
            linewidths=2,
            zorder=5,
            label=label,
        )

    plt.xlabel("tokens/s/user")
    plt.ylabel("tokens/s/10k")
    plt.title(f"{model_label} tokens/s/10k vs tokens/s/user")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        fontsize=8,
    )
    plt.tight_layout()
    out_path = model_dir / output_name
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "用法: python scripts/plot_custom_pareto_cost_v3.py <model_dir>",
            file=sys.stderr,
        )
        sys.exit(1)

    model_dir = Path(sys.argv[1])
    if not model_dir.is_dir():
        print(f"错误: {model_dir} 不是有效目录", file=sys.stderr)
        sys.exit(1)

    out_path = plot_tokens_per_gpu_vs_user(model_dir)
    print(f"已生成: {out_path}")


if __name__ == "__main__":
    main()