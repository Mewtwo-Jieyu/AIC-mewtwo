#!/usr/bin/env python3
"""
根据 aiconfigurator 生成的 pareto.csv 和 best_config_topn.csv，
重新绘制 tokens/s/gpu vs tokens/s/user 图。

- 曲线使用每个实验目录下的 pareto.csv 中的点（按 x 排序，连续连接）。
- 如果存在 best_config_topn.csv，则在同一图上用 "x" 标记该文件中的
  所有行，这些点通常代表满足 SLA 时的最佳配置。

原先版本中会把两个文件的数据合并再画图，现在我们不再混合，确保
曲线只反映 pareto 数据、标记只来自 best_config。这样避免了线条
中出现 "x" 点带来的重复，并且更清晰地区分两个来源。

用法示例：

  python scripts/plot_custom_pareto_with_best.py \
    output_qwen32b/a100_l40_short/Qwen/\
Qwen3-32B_a100_sxm_trtllm_isl512_osl128_ttft600_tpot20_58220

会在该目录下生成：
  custom_pareto_tokens_gpu_vs_user_merged.png
"""

import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd


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

    # fallback: Qwen3-32B_a100_sxm_trtllm_isl... -> Qwen3-32B
    return model_dir.name.split("_")[0]


def load_pareto_and_best_frames(
    model_dir: Path,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    遍历给定的 model_dir 下的所有子目录，分别读取每个实验目录下的
    * `pareto.csv` （必需，用来绘制曲线）
    * `best_config_topn.csv`（可选，用来标记 "x" 点）

    返回两个字典：
      - pareto_frames: {experiment_name: pareto_df}
      - best_frames:   {experiment_name: best_df} （只包含存在且非空的文件）

    这样绘图时曲线仅来源于 pareto.csv，而 "x" 标记则来自
    best_config_topn.csv 中的所有行。
    """
    pareto_frames: Dict[str, pd.DataFrame] = {}
    best_frames: Dict[str, pd.DataFrame] = {}

    for pareto_path in model_dir.rglob("pareto.csv"):
        exp_dir = pareto_path.parent
        exp_name = exp_dir.name  # e.g. a100_short_agg, l40_p_a100_d_short_disagg
        best_path = exp_dir / "best_config_topn.csv"

        try:
            pareto_df = pd.read_csv(pareto_path)
        except Exception as e:
            print(f"[WARN] 读取 {pareto_path} 失败: {e}", file=sys.stderr)
            continue

        if pareto_df.empty:
            continue

        # 仅保证 pareto.csv 包含必要列
        required_cols = ["tokens/s/gpu", "tokens/s/user"]
        missing = [c for c in required_cols if c not in pareto_df.columns]
        if missing:
            print(f"[WARN] {pareto_path} 缺少列: {missing}，跳过", file=sys.stderr)
            continue

        pareto_frames[exp_name] = pareto_df

        if best_path.is_file():
            try:
                best_df = pd.read_csv(best_path)
            except pd.errors.EmptyDataError:
                # 空文件，忽略
                continue
            except Exception as e:
                print(f"[WARN] 读取 {best_path} 失败: {e}", file=sys.stderr)
                continue
            else:
                if best_df.empty:
                    continue

                # 同样保证列存在
                missing = [c for c in required_cols if c not in best_df.columns]
                if missing:
                    print(f"[WARN] {best_path} 缺少列: {missing}，忽略其中的数据", file=sys.stderr)
                else:
                    best_frames[exp_name] = best_df

    return pareto_frames, best_frames


def nice_label(exp_name: str) -> str:
    """
    把实验目录名转成稍微好一点的图例文本。
    """
    return exp_name.replace("_", " ")


def plot_tokens_per_gpu_vs_user(
    model_dir: Path,
    output_name: str = "custom_pareto_tokens_gpu_vs_user_merged.png",
) -> Path:
    pareto_frames, best_frames = load_pareto_and_best_frames(model_dir)
    if not pareto_frames:
        raise RuntimeError(f"在 {model_dir} 下没有找到任何 pareto.csv")

    model_label = infer_model_label(pareto_frames, model_dir)

    # 预定义一些颜色和 marker，循环使用
    colors = plt.cm.tab20.colors  # 20 种颜色
    markers = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*"]

    plt.figure(figsize=(10, 6), dpi=120)

    sorted_items = sorted(pareto_frames.items(), key=lambda kv: kv[0])

    # 记录每条曲线使用的颜色，方便后面给 CLI 选出的点加上同色 "x" 标记
    exp_color_map: Dict[str, tuple] = {}

    for idx, (exp_name, df) in enumerate(sorted_items):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]

        exp_color_map[exp_name] = color

        # 为避免线条太密，可以按 x 排序
        df_sorted = df.sort_values(by="tokens/s/user")

        plt.plot(
            df_sorted["tokens/s/user"],
            df_sorted["tokens/s/gpu"],
            label=nice_label(exp_name),
            color=color,
            marker=marker,
            markersize=4,
            linewidth=1.4,
            alpha=0.9,
        )

    # 叠加 CLI 可见的“最优点”（来自 best_config_topn.csv），用 x 标记出来
    # 用户可能会看到多个点完全重合，我们通过稍微水平抖动并在图例中添加一次说明
    # 以便可以辨别哪些点是完全相同的。
    all_best_points = []  # (x,y,color)
    for exp_name, best_df in best_frames.items():
        color = exp_color_map.get(exp_name, "black")
        for _, best_row in best_df.iterrows():
            if "tokens/s/user" not in best_row or "tokens/s/gpu" not in best_row:
                continue
            all_best_points.append((best_row["tokens/s/user"], best_row["tokens/s/gpu"], color))

    # 找出重复 (x,y) 的数量
    from collections import Counter
    counts = Counter((x, y) for x, y, _ in all_best_points)

    # 用于给重复的点按顺序分配偏移
    positions = {}
    added_best_legend = False
    for x_val, y_val, color in all_best_points:
        label = "CLI best (SLA)" if not added_best_legend else None
        added_best_legend = True

        if counts[(x_val, y_val)] > 1:
            idx = positions.get((x_val, y_val), 0)
            # 将重复数量均匀分布在 [-0.5, 0.5] 的范围内，单位与 x 轴的量级一致
            span = 0.2 if counts[(x_val, y_val)] > 1 else 0
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
    plt.ylabel("tokens/s/gpu")
    plt.title(f"{model_label}  tokens/s/gpu vs tokens/s/user")
    plt.grid(True, linestyle="--", alpha=0.3)

    # 图例放到外侧，避免遮挡曲线
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
            "用法: python scripts/plot_custom_pareto_with_best.py <model_dir>\n"
            "示例:\n"
            "  python scripts/plot_custom_pareto_with_best.py "
            "output_qwen32b/a100_l40_short/Qwen/"
            "Qwen3-32B_a100_sxm_trtllm_isl512_osl128_ttft600_tpot20_58220",
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

