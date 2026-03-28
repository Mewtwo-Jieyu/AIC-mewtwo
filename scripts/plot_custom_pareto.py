#!/usr/bin/env python3
"""
根据 aiconfigurator 生成的 pareto.csv，重新绘制 tokens/s/gpu vs tokens/s/user 图，
使用更清晰的配色和 marker，方便区分不同实验（尤其是混推场景）。

用法示例：

  python scripts/plot_custom_pareto.py \
    output/a100_l40_full_comparison_mix_all/Qwen/Qwen3-32B-FP8_a100_sxm_trtllm_isl512_osl128_ttft600_tpot20_870245

会在该目录下生成：
  custom_pareto_tokens_gpu_vs_user.png
"""

import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def load_pareto_frames(model_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    从给定的 model_dir（例如 Qwen3-32B-FP8_a100_sxm_trtllm_isl512_osl128_ttft600_tpot20_xxxxxx）
    下面收集所有子目录中的 pareto.csv，返回 {experiment_name: df}。
    """
    frames: Dict[str, pd.DataFrame] = {}

    for pareto_path in model_dir.rglob("pareto.csv"):
        exp_dir = pareto_path.parent
        exp_name = exp_dir.name  # e.g. a100_short_agg, l40_p_a100_d_short_disagg

        try:
            df = pd.read_csv(pareto_path)
        except Exception as e:
            print(f"[WARN] 读取 {pareto_path} 失败: {e}", file=sys.stderr)
            continue

        if df.empty:
            continue

        # 仅保留需要的列
        required_cols = ["tokens/s/gpu", "tokens/s/user"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"[WARN] {pareto_path} 缺少列: {missing}，跳过", file=sys.stderr)
            continue

        frames[exp_name] = df

    return frames


def nice_label(exp_name: str) -> str:
    """
    把实验目录名转成稍微好一点的图例文本。
    """
    return exp_name.replace("_", " ")


def plot_tokens_per_gpu_vs_user(model_dir: Path, output_name: str = "custom_pareto_tokens_gpu_vs_user.png") -> Path:
    frames = load_pareto_frames(model_dir)
    if not frames:
        raise RuntimeError(f"在 {model_dir} 下没有找到任何 pareto.csv")

    # 预定义一些颜色和 marker，循环使用
    colors = plt.cm.tab20.colors  # 20 种颜色
    markers = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*"]

    plt.figure(figsize=(10, 6), dpi=120)

    sorted_items = sorted(frames.items(), key=lambda kv: kv[0])

    for idx, (exp_name, df) in enumerate(sorted_items):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]

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

    plt.xlabel("tokens/s/user")
    plt.ylabel("tokens/s/gpu")
    plt.title(f"{model_dir.name} tokens/s/gpu vs tokens/s/user (custom view)")
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
            "用法: python scripts/plot_custom_pareto.py <model_dir>\n"
            "示例:\n"
            "  python scripts/plot_custom_pareto.py "
            "output/a100_l40_full_comparison_mix_all/Qwen/"
            "Qwen3-32B-FP8_a100_sxm_trtllm_isl512_osl128_ttft600_tpot20_870245",
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

