#!/usr/bin/env python3
"""
根据 aiconfigurator 生成的 pareto.csv，绘制带有 SLA 边界线的 Pareto 图。

在 tokens/s/gpu vs tokens/s/user 图中添加 TPOT SLA 约束边界线，
直观展示最优点距离 SLA 边界的安全边际。

注意：图中仅显示 TPOT 边界线（垂直红线），因为：
- TPOT 边界决定了 tokens/s/user 的下限，直接对应用户感知的生成速度
- TTFT 约束在 Pareto 图上表现为水平方向的限制，与 TPOT 边界共同围成可行区域
- 安全边际表中展示的 TTFT/TPOT 余量可在边界线的相对位置中直观体现

用法示例（SLA 约束以报告文档中的表格为准）：

  # Short 场景：TTFT=600ms, TPOT=20ms → tokens/s/user ≥ 50
  python scripts/plot_sla_pareto.py \\
    output_qwen32b/a100_l40_short/Qwen/... \\
    --scene short

  # Medium 场景：TTFT=1500ms, TPOT=40ms → tokens/s/user ≥ 25
  python scripts/plot_sla_pareto.py \\
    output_qwen32b/a100_l40_medium/Qwen/... \\
    --scene medium

  # Long 场景：TTFT=2500ms, TPOT=60ms → tokens/s/user ≥ 16.7
  python scripts/plot_sla_pareto.py \\
    output_qwen32b/a100_l40_long/Qwen/... \\
    --scene long

会在该目录下生成：
  sla_pareto_tokens_gpu_vs_user.png
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

# 配置中文字体，避免方框问题
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 报告中定义的 SLA 约束（敲定版推理系统性能优化分析报告）
SCENE_SLA_PRESETS = {
    'short': {'ttft': 600, 'tpot': 20},
    'medium': {'ttft': 1500, 'tpot': 40},
    'long': {'ttft': 2500, 'tpot': 60},
}


def parse_sla_from_dir_name(model_dir: Path) -> Optional[Tuple[float, float, float]]:
    """
    从目录名解析 SLA 约束和 OSL。（已废弃，仅保留向后兼容）

    目录名格式示例:
    Qwen3-32B_a100_sxm_trtllm_isl512_osl128_ttft600_tpot20_58220

    返回：(ttft_limit_ms, tpot_limit_ms, osl)

    注意：目录名中的 SLA 可能与报告中实际使用的 SLA 不一致，
    建议始终使用 --scene 参数或手动指定 --ttft/--tpot/--osl。
    """
    dir_name = model_dir.name

    # 使用正则表达式提取 ttft, tpot, osl
    ttft_match = re.search(r"ttft(\d+)", dir_name)
    tpot_match = re.search(r"tpot(\d+)", dir_name)
    osl_match = re.search(r"osl(\d+)", dir_name)

    if ttft_match and tpot_match and osl_match:
        return (
            float(ttft_match.group(1)),
            float(tpot_match.group(1)),
            float(osl_match.group(1)),
        )

    return None


def load_pareto_frames(model_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    从给定的 model_dir 下面收集所有子目录中的 pareto.csv，
    同时也会加载 pareto_with_replica.csv（如果存在）。
    返回 {experiment_name: df}。
    """
    frames: Dict[str, pd.DataFrame] = {}

    # 同时查找 pareto.csv 和 pareto_with_replica.csv
    for pareto_pattern in ["pareto.csv", "pareto_with_replica.csv"]:
        for pareto_path in model_dir.rglob(pareto_pattern):
            exp_dir = pareto_path.parent
            exp_name = exp_dir.name

            if exp_name in frames:
                continue  # 已经加载过

            try:
                df = pd.read_csv(pareto_path)
            except Exception as e:
                print(f"[WARN] 读取 {pareto_path} 失败：{e}", file=sys.stderr)
                continue

            if df.empty:
                continue

            # 仅保留需要的列
            required_cols = ["tokens/s/gpu", "tokens/s/user"]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                print(f"[WARN] {pareto_path} 缺少列：{missing}，跳过", file=sys.stderr)
                continue

            frames[exp_name] = df

    return frames


def load_best_configs(model_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    加载每个实验的 best_config_topn.csv，用于标记最优点。
    返回 {experiment_name: df}。
    """
    frames: Dict[str, pd.DataFrame] = {}

    for best_path in model_dir.rglob("best_config_topn.csv"):
        exp_dir = best_path.parent
        exp_name = exp_dir.name

        try:
            df = pd.read_csv(best_path)
        except Exception as e:
            continue

        if df.empty:
            continue

        required_cols = ["tokens/s/gpu", "tokens/s/user"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            continue

        frames[exp_name] = df

    return frames


def nice_label(exp_name: str) -> str:
    """
    把实验目录名转成稍微好一点的图例文本。
    """
    # 简化标签，只保留关键信息
    label = exp_name.replace("_", " ")
    # 缩短过长的标签
    if len(label) > 40:
        label = label[:37] + "..."
    return label


def plot_sla_pareto(
    model_dir: Path,
    ttft_limit: float,
    tpot_limit: float,
    scene: Optional[str] = None,
    output_name: str = "sla_pareto_tokens_gpu_vs_user.png",
) -> Path:
    """
    绘制带 TPOT SLA 边界线的 Pareto 图。

    图中仅显示 TPOT 边界线（垂直红线），因为：
    - tokens/s/gpu vs tokens/s/user 图中，TPOT 约束表现为垂直边界
    - TTFT 约束影响水平方向，在实际数据中通常已满足
    - 安全边际的直观展示主要通过 TPOT 边界体现

    参数:
    - model_dir: 模型输出目录
    - ttft_limit: TTFT SLA 约束（毫秒），用于标题显示
    - tpot_limit: TPOT SLA 约束（毫秒），用于绘制边界线
    - scene: 场景名称（用于标题显示）
    - output_name: 输出文件名
    """
    frames = load_pareto_frames(model_dir)
    if not frames:
        raise RuntimeError(f"在 {model_dir} 下没有找到任何 pareto.csv")

    best_configs = load_best_configs(model_dir)

    # 计算每个配置的安全边际
    # 安全边际 = (SLA 约束 - 仿真达成) / SLA 约束
    # 找出满足 SLA 约束的配置
    feasible_configs = []
    for exp_name, df in frames.items():
        for _, row in df.iterrows():
            # 从数据中读取实际 TTFT 和 TPOT
            actual_ttft = row.get('ttft', 0)
            actual_tpot = row.get('tpot', 0)

            # 检查是否满足 SLA 约束
            if actual_ttft <= ttft_limit and actual_tpot <= tpot_limit:
                feasible_configs.append({
                    'exp_name': exp_name,
                    'tokens/s/gpu': row['tokens/s/gpu'],
                    'tokens/s/user': row['tokens/s/user'],
                    'ttft': actual_ttft,
                    'tpot': actual_tpot,
                    'safety_margin_ttft': (ttft_limit - actual_ttft) / ttft_limit,
                    'safety_margin_tpot': (tpot_limit - actual_tpot) / tpot_limit,
                })

    if not feasible_configs:
        print(f"[WARN] 没有找到满足 SLA 约束的配置 (TTFT≤{ttft_limit}ms, TPOT≤{tpot_limit}ms)")

    # 计算 TPOT SLA 边界在图上的位置
    # tokens/s/user 表示用户感知的生成速度
    # 根据 TPOT SLA 约束计算边界：tokens/s/user = 1000 / tpot_limit
    # 例如：TPOT ≤ 20ms → tokens/s/user ≥ 1000/20 = 50
    # 注意：图中仅绘制 TPOT 边界线，TTFT 约束影响的是水平方向的可行区域
    min_tokens_per_user = 1000.0 / tpot_limit

    # 预定义一些颜色和 marker，循环使用
    colors = plt.cm.tab20.colors
    markers = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*"]

    plt.figure(figsize=(12, 8), dpi=120)

    # 建立实验名到颜色的映射
    exp_color_map = {}
    sorted_items = sorted(frames.items(), key=lambda kv: kv[0])
    for idx, (exp_name, df) in enumerate(sorted_items):
        exp_color_map[exp_name] = {
            'color': colors[idx % len(colors)],
            'marker': markers[idx % len(markers)],
        }

    for idx, (exp_name, df) in enumerate(sorted_items):
        color = exp_color_map[exp_name]['color']
        marker = exp_color_map[exp_name]['marker']

        # 按 x 排序让线条更清晰
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

    # 标记最优点（best_config_topn.csv 中的配置）- 使用与曲线相同的颜色
    for exp_name, df_best in best_configs.items():
        if df_best.empty:
            continue

        # 如果未在曲线中找到对应实验，使用灰色衬底以保证可见性
        color = exp_color_map.get(exp_name, {}).get('color', 'gray')

        for _, row in df_best.iterrows():
            plt.plot(
                row["tokens/s/user"],
                row["tokens/s/gpu"],
                marker="X",
                markersize=10,
                markerfacecolor="none",
                markeredgecolor=color,
                markeredgewidth=2,
                alpha=1.0,
            )

    # 绘制 TPOT SLA 边界线（仅绘制 TPOT 边界，TTFT 约束在标题中说明）
    if min_tokens_per_user is not None:
        # TPOT 边界：tokens/s/user 的最小值（垂直线）
        # TPOT 越小越好，所以 TPOT ≤ limit 意味着 tokens/s/user ≥ min_value
        # 在图中，可行区域是边界线右侧（上方为 TTFT 约束区域）
        plt.axvline(
            x=min_tokens_per_user,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"TPOT SLA (≤{tpot_limit}ms)",
        )
        print(f"[INFO] 绘制 TPOT SLA 边界线：x = {min_tokens_per_user:.1f} tokens/s/user")

    # 添加图例说明
    plt.xlabel("tokens/s/user (用户感知生成速度)", fontsize=11)
    plt.ylabel("tokens/s/gpu (GPU 吞吐效率)", fontsize=11)

    scene_title = f" - {scene.title()} Scene" if scene else ""
    plt.title(f"tokens/s/gpu vs tokens/s/user with SLA Boundaries{scene_title}\n(SLA: TTFT≤{ttft_limit}ms, TPOT≤{tpot_limit}ms)", fontsize=12)

    plt.grid(True, linestyle="--", alpha=0.3)

    # 图例放到外侧
    plt.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        fontsize=8,
    )

    plt.tight_layout()
    out_path = model_dir / output_name
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="绘制带 TPOT SLA 边界线的 Pareto 图",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例（SLA 约束以报告文档中的表格为准）:

  # Short 场景：TTFT=600ms, TPOT=20ms → TPOT 边界 tokens/s/user = 50
  python scripts/plot_sla_pareto.py \\
    output_qwen32b/a100_l40_short/Qwen/... \\
    --scene short

  # Medium 场景：TTFT=1500ms, TPOT=40ms → TPOT 边界 tokens/s/user = 25
  python scripts/plot_sla_pareto.py \\
    output_qwen32b/a100_l40_medium/Qwen/... \\
    --scene medium

  # Long 场景：TTFT=2500ms, TPOT=60ms → TPOT 边界 tokens/s/user = 16.7
  python scripts/plot_sla_pareto.py \\
    output_qwen32b/a100_l40_long/Qwen/... \\
    --scene long

图例说明:
  - 实线/散点：Pareto 前沿配置
  - X 标记：最优点 (best_config_topn)
  - 红色虚线：TPOT SLA 边界（右侧为可行区域）
  - 安全边际 = (最优点 TPOT - TPOT SLA) / TPOT SLA
        """,
    )
    parser.add_argument("model_dir", type=Path, help="模型输出目录")
    parser.add_argument(
        "--scene",
        type=str,
        choices=['short', 'medium', 'long'],
        default=None,
        help="场景预设（根据报告中的 SLA 约束自动设置）",
    )
    parser.add_argument(
        "--ttft",
        type=float,
        default=None,
        help="TTFT SLA 约束（毫秒），如不指定则根据 --scene 预设",
    )
    parser.add_argument(
        "--tpot",
        type=float,
        default=None,
        help="TPOT SLA 约束（毫秒），如不指定则根据 --scene 预设",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sla_pareto_tokens_gpu_vs_user.png",
        help="输出文件名（默认：sla_pareto_tokens_gpu_vs_user.png）",
    )

    args = parser.parse_args()

    # 使用场景预设填充 SLA 参数
    if args.scene:
        preset = SCENE_SLA_PRESETS[args.scene]
        ttft = args.ttft if args.ttft is not None else preset['ttft']
        tpot = args.tpot if args.tpot is not None else preset['tpot']
        scene_name = args.scene.title()
    else:
        # 没有指定 scene，则必须手动指定所有 SLA 参数
        if args.ttft is None or args.tpot is None:
            print("错误：未指定 --scene 时，必须使用 --ttft/--tpot 手动指定所有 SLA 参数", file=sys.stderr)
            print("使用 --scene short|medium|long 可自动加载报告中的 SLA 约束", file=sys.stderr)
            sys.exit(1)
        ttft = args.ttft
        tpot = args.tpot
        scene_name = None

    if not args.model_dir.is_dir():
        print(f"错误：{args.model_dir} 不是有效目录", file=sys.stderr)
        sys.exit(1)

    try:
        out_path = plot_sla_pareto(
            args.model_dir,
            ttft_limit=ttft,
            tpot_limit=tpot,
            scene=scene_name,
            output_name=args.output,
        )
        print(f"已生成：{out_path}")
        print(f"\nTPOT SLA 边界线位置: x = {1000.0/tpot:.1f} tokens/s/user")
    except RuntimeError as e:
        print(f"错误：{e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
