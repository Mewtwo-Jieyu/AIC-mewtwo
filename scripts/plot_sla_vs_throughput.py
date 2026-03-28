#!/usr/bin/env python3
"""
Plot SLA constraint vs. unconstrained throughput from AIC pareto.csv.

Illustrates why performance comparison must use TTFT/TPOT SLA: without SLA,
one can report 2-3x higher throughput (by picking high-latency configs);
with SLA, only configs inside the feasible region count.

Usage:
  python scripts/plot_sla_vs_throughput.py <model_dir>

Reads <model_dir>/pareto.csv and writes:
  <model_dir>/sla_vs_throughput.png
  <model_dir>/sla_vs_throughput_bar.png (optional bar chart)

Example:
  python scripts/plot_sla_vs_throughput.py output_qwen32b/a100_l40_mid/Qwen/.../a100_medium_disagg
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_pareto(model_dir: Path) -> pd.DataFrame:
    path = model_dir / "pareto.csv"
    if not path.exists():
        raise FileNotFoundError(f"pareto.csv not found under {model_dir}")
    df = pd.read_csv(path)
    for col in ("ttft", "tpot", "tokens/s/gpu"):
        if col not in df.columns:
            raise ValueError(f"pareto.csv must contain column '{col}'")
    return df


def scatter_sla_vs_throughput(
    df: pd.DataFrame,
    target_tpot: float = 50.0,
    target_ttft: float | None = None,
    out_path: Path | None = None,
) -> None:
    """Scatter: TPOT vs tokens/s/gpu, SLA line, and annotate no-SLA vs under-SLA max."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    tpot = df["tpot"]
    throughput = df["tokens/s/gpu"]

    # All points
    ax.scatter(tpot, throughput, alpha=0.7, s=36, c="steelblue", edgecolors="navy", linewidths=0.5, label="Config points")

    # SLA vertical line (TPOT)
    ax.axvline(x=target_tpot, color="red", linestyle="--", linewidth=1.5, label=f"SLA: TPOT ≤ {target_tpot} ms")
    if target_ttft is not None:
        # Optional: shade region where ttft also exceeds (we don't shade here for simplicity)
        pass

    # No-SLA max: global max throughput (often at high TPOT)
    max_no_sla = throughput.max()
    row_no_sla = df.loc[throughput.idxmax()]
    tpot_no_sla = row_no_sla["tpot"]
    ax.scatter(
        [tpot_no_sla],
        [max_no_sla],
        color="darkorange",
        s=120,
        marker="*",
        zorder=5,
        edgecolors="black",
        linewidths=1,
        label=f"Max (no SLA): {max_no_sla:.1f} tokens/s/gpu",
    )
    ax.annotate(
        f"No SLA\n{max_no_sla:.0f}",
        xy=(tpot_no_sla, max_no_sla),
        xytext=(10, 15),
        textcoords="offset points",
        fontsize=9,
        color="darkorange",
        arrowprops=dict(arrowstyle="->", color="darkorange", lw=1),
    )

    # Under-SLA max: max among points with tpot <= target_tpot (and optionally ttft <= target_ttft)
    if target_ttft is not None:
        mask = (df["ttft"] <= target_ttft) & (df["tpot"] <= target_tpot)
    else:
        mask = df["tpot"] <= target_tpot
    feasible = df[mask]
    if feasible.empty:
        max_sla = float("nan")
        ax.set_title(f"TPOT vs Throughput (no point satisfies TPOT≤{target_tpot} ms)")
    else:
        max_sla = feasible["tokens/s/gpu"].max()
        row_sla = feasible.loc[feasible["tokens/s/gpu"].idxmax()]
        tpot_sla = row_sla["tpot"]
        ax.scatter(
            [tpot_sla],
            [max_sla],
            color="green",
            s=120,
            marker="s",
            zorder=5,
            edgecolors="black",
            linewidths=1,
            label=f"Max (under SLA): {max_sla:.1f} tokens/s/gpu",
        )
        ax.annotate(
            f"Under SLA\n{max_sla:.0f}",
            xy=(tpot_sla, max_sla),
            xytext=(10, -25),
            textcoords="offset points",
            fontsize=9,
            color="green",
            arrowprops=dict(arrowstyle="->", color="green", lw=1),
        )
        ratio = max_no_sla / max_sla if max_sla > 0 else float("nan")
        ax.set_title(f"TPOT vs Throughput · No-SLA/With-SLA ratio ≈ {ratio:.2f}x")

    ax.set_xlabel("TPOT (ms)")
    ax.set_ylabel("tokens/s/gpu")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")
    else:
        plt.show()


def bar_sla_vs_throughput(
    df: pd.DataFrame,
    tpot_targets: list[float] = (20, 40, 50, 60, 80),
    target_ttft: float | None = None,
    out_path: Path | None = None,
) -> None:
    """Bar: for each TPOT SLA (and optional TTFT), max tokens/s/gpu under that SLA vs no constraint."""
    max_no_sla = df["tokens/s/gpu"].max()
    labels = ["No SLA"] + [f"TPOT≤{t} ms" for t in tpot_targets]
    values = [max_no_sla]
    for t in tpot_targets:
        if target_ttft is not None:
            mask = (df["ttft"] <= target_ttft) & (df["tpot"] <= t)
        else:
            mask = df["tpot"] <= t
        subset = df[mask]
        values.append(subset["tokens/s/gpu"].max() if not subset.empty else 0.0)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["darkorange"] + ["steelblue"] * len(tpot_targets)
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.8)
    ax.axhline(y=max_no_sla, color="darkorange", linestyle=":", alpha=0.8)
    ax.set_ylabel("tokens/s/gpu (max)")
    ax.set_xlabel("Constraint")
    ax.set_title("Max Throughput vs SLA Strictness")
    ax.set_ylim(bottom=0)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot SLA vs throughput from pareto.csv")
    parser.add_argument("model_dir", type=str, help="Directory containing pareto.csv (e.g. .../a100_medium_disagg)")
    parser.add_argument("--ttft", type=float, default=None, help="Optional TTFT SLA (ms) for feasible region")
    parser.add_argument("--tpot", type=float, default=50.0, help="TPOT SLA (ms) for scatter plot")
    parser.add_argument("--no-bar", action="store_true", help="Skip bar chart")
    parser.add_argument("-o", "--output", type=str, default=None, help="Override output directory")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        print(f"Error: not a directory: {model_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output) if args.output else model_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_pareto(model_dir)
    scatter_sla_vs_throughput(
        df,
        target_tpot=args.tpot,
        target_ttft=args.ttft,
        out_path=out_dir / "sla_vs_throughput.png",
    )
    if not args.no_bar:
        bar_sla_vs_throughput(
            df,
            target_ttft=args.ttft,
            out_path=out_dir / "sla_vs_throughput_bar.png",
        )


if __name__ == "__main__":
    main()
