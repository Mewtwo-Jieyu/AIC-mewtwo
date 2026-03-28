import pandas as pd
import matplotlib.pyplot as plt


def _select_best_under_sla(df: pd.DataFrame, sla_ttft: float, sla_tpot: float) -> pd.Series:
    """
    Emulate the selection logic used to build best_config_topn.csv:
    - filter by TTFT and TPOT SLA
    - pick the configuration with the highest tokens/s per GPU (or cluster)
    """
    candidates = df[(df["ttft"] <= sla_ttft) & (df["tpot"] <= sla_tpot)].copy()
    if candidates.empty:
        raise ValueError("No configuration satisfies TTFT and TPOT SLA.")

    sort_key = "tokens/s/gpu_cluster" if "tokens/s/gpu_cluster" in candidates.columns else "tokens/s/gpu"
    return candidates.sort_values(sort_key, ascending=False).iloc[0]


def plot_medium_stability():
    """
    Medium scenario (2048/512): highlight feasible region and show how narrow the
    TTFT+TPOT-feasible band is around the selected optimal configuration.
    """
    med_path = (
        "output_qwen32b/a100_l40/"
        "Qwen/Qwen3-32B_a100_sxm_trtllm_isl512_osl128_ttft600_tpot20_229002/"
        "l40_p_a100_d_medium_disagg/pareto.csv"
    )
    med_df = pd.read_csv(med_path)

    sla_ttft = 1500.0
    sla_tpot = 40.0

    med_sub = med_df[med_df["ttft"] <= sla_ttft].copy()
    thr_user = 1000 / sla_tpot
    med_mask_tpot = med_sub["tpot"] <= sla_tpot

    # Selected optimal config using the same SLA-constrained objective
    med_best = _select_best_under_sla(med_df, sla_ttft, sla_tpot)

    # Local neighborhood around the optimal point to illustrate
    # how many near-optimal, TTFT+TPOT-feasible alternatives exist.
    neighbor_mask = (
        med_mask_tpot
        & med_sub["tokens/s/gpu"].between(
            med_best["tokens/s/gpu"] - 40, med_best["tokens/s/gpu"] + 40
        )
        & med_sub["tokens/s/user"].between(
            med_best["tokens/s/user"] - 5, med_best["tokens/s/user"] + 5
        )
    )

    plt.figure(figsize=(6, 4))

    # All TTFT-feasible points
    plt.scatter(
        med_sub["tokens/s/gpu"],
        med_sub["tokens/s/user"],
        c="lightgray",
        label="TTFT feasible",
    )

    # TTFT + TPOT feasible band
    plt.scatter(
        med_sub[med_mask_tpot]["tokens/s/gpu"],
        med_sub[med_mask_tpot]["tokens/s/user"],
        c="tab:blue",
        label="TTFT+TPOT feasible",
    )

    # Near-optimal alternatives around the chosen point
    plt.scatter(
        med_sub[neighbor_mask]["tokens/s/gpu"],
        med_sub[neighbor_mask]["tokens/s/user"],
        facecolors="none",
        edgecolors="green",
        s=80,
        linewidths=1.2,
        label="Near-optimal alternatives",
    )

    # Selected optimal configuration
    plt.scatter(
        med_best["tokens/s/gpu"],
        med_best["tokens/s/user"],
        c="red",
        marker="x",
        s=80,
        label="Selected optimal config",
    )

    # TPOT SLA line (via tokens/s/user = 1000 / TPOT_SLA)
    plt.axhline(
        thr_user,
        color="orange",
        linestyle="--",
        label=f"TPOT SLA line (y={thr_user:.1f})",
    )

    plt.xlabel("tokens/s per GPU")
    plt.ylabel("tokens/s per user")
    plt.title("Medium scenario: feasible region and stability (L40_P_A100_D)")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("documentations/medium_stability_example.png", dpi=200)
    plt.close()


def plot_short_stability():
    """
    Short scenario (512/128): highlight feasible region and show that there is
    a wider band of TTFT+TPOT-feasible, near-optimal alternatives.
    """
    short_path = (
        "output_qwen32b/a100_l40/"
        "Qwen/Qwen3-32B_a100_sxm_trtllm_isl512_osl128_ttft600_tpot20_229002/"
        "l40_p_a100_d_short_disagg/pareto.csv"
    )
    short_df = pd.read_csv(short_path)

    sla_ttft = 600.0
    sla_tpot = 20.0

    short_sub = short_df[short_df["ttft"] <= sla_ttft].copy()
    thr_user = 1000 / sla_tpot
    short_mask_tpot = short_sub["tpot"] <= sla_tpot

    # Select optimal config using the same SLA-constrained objective
    short_best = _select_best_under_sla(short_df, sla_ttft, sla_tpot)

    # Local neighborhood around the chosen optimal config
    neighbor_mask = (
        short_mask_tpot
        & short_sub["tokens/s/gpu"].between(
            short_best["tokens/s/gpu"] - 60, short_best["tokens/s/gpu"] + 60
        )
        & short_sub["tokens/s/user"].between(
            short_best["tokens/s/user"] - 25, short_best["tokens/s/user"] + 10
        )
    )

    plt.figure(figsize=(6, 4))

    # All TTFT-feasible points
    plt.scatter(
        short_sub["tokens/s/gpu"],
        short_sub["tokens/s/user"],
        c="lightgray",
        label="TTFT feasible",
    )

    # TTFT + TPOT feasible band
    plt.scatter(
        short_sub[short_mask_tpot]["tokens/s/gpu"],
        short_sub[short_mask_tpot]["tokens/s/user"],
        c="tab:blue",
        label="TTFT+TPOT feasible",
    )

    # Near-optimal alternatives around the chosen point
    plt.scatter(
        short_sub[neighbor_mask]["tokens/s/gpu"],
        short_sub[neighbor_mask]["tokens/s/user"],
        facecolors="none",
        edgecolors="green",
        s=80,
        linewidths=1.2,
        label="Near-optimal alternatives",
    )

    # Selected optimal configuration
    plt.scatter(
        short_best["tokens/s/gpu"],
        short_best["tokens/s/user"],
        c="red",
        marker="x",
        s=80,
        label="Selected optimal config",
    )

    # TPOT SLA line
    plt.axhline(
        thr_user,
        color="orange",
        linestyle="--",
        label=f"TPOT SLA line (y={thr_user:.0f})",
    )

    plt.xlabel("tokens/s per GPU")
    plt.ylabel("tokens/s per user")
    plt.title("Short scenario: feasible region and stability (L40_P_A100_D)")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("documentations/short_stability_example.png", dpi=200)
    plt.close()


def main():
    plot_medium_stability()
    plot_short_stability()


if __name__ == "__main__":
    main()

