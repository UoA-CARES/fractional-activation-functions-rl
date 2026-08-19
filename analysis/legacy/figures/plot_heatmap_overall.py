from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.expanduser("~/Desktop/new-Fr")
IN_SEED = os.path.join(ROOT, "outputs/analysis/auc_by_seed_normalized.csv")

OUT_PAPER = os.path.join(ROOT, "outputs/paper")
os.makedirs(OUT_PAPER, exist_ok=True)

OUT_TD3 = os.path.join(OUT_PAPER, "fig_heatmap_overall_td3.pdf")
OUT_SAC = os.path.join(OUT_PAPER, "fig_heatmap_overall_sac.pdf")


def make_heatmap(df_sub: pd.DataFrame, algo: str, outpath: str) -> None:
    tasks = sorted(df_sub["task"].unique().tolist())
    acts = sorted(df_sub["activation"].unique().tolist())

    pivot = (
        df_sub.pivot(index="task", columns="activation", values="best_mean_delta")
        .reindex(index=tasks, columns=acts)
    )
    mat = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(1.2 + 1.0 * len(acts), 1.2 + 0.35 * len(tasks)))
    im = ax.imshow(mat, aspect="auto")

    ax.set_title(f"Overall Performance (best mean $\\Delta\\%$ vs ReLU) [{algo}]")
    ax.set_xlabel("Activation")
    ax.set_ylabel("Task")

    ax.set_xticks(np.arange(len(acts)))
    ax.set_xticklabels(acts, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(tasks)))
    ax.set_yticklabels(tasks)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("best mean $\\Delta\\%$ vs ReLU")

    for i in range(len(tasks)):
        for j in range(len(acts)):
            v = mat[i, j]
            ax.text(j, i, "NA" if np.isnan(v) else f"{v:+.1f}", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(IN_SEED)

    required = ["algo","task","activation","layers","placement","alpha","delta_pct_vs_relu"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available: {list(df.columns)}")

    # Only fractional activations for this heatmap
    dff = df[df["activation"].astype(str).str.lower().isin(["frelu","flrelu","fprelu"])].copy()

    # For each (algo, task, activation), take best mean Δ% among all configs (layers, placement, alpha)
    cfg_cols = ["algo","task","activation","layers","placement","alpha"]
    mean_cfg = (
        dff.groupby(cfg_cols, as_index=False)["delta_pct_vs_relu"]
        .mean()
        .rename(columns={"delta_pct_vs_relu": "mean_delta"})
    )
    best = (
        mean_cfg.groupby(["algo","task","activation"], as_index=False)["mean_delta"]
        .max()
        .rename(columns={"mean_delta": "best_mean_delta"})
    )

    for algo, outpath in [("TD3", OUT_TD3), ("SAC", OUT_SAC)]:
        sub = best[best["algo"].astype(str).str.upper() == algo].copy()
        if len(sub) == 0:
            print(f"No rows for {algo}, skipping.")
            continue
        make_heatmap(sub, algo, outpath)
        print("Saved:", outpath)


if __name__ == "__main__":
    main()
