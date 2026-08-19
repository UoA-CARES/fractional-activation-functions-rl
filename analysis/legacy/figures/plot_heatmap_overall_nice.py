from __future__ import annotations

import os
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import TwoSlopeNorm

ROOT = os.path.expanduser("~/Desktop/new-Fr")
IN_SEED = os.path.join(ROOT, "outputs/analysis/auc_by_seed_normalized.csv")

OUT_PAPER = os.path.join(ROOT, "outputs/paper")
os.makedirs(OUT_PAPER, exist_ok=True)

def style_rcparams() -> None:
    mpl.rcParams.update({
        "font.size": 12,
        "font.weight": "bold",
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
    })

def build_matrix(df: pd.DataFrame, algo: str):
    sub = df[df["algo"].astype(str).str.upper() == algo].copy()
    sub["activation"] = sub["activation"].astype(str).str.lower()
    sub = sub[sub["activation"].isin(["frelu", "flrelu", "fprelu", "fswish", "fgelu"])].copy()

    cfg_cols = ["algo", "task", "activation", "layers", "placement", "alpha"]
    mean_cfg = (
        sub.groupby(cfg_cols, as_index=False)["delta_pct_vs_relu"]
        .mean()
        .rename(columns={"delta_pct_vs_relu": "mean_delta"})
    )

    best = (
        mean_cfg.groupby(["activation", "task"], as_index=False)["mean_delta"]
        .max()
        .rename(columns={"mean_delta": "best_mean_delta"})
    )

    acts = ["frelu", "flrelu", "fprelu", "fswish", "fgelu"]
    tasks = sorted(best["task"].unique().tolist())

    pivot = (
        best.pivot(index="activation", columns="task", values="best_mean_delta")
        .reindex(index=acts, columns=tasks)
    )

    return pivot.to_numpy(dtype=float), acts, tasks


def save_heatmap(data, row_labels, col_labels, title, base_outpath):
    finite = data[np.isfinite(data)]
    max_abs = float(np.max(np.abs(finite))) if finite.size else 1.0
    if max_abs == 0:
        max_abs = 1.0

    norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=+max_abs)

    fig_w = max(7.8, 0.95 * len(col_labels) + 3.0)
    fig_h = 3.2

    # ==========================================================
    # 1) NORMAL version (as before): PDF + PNG with labels/title
    # ==========================================================
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(data, cmap="coolwarm", aspect="auto", norm=norm)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=25, ha="right", fontsize=11, fontweight="bold")
    ax.set_yticklabels(row_labels, fontsize=12, fontweight="bold")

    ax.set_xlabel("Task", fontsize=13, fontweight="bold")
    ax.set_ylabel("Activation", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")

    threshold = 0.5 * max_abs
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isfinite(v):
                continue
            color = "white" if abs(v) > threshold else "black"
            ax.text(j, i, f"{v:+.1f}%", ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold")

    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.8%", pad=0.08)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Best mean Δ% vs ReLU (normalized AUC)",
                   fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig.savefig(base_outpath + ".pdf", bbox_inches="tight")
    fig.savefig(base_outpath + ".png", dpi=400, bbox_inches="tight")
    plt.close(fig)

    # ==========================================================
    # 2) NO-LABEL version: keep ticks + cell text, remove labels
    # ==========================================================
    fig2, ax2 = plt.subplots(figsize=(fig_w, fig_h))
    im2 = ax2.imshow(data, cmap="coolwarm", aspect="auto", norm=norm)

    ax2.set_xticks(np.arange(len(col_labels)))
    ax2.set_yticks(np.arange(len(row_labels)))
    ax2.set_xticklabels(col_labels, rotation=25, ha="right", fontsize=11, fontweight="bold")
    ax2.set_yticklabels(row_labels, fontsize=12, fontweight="bold")

    # Remove axis labels and title
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    ax2.set_title("")

    threshold = 0.5 * max_abs
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isfinite(v):
                continue
            color = "white" if abs(v) > threshold else "black"
            ax2.text(j, i, f"{v:+.1f}%", ha="center", va="center",
                     color=color, fontsize=11, fontweight="bold")

    ax2.spines[:].set_visible(False)
    ax2.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax2.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax2.tick_params(which="minor", bottom=False, left=False)

    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="3.8%", pad=0.08)
    cbar2 = fig2.colorbar(im2, cax=cax2)

    # Remove colorbar label (keep ticks)
    cbar2.set_label("")

    plt.tight_layout()
    fig2.savefig(base_outpath + "_nolabel.png", dpi=400, bbox_inches="tight")
    plt.close(fig2)

def main():
    style_rcparams()
    df = pd.read_csv(IN_SEED)

    for algo in ["TD3", "SAC"]:
        data, acts, tasks = build_matrix(df, algo)
        base_out = os.path.join(OUT_PAPER, f"fig_heatmap_overall_{algo.lower()}")
        save_heatmap(
            data,
            acts,
            tasks,
            title=f"Overall Performance (best mean Δ% vs ReLU) [{algo}]",
            base_outpath=base_out,
        )
        print("Saved:", base_out + ".pdf")
        print("Saved:", base_out + ".png")


if __name__ == "__main__":
    main()
