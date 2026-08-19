from __future__ import annotations

import os
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import TwoSlopeNorm

ROOT = os.path.expanduser("~/Desktop/new-Fr")
IN_SEED = os.path.join(ROOT, "outputs/analysis/auc_by_seed_normalized_layers.csv")

OUT_ROOT = os.path.join(ROOT, "outputs/paper/heatmaps")
os.makedirs(OUT_ROOT, exist_ok=True)

def style_rcparams() -> None:
    mpl.rcParams.update({
        "font.size": 12,
        "font.weight": "bold",
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
    })

def make_outdir(*parts: str) -> str:
    outdir = os.path.join(OUT_ROOT, *parts)
    os.makedirs(outdir, exist_ok=True)
    return outdir

def activation_order(available: list[str]) -> list[str]:
    preferred = ["relu", "lrelu", "prelu", "swish", "gelu", "frelu", "flrelu", "fprelu", "fswish", "fgelu"]
    aset = set(available)
    ordered = [a for a in preferred if a in aset]
    rest = sorted([a for a in available if a not in set(ordered)])
    return ordered + rest

def draw_heatmap(ax, data, row_labels, col_labels, norm, title, cbar_label):
    im = ax.imshow(data, cmap="coolwarm", aspect="auto", norm=norm)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=25, ha="right", fontsize=10, fontweight="bold")
    ax.set_yticklabels(row_labels, fontsize=11, fontweight="bold")

    ax.set_xlabel("Task", fontsize=12, fontweight="bold")
    ax.set_ylabel("Activation", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")

    finite = data[np.isfinite(data)]
    max_abs = float(np.max(np.abs(finite))) if finite.size else 1.0
    if max_abs == 0:
        max_abs = 1.0
    threshold = 0.50 * max_abs

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isfinite(v):
                continue
            color = "white" if abs(v) > threshold else "black"
            ax.text(j, i, f"{v:+.1f}%", ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold")

    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.6%", pad=0.08)
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.set_label(cbar_label, fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=10, width=1.1)

def build_matrix(df: pd.DataFrame, algo: str, placement: str):
    sub = df[df["algo"].astype(str).str.upper() == algo].copy()
    sub["activation"] = sub["activation"].astype(str).str.lower()
    sub["layers"] = sub["layers"].astype(str)

    # 2layers only
    sub = sub[sub["layers"] == "2"].copy()

    # placement filter
    sub["placement"] = sub["placement"].astype(str)
    sub = sub[sub["placement"] == placement].copy()

    # preserve NaN alpha groups
    sub["alpha_key"] = sub["alpha"].astype(object).where(sub["alpha"].notna(), "NA")

    value_col = "delta_pct_vs_relu_layers"
    if value_col not in sub.columns:
        raise KeyError(f"Missing column {value_col}. Did you run make_delta_normalized_by_layers.py ?")

    cfg_cols = ["algo", "task", "activation", "layers", "placement", "alpha_key"]
    mean_cfg = (
        sub.groupby(cfg_cols, as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: "mean_delta"})
    )

    # Best config per activation per task, inside THIS placement
    best = (
        mean_cfg.groupby(["activation", "task"], as_index=False)["mean_delta"]
        .max()
        .rename(columns={"mean_delta": "best_mean_delta"})
    )

    acts_available = sorted(best["activation"].unique().tolist())
    tasks = sorted(best["task"].unique().tolist())
    acts = activation_order(acts_available)

    # Remove ReLU row because it is always 0% in delta plots
    best = best[best["activation"] != "relu"].copy()
    acts = [a for a in acts if a != "relu"]

    pivot = (
        best.pivot(index="activation", columns="task", values="best_mean_delta")
        .reindex(index=acts, columns=tasks)
    )
    data = pivot.to_numpy(dtype=float)

    finite = data[np.isfinite(data)]
    max_abs = float(np.max(np.abs(finite))) if finite.size else 1.0
    if max_abs == 0:
        max_abs = 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=+max_abs)

    return data, acts, tasks, norm

def save_heatmap(data, acts, tasks, norm, out_base, title):
    fig_w = max(8.2, 0.85 * len(tasks) + 3.4)
    fig_h = max(3.2, 0.55 * len(acts) + 1.8)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    draw_heatmap(
        ax, data, acts, tasks, norm,
        title=title,
        cbar_label="Best mean Δ% vs ReLU (2layers normalisation)"
    )
    plt.tight_layout()
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    fig.savefig(out_base + ".png", dpi=400, bbox_inches="tight")
    plt.close(fig)

def main():
    style_rcparams()
    df = pd.read_csv(IN_SEED)

    placements = ["all-actor", "all-critic", "all-both", "first-actor", "first-both"]
    algos = ["TD3", "SAC"]

    for pl in placements:
        outdir = make_outdir("2layers", pl)
        for algo in algos:
            data, acts, tasks, norm = build_matrix(df, algo, pl)
            out_base = os.path.join(outdir, f"fig_heatmap_{algo.lower()}_2layers_{pl}_layersnorm")
            save_heatmap(
                data, acts, tasks, norm, out_base,
                title=f"2layers Placement: {pl} (best mean Δ% vs 2layers ReLU) [{algo}]"
            )
            print("Saved:", out_base + ".pdf/.png")

if __name__ == "__main__":
    main()
