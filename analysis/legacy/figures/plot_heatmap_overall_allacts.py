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

def activation_order(available: list[str]) -> list[str]:
    preferred = ["relu", "lrelu", "prelu", "swish", "gelu", "frelu", "flrelu", "fprelu", "fswish", "fgelu"]
    aset = set(available)
    ordered = [a for a in preferred if a in aset]
    rest = sorted([a for a in available if a not in set(ordered)])
    return ordered + rest

def build_matrix(df: pd.DataFrame, algo: str):
    sub = df[df["algo"].astype(str).str.upper() == algo].copy()
    sub["activation"] = sub["activation"].astype(str).str.lower()

    # IMPORTANT: keep non-fractional activations in groupby by avoiding NaN in alpha
    # pandas groupby drops NaN groups by default
    sub["alpha_key"] = sub["alpha"].astype(object).where(sub["alpha"].notna(), "NA")

    # Debug to confirm what exists
    print(f"[DEBUG] {algo} activations in normalized CSV:",
          sorted(sub["activation"].unique().tolist()))

    cfg_cols = ["algo", "task", "activation", "layers", "placement", "alpha_key"]
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

    acts_available = sorted(best["activation"].unique().tolist())

    # REMOVE RELU FROM VISUALIZATION
    acts_available = [a for a in acts_available if a != "relu"]

    tasks = sorted(best["task"].unique().tolist())
    acts = activation_order(acts_available)

    pivot = best.pivot(index="activation", columns="task", values="best_mean_delta").reindex(index=acts, columns=tasks)
    return pivot.to_numpy(dtype=float), acts, tasks

def draw_heatmap(ax, data, row_labels, col_labels, norm, show_axis_labels, show_title, title, show_cbar_label):
    im = ax.imshow(data, cmap="coolwarm", aspect="auto", norm=norm)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=25, ha="right", fontsize=10, fontweight="bold")
    ax.set_yticklabels(row_labels, fontsize=11, fontweight="bold")

    ax.set_xlabel("Task" if show_axis_labels else "", fontsize=12, fontweight="bold")
    ax.set_ylabel("Activation" if show_axis_labels else "", fontsize=12, fontweight="bold")
    ax.set_title(title if show_title else "", fontsize=13, fontweight="bold")

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
    cbar.set_label("Best mean Δ% vs ReLU (normalized AUC)" if show_cbar_label else "",
                   fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=10, width=1.1)

def save_all_versions(data, acts, tasks, algo: str):
    finite = data[np.isfinite(data)]
    max_abs = float(np.max(np.abs(finite))) if finite.size else 1.0
    if max_abs == 0:
        max_abs = 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=+max_abs)

    fig_w = max(8.2, 0.85 * len(tasks) + 3.4)
    fig_h = max(3.2, 0.55 * len(acts) + 1.8)

    base_out = os.path.join(OUT_PAPER, f"fig_heatmap_overall_{algo.lower()}_allacts")

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    draw_heatmap(ax, data, acts, tasks, norm, True, True,
                 f"Overall Performance (best mean Δ% vs ReLU) [{algo}]",
                 True)
    plt.tight_layout()
    fig.savefig(base_out + ".pdf", bbox_inches="tight")
    fig.savefig(base_out + ".png", dpi=400, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(fig_w, fig_h))
    draw_heatmap(ax2, data, acts, tasks, norm, False, False, "", False)
    plt.tight_layout()
    fig2.savefig(base_out + "_nolabel.png", dpi=400, bbox_inches="tight")
    plt.close(fig2)

    print("Saved:", base_out + ".pdf")
    print("Saved:", base_out + ".png")
    print("Saved:", base_out + "_nolabel.png")

def main():
    style_rcparams()
    df = pd.read_csv(IN_SEED)

    print("[DEBUG] All activations in normalized CSV:",
          sorted(df["activation"].astype(str).str.lower().unique().tolist()))

    for algo in ["TD3", "SAC"]:
        data, acts, tasks = build_matrix(df, algo)
        print(f"[DEBUG] {algo} heatmap rows (activations):", acts)
        save_all_versions(data, acts, tasks, algo)

if __name__ == "__main__":
    main()
