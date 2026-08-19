#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FixedLocator, FuncFormatter

ROOT = os.path.expanduser("~/Desktop/new-Fr")
IN_NORM = os.path.join(ROOT, "outputs", "analysis", "auc_by_seed_normalized.csv")

OUT_DIR = os.path.join(ROOT, "outputs", "paper", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

ALGOS = ["SAC", "TD3"]
PLACEMENTS_2L = ["all-actor", "all-critic", "all-both", "first-actor", "first-both"]

# Heatmap row order
ROWS = ["1layer"] + PLACEMENTS_2L


def require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing columns: {missing}\nFound: {list(df.columns)}")


def style_rcparams() -> None:
    mpl.rcParams.update({
        "font.size": 18,
        "font.weight": "bold",
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    })


def draw_heatmap(ax, data: np.ndarray, rows: list[str], cols: list[str], title: str) -> None:
    finite = data[np.isfinite(data)]
    max_abs = float(np.max(np.abs(finite))) if finite.size else 1.0
    if max_abs == 0:
        max_abs = 1.0

    norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=max_abs)
    im = ax.imshow(data, cmap="coolwarm", norm=norm, aspect="equal")

    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels(cols, rotation=25, ha="right", fontweight="bold")
    ax.set_yticklabels(rows, fontweight="bold")
    ax.set_title(title, fontsize=20, fontweight="bold")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.1f}%", ha="center", va="center", fontsize=16, fontweight="bold")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.05)
    cb = plt.colorbar(im, cax=cax)

    ticks = np.linspace(-max_abs, max_abs, 7)
    cb.ax.yaxis.set_major_locator(FixedLocator(ticks))
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x:.1f}"))
    cb.ax.yaxis.set_ticks_position("right")
    cb.ax.tick_params(labelsize=20, width=1.5, length=6, colors="black")
    for lab in cb.ax.get_yticklabels():
        lab.set_fontweight("bold")
        lab.set_color("black")
    cb.outline.set_linewidth(1.5)


def main() -> None:
    style_rcparams()

    if not os.path.exists(IN_NORM):
        raise FileNotFoundError(f"Missing input: {IN_NORM}")

    df = pd.read_csv(IN_NORM)
    require_cols(
        df,
        ["algo", "task", "layers", "placement", "activation", "seed", "delta_pct_vs_relu"],
        "auc_by_seed_normalized.csv",
    )

    df["algo"] = df["algo"].astype(str).str.upper()
    df["task"] = df["task"].astype(str)
    df["layers"] = df["layers"].astype(str).str.lower()
    df["placement"] = df["placement"].astype(str).str.lower()
    df["activation"] = df["activation"].astype(str).str.lower()

    # Ensure alpha_key exists for best over alpha
    if "alpha_key" not in df.columns:
        if "alpha" in df.columns:
            df["alpha_key"] = df["alpha"].astype(object).where(df["alpha"].notna(), "NA")
        else:
            df["alpha_key"] = "NA"

    # Force 1layer placement to none
    df.loc[df["layers"] == "1layer", "placement"] = "none"

    # -----------------------------
    # 1) Aggregate mean across seeds per full config (incl alpha)
    # -----------------------------
    cfg = (
        df.groupby(["algo", "task", "layers", "placement", "activation", "alpha_key"], as_index=False)["delta_pct_vs_relu"]
        .mean()
        .rename(columns={"delta_pct_vs_relu": "mean_delta_pct"})
    )

    # -----------------------------
    # 2) Build best delta for 1layer row per task:
    #    best over activation and alpha within 1layer
    # -----------------------------
    cfg_1 = cfg[cfg["layers"] == "1layer"].copy()
    best_1 = (
        cfg_1.groupby(["algo", "task"], as_index=False)["mean_delta_pct"]
        .max()
        .rename(columns={"mean_delta_pct": "best_delta_pct"})
    )
    best_1["row"] = "1layer"

    # -----------------------------
    # 3) Build best delta for each 2layers placement per task:
    #    best over activation and alpha within that placement
    # -----------------------------
    cfg_2 = cfg[(cfg["layers"] == "2layers") & (cfg["placement"].isin(PLACEMENTS_2L))].copy()

    best_2 = (
        cfg_2.groupby(["algo", "task", "placement"], as_index=False)["mean_delta_pct"]
        .max()
        .rename(columns={"mean_delta_pct": "best_delta_pct", "placement": "row"})
    )

    # Combine into one table of heatmap values
    best_all = pd.concat([best_1[["algo", "task", "row", "best_delta_pct"]],
                          best_2[["algo", "task", "row", "best_delta_pct"]]],
                         ignore_index=True)

    # -----------------------------
    # 4) Plot per algo
    # -----------------------------
    for algo in ALGOS:
        sub = best_all[best_all["algo"] == algo].copy()
        if sub.empty:
            print(f"[WARN] Empty for {algo}")
            continue

        tasks = sorted(sub["task"].unique().tolist())
        rows = ROWS

        pivot = (
            sub.pivot(index="row", columns="task", values="best_delta_pct")
            .reindex(index=rows, columns=tasks)
        )
        data = pivot.to_numpy(dtype=float)

        fig_w = max(10.0, 1.25 * len(tasks) + 6.0)
        fig_h = max(6.0, 1.15 * len(rows) + 4.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        draw_heatmap(
            ax,
            data,
            rows,
            tasks,
            title=f"Best mean Δ% vs ReLU by architecture and placement [{algo}]",
        )

        out_base = os.path.join(OUT_DIR, f"fig_placement_heatmap_with_1layer_{algo.lower()}")
        fig.savefig(out_base + ".png", dpi=400, bbox_inches="tight", pad_inches=0.20)
        fig.savefig(out_base + ".pdf", bbox_inches="tight", pad_inches=0.20)
        plt.close(fig)

        print("Saved:", out_base + ".png")
        print("Saved:", out_base + ".pdf")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e), file=sys.stderr)
        raise
