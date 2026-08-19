#!/usr/bin/env python3
from __future__ import annotations

import os
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FuncFormatter, FixedLocator

ROOT = os.path.expanduser("~/Desktop/new-Fr")
IN_ROOT = os.path.join(ROOT, "outputs", "analysis", "auc_subsets")

# Separate output folder so it won't mix with per-placement heatmaps
OUT_ROOT = os.path.join(ROOT, "outputs", "paper", "heatmaps_auc_best2layers")
os.makedirs(OUT_ROOT, exist_ok=True)

ALGOS = ["TD3", "SAC"]
PLACEMENTS_2L = ["all-actor", "all-critic", "all-both", "first-actor", "first-both"]

# ReLU is baseline, usually not shown as a row here
ACT_ORDER = ["relu", "lrelu", "prelu", "swish", "gelu", "frelu", "flrelu", "fprelu", "fswish", "fgelu"]

# Show placement info under the value, like "+12.3%\nAC"
SHOW_PLACEMENT_IN_CELL = True
PL_ABBR = {
    "all-actor": "AA",
    "all-critic": "AC",
    "all-both": "AB",
    "first-actor": "FA",
    "first-both": "FB",
}


def style_rcparams() -> None:
    mpl.rcParams.update({
        "font.size": 18,
        "font.weight": "bold",
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "savefig.dpi": 300,
    })


def load_best_and_baseline_for_placement(placement: str) -> pd.DataFrame:
    best_path = os.path.join(IN_ROOT, "2layers", placement, "auc_best_by_act_task.csv")
    base_path = os.path.join(IN_ROOT, "2layers", placement, "baseline_relu_by_task.csv")

    if not os.path.exists(best_path):
        raise FileNotFoundError(f"Missing: {best_path}")
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Missing: {base_path}")

    best = pd.read_csv(best_path)
    base = pd.read_csv(base_path)

    best["algo"] = best["algo"].astype(str).str.upper()
    best["task"] = best["task"].astype(str)
    best["layers"] = best["layers"].astype(str).str.lower()
    best["placement"] = best["placement"].astype(str).str.lower()
    best["activation"] = best["activation"].astype(str).str.lower()

    base["algo"] = base["algo"].astype(str).str.upper()
    base["task"] = base["task"].astype(str)
    base["layers"] = base["layers"].astype(str).str.lower()
    base["placement"] = base["placement"].astype(str).str.lower()

    key = ["algo", "task", "layers", "placement"]
    out = best.merge(base[key + ["mean_auc_relu"]], on=key, how="left")

    if out["mean_auc_relu"].isna().any():
        missing = out[out["mean_auc_relu"].isna()][key].drop_duplicates().head(20)
        raise ValueError(f"Missing baseline for placement={placement}. Example keys:\n{missing}")

    out["delta_pct"] = 100.0 * ((out["best_mean_auc"] - out["mean_auc_relu"]) / out["mean_auc_relu"])
    out["placement_src"] = placement
    return out


def build_best_across_placements(algo: str) -> pd.DataFrame:
    all_rows = []
    for pl in PLACEMENTS_2L:
        all_rows.append(load_best_and_baseline_for_placement(pl))
    df = pd.concat(all_rows, ignore_index=True)

    df = df[(df["layers"] == "2layers") & (df["algo"] == algo)].copy()
    df = df[df["activation"].isin(ACT_ORDER)].copy()

    # Pick placement that maximizes delta_pct for each (activation, task)
    idx = df.groupby(["activation", "task"])["delta_pct"].idxmax()
    best = df.loc[idx, ["activation", "task", "delta_pct", "placement_src"]].copy()
    return best


def draw_heatmap(ax, data: np.ndarray, acts: list[str], tasks: list[str], cell_text: np.ndarray, title: str) -> None:
    finite_vals = data[np.isfinite(data)]
    max_abs = float(np.max(np.abs(finite_vals))) if finite_vals.size else 1.0
    if max_abs == 0:
        max_abs = 1.0

    norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=max_abs)
    im = ax.imshow(data, cmap="coolwarm", norm=norm, aspect="equal")

    ax.set_xticks(np.arange(len(tasks)))
    ax.set_yticks(np.arange(len(acts)))
    ax.set_xticklabels(tasks, rotation=25, ha="right", fontsize=18, fontweight="bold")
    ax.set_yticklabels(acts, fontsize=18, fontweight="bold")
    ax.set_title(title, fontsize=20, fontweight="bold")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            txt = cell_text[i, j]
            if txt:
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=18, fontweight="bold", linespacing=0.9)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.05)
    cb = plt.colorbar(im, cax=cax)

    ticks = np.linspace(-max_abs, max_abs, 7)
    cb.ax.yaxis.set_major_locator(FixedLocator(ticks))
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x:.1f}"))
    cb.ax.yaxis.set_ticks_position("right")

    cb.ax.tick_params(labelsize=22, width=1.5, length=6, colors="black")
    for lab in cb.ax.get_yticklabels():
        lab.set_fontweight("bold")
        lab.set_color("black")
    cb.outline.set_linewidth(1.5)


def save_heatmap(data, acts, tasks, cell_text, out_base: str, title: str) -> None:
    fig_w = max(10.0, 1.25 * len(tasks) + 6.0)
    fig_h = max(6.5, 1.25 * len(acts) + 5.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    draw_heatmap(ax, data, acts, tasks, cell_text, title)

    fig.savefig(out_base + ".pdf", bbox_inches="tight", pad_inches=0.20)
    fig.savefig(out_base + ".png", dpi=400, bbox_inches="tight", pad_inches=0.20)
    plt.close(fig)
    print("Saved:", out_base)


def save_exact_cell_outputs(algo: str, pivot: pd.DataFrame, cell_text: np.ndarray, acts: list[str], tasks: list[str]) -> None:
    """
    Saves EXACTLY what appears in the heatmap cells.

    - heatmap_values_<algo>.csv: raw Δ% values (full precision floats)
    - heatmap_cells_<algo>.csv: exact cell strings, including the \n placement label
    - heatmap_cells_<algo>.txt: readable grid with exact cell strings
    """
    algo_l = algo.lower()

    # 1) Raw numeric values (not rounded)
    values_path = os.path.join(OUT_ROOT, f"heatmap_values_{algo_l}.csv")
    pivot.to_csv(values_path, index=True)

    # 2) Exact cell strings (what you draw on the heatmap)
    cells_df = pd.DataFrame(cell_text, index=acts, columns=tasks)
    cells_csv_path = os.path.join(OUT_ROOT, f"heatmap_cells_{algo_l}.csv")
    cells_df.to_csv(cells_csv_path, index=True)

    # 3) TXT grid (exact strings)
    txt_path = os.path.join(OUT_ROOT, f"heatmap_cells_{algo_l}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Heatmap cells for {algo} (exact strings as plotted)\n")
        f.write("Format: +X.X% then placement abbreviation on next line (if enabled)\n")
        f.write("\n")

        # Header
        f.write("ACTIVATION/TASK\n")
        f.write("\t" + "\t".join(tasks) + "\n")

        # Rows
        for i, a in enumerate(acts):
            row_items = [cells_df.loc[a, t] if isinstance(cells_df.loc[a, t], str) else "" for t in tasks]
            # Replace literal newlines with "\n" so the grid stays 1-line per cell in the txt table
            row_items = [s.replace("\n", "\\n") for s in row_items]
            f.write(a + "\t" + "\t".join(row_items) + "\n")

    print("Saved exact cell outputs:")
    print(" ", values_path)
    print(" ", cells_csv_path)
    print(" ", txt_path)


def main() -> None:
    style_rcparams()

    for algo in ALGOS:
        best = build_best_across_placements(algo)

        tasks = sorted(best["task"].unique().tolist())
        acts = [a for a in ACT_ORDER if a in set(best["activation"])]

        pivot = best.pivot(index="activation", columns="task", values="delta_pct").reindex(index=acts, columns=tasks)
        data = pivot.to_numpy(dtype=float)

        pl_map = best.set_index(["activation", "task"])["placement_src"].to_dict()

        cell_text = np.empty((len(acts), len(tasks)), dtype=object)
        cell_text[:] = ""
        for i, a in enumerate(acts):
            for j, t in enumerate(tasks):
                val = pivot.loc[a, t]
                if pd.isna(val):
                    continue
                if SHOW_PLACEMENT_IN_CELL:
                    pl = pl_map.get((a, t), "")
                    ab = PL_ABBR.get(pl, pl)
                    # EXACT formatting used in the heatmap
                    cell_text[i, j] = f"{val:+.1f}%\n{ab}"
                else:
                    cell_text[i, j] = f"{val:+.1f}%"

        out_base = os.path.join(OUT_ROOT, f"fig_heatmap_{algo.lower()}_2layers_bestplacement_auc_rel_nolabel")
        title = f"Best Mean Δ% vs ReLU [{algo}]"
        save_heatmap(data, acts, tasks, cell_text, out_base, title)

        # Save exact numbers/strings used in heatmap cells
        save_exact_cell_outputs(algo, pivot, cell_text, acts, tasks)

    print("Done.")


if __name__ == "__main__":
    main()
