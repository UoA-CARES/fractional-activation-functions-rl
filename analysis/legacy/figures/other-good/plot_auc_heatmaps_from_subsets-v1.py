#!/usr/bin/env python3
from __future__ import annotations

import os
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = os.path.expanduser("~/Desktop/new-Fr")
IN_ROOT = os.path.join(ROOT, "outputs", "analysis", "auc_subsets")
OUT_ROOT = os.path.join(ROOT, "outputs", "paper", "heatmaps_auc")
os.makedirs(OUT_ROOT, exist_ok=True)

ALGOS = ["TD3", "SAC"]

# Only the placements that actually exist under outputs/analysis/auc_subsets/2layers/
PLACEMENTS_2L = ["all-actor", "all-critic", "all-both", "first-actor", "first-both"]

DROP_RELU_ROW = True  # for 2layers you usually want this True


def style_rcparams() -> None:
    mpl.rcParams.update({
        "font.size": 14,
        "font.weight": "bold",
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
    })


def activation_order(available: list[str]) -> list[str]:
    preferred = ["relu", "lrelu", "prelu", "swish", "gelu", "frelu", "flrelu", "fprelu", "fswish", "fgelu"]
    aset = set(available)
    ordered = [a for a in preferred if a in aset]
    rest = sorted([a for a in available if a not in ordered])
    return ordered + rest


def cfg_csv_path(layers_tag: str, placement: str | None) -> str:
    if layers_tag == "1layer":
        return os.path.join(IN_ROOT, "1layer", "auc_mean_by_config.csv")
    return os.path.join(IN_ROOT, "2layers", str(placement), "auc_mean_by_config.csv")


def baseline_csv_path(layers_tag: str, placement: str | None) -> str:
    if layers_tag == "1layer":
        return os.path.join(IN_ROOT, "1layer", "baseline_relu_by_task.csv")
    return os.path.join(IN_ROOT, "2layers", str(placement), "baseline_relu_by_task.csv")


def read_baseline(layers_tag: str, placement: str | None) -> pd.DataFrame:
    bpath = baseline_csv_path(layers_tag, placement)
    if not os.path.exists(bpath):
        raise FileNotFoundError(f"Baseline file missing: {bpath}")

    base = pd.read_csv(bpath)
    if base.empty:
        raise ValueError(f"Baseline file is empty: {bpath}")

    base["algo"] = base["algo"].astype(str).str.upper()
    base["task"] = base["task"].astype(str)
    base["layers"] = base["layers"].astype(str).str.lower()
    base["placement"] = base["placement"].astype(str).str.lower()

    return base.rename(columns={"mean_auc_relu": "baseline_auc"})


def build_matrix_best_delta_pct(layers_tag: str, placement: str | None, algo: str):
    cfg_path = cfg_csv_path(layers_tag, placement)
    if not os.path.exists(cfg_path):
        print(f"[WARN] Missing cfg file: {cfg_path}")
        return None, None, None

    df = pd.read_csv(cfg_path)
    if df.empty:
        return None, None, None

    df["algo"] = df["algo"].astype(str).str.upper()
    df["activation"] = df["activation"].astype(str).str.lower()
    df["task"] = df["task"].astype(str)
    df["layers"] = df["layers"].astype(str).str.lower()
    df["placement"] = df["placement"].astype(str).str.lower()

    sub = df[df["algo"] == algo].copy()
    if sub.empty:
        return None, None, None

    # Load baseline for this subset (now exists for 1layer and each 2layers placement)
    base = read_baseline(layers_tag, placement)

    # Safer merge key (prevents accidental mixing)
    key = ["algo", "task", "layers", "placement"]
    sub = sub.merge(base[key + ["baseline_auc"]], on=key, how="left")

    if sub["baseline_auc"].isna().any():
        missing = sub[sub["baseline_auc"].isna()][key].drop_duplicates().head(20)
        raise ValueError(
            f"Missing baseline values for {layers_tag} {placement} {algo}. Example keys:\n{missing}"
        )

    sub["delta_pct"] = 100.0 * ((sub["mean_auc"] - sub["baseline_auc"]) / sub["baseline_auc"])

    # best over alpha_key within each activation-task
    best = (
        sub.groupby(["activation", "task"], as_index=False)["delta_pct"]
        .max()
        .rename(columns={"delta_pct": "best_mean_delta_pct"})
    )

    # Usually drop relu row (especially for 2layers where relu isn't in cfg anyway)
    if DROP_RELU_ROW:
        best = best[best["activation"] != "relu"]

    acts = activation_order(best["activation"].unique().tolist())
    tasks = sorted(best["task"].unique().tolist())

    pivot = (
        best.pivot(index="activation", columns="task", values="best_mean_delta_pct")
        .reindex(index=acts, columns=tasks)
    )

    return pivot.to_numpy(dtype=float), acts, tasks


def draw_heatmap(ax, data: np.ndarray, acts: list[str], tasks: list[str], title: str) -> None:
    finite_vals = data[np.isfinite(data)]
    max_abs = float(np.max(np.abs(finite_vals))) if finite_vals.size else 1.0
    if max_abs == 0:
        max_abs = 1.0

    norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=max_abs)
    im = ax.imshow(data, cmap="coolwarm", norm=norm, aspect="equal")

    ax.set_xticks(np.arange(len(tasks)))
    ax.set_yticks(np.arange(len(acts)))
    ax.set_xticklabels(tasks, rotation=25, ha="right")
    ax.set_yticklabels(acts)
    ax.set_title(title)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:+.1f}%", ha="center", va="center", fontsize=12)

    plt.colorbar(im, ax=ax)


def save_heatmap(data, acts, tasks, out_base: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(1.2 * len(tasks) + 3, 1.2 * len(acts) + 3))
    draw_heatmap(ax, data, acts, tasks, title)
    plt.tight_layout()
    fig.savefig(out_base + ".pdf")
    fig.savefig(out_base + ".png", dpi=300)
    plt.close(fig)
    print("Saved:", out_base)


def plot_subset(layers_tag: str, placement: str | None) -> None:
    for algo in ALGOS:
        data, acts, tasks = build_matrix_best_delta_pct(layers_tag, placement, algo)
        if data is None:
            print(f"[WARN] Empty subset: {layers_tag} {placement} [{algo}]")
            continue

        if layers_tag == "1layer":
            out_base = os.path.join(OUT_ROOT, "1layer", f"fig_heatmap_{algo.lower()}_1layer")
            title = f"1layer Best Mean Δ% vs ReLU [{algo}]"
        else:
            out_base = os.path.join(
                OUT_ROOT, "2layers", str(placement), f"fig_heatmap_{algo.lower()}_2layers_{placement}"
            )
            title = f"2layers ({placement}) Best Mean Δ% vs ReLU [{algo}]"

        os.makedirs(os.path.dirname(out_base), exist_ok=True)
        save_heatmap(data, acts, tasks, out_base, title)


def main() -> None:
    style_rcparams()

    # 1layer
    plot_subset("1layer", "none")

    # 2layers placements
    for pl in PLACEMENTS_2L:
        plot_subset("2layers", pl)

    print("Done.")


if __name__ == "__main__":
    main()
