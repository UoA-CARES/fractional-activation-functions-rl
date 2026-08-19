#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FuncFormatter, FixedLocator


# =========================
# Paths
# =========================
ROOT = os.path.expanduser("~/Desktop/new-Fr")
IN_ROOT = os.path.join(ROOT, "outputs", "analysis", "auc_subsets")

# Output folder for paper artifacts
OUT_ROOT = os.path.join(ROOT, "outputs", "paper", "heatmaps_auc_best2layers")
os.makedirs(OUT_ROOT, exist_ok=True)


# =========================
# Config
# =========================
ALGOS = ["TD3", "SAC"]

PLACEMENTS_2L = [
    "all-actor",
    "all-critic",
    "all-both",
    "first-actor",
    "first-both",
]

# ReLU is baseline. I keep it here because your pasted script already included it.
# If you do not want ReLU as a row in the heatmap, remove "relu" from this list.
ACT_ORDER = [
    "lrelu",
    "prelu",
    "gelu",
    "swish",
    "frelu",
    "flrelu",
    "fprelu",
    "fgelu",
    "fswish",
]

FRACTIONAL_SET = {
    "frelu",
    "flrelu",
    "fprelu",
    "fswish",
    "fgelu",
}

ACT_LABEL = {
    "relu": "ReLU",
    "lrelu": "LReLU",
    "prelu": "PReLU",
    "swish": "Swish",
    "gelu": "GELU",
    "frelu": "FReLU",
    "flrelu": "FLReLU",
    "fprelu": "FPReLU",
    "fswish": "FSwish",
    "fgelu": "FGELU",
}

# Show placement info under the value, like "+12.3%\nAC"
SHOW_PLACEMENT_IN_CELL = True

PL_ABBR = {
    "all-actor": "AA",
    "all-critic": "AC",
    "all-both": "AB",
    "first-actor": "FA",
    "first-both": "FB",
}

# If you want the TXT to include only fractional acts, set this to True.
ONLY_FRACTIONAL_IN_TXT = False


# =========================
# Normalization helpers
# =========================
def norm_activation(x) -> str:
    s = str(x).strip().lower()

    amap = {
        "r": "relu",
        "relu": "relu",

        "l": "lrelu",
        "lrelu": "lrelu",
        "leakyrelu": "lrelu",
        "leaky_relu": "lrelu",

        "p": "prelu",
        "prelu": "prelu",
        "parametricrelu": "prelu",
        "parametric_relu": "prelu",

        "s": "swish",
        "swish": "swish",
        "silu": "swish",

        "g": "gelu",
        "gelu": "gelu",

        "fr": "frelu",
        "frelu": "frelu",
        "fractionalrelu": "frelu",
        "fractional_relu": "frelu",

        "fl": "flrelu",
        "flrelu": "flrelu",
        "fractionalleakyrelu": "flrelu",
        "fractional_leaky_relu": "flrelu",

        "fp": "fprelu",
        "fprelu": "fprelu",
        "fractionalprelu": "fprelu",
        "fractional_prelu": "fprelu",

        "fs": "fswish",
        "fswish": "fswish",
        "fractionalswish": "fswish",
        "fractional_swish": "fswish",
        "fractionalswishbeta": "fswish",
        "fractional_swish_beta": "fswish",

        "fg": "fgelu",
        "fgelu": "fgelu",
        "fractionalgelu": "fgelu",
        "fractional_gelu": "fgelu",
        "fractionalgelubeta": "fgelu",
        "fractional_gelu_beta": "fgelu",
    }

    return amap.get(s, s)


def norm_placement(x) -> str:
    s = str(x).strip().lower()

    if s in {"nan", "none", "na", ""}:
        return "none"

    return s


# =========================
# Plot style
# =========================
def style_rcparams() -> None:
    mpl.rcParams.update(
        {
            "font.size": 18,
            "font.weight": "bold",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "savefig.dpi": 300,
        }
    )


# =========================
# Loading helpers
# =========================
def _require_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")


def load_best_and_baseline_for_placement(placement: str) -> pd.DataFrame:
    """
    Loads:
      - auc_best_by_act_task.csv
      - baseline_relu_by_task.csv

    Then computes delta_pct vs ReLU for each row.
    """
    best_path = os.path.join(IN_ROOT, "2layers", placement, "auc_best_by_act_task.csv")
    base_path = os.path.join(IN_ROOT, "2layers", placement, "baseline_relu_by_task.csv")

    _require_exists(best_path)
    _require_exists(base_path)

    best = pd.read_csv(best_path)
    base = pd.read_csv(base_path)

    # Normalize key columns
    for df in (best, base):
        df["algo"] = df["algo"].astype(str).str.upper().str.strip()
        df["task"] = df["task"].astype(str).str.strip()
        df["layers"] = df["layers"].astype(str).str.lower().str.strip()
        df["placement"] = df["placement"].apply(norm_placement)

    best["activation"] = best["activation"].apply(norm_activation)

    # Required columns
    if "best_mean_auc" not in best.columns:
        if "mean_auc" in best.columns:
            best = best.rename(columns={"mean_auc": "best_mean_auc"})
        else:
            raise KeyError(
                f"{best_path} missing best_mean_auc or mean_auc column. "
                f"Found: {list(best.columns)}"
            )

    if "mean_auc_relu" not in base.columns:
        raise KeyError(
            f"{base_path} missing mean_auc_relu column. "
            f"Found: {list(base.columns)}"
        )

    # Optional alpha columns
    alpha_col = None
    for c in ["best_alpha", "alpha", "alpha_key"]:
        if c in best.columns:
            alpha_col = c
            break

    if alpha_col and alpha_col != "best_alpha":
        best = best.rename(columns={alpha_col: "best_alpha"})

    if "best_alpha" not in best.columns:
        best["best_alpha"] = np.nan

    key = ["algo", "task", "layers", "placement"]

    out = best.merge(
        base[key + ["mean_auc_relu"]],
        on=key,
        how="left",
    )

    if out["mean_auc_relu"].isna().any():
        missing = out[out["mean_auc_relu"].isna()][key].drop_duplicates().head(20)
        raise ValueError(
            f"Missing baseline rows after merge for placement={placement}. "
            f"Example keys:\n{missing}"
        )

    # Keep the original delta formula.
    out["delta_pct"] = 100.0 * (
        (out["best_mean_auc"] - out["mean_auc_relu"]) / out["mean_auc_relu"]
    )

    out["placement_src"] = placement

    return out


def load_all_2layer_rows() -> pd.DataFrame:
    rows = []

    for pl in PLACEMENTS_2L:
        rows.append(load_best_and_baseline_for_placement(pl))

    df = pd.concat(rows, ignore_index=True)

    df = df[df["layers"] == "2layers"].copy()
    df["activation"] = df["activation"].apply(norm_activation)

    df = df[df["activation"].isin(ACT_ORDER)].copy()

    if ONLY_FRACTIONAL_IN_TXT:
        df = df[df["activation"].isin(FRACTIONAL_SET)].copy()

    print("\nActivations found in 2-layer heatmap input:")
    print(sorted(df["activation"].dropna().unique()))

    print("\nRows per activation:")
    print(df["activation"].value_counts().sort_index())

    return df


# =========================
# Best selection logic
# =========================
def best_per_activation_task_across_placements(
    df_all: pd.DataFrame,
    algo: str,
) -> pd.DataFrame:
    """
    Pick placement that maximizes delta_pct for each activation and task.
    """
    df = df_all[df_all["algo"] == algo].copy()

    idx = df.groupby(["activation", "task"])["delta_pct"].idxmax()

    best = df.loc[
        idx,
        ["activation", "task", "delta_pct", "placement_src"],
    ].copy()

    return best


def best_per_task_overall(df_all: pd.DataFrame, algo: str) -> pd.DataFrame:
    """
    Pick the single best configuration per task across activations and placements.
    """
    df = df_all[df_all["algo"] == algo].copy()

    idx = df.groupby(["task"])["delta_pct"].idxmax()

    cols = ["task", "activation", "placement_src", "delta_pct", "best_alpha"]

    best = df.loc[idx, cols].copy()
    best = best.sort_values("task", ascending=True).reset_index(drop=True)

    return best


# =========================
# Heatmap plotting
# =========================
def draw_heatmap(
    ax,
    data: np.ndarray,
    acts: List[str],
    tasks: List[str],
    cell_text: np.ndarray,
    title: str,
) -> None:
    finite_vals = data[np.isfinite(data)]

    max_abs = float(np.max(np.abs(finite_vals))) if finite_vals.size else 1.0

    if max_abs == 0:
        max_abs = 1.0

    norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=max_abs)
    im = ax.imshow(data, cmap="coolwarm", norm=norm, aspect="equal")

    act_labels = [ACT_LABEL.get(a, a) for a in acts]

    ax.set_xticks(np.arange(len(tasks)))
    ax.set_yticks(np.arange(len(acts)))

    ax.set_xticklabels(
        tasks,
        rotation=25,
        ha="right",
        fontsize=18,
        fontweight="bold",
    )

    ax.set_yticklabels(
        act_labels,
        fontsize=18,
        fontweight="bold",
    )

    ax.set_title(title, fontsize=20, fontweight="bold")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            txt = cell_text[i, j]

            if txt:
                ax.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=18,
                    fontweight="bold",
                    linespacing=0.9,
                )

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


def save_heatmap(
    data: np.ndarray,
    acts: List[str],
    tasks: List[str],
    cell_text: np.ndarray,
    out_base: str,
    title: str,
) -> None:
    fig_w = max(10.0, 1.25 * len(tasks) + 6.0)
    fig_h = max(6.5, 1.25 * len(acts) + 5.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    draw_heatmap(
        ax=ax,
        data=data,
        acts=acts,
        tasks=tasks,
        cell_text=cell_text,
        title=title,
    )

    fig.savefig(out_base + ".pdf", bbox_inches="tight", pad_inches=0.20)
    fig.savefig(out_base + ".png", dpi=400, bbox_inches="tight", pad_inches=0.20)

    plt.close(fig)

    print("Saved:", out_base + ".pdf/.png")


# =========================
# TXT report helpers
# =========================
def _act_pretty(a: str) -> str:
    return ACT_LABEL.get(a, a)


def _alpha_pretty(alpha) -> str:
    try:
        if pd.isna(alpha):
            return "--"
    except Exception:
        pass

    if isinstance(alpha, str):
        return alpha

    try:
        return f"{float(alpha):.1f}"
    except Exception:
        return str(alpha)


def save_txt_report(df_all: pd.DataFrame, out_path: str) -> None:
    lines: List[str] = []

    lines.append("2-Layer Best-Placement Summary (Delta% AUC vs ReLU)")
    lines.append(f"Input root: {IN_ROOT}")
    lines.append(f"Output root: {OUT_ROOT}")
    lines.append("")
    lines.append("Placements considered: " + ", ".join(PLACEMENTS_2L))
    lines.append("Activations considered: " + ", ".join(ACT_ORDER))

    if ONLY_FRACTIONAL_IN_TXT:
        lines.append(
            "TXT filtered: ONLY fractional activations "
            "(frelu, flrelu, fprelu, fswish, fgelu)"
        )

    lines.append("")
    lines.append(
        "Placement abbreviations: "
        + ", ".join([f"{k}={v}" for k, v in PL_ABBR.items()])
    )

    lines.append("")
    lines.append(
        "Selection rule: For each task, we choose the configuration "
        "(activation, placement) with the maximum Delta% relative to the "
        "placement-matched ReLU baseline."
    )
    lines.append(
        "Note: best_alpha is used when present in auc_best_by_act_task.csv. "
        "Otherwise alpha is reported as '--'."
    )
    lines.append("")

    for algo in ALGOS:
        best_task = best_per_task_overall(df_all, algo)

        lines.append("=" * 78)
        lines.append(f"BEST CONFIGURATION PER TASK [{algo}] (2 layers)")
        lines.append("=" * 78)

        header = f"{'Task':<20}  {'Act.':<8}  {'alpha':<5}  {'Place':<12}  {'Delta%':>8}"
        lines.append(header)
        lines.append("-" * len(header))

        for _, r in best_task.iterrows():
            task = str(r["task"])
            act = _act_pretty(str(r["activation"]))
            alpha = _alpha_pretty(r.get("best_alpha", np.nan))
            pl = str(r["placement_src"])
            delta = float(r["delta_pct"])

            lines.append(
                f"{task:<20}  {act:<8}  {alpha:<5}  {pl:<12}  {delta:>+8.2f}"
            )

        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("Saved TXT report:", out_path)


# =========================
# Main
# =========================
def main() -> None:
    style_rcparams()

    df_all = load_all_2layer_rows()

    for algo in ALGOS:
        best = best_per_activation_task_across_placements(df_all, algo)

        tasks = sorted(best["task"].unique().tolist())
        acts = [a for a in ACT_ORDER if a in set(best["activation"])]

        pivot = best.pivot(
            index="activation",
            columns="task",
            values="delta_pct",
        ).reindex(index=acts, columns=tasks)

        data = pivot.to_numpy(dtype=float)

        pl_map: Dict[Tuple[str, str], str] = best.set_index(
            ["activation", "task"]
        )["placement_src"].to_dict()

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
                    cell_text[i, j] = f"{val:+.1f}%\n{ab}"
                else:
                    cell_text[i, j] = f"{val:+.1f}%"

        out_base = os.path.join(
            OUT_ROOT,
            f"fig_heatmap_{algo.lower()}_2layers_bestplacement_auc_rel_nolabel",
        )

        title = f"Best Mean Delta% vs ReLU [{algo}]"

        save_heatmap(
            data=data,
            acts=acts,
            tasks=tasks,
            cell_text=cell_text,
            out_base=out_base,
            title=title,
        )

    txt_path = os.path.join(OUT_ROOT, "best_2layers_summary.txt")
    save_txt_report(df_all, txt_path)

    print("Done.")


if __name__ == "__main__":
    main()
