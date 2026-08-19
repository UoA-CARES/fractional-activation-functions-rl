#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(os.path.expanduser("~/Desktop/new-Fr"))

NEW_ALPHA_CSV = ROOT / "outputs" / "paper" / "alpha_sensitivity" / "table_alpha_sensitivity.csv"

OUT_APPENDIX = ROOT / "Appendix" / "alpha_sensitivity_best_task"
OUT_BACKUP = ROOT / "outputs" / "paper" / "alpha_sensitivity" / "appendix_figures_manual_old_plus_new"
OUT_VALUES = OUT_BACKUP / "used_values"

OUT_APPENDIX.mkdir(parents=True, exist_ok=True)
OUT_BACKUP.mkdir(parents=True, exist_ok=True)
OUT_VALUES.mkdir(parents=True, exist_ok=True)

ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5]

OLD_ACTS = ["frelu", "flrelu", "fprelu"]
NEW_ACTS = ["fgelu", "fswish"]
ACT_ORDER = ["frelu", "flrelu", "fprelu", "fgelu", "fswish"]

ACT_LABEL = {
    "frelu": "FReLU",
    "flrelu": "FLReLU",
    "fprelu": "FPReLU",
    "fgelu": "FGELU",
    "fswish": "FSwish",
}

ACT_COLOR = {
    "frelu": "#1f77b4",   # blue
    "flrelu": "#2ca02c",  # green
    "fprelu": "#d62728",  # red
    "fgelu": "#9467bd",   # purple
    "fswish": "#ffbf00",  # yellow
}

REAL_PLACEMENTS = [
    "all-both",
    "all-actor",
    "all-critic",
    "first-both",
    "first-actor",
]

PLACEMENTS = [
    "all-both",
    "all-actor",
    "all-critic",
    "first-both",
    "first-actor",
    "best-placement",
]


# ---------------------------------------------------------------------
# Manual old values read from your old appendix plots.
# These are the raw labels shown on the old plots.
# ---------------------------------------------------------------------
OLD_VALUES = {
    "SAC": {
        "all-both": {
            "frelu":  [3.0, 21.2, 3.8, 7.7, 5.5],
            "flrelu": [13.9, 3.5, 6.5, 15.8, 13.2],
            "fprelu": [3.0, 21.2, 15.6, 15.8, 22.6],
        },
        "all-actor": {
            "frelu":  [13.0, 23.2, 18.8, 19.8, 13.6],
            "flrelu": [13.6, 19.8, 18.1, 42.9, 34.6],
            "fprelu": [13.0, 26.2, 20.4, 42.9, 38.0],
        },
        "all-critic": {
            "frelu":  [33.8, 12.3, 7.2, 1.3, 1.3],
            "flrelu": [8.5, 52.6, 32.2, 1.3, 1.4],
            "fprelu": [33.8, 52.6, 45.6, 1.3, 14.0],
        },
        "first-both": {
            "frelu":  [38.9, 21.5, 6.8, 11.1, 12.3],
            "flrelu": [38.9, 17.9, 15.7, 13.6, 13.6],
            "fprelu": [38.9, 12.1, 20.7, 10.4, 14.5],
        },
        "first-actor": {
            "frelu":  [32.0, 48.8, 46.8, 26.2, 11.4],
            "flrelu": [9.1, 31.6, 8.0, 34.2, 35.9],
            "fprelu": [8.1, 48.8, 16.0, 13.6, 37.5],
        },
        "best-placement": {
            "frelu":  [33.8, 56.8, 46.8, 26.2, 13.6],
            "flrelu": [21.5, 31.6, 18.1, 42.2, 35.9],
            "fprelu": [33.8, 56.8, 45.6, 43.9, 38.0],
        },
    },

    "TD3": {
        "all-both": {
            "frelu":  [2.2, 1.2, 3.9, 39.0, -2.3],
            "flrelu": [8.3, 1.2, 11.5, -1.4, 11.2],
            "fprelu": [25.9, 3.8, -5.8, -67.1, -103.0],
        },
        "all-actor": {
            "frelu":  [2.6, 31.6, 8.9, 8.2, 21.3],
            "flrelu": [5.8, -2.2, 66.2, 6.0, 5.8],
            "fprelu": [0.5, 32.0, 66.2, 0.6, 34.2],
        },
        "all-critic": {
            "frelu":  [3.3, 5.2, -1.9, -12.6, -80.8],
            "flrelu": [1.9, 7.7, 6.9, -12.6, -80.8],
            "fprelu": [54.3, -8.0, -8.9, 14.0, -80.8],
        },
        "first-both": {
            "frelu":  [3.5, 51.8, -4.2, -21.6, -23.6],
            "flrelu": [5.9, 9.4, 4.2, -14.7, -52.9],
            "fprelu": [19.7, 52.8, 52.8, -42.4, -52.9],
        },
        "first-actor": {
            "frelu":  [4.8, 10.9, 3.5, 41.6, 1.1],
            "flrelu": [103.0, -2.7, 3.9, 0.0, -0.7],
            "fprelu": [1.4, 11.5, 44.6, 33.8, 22.0],
        },
        "best-placement": {
            "frelu":  [8.3, 43.8, 8.9, 41.6, 31.3],
            "flrelu": [103.0, 9.4, 11.5, 6.0, 11.2],
            "fprelu": [54.3, 52.0, 66.2, 14.0, 34.2],
        },
    },
}


def load_current_new() -> pd.DataFrame:
    df = pd.read_csv(NEW_ALPHA_CSV)

    df["algo"] = df["algo"].astype(str).str.upper().str.strip()
    df["task"] = df["task"].astype(str).str.strip()
    df["layers"] = df["layers"].astype(str).str.lower().str.strip()
    df["placement"] = df["placement"].astype(str).str.lower().str.strip()
    df["activation"] = df["activation"].astype(str).str.lower().str.strip()
    df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")
    df["mean_delta_pct"] = pd.to_numeric(df["mean_delta_pct"], errors="coerce")

    df = df.dropna(subset=["alpha", "mean_delta_pct"]).copy()

    df = df[
        (df["layers"] == "2layers")
        & (df["placement"].isin(REAL_PLACEMENTS))
        & (df["activation"].isin(NEW_ACTS))
    ].copy()

    return df


def old_manual_df(algo: str, placement: str) -> pd.DataFrame:
    rows = []

    for act in OLD_ACTS:
        vals = OLD_VALUES[algo][placement][act]
        for alpha, y in zip(ALPHAS, vals):
            rows.append(
                {
                    "algo": algo,
                    "placement": placement,
                    "activation": act,
                    "alpha": alpha,
                    "y": float(y),
                    "source": "manual_old_plot",
                }
            )

    return pd.DataFrame(rows)


def compute_new_df(current: pd.DataFrame, algo: str, placement: str) -> pd.DataFrame:
    sub = current[current["algo"] == algo].copy()

    if placement == "best-placement":
        # For new FGELU and FSwish, use the same appendix idea:
        # strongest task-level improvement across placements.
        d = (
            sub.groupby(["task", "activation", "alpha"], as_index=False)["mean_delta_pct"]
            .max()
        )
        curve = (
            d.groupby(["activation", "alpha"], as_index=False)["mean_delta_pct"]
            .max()
            .rename(columns={"mean_delta_pct": "y"})
        )
    else:
        d = sub[sub["placement"] == placement].copy()
        curve = (
            d.groupby(["activation", "alpha"], as_index=False)["mean_delta_pct"]
            .max()
            .rename(columns={"mean_delta_pct": "y"})
        )

    curve["algo"] = algo
    curve["placement"] = placement
    curve["source"] = "computed_new_plot"

    return curve[["algo", "placement", "activation", "alpha", "y", "source"]]


def add_norm(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for act, g in df.groupby("activation", sort=False):
        g = g.copy()
        max_abs = g["y"].abs().max()

        if max_abs == 0 or pd.isna(max_abs):
            g["norm_y"] = 0.0
        else:
            g["norm_y"] = g["y"] / max_abs

        rows.append(g)

    return pd.concat(rows, ignore_index=True)


def make_values(current: pd.DataFrame, algo: str, placement: str) -> pd.DataFrame:
    old = old_manual_df(algo, placement)
    new = compute_new_df(current, algo, placement)

    df = pd.concat([old, new], ignore_index=True)

    df["activation"] = pd.Categorical(df["activation"], categories=ACT_ORDER, ordered=True)
    df = df.sort_values(["activation", "alpha"]).reset_index(drop=True)

    df = add_norm(df)

    return df


def plot_one(values: pd.DataFrame, algo: str, placement: str) -> None:
    out_dir = OUT_APPENDIX / algo
    backup_dir = OUT_BACKUP / algo

    out_dir.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5.5))

    for act in ACT_ORDER:
        g = values[values["activation"] == act].sort_values("alpha")

        if g.empty:
            continue

        xs = g["alpha"].tolist()
        ys = g["norm_y"].tolist()
        raw = g["y"].tolist()

        plt.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.2,
            markersize=6,
            color=ACT_COLOR[act],
            label=ACT_LABEL[act],
        )

        for x, y, r in zip(xs, ys, raw):
            sign = "+" if r >= 0 else ""
            va = "bottom" if y >= 0 else "top"
            plt.text(
                x,
                y,
                f"{sign}{r:.1f}",
                fontsize=7.5,
                fontweight="bold",
                ha="center",
                va=va,
            )

    plt.axhline(0, linewidth=1.2)
    plt.title(f"Alpha Sensitivity, {algo} 2-layer ({placement})", fontsize=10)
    plt.xlabel("alpha", fontsize=9)
    plt.ylabel("Normalized best Δ%", fontsize=9)
    plt.xticks(ALPHAS, fontsize=8)
    plt.yticks(fontsize=8)
    plt.ylim(-1.08, 1.08)
    plt.grid(True, linewidth=0.4, alpha=0.3)
    plt.legend(fontsize=7, loc="best")
    plt.tight_layout()

    filename = f"{placement}.png"

    plt.savefig(out_dir / filename, dpi=300, bbox_inches="tight")
    plt.savefig(backup_dir / filename, dpi=300, bbox_inches="tight")

    if placement == "best-placement":
        plt.savefig(out_dir / "best placement.png", dpi=300, bbox_inches="tight")
        plt.savefig(backup_dir / "best placement.png", dpi=300, bbox_inches="tight")

    if placement == "first-both":
        plt.savefig(out_dir / "fisrt-both.png", dpi=300, bbox_inches="tight")
        plt.savefig(backup_dir / "fisrt-both.png", dpi=300, bbox_inches="tight")

    plt.close()

    print("Wrote:", out_dir / filename)


def main() -> None:
    current = load_current_new()

    for algo in ["SAC", "TD3"]:
        for placement in PLACEMENTS:
            values = make_values(current, algo, placement)

            out_csv = OUT_VALUES / f"manual_values_{algo}_{placement}.csv"
            values.to_csv(out_csv, index=False)

            print()
            print(f"===== {algo} {placement} values used =====")
            print(values[["activation", "alpha", "y", "norm_y", "source"]].to_string(index=False))

            plot_one(values, algo, placement)

    print()
    print("Done.")
    print("Manual value tables written to:", OUT_VALUES)


if __name__ == "__main__":
    main()
