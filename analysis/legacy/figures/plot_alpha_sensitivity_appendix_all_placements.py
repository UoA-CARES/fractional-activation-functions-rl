#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(os.path.expanduser("~/Desktop/new-Fr"))

IN_CSV = ROOT / "outputs" / "paper" / "alpha_sensitivity" / "table_alpha_sensitivity.csv"

OUT_APPENDIX = ROOT / "Appendix" / "alpha_sensitivity_best_task"
OUT_BACKUP = ROOT / "outputs" / "paper" / "alpha_sensitivity" / "appendix_figures_normalized_best"

OUT_APPENDIX.mkdir(parents=True, exist_ok=True)
OUT_BACKUP.mkdir(parents=True, exist_ok=True)

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

PLOT_PLACEMENTS = [
    "all-both",
    "all-actor",
    "all-critic",
    "first-both",
    "first-actor",
    "best-placement",
]

ALGOS = ["SAC", "TD3"]
ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5]


def load_df() -> pd.DataFrame:
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Missing input file: {IN_CSV}")

    df = pd.read_csv(IN_CSV)

    required = [
        "algo",
        "task",
        "layers",
        "placement",
        "activation",
        "alpha",
        "mean_delta_pct",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}\nFound: {list(df.columns)}")

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
        & (df["activation"].isin(ACT_ORDER))
    ].copy()

    if df.empty:
        raise ValueError("No valid two-layer alpha-sensitivity rows found.")

    return df


def make_normalized_best_curve(df: pd.DataFrame, algo: str, placement: str) -> pd.DataFrame:
    sub = df[df["algo"] == algo].copy()

    if placement == "best-placement":
        # First choose best placement for each task, activation, and alpha.
        per_task = (
            sub.groupby(["task", "activation", "alpha"], as_index=False)["mean_delta_pct"]
            .max()
            .rename(columns={"mean_delta_pct": "task_delta"})
        )
    else:
        # Fixed placement.
        per_task = (
            sub[sub["placement"] == placement]
            .groupby(["task", "activation", "alpha"], as_index=False)["mean_delta_pct"]
            .mean()
            .rename(columns={"mean_delta_pct": "task_delta"})
        )

    # Previous appendix definition:
    # For each activation and alpha, take the best task-level improvement.
    curve = (
        per_task.groupby(["activation", "alpha"], as_index=False)["task_delta"]
        .max()
        .rename(columns={"task_delta": "best_delta_pct"})
    )

    # Normalize within each activation family by maximum absolute value.
    rows = []
    for act, g in curve.groupby("activation", sort=False):
        max_abs = g["best_delta_pct"].abs().max()
        g = g.copy()
        if max_abs == 0 or pd.isna(max_abs):
            g["norm_y"] = 0.0
        else:
            g["norm_y"] = g["best_delta_pct"] / max_abs
        rows.append(g)

    out = pd.concat(rows, ignore_index=True)
    return out


def plot_one(curve: pd.DataFrame, algo: str, placement: str) -> None:
    out_dir = OUT_APPENDIX / algo
    backup_dir = OUT_BACKUP / algo

    out_dir.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 7))

    for act in ACT_ORDER:
        g = curve[curve["activation"] == act].sort_values("alpha")

        if g.empty:
            print(f"[SKIP] {algo} {placement} {act}")
            continue

        xs = g["alpha"].tolist()
        ys = g["norm_y"].tolist()
        raw = g["best_delta_pct"].tolist()

        plt.plot(
            xs,
            ys,
            marker="o",
            linewidth=3,
            markersize=8,
            color=ACT_COLOR[act],
            label=ACT_LABEL[act],
        )

        # Labels show raw best Δ%, while y-axis shows normalized value.
        for x, y, r in zip(xs, ys, raw):
            sign = "+" if r >= 0 else ""
            va = "bottom" if y >= 0 else "top"
            plt.text(
                x,
                y,
                f"{sign}{r:.1f}",
                fontsize=10,
                fontweight="bold",
                ha="center",
                va=va,
            )

    plt.axhline(0, linewidth=2)
    plt.title(
        f"Alpha Sensitivity — {algo} ({placement})",
        fontsize=18,
        fontweight="bold",
    )
    plt.xlabel(r"$\alpha$", fontsize=15, fontweight="bold")
    plt.ylabel(r"Normalized best $\Delta\%$", fontsize=15, fontweight="bold")
    plt.xticks(ALPHAS, fontsize=12, fontweight="bold")
    plt.yticks(fontsize=12, fontweight="bold")
    plt.ylim(-1.12, 1.12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc="lower left")
    plt.tight_layout()

    filename = f"{placement}.png"

    plt.savefig(out_dir / filename, dpi=300, bbox_inches="tight")
    plt.savefig(backup_dir / filename, dpi=300, bbox_inches="tight")

    # Compatibility copies for old LaTeX paths.
    if placement == "best-placement":
        plt.savefig(out_dir / "best placement.png", dpi=300, bbox_inches="tight")
        plt.savefig(backup_dir / "best placement.png", dpi=300, bbox_inches="tight")

    if placement == "first-both":
        plt.savefig(out_dir / "fisrt-both.png", dpi=300, bbox_inches="tight")
        plt.savefig(backup_dir / "fisrt-both.png", dpi=300, bbox_inches="tight")

    plt.close()

    print("Wrote:", out_dir / filename)
    print("Wrote:", backup_dir / filename)


def main() -> None:
    df = load_df()

    print("Input rows:", len(df))
    print("Activations:", sorted(df["activation"].unique().tolist()))
    print("Placements:", sorted(df["placement"].unique().tolist()))
    print("Alphas:", sorted(df["alpha"].unique().tolist()))

    for algo in ALGOS:
        for placement in PLOT_PLACEMENTS:
            curve = make_normalized_best_curve(df, algo, placement)
            plot_one(curve, algo, placement)

    print("Done.")


if __name__ == "__main__":
    main()
