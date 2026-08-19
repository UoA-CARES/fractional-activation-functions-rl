#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(os.path.expanduser("~/Desktop/new-Fr"))

IN_CSV = ROOT / "outputs" / "paper" / "alpha_sensitivity" / "table_alpha_sensitivity.csv"

OUT_IMG = ROOT / "Images" / "alpha-sensitivity"
OUT_PAPER = ROOT / "outputs" / "paper" / "alpha_sensitivity" / "figures"

OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_PAPER.mkdir(parents=True, exist_ok=True)

PLACEMENTS_2L = ["all-actor", "all-critic", "all-both", "first-actor", "first-both"]

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


def load_summary() -> pd.DataFrame:
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Missing input: {IN_CSV}")

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
        & (df["placement"].isin(PLACEMENTS_2L))
        & (df["activation"].isin(ACT_ORDER))
    ].copy()

    if df.empty:
        raise ValueError("No valid 2-layer fractional alpha rows found.")

    # For each algo, task, activation, and alpha, select the best placement.
    best_place = (
        df.groupby(["algo", "task", "activation", "alpha"], as_index=False)["mean_delta_pct"]
        .max()
        .rename(columns={"mean_delta_pct": "best_delta_pct"})
    )

    # Then average across tasks.
    summary = (
        best_place.groupby(["algo", "activation", "alpha"], as_index=False)["best_delta_pct"]
        .mean()
        .rename(columns={"best_delta_pct": "avg_delta_pct"})
    )

    summary["avg_delta_pct"] = summary["avg_delta_pct"].round(4)

    summary.to_csv(OUT_PAPER / "alpha_sensitivity_2layers_bestplacement_summary.csv", index=False)

    return summary


def plot_algo(summary: pd.DataFrame, algo: str) -> None:
    sub = summary[summary["algo"] == algo].copy()

    if sub.empty:
        print(f"[SKIP] no rows for {algo}")
        return

    plt.figure(figsize=(12, 8))

    for act in ACT_ORDER:
        g = sub[sub["activation"] == act].sort_values("alpha")

        if g.empty:
            print(f"[SKIP] no rows for {algo} {act}")
            continue

        xs = g["alpha"].tolist()
        ys = g["avg_delta_pct"].tolist()

        plt.plot(
            xs,
            ys,
            marker="o",
            linewidth=3,
            markersize=9,
            color=ACT_COLOR.get(act, None),
            label=ACT_LABEL.get(act, act),
        )

        for x, y in zip(xs, ys):
            sign = "+" if y >= 0 else ""
            plt.text(
                x,
                y,
                f"{sign}{y:.1f}",
                fontsize=13,
                fontweight="bold",
                ha="center",
                va="bottom",
            )

    plt.axhline(0, linewidth=2)
    plt.title(
        f"Alpha Sensitivity — {algo} (2 layers, best placement per task)",
        fontsize=20,
        fontweight="bold",
    )
    plt.xlabel(r"$\alpha$", fontsize=16, fontweight="bold")
    plt.ylabel(r"Average $\Delta\%$ vs ReLU", fontsize=16, fontweight="bold")
    plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5], fontsize=13, fontweight="bold")
    plt.yticks(fontsize=13, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=14, loc="lower left")
    plt.tight_layout()

    name = f"fig_alpha_{algo.lower()}_2layers_bestplacement.png"

    plt.savefig(OUT_IMG / name, dpi=300, bbox_inches="tight")
    plt.savefig(OUT_PAPER / name, dpi=300, bbox_inches="tight")
    plt.close()

    print("Wrote:", OUT_IMG / name)
    print("Wrote:", OUT_PAPER / name)


def main() -> None:
    summary = load_summary()

    print("Summary rows:", len(summary))
    print("Activations:", sorted(summary["activation"].unique().tolist()))
    print("Alphas:", sorted(summary["alpha"].unique().tolist()))

    plot_algo(summary, "SAC")
    plot_algo(summary, "TD3")


if __name__ == "__main__":
    main()
