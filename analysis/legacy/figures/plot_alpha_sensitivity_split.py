#!/usr/bin/env python3
from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.expanduser("~/Desktop/new-Fr")
IN_CSV = os.path.join(ROOT, "outputs", "paper", "table_alpha_sensitivity.csv")
OUT_DIR = os.path.join(ROOT, "outputs", "paper", "figures", "alpha_sensitivity_split")
os.makedirs(OUT_DIR, exist_ok=True)

ACT_ORDER = ["frelu", "flrelu", "fprelu", "fswish", "fgelu"]
PLACEMENTS_2L = ["all-actor", "all-critic", "all-both", "first-actor", "first-both"]


def plot_curves(df: pd.DataFrame, title: str, out_path: str) -> None:
    plt.figure(figsize=(8, 6))

    for act in ACT_ORDER:
        s2 = df[df["activation"] == act].sort_values("alpha")
        if s2.empty:
            continue

        plt.plot(s2["alpha"], s2["y"], marker="o", label=act)

        for x, y in zip(s2["alpha"], s2["y"]):
            plt.text(x, y, f"{y:+.1f}", fontsize=11, fontweight="bold", ha="center", va="bottom")

    plt.axhline(0)
    plt.xlabel("alpha")
    plt.ylabel("Average Δ% vs ReLU")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved:", out_path)


def main() -> None:
    df = pd.read_csv(IN_CSV)

    df["algo"] = df["algo"].astype(str).str.upper()
    df["layers"] = df["layers"].astype(str).str.lower()
    df["placement"] = df["placement"].astype(str).str.lower()
    df["activation"] = df["activation"].astype(str).str.lower()

    for algo in sorted(df["algo"].unique()):
        dfa = df[df["algo"] == algo].copy()

        # -----------------------------
        # A) 1layer (main)
        # -----------------------------
        d1 = dfa[dfa["layers"] == "1layer"].copy()
        curve_1 = (
            d1.groupby(["activation", "alpha"], as_index=False)["mean_delta_pct"]
            .mean()
            .rename(columns={"mean_delta_pct": "y"})
        )
        out1 = os.path.join(OUT_DIR, f"fig_alpha_{algo.lower()}_1layer.png")
        plot_curves(curve_1, f"Alpha Sensitivity — {algo} (1layer)", out1)

        # -----------------------------
        # B) 2layers best placement per task (main)
        # For each (task, activation, alpha): take max over placements, then average across tasks
        # -----------------------------
        d2 = dfa[(dfa["layers"] == "2layers") & (dfa["placement"].isin(PLACEMENTS_2L))].copy()

        # best placement per (task, activation, alpha)
        best_pl = (
            d2.groupby(["task", "activation", "alpha"], as_index=False)["mean_delta_pct"]
            .max()
        )

        curve_2_best = (
            best_pl.groupby(["activation", "alpha"], as_index=False)["mean_delta_pct"]
            .mean()
            .rename(columns={"mean_delta_pct": "y"})
        )
        out2 = os.path.join(OUT_DIR, f"fig_alpha_{algo.lower()}_2layers_bestplacement.png")
        plot_curves(curve_2_best, f"Alpha Sensitivity — {algo} (2layers, best placement per task)", out2)

        # -----------------------------
        # C) 2layers per placement (appendix)
        # -----------------------------
        for pl in PLACEMENTS_2L:
            dpl = d2[d2["placement"] == pl].copy()
            curve_pl = (
                dpl.groupby(["activation", "alpha"], as_index=False)["mean_delta_pct"]
                .mean()
                .rename(columns={"mean_delta_pct": "y"})
            )
            outp = os.path.join(OUT_DIR, f"fig_alpha_{algo.lower()}_2layers_{pl}.png")
            plot_curves(curve_pl, f"Alpha Sensitivity — {algo} (2layers, {pl})", outp)

    print("Done.")


if __name__ == "__main__":
    main()
