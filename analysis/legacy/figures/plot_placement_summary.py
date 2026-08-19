#!/usr/bin/env python3
from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.expanduser("~/Desktop/new-Fr")
IN_CSV = os.path.join(ROOT, "outputs", "paper", "table_placement_summary.csv")
OUT_DIR = os.path.join(ROOT, "outputs", "paper", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

PLACEMENT_ORDER = ["all-actor", "all-critic", "all-both", "first-actor", "first-both"]


def main() -> None:
    df = pd.read_csv(IN_CSV)
    df["algo"] = df["algo"].astype(str).str.upper()
    df["placement"] = df["placement"].astype(str).str.lower()

    for algo in sorted(df["algo"].unique()):
        sub = df[df["algo"] == algo].copy()
        sub["placement"] = pd.Categorical(sub["placement"], categories=PLACEMENT_ORDER, ordered=True)
        sub = sub.sort_values("placement")

        plt.figure(figsize=(9, 5))
        bars = plt.bar(sub["placement"], sub["avg_delta_pct"])

        plt.axhline(0)
        plt.xticks(rotation=20, ha="right")
        plt.ylabel("Average Δ% vs ReLU")
        plt.title(f"2layers Placement Ablation — {algo}")

        # value labels
        for b, v in zip(bars, sub["avg_delta_pct"]):
            plt.text(
                b.get_x() + b.get_width() / 2,
                v,
                f"{v:+.1f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontweight="bold",
            )

        out_path = os.path.join(OUT_DIR, f"fig_placement_summary_{algo.lower()}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        print("Saved:", out_path)


if __name__ == "__main__":
    main()
