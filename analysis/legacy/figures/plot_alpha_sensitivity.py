#!/usr/bin/env python3
from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.expanduser("~/Desktop/new-Fr")
IN_CSV = os.path.join(ROOT, "outputs", "paper", "table_alpha_sensitivity.csv")
OUT_DIR = os.path.join(ROOT, "outputs", "paper", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

ACT_ORDER = ["frelu", "flrelu", "fprelu", "fswish", "fgelu"]


def main() -> None:
    df = pd.read_csv(IN_CSV)
    df["algo"] = df["algo"].astype(str).str.upper()
    df["activation"] = df["activation"].astype(str).str.lower()

    # Aggregate across tasks, placements, layers
    curve = (
        df.groupby(["algo", "activation", "alpha"], as_index=False)["mean_delta_pct"]
        .mean()
        .rename(columns={"mean_delta_pct": "avg_delta_pct_across_all"})
    )

    for algo in sorted(curve["algo"].unique()):
        sub = curve[curve["algo"] == algo].copy()

        plt.figure(figsize=(8, 6))

        for act in ACT_ORDER:
            s2 = sub[sub["activation"] == act].sort_values("alpha")
            if s2.empty:
                continue
            plt.plot(s2["alpha"], s2["avg_delta_pct_across_all"], marker="o", label=act)

            # label points
            for x, y in zip(s2["alpha"], s2["avg_delta_pct_across_all"]):
                plt.text(x, y, f"{y:+.1f}", fontsize=11, fontweight="bold", ha="center", va="bottom")

        plt.axhline(0)
        plt.xlabel("alpha")
        plt.ylabel("Average Δ% vs ReLU")
        plt.title(f"Alpha Sensitivity (mean across tasks/placements/layers) — {algo}")
        plt.legend()

        out_path = os.path.join(OUT_DIR, f"fig_alpha_sensitivity_{algo.lower()}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        print("Saved:", out_path)


if __name__ == "__main__":
    main()
