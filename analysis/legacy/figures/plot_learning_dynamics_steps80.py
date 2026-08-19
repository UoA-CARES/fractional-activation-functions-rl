#!/usr/bin/env python3
from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.expanduser("~/Desktop/new-Fr")
IN_CSV = os.path.join(ROOT, "outputs", "paper", "table_stability_summary.csv")
OUT_DIR = os.path.join(ROOT, "outputs", "paper", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# If you want a consistent order
ACT_ORDER = ["relu", "lrelu", "prelu", "swish", "gelu", "frelu", "flrelu", "fprelu", "fswish", "fgelu"]


def main() -> None:
    df = pd.read_csv(IN_CSV)
    df["algo"] = df["algo"].astype(str).str.upper()
    df["activation"] = df["activation"].astype(str).str.lower()
    df["layers"] = df["layers"].astype(str).str.lower()

    # Focus on 1layer and best 2layers placement only, if present.
    # If you want all placements, remove the placement filter.
    # Here we keep all rows and let you decide later.
    for algo in sorted(df["algo"].unique()):
        sub = df[df["algo"] == algo].copy()

        # Plot for each layers setting separately
        for layers in sorted(sub["layers"].unique()):
            s2 = sub[sub["layers"] == layers].copy()
            if s2.empty:
                continue

            # Order activations if possible
            s2["activation"] = pd.Categorical(s2["activation"], categories=ACT_ORDER, ordered=True)
            s2 = s2.sort_values("activation")

            plt.figure(figsize=(9, 5))
            bars = plt.bar(s2["activation"].astype(str), s2["avg_steps_to_80pct"].astype(float))

            plt.axhline(0)
            plt.ylabel("Average steps to 80% progress")
            plt.title(f"Learning speed summary (steps to 80%) [{algo}, {layers}]")
            plt.xticks(rotation=20, ha="right")

            for b, v in zip(bars, s2["avg_steps_to_80pct"].astype(float)):
                if pd.isna(v):
                    continue
                plt.text(
                    b.get_x() + b.get_width() / 2,
                    v,
                    f"{v:.0f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=11,
                )

            out_path = os.path.join(OUT_DIR, f"fig_steps_to_80pct_{algo.lower()}_{layers}.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=300)
            plt.close()
            print("Saved:", out_path)

    print("Done.")


if __name__ == "__main__":
    main()
