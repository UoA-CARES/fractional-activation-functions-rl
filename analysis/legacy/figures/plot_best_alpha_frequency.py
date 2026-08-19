#!/usr/bin/env python3
from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.expanduser("~/Desktop/new-Fr")
IN_CSV = os.path.join(ROOT, "outputs", "paper", "table_best_alpha_frequency.csv")
OUT_DIR = os.path.join(ROOT, "outputs", "paper", "figures", "alpha_bestfreq")
os.makedirs(OUT_DIR, exist_ok=True)

ACT_ORDER = ["frelu", "flrelu", "fprelu", "fswish", "fgelu"]


def main() -> None:
    df = pd.read_csv(IN_CSV)
    df["algo"] = df["algo"].astype(str).str.upper()
    df["layers"] = df["layers"].astype(str).str.lower()
    df["placement"] = df["placement"].astype(str).str.lower()
    df["activation"] = df["activation"].astype(str).str.lower()

    # We will plot for two cases per algo:
    # - 1layer (placement none)
    # - 2layers best-placement (placement best-placement)
    cases = [
        ("1layer", "none"),
        ("2layers", "best-placement"),
    ]

    for algo in sorted(df["algo"].unique()):
        for layers, placement in cases:
            sub = df[(df["algo"] == algo) & (df["layers"] == layers) & (df["placement"] == placement)].copy()
            if sub.empty:
                continue

            # One plot per activation
            for act in ACT_ORDER:
                s2 = sub[sub["activation"] == act].copy()
                if s2.empty:
                    continue
                s2 = s2.sort_values("alpha")

                plt.figure(figsize=(7.5, 4.5))
                bars = plt.bar(s2["alpha"].astype(str), s2["win_count"])

                plt.xlabel("alpha")
                plt.ylabel("Win count across tasks")
                plt.title(f"Best alpha frequency, {algo}, {layers}, {act}")

                for b, c, p in zip(bars, s2["win_count"], s2["win_pct"]):
                    plt.text(
                        b.get_x() + b.get_width() / 2,
                        c,
                        f"{int(c)} ({p:.1f}%)",
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )

                out_path = os.path.join(OUT_DIR, f"fig_bestalpha_{algo.lower()}_{layers}_{act}.png")
                plt.tight_layout()
                plt.savefig(out_path, dpi=300)
                plt.close()
                print("Saved:", out_path)

    print("Done.")


if __name__ == "__main__":
    main()
