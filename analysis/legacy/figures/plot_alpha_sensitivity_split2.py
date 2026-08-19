#!/usr/bin/env python3
from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.expanduser("~/Desktop/new-Fr")
IN_CSV = os.path.join(ROOT, "outputs", "paper", "table_alpha_sensitivity.csv")

FIG_DIR = os.path.join(ROOT, "outputs", "paper", "figures", "alpha_sensitivity_split")
TAB_DIR = os.path.join(ROOT, "outputs", "paper", "tables_alpha_sensitivity")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

ACT_ORDER = ["frelu", "flrelu", "fprelu", "fswish", "fgelu"]
PLACEMENTS_2L = ["all-actor", "all-critic", "all-both", "first-actor", "first-both"]


# ------------------------------------------------
# Plot function
# ------------------------------------------------
def plot_curves(df: pd.DataFrame, title: str, out_path: str) -> None:
    plt.figure(figsize=(8, 6))

    for act in ACT_ORDER:
        s2 = df[df["activation"] == act].sort_values("alpha")
        if s2.empty:
            continue

        plt.plot(s2["alpha"], s2["y"], marker="o", label=act)

        for x, y in zip(s2["alpha"], s2["y"]):
            plt.text(x, y, f"{y:+.1f}", fontsize=11, ha="center", va="bottom")

    plt.axhline(0)
    plt.xlabel("alpha")
    plt.ylabel("Average Δ% vs ReLU")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved figure:", out_path)


# ------------------------------------------------
# Save tables (CSV + TEX + TXT)
# ------------------------------------------------
def save_tables(df: pd.DataFrame, name: str):

    csv_path = os.path.join(TAB_DIR, f"{name}.csv")
    tex_path = os.path.join(TAB_DIR, f"{name}.tex")
    txt_path = os.path.join(TAB_DIR, f"{name}.txt")

    df.to_csv(csv_path, index=False)

    # --- save latex table
    with open(tex_path, "w") as f:
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\n")
        f.write("Activation & $\\alpha$ & Avg $\\Delta\\%$ \\\\\n")
        f.write("\\midrule\n")

        for _, r in df.iterrows():
            f.write(f"{r['activation']} & {r['alpha']} & {r['y']:.2f} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    # --- save readable text summary
    with open(txt_path, "w") as f:
        for act in ACT_ORDER:
            sub = df[df.activation == act]
            if sub.empty:
                continue

            best = sub.loc[sub["y"].idxmax()]
            f.write(
                f"{act.upper()} best alpha = {best['alpha']} "
                f"(avg Δ% = {best['y']:.2f})\n"
            )

    print("Saved table:", csv_path)
    print("Saved latex:", tex_path)
    print("Saved summary:", txt_path)


# ------------------------------------------------
# Main
# ------------------------------------------------
def main() -> None:

    df = pd.read_csv(IN_CSV)

    df["algo"] = df["algo"].astype(str).str.upper()
    df["layers"] = df["layers"].astype(str).str.lower()
    df["placement"] = df["placement"].astype(str).str.lower()
    df["activation"] = df["activation"].astype(str).str.lower()

    for algo in sorted(df["algo"].unique()):

        dfa = df[df["algo"] == algo].copy()

        # -----------------------------
        # 1layer
        # -----------------------------
        d1 = dfa[dfa["layers"] == "1layer"].copy()

        curve_1 = (
            d1.groupby(["activation", "alpha"], as_index=False)["mean_delta_pct"]
            .mean()
            .rename(columns={"mean_delta_pct": "y"})
        )

        plot_path = os.path.join(FIG_DIR, f"fig_alpha_{algo.lower()}_1layer.png")
        plot_curves(curve_1, f"Alpha Sensitivity — {algo} (1layer)", plot_path)

        save_tables(curve_1, f"alpha_{algo.lower()}_1layer")


        # -----------------------------
        # 2layer best placement
        # -----------------------------
        d2 = dfa[(dfa["layers"] == "2layers") &
                 (dfa["placement"].isin(PLACEMENTS_2L))].copy()

        best_pl = (
            d2.groupby(["task", "activation", "alpha"], as_index=False)["mean_delta_pct"]
            .max()
        )

        curve_2 = (
            best_pl.groupby(["activation", "alpha"], as_index=False)["mean_delta_pct"]
            .mean()
            .rename(columns={"mean_delta_pct": "y"})
        )

        plot_path = os.path.join(FIG_DIR, f"fig_alpha_{algo.lower()}_2layers_bestplacement.png")

        plot_curves(curve_2,
                    f"Alpha Sensitivity — {algo} (2layers best placement)",
                    plot_path)

        save_tables(curve_2, f"alpha_{algo.lower()}_2layers_bestplacement")


        # -----------------------------
        # 2layer per placement (appendix)
        # -----------------------------
        for pl in PLACEMENTS_2L:

            dpl = d2[d2["placement"] == pl].copy()

            curve_pl = (
                dpl.groupby(["activation", "alpha"], as_index=False)["mean_delta_pct"]
                .mean()
                .rename(columns={"mean_delta_pct": "y"})
            )

            plot_path = os.path.join(
                FIG_DIR,
                f"fig_alpha_{algo.lower()}_2layers_{pl}.png"
            )

            plot_curves(
                curve_pl,
                f"Alpha Sensitivity — {algo} ({pl})",
                plot_path
            )

            save_tables(curve_pl, f"alpha_{algo.lower()}_{pl}")


    print("Done.")


if __name__ == "__main__":
    main()
