#!/usr/bin/env python3
from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.expanduser("~/Desktop/new-Fr")
IN_CSV = os.path.join(ROOT, "outputs", "paper", "table_alpha_sensitivity.csv")

OUT_FIG_DIR = os.path.join(ROOT, "outputs", "paper", "figures")
OUT_TEX_DIR = os.path.join(ROOT, "outputs", "paper", "latex")
os.makedirs(OUT_FIG_DIR, exist_ok=True)
os.makedirs(OUT_TEX_DIR, exist_ok=True)

ACT_ORDER = ["frelu", "flrelu", "fprelu", "fswish", "fgelu"]
PLACEMENTS_2L = ["all-actor", "all-critic", "all-both", "first-actor", "first-both"]
ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5]


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["algo"] = df["algo"].astype(str).str.upper()
    df["layers"] = df["layers"].astype(str).str.lower()
    df["placement"] = df["placement"].astype(str).str.lower()
    df["activation"] = df["activation"].astype(str).str.lower()
    return df


def build_curve_1layer(dfa: pd.DataFrame) -> pd.DataFrame:
    d1 = dfa[dfa["layers"] == "1layer"].copy()
    curve = (
        d1.groupby(["activation", "alpha"], as_index=False)["mean_delta_pct"]
        .mean()
        .rename(columns={"mean_delta_pct": "y"})
    )
    return curve


def build_curve_2layer_bestplacement(dfa: pd.DataFrame) -> pd.DataFrame:
    d2 = dfa[(dfa["layers"] == "2layers") & (dfa["placement"].isin(PLACEMENTS_2L))].copy()

    # best placement per (task, activation, alpha)
    best_pl = d2.groupby(["task", "activation", "alpha"], as_index=False)["mean_delta_pct"].max()

    # average across tasks
    curve = (
        best_pl.groupby(["activation", "alpha"], as_index=False)["mean_delta_pct"]
        .mean()
        .rename(columns={"mean_delta_pct": "y"})
    )
    return curve


def plot_curve(ax, curve: pd.DataFrame, title: str) -> None:
    for act in ACT_ORDER:
        s = curve[curve["activation"] == act].sort_values("alpha")
        if s.empty:
            continue
        ax.plot(s["alpha"], s["y"], marker="o", label=act)

        # annotate values
        for x, y in zip(s["alpha"], s["y"]):
            ax.text(x, y, f"{y:+.1f}", fontsize=9, ha="center", va="bottom")

    ax.axhline(0, linewidth=1)
    ax.set_xlabel("alpha")
    ax.set_ylabel("Average Δ% vs ReLU")
    ax.set_title(title)
    ax.set_xticks(ALPHAS)
    ax.grid(True, linewidth=0.4, alpha=0.3)


def export_alpha_frequency_tables(df: pd.DataFrame) -> None:
    """
    Creates:
      - alpha_frequency_sac_2layers_bestplacement.csv/.tex
      - alpha_frequency_td3_2layers_bestplacement.csv/.tex

    Definition:
      For each algo and activation, we:
        (1) take max over placements for each (task, activation, alpha),
        (2) choose the alpha that maximizes mean_delta_pct for that (task, activation),
        (3) count how often each alpha is selected across tasks.
    """
    df = df.copy()
    df = df[(df["layers"] == "2layers") & (df["placement"].isin(PLACEMENTS_2L))].copy()

    # max over placements per (algo, task, activation, alpha)
    best_pl = df.groupby(["algo", "task", "activation", "alpha"], as_index=False)["mean_delta_pct"].max()

    # winner alpha per (algo, task, activation)
    idx = best_pl.groupby(["algo", "task", "activation"])["mean_delta_pct"].idxmax()
    winners = best_pl.loc[idx, ["algo", "task", "activation", "alpha"]].copy()

    # counts per (algo, activation, alpha)
    freq = (
        winners.groupby(["algo", "activation", "alpha"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )

    # ensure full grid for display
    rows = []
    for algo in sorted(freq["algo"].unique()):
        for act in ACT_ORDER:
            for a in ALPHAS:
                rows.append((algo, act, a))
    full = pd.DataFrame(rows, columns=["algo", "activation", "alpha"])
    full = full.merge(freq, on=["algo", "activation", "alpha"], how="left").fillna({"count": 0})
    full["count"] = full["count"].astype(int)

    # export per algo as wide table
    for algo in sorted(full["algo"].unique()):
        sub = full[full["algo"] == algo].copy()

        wide = sub.pivot_table(index="activation", columns="alpha", values="count", aggfunc="sum").reset_index()
        # consistent column order
        wide = wide[["activation"] + ALPHAS]
        wide = wide.sort_values("activation", key=lambda s: s.map({a: i for i, a in enumerate(ACT_ORDER)}))

        out_csv = os.path.join(OUT_TEX_DIR, f"table_alpha_frequency_{algo.lower()}_2layers_bestplacement.csv")
        wide.to_csv(out_csv, index=False)

        out_tex = os.path.join(OUT_TEX_DIR, f"table_alpha_frequency_{algo.lower()}_2layers_bestplacement.tex")
        caption = (
            f"Fractional order frequency for each activation in the two-layer experiments ({algo}). "
            "Counts indicate how often each fractional order is selected among the best configurations across tasks."
        )
        label = f"tab:alpha-frequency-{algo.lower()}"

        with open(out_tex, "w") as f:
            f.write("\\begin{table}[t]\n\\centering\n")
            f.write(f"\\caption{{{caption}}}\n")
            f.write(f"\\label{{{label}}}\n")
            f.write("\\small\n")
            f.write("\\setlength{\\tabcolsep}{6pt}\n")
            f.write("\\renewcommand{\\arraystretch}{1.1}\n")
            f.write("\\begin{tabular}{lccccc}\n\\toprule\n")
            f.write("Activation & $\\alpha=0.1$ & $\\alpha=0.2$ & $\\alpha=0.3$ & $\\alpha=0.4$ & $\\alpha=0.5$ \\\\\n")
            f.write("\\midrule\n")
            for _, r in wide.iterrows():
                act = r["activation"]
                counts = [int(r[a]) for a in ALPHAS]
                f.write(f"{act} & {counts[0]} & {counts[1]} & {counts[2]} & {counts[3]} & {counts[4]} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

        print("Wrote:", out_csv)
        print("Wrote:", out_tex)


def export_panel_tex(fig_filename_pdf: str) -> None:
    """
    Writes a LaTeX figure* block that includes the panel.
    """
    out_tex = os.path.join(OUT_TEX_DIR, "fig_alpha_sensitivity_panel.tex")
    with open(out_tex, "w") as f:
        f.write("\\begin{figure*}[t]\n")
        f.write("\\centering\n")
        f.write(f"\\includegraphics[width=\\textwidth]{{{fig_filename_pdf}}}\n")
        f.write("\\caption{Alpha sensitivity of fractional activations. "
                "Top row: SAC; bottom row: TD3. "
                "Left column: 1-layer architecture; right column: 2-layer architecture using the best placement per task. "
                "Points show average $\\Delta\\%$ (normalised AUC vs ReLU) aggregated across tasks.}\n")
        f.write("\\label{fig:alpha_sensitivity_panel}\n")
        f.write("\\end{figure*}\n")
    print("Wrote:", out_tex)


def main() -> None:
    df = pd.read_csv(IN_CSV)
    df = _prep(df)

    # Build curves
    sac = df[df["algo"] == "SAC"].copy()
    td3 = df[df["algo"] == "TD3"].copy()

    sac_1 = build_curve_1layer(sac)
    sac_2 = build_curve_2layer_bestplacement(sac)

    td3_1 = build_curve_1layer(td3)
    td3_2 = build_curve_2layer_bestplacement(td3)

    # Plot 2x2 panel
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)

    plot_curve(axes[0, 0], sac_1, "SAC (1-layer)")
    plot_curve(axes[0, 1], sac_2, "SAC (2-layer, best placement per task)")
    plot_curve(axes[1, 0], td3_1, "TD3 (1-layer)")
    plot_curve(axes[1, 1], td3_2, "TD3 (2-layer, best placement per task)")

    # single legend for whole figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    out_png = os.path.join(OUT_FIG_DIR, "fig_alpha_sensitivity_panel.png")
    out_pdf = os.path.join(OUT_FIG_DIR, "fig_alpha_sensitivity_panel.pdf")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)

    print("Saved:", out_png)
    print("Saved:", out_pdf)

    # Export LaTeX snippet for the panel
    # Use the path relative to your paper folder as you prefer; this is a common one:
    export_panel_tex("outputs/paper/figures/fig_alpha_sensitivity_panel.pdf")

    # Export alpha frequency tables per algo as .tex
    export_alpha_frequency_tables(df)

    print("Done.")


if __name__ == "__main__":
    main()
