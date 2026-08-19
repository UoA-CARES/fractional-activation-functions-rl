#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import pandas as pd
import numpy as np

ROOT = os.path.expanduser("~/Desktop/new-Fr")

IN_DYN = os.path.join(ROOT, "outputs", "paper", "table_learning_dynamics.csv")
IN_STAB = os.path.join(ROOT, "outputs", "paper", "table_stability_summary.csv")

OUT_DIR = os.path.join(ROOT, "outputs", "paper")
OUT_TEX_DIR = os.path.join(OUT_DIR, "tex_tables")
os.makedirs(OUT_TEX_DIR, exist_ok=True)

OUT_A_CSV = os.path.join(OUT_DIR, "table_learning_dynamics_best_by_task.csv")
OUT_B_CSV = os.path.join(OUT_DIR, "table_learning_dynamics_relu_vs_best_fractional.csv")
OUT_C_CSV = os.path.join(OUT_DIR, "table_learning_dynamics_activation_ranking.csv")

OUT_A_TEX = os.path.join(OUT_TEX_DIR, "table_learning_dynamics_best_by_task.tex")
OUT_B_TEX = os.path.join(OUT_TEX_DIR, "table_learning_dynamics_relu_vs_best_fractional.tex")
OUT_C_TEX = os.path.join(OUT_TEX_DIR, "table_learning_dynamics_activation_ranking.tex")

FRACTIONAL = {"frelu", "flrelu", "fprelu", "fswish", "fgelu"}


def require(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing input: {path}")


def latex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def df_to_booktabs(df: pd.DataFrame, caption: str, label: str, colspec: str | None = None) -> str:
    cols = df.columns.tolist()
    if colspec is None:
        colspec = "l" * len(cols)

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{" + latex_escape(caption) + "}")
    lines.append("\\label{" + latex_escape(label) + "}")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.12}")
    lines.append("\\begin{tabular}{" + colspec + "}")
    lines.append("\\toprule")
    lines.append(" & ".join([latex_escape(c) for c in cols]) + " \\\\")
    lines.append("\\midrule")
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if pd.isna(v):
                vals.append("")
            else:
                vals.append(latex_escape(str(v)))
        lines.append(" & ".join(vals) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def fmt_delta(x) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x):+0.1f}"


def fmt_steps(x) -> str:
    if pd.isna(x):
        return ""
    return f"{int(round(float(x))):d}"


def fmt_cv(x) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x):.3f}"


def main() -> None:
    require(IN_DYN)
    require(IN_STAB)

    dyn = pd.read_csv(IN_DYN)
    stab = pd.read_csv(IN_STAB)

    # Normalize strings
    for df in (dyn, stab):
        df["algo"] = df["algo"].astype(str).str.upper()
        df["layers"] = df["layers"].astype(str).str.lower()
        df["placement"] = df["placement"].astype(str).str.lower()
        df["activation"] = df["activation"].astype(str).str.lower()

    # -----------------------------
    # Table A: Best config per (algo, task)
    # Criterion: max mean_delta_pct
    # Show: best activation, layers, placement, mean_delta_pct, steps_to_80, CV
    # -----------------------------
    needed = [
        "algo", "task", "layers", "placement", "activation", "alpha_key",
        "mean_delta_pct", "mean_steps_to_80pct", "cv_final_return"
    ]
    missing = [c for c in needed if c not in dyn.columns]
    if missing:
        raise KeyError(f"table_learning_dynamics.csv missing columns: {missing}\nFound: {list(dyn.columns)}")

    dyn_best = dyn.copy()
    idx = dyn_best.groupby(["algo", "task"])["mean_delta_pct"].idxmax()
    best_by_task = dyn_best.loc[idx, needed].copy()

    best_by_task = best_by_task.rename(columns={
        "algo": "Algo",
        "task": "Task",
        "layers": "Layers",
        "placement": "Placement",
        "activation": "Act",
        "alpha_key": "α",
        "mean_delta_pct": "Δ%",
        "mean_steps_to_80pct": "Steps80",
        "cv_final_return": "CV",
    })

    best_by_task["Δ%"] = best_by_task["Δ%"].map(fmt_delta)
    best_by_task["Steps80"] = best_by_task["Steps80"].map(fmt_steps)
    best_by_task["CV"] = best_by_task["CV"].map(fmt_cv)

    best_by_task = best_by_task.sort_values(["Algo", "Task"])
    best_by_task.to_csv(OUT_A_CSV, index=False)

    texA = df_to_booktabs(
        best_by_task,
        caption="Best learning-dynamics configuration per task (chosen by mean Δ% vs ReLU). Steps80 is the mean steps to reach 80% of final return; CV is coefficient of variation of final return across seeds.",
        label="tab:learning-dynamics-best-by-task",
        colspec="lll" + "l" + "l" + "l" + "l" + "l",
    )
    with open(OUT_A_TEX, "w", encoding="utf-8") as f:
        f.write(texA)

    # Print A to terminal
    print("\n" + "=" * 90)
    print("TABLE A: Best config per (Algo, Task) by Δ%")
    print("=" * 90)
    print(best_by_task.to_string(index=False))

    # -----------------------------
    # Table B: ReLU vs Best Fractional summary per (algo, layers)
    # - ReLU row from stability summary (activation == relu)
    # - Best fractional chosen by avg_mean_delta_pct among fractional activations
    # Show avg Δ%, avg Steps80, avg CV
    # -----------------------------
    neededS = ["algo", "layers", "activation", "avg_mean_delta_pct", "avg_steps_to_80pct", "avg_cv_final_return"]
    missingS = [c for c in neededS if c not in stab.columns]
    if missingS:
        raise KeyError(f"table_stability_summary.csv missing columns: {missingS}\nFound: {list(stab.columns)}")

    relu = stab[stab["activation"] == "relu"].copy()

    frac = stab[stab["activation"].isin(FRACTIONAL)].copy()
    # pick best fractional per (algo, layers) by avg_mean_delta_pct
    idx2 = frac.groupby(["algo", "layers"])["avg_mean_delta_pct"].idxmax()
    best_frac = frac.loc[idx2].copy()

    relu = relu.rename(columns={
        "avg_mean_delta_pct": "relu_avg_Δ%",
        "avg_steps_to_80pct": "relu_avg_Steps80",
        "avg_cv_final_return": "relu_avg_CV",
    })
    best_frac = best_frac.rename(columns={
        "activation": "best_frac_act",
        "avg_mean_delta_pct": "best_frac_avg_Δ%",
        "avg_steps_to_80pct": "best_frac_avg_Steps80",
        "avg_cv_final_return": "best_frac_avg_CV",
    })

    B = relu.merge(
        best_frac[["algo", "layers", "best_frac_act", "best_frac_avg_Δ%", "best_frac_avg_Steps80", "best_frac_avg_CV"]],
        on=["algo", "layers"],
        how="left",
    )

    B["gain_Δ%"] = B["best_frac_avg_Δ%"] - B["relu_avg_Δ%"]
    B["gain_Steps80"] = B["best_frac_avg_Steps80"] - B["relu_avg_Steps80"]
    B["gain_CV"] = B["best_frac_avg_CV"] - B["relu_avg_CV"]

    # Format and rename for paper
    B2 = B.rename(columns={
        "algo": "Algo",
        "layers": "Layers",
        "best_frac_act": "Best fractional",
    }).copy()

    for c in ["relu_avg_Δ%", "best_frac_avg_Δ%", "gain_Δ%"]:
        B2[c] = B2[c].map(fmt_delta)

    for c in ["relu_avg_Steps80", "best_frac_avg_Steps80", "gain_Steps80"]:
        B2[c] = B2[c].map(fmt_steps)

    for c in ["relu_avg_CV", "best_frac_avg_CV", "gain_CV"]:
        B2[c] = B2[c].map(fmt_cv)

    B2 = B2[[
        "Algo", "Layers", "Best fractional",
        "relu_avg_Δ%", "best_frac_avg_Δ%", "gain_Δ%",
        "relu_avg_Steps80", "best_frac_avg_Steps80", "gain_Steps80",
        "relu_avg_CV", "best_frac_avg_CV", "gain_CV",
    ]].sort_values(["Algo", "Layers"])

    B2.to_csv(OUT_B_CSV, index=False)

    texB = df_to_booktabs(
        B2,
        caption="Learning-dynamics summary comparing ReLU to the best fractional activation (selected by average Δ% across tasks) for each algorithm and architecture.",
        label="tab:learning-dynamics-relu-vs-fractional",
        colspec="ll" + "l" + "ccc" + "ccc" + "ccc",
    )
    with open(OUT_B_TEX, "w", encoding="utf-8") as f:
        f.write(texB)

    print("\n" + "=" * 90)
    print("TABLE B: ReLU vs best fractional summary per (Algo, Layers)")
    print("=" * 90)
    print(B2.to_string(index=False))

    # -----------------------------
    # Table C: Activation ranking (stability summary) per (algo, layers)
    # Keep compact: top 3 activations by avg_mean_delta_pct
    # -----------------------------
    stab_rank = stab.copy()
    stab_rank = stab_rank.sort_values(["algo", "layers", "avg_mean_delta_pct"], ascending=[True, True, False])

    top3 = (
        stab_rank.groupby(["algo", "layers"], as_index=False)
        .head(3)
        .copy()
    )

    C = top3.rename(columns={
        "algo": "Algo",
        "layers": "Layers",
        "placement": "Placement",
        "activation": "Act",
        "avg_mean_delta_pct": "Avg Δ%",
        "avg_steps_to_80pct": "Avg Steps80",
        "avg_cv_final_return": "Avg CV",
    })

    C["Avg Δ%"] = C["Avg Δ%"].map(fmt_delta)
    C["Avg Steps80"] = C["Avg Steps80"].map(fmt_steps)
    C["Avg CV"] = C["Avg CV"].map(fmt_cv)

    C = C[["Algo", "Layers", "Placement", "Act", "Avg Δ%", "Avg Steps80", "Avg CV"]]
    C.to_csv(OUT_C_CSV, index=False)

    texC = df_to_booktabs(
        C,
        caption="Top 3 activations by average Δ% across tasks, with learning speed (Steps80) and stability (CV).",
        label="tab:learning-dynamics-top3",
        colspec="lll" + "l" + "l" + "l" + "l",
    )
    with open(OUT_C_TEX, "w", encoding="utf-8") as f:
        f.write(texC)

    print("\n" + "=" * 90)
    print("TABLE C: Top 3 activations by avg Δ% (with Steps80 and CV)")
    print("=" * 90)
    print(C.to_string(index=False))

    print("\nWrote:")
    print(" ", OUT_A_CSV)
    print(" ", OUT_B_CSV)
    print(" ", OUT_C_CSV)
    print(" ", OUT_A_TEX)
    print(" ", OUT_B_TEX)
    print(" ", OUT_C_TEX)
    print("\nOverleaf: add \\usepackage{booktabs} (and \\usepackage{multirow} only if you use multirow elsewhere).")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e), file=sys.stderr)
        raise
