#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import pandas as pd

ROOT = os.path.expanduser("~/Desktop/new-Fr")

IN_BEST_BY_TASK = os.path.join(ROOT, "outputs", "paper", "table_best_by_task.csv")
IN_1V2 = os.path.join(ROOT, "outputs", "paper", "table_1layer_vs_2layers.csv")
IN_BEST_PLACE = os.path.join(ROOT, "outputs", "paper", "table_best_placement_by_task.csv")

OUT_CSV = os.path.join(ROOT, "outputs", "paper", "table_main_results.csv")
OUT_TEX_DIR = os.path.join(ROOT, "outputs", "paper", "tex_tables")
OUT_TEX = os.path.join(OUT_TEX_DIR, "table_main_results.tex")
os.makedirs(OUT_TEX_DIR, exist_ok=True)


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


def to_booktabs(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = df.columns.tolist()
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


def main() -> None:
    require(IN_BEST_BY_TASK)
    require(IN_1V2)
    require(IN_BEST_PLACE)

    best = pd.read_csv(IN_BEST_BY_TASK)
    v12 = pd.read_csv(IN_1V2)
    bp = pd.read_csv(IN_BEST_PLACE)

    # Normalize
    for df in (best, v12, bp):
        df["algo"] = df["algo"].astype(str).str.upper()
        df["task"] = df["task"].astype(str)

    # best config overall per task is already in best
    # It includes layers, placement, activation, mean_delta_pct
    best = best.rename(columns={"mean_delta_pct": "best_overall_delta_pct"})

    # 1layer vs 2layers best deltas
    # Columns: best_1layer_delta_pct, best_2layers_delta_pct, diff_2minus1
    v12 = v12.copy()

    # best placement (overall best over activation and alpha for each placement)
    # Columns: placement, best_delta_pct_this_placement
    bp = bp.rename(columns={"placement": "best_2layers_placement"})
    bp = bp.rename(columns={"best_delta_pct_this_placement": "best_2layers_delta_pct_at_bestplacement"})

    # Merge into one
    out = v12.merge(bp[["algo", "task", "best_2layers_placement", "best_2layers_delta_pct_at_bestplacement"]],
                    on=["algo", "task"], how="left")

    out = out.merge(best[["algo", "task", "layers", "placement", "activation", "best_overall_delta_pct"]],
                    on=["algo", "task"], how="left")

    # Identify where best overall came from
    out["best_overall_source"] = out["layers"].astype(str) + "/" + out["placement"].astype(str)

    # Format for paper
    out = out.rename(columns={
        "best_1layer_delta_pct": "Best 1L Δ%",
        "best_2layers_delta_pct": "Best 2L Δ%",
        "diff_2minus1": "Δ(2L-1L)",
        "best_2layers_placement": "Best 2L placement",
        "best_overall_delta_pct": "Best overall Δ%",
        "activation": "Best activation",
        "best_overall_source": "Source",
    })

    # Apply formatting to delta columns
    for c in ["Best 1L Δ%", "Best 2L Δ%", "Δ(2L-1L)", "Best overall Δ%"]:
        out[c] = out[c].map(fmt_delta)

    # Keep compact columns
    out = out[[
        "task",
        "algo",
        "Best 1L Δ%",
        "Best 2L Δ%",
        "Δ(2L-1L)",
        "Best 2L placement",
        "Best activation",
        "Best overall Δ%",
        "Source",
    ]].sort_values(["algo", "task"])

    out.to_csv(OUT_CSV, index=False)

    tex = to_booktabs(
        out,
        caption="Main results summary (Δ% vs ReLU). Best 1layer and best 2layers values are reported per task, along with the best overall configuration.",
        label="tab:main-results",
    )
    with open(OUT_TEX, "w", encoding="utf-8") as f:
        f.write(tex)

    print("Wrote:", OUT_CSV)
    print("Wrote:", OUT_TEX)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e), file=sys.stderr)
        raise
