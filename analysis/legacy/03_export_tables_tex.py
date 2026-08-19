#!/usr/bin/env python3
from __future__ import annotations

import os
import pandas as pd


ROOT = os.path.expanduser("~/Desktop/new-Fr")
IN_DIR = os.path.join(ROOT, "outputs", "paper")
OUT_DIR = os.path.join(ROOT, "outputs", "paper", "tex_tables")
os.makedirs(OUT_DIR, exist_ok=True)

FILES = [
    ("table_best_by_task.csv", "table_best_by_task.tex", "Best configuration per task (mean Δ% vs ReLU).", "tab:best-by-task"),
    ("table_activation_family_summary.csv", "table_activation_family_summary.tex", "Activation-family summary across tasks.", "tab:act-family-summary"),
    ("table_1layer_vs_2layers.csv", "table_1layer_vs_2layers.tex", "Best Δ% comparison: 1layer vs 2layers.", "tab:1layer-vs-2layers"),
]


def latex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
         .replace("_", "\\_")
         .replace("%", "\\%")
         .replace("&", "\\&")
         .replace("#", "\\#")
    )


def df_to_booktabs_tex(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = df.columns.tolist()

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{" + latex_escape(caption) + "}")
    lines.append("\\label{" + latex_escape(label) + "}")
    lines.append("\\begin{tabular}{" + "l" * len(cols) + "}")
    lines.append("\\toprule")
    lines.append(" & ".join([latex_escape(c) for c in cols]) + " \\\\")
    lines.append("\\midrule")

    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if pd.isna(v):
                vals.append("")
            elif isinstance(v, float):
                # keep 1 decimal if it looks like a delta column, else raw
                if "delta" in c.lower() or "diff" in c.lower():
                    vals.append(f"{v:+.1f}")
                else:
                    vals.append(f"{v:.3f}")
            else:
                vals.append(latex_escape(str(v)))
        lines.append(" & ".join(vals) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def main() -> None:
    for in_name, out_name, caption, label in FILES:
        in_path = os.path.join(IN_DIR, in_name)
        out_path = os.path.join(OUT_DIR, out_name)

        df = pd.read_csv(in_path)

        # nicer column names for paper
        rename_map = {
            "mean_delta_pct": "Δ% (mean)",
            "avg_delta_pct": "Avg Δ%",
            "win_count": "Wins",
            "best_1layer_delta_pct": "Best 1layer Δ%",
            "best_2layers_delta_pct": "Best 2layers Δ%",
            "diff_2minus1": "Δ(2-1)",
            "placement": "Placement",
            "activation": "Activation",
            "act_family": "Activation",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        tex = df_to_booktabs_tex(df, caption=caption, label=label)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(tex)

        print("Wrote:", out_path)

    print("Done. Add this to Overleaf preamble: \\usepackage{booktabs}")


if __name__ == "__main__":
    main()
