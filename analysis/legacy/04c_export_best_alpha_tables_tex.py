#!/usr/bin/env python3
from __future__ import annotations

import os
import pandas as pd

ROOT = os.path.expanduser("~/Desktop/new-Fr")
IN_CSV = os.path.join(ROOT, "outputs", "paper", "table_best_alpha_frequency.csv")
OUT_DIR = os.path.join(ROOT, "outputs", "paper", "tex_tables")
os.makedirs(OUT_DIR, exist_ok=True)

ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5]
ACT_ORDER = ["frelu", "flrelu", "fprelu", "fswish", "fgelu"]
CASES = [("1layer", "none"), ("2layers", "best-placement")]


def fmt_cell(win_count: int, win_pct: float) -> str:
    return f"{int(win_count)} ({win_pct:.0f}\\%)"


def pivot_case(df: pd.DataFrame, algo: str, layers: str, placement: str) -> pd.DataFrame:
    sub = df[(df["algo"] == algo) & (df["layers"] == layers) & (df["placement"] == placement)].copy()
    if sub.empty:
        return pd.DataFrame()

    sub["alpha"] = sub["alpha"].astype(float)

    pivot = sub.pivot_table(
        index="activation",
        columns="alpha",
        values=["win_count", "win_pct"],
        aggfunc="first",
    )

    # build formatted table
    out = pd.DataFrame(index=ACT_ORDER, columns=ALPHAS)
    for act in ACT_ORDER:
        for a in ALPHAS:
            try:
                wc = pivot.loc[act, ("win_count", a)]
                wp = pivot.loc[act, ("win_pct", a)]
                if pd.isna(wc) or pd.isna(wp):
                    out.loc[act, a] = "-"
                else:
                    out.loc[act, a] = fmt_cell(int(wc), float(wp))
            except Exception:
                out.loc[act, a] = "-"

    out.index.name = "Activation"
    out.columns = [f"$\\alpha={a}$" for a in ALPHAS]
    return out


def to_booktabs(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = ["Activation"] + list(df.columns)
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{" + caption + "}")
    lines.append("\\label{" + label + "}")
    lines.append("\\begin{tabular}{" + "l" + "c" * (len(cols) - 1) + "}")
    lines.append("\\toprule")
    lines.append(" & ".join(cols) + " \\\\")
    lines.append("\\midrule")

    for act, row in df.iterrows():
        vals = [act] + [str(row[c]) for c in df.columns]
        lines.append(" & ".join(vals) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def main() -> None:
    df = pd.read_csv(IN_CSV)

    df["algo"] = df["algo"].astype(str).str.upper()
    df["layers"] = df["layers"].astype(str).str.lower()
    df["placement"] = df["placement"].astype(str).str.lower()
    df["activation"] = df["activation"].astype(str).str.lower()

    for algo in sorted(df["algo"].unique()):
        for layers, placement in CASES:
            tab = pivot_case(df, algo, layers, placement)
            if tab.empty:
                continue

            fname = f"table_best_alpha_frequency_{algo.lower()}_{layers}.tex"
            out_path = os.path.join(OUT_DIR, fname)

            caption = f"Best $\\alpha$ frequency across tasks for {algo} ({layers}). Cells show wins (percentage)."
            label = f"tab:best-alpha-{algo.lower()}-{layers}"

            tex = to_booktabs(tab, caption=caption, label=label)

            with open(out_path, "w", encoding="utf-8") as f:
                f.write(tex)

            print("Wrote:", out_path)

    print("Done. Requires: \\usepackage{booktabs}")


if __name__ == "__main__":
    main()
