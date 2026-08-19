#!/usr/bin/env python3
from __future__ import annotations

import os
import pandas as pd

ROOT = os.path.expanduser("~/Desktop/new-Fr")

ALPHA_DIR = os.path.join(ROOT, "outputs", "paper", "alpha_sensitivity")
TEX_DIR = os.path.join(ALPHA_DIR, "tex_tables")
os.makedirs(TEX_DIR, exist_ok=True)

IN_CSV = os.path.join(ALPHA_DIR, "table_best_alpha_frequency.csv")
OUT_PATH = os.path.join(TEX_DIR, "table_best_alpha_frequency_main.tex")

ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5]

ACT_ORDER = ["frelu", "flrelu", "fprelu", "fgelu", "fswish"]

ACT_LABEL = {
    "frelu": "FReLU",
    "flrelu": "FLReLU",
    "fprelu": "FPReLU",
    "fgelu": "FGELU",
    "fswish": "FSwish",
}

CASES = [
    ("SAC", "1layer", "none", "SAC 1-layer"),
    ("SAC", "2layers", "best-placement", "SAC 2-layer"),
    ("TD3", "1layer", "none", "TD3 1-layer"),
    ("TD3", "2layers", "best-placement", "TD3 2-layer"),
]


def fmt_cell(win_count, win_pct) -> str:
    if pd.isna(win_count) or pd.isna(win_pct):
        return "-"
    return f"{int(win_count)} ({float(win_pct):.0f}\\%)"


def get_cell(df: pd.DataFrame, algo: str, layers: str, placement: str, act: str, alpha: float) -> str:
    sub = df[
        (df["algo"] == algo)
        & (df["layers"] == layers)
        & (df["placement"] == placement)
        & (df["activation"] == act)
        & (df["alpha"] == float(alpha))
    ]

    if sub.empty:
        return "-"

    r = sub.iloc[0]
    return fmt_cell(r["win_count"], r["win_pct"])


def main() -> None:
    if not os.path.exists(IN_CSV):
        raise FileNotFoundError(f"Missing input: {IN_CSV}")

    df = pd.read_csv(IN_CSV)

    df["algo"] = df["algo"].astype(str).str.upper().str.strip()
    df["layers"] = df["layers"].astype(str).str.lower().str.strip()
    df["placement"] = df["placement"].astype(str).str.lower().str.strip()
    df["activation"] = df["activation"].astype(str).str.lower().str.strip()
    df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")

    cols_per_case = len(ALPHAS)
    total_cols = 1 + cols_per_case * len(CASES)
    colspec = "l" + "c" * (total_cols - 1)

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Best $\alpha$ frequency across tasks for fractional activations. Each cell reports the number of tasks for which a given $\alpha$ is selected as best, with the corresponding percentage in parentheses. For the two-layer setting, the best placement is selected for each task, activation, and $\alpha$.}")
    lines.append(r"\label{tab:best-alpha-frequency-main}")
    lines.append(r"\small")
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\toprule")

    header1 = [r"\multirow{2}{*}{Activation}"]
    for _, _, _, label in CASES:
        header1.append(rf"\multicolumn{{{cols_per_case}}}{{c}}{{{label}}}")
    lines.append(" & ".join(header1) + r" \\")

    lines.append(r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}\cmidrule(lr){12-16}\cmidrule(lr){17-21}")

    header2 = [""]
    for _ in CASES:
        header2.extend([rf"$\alpha={a}$" for a in ALPHAS])
    lines.append(" & ".join(header2) + r" \\")
    lines.append(r"\midrule")

    for act in ACT_ORDER:
        row = [ACT_LABEL.get(act, act)]
        for algo, layers, placement, _ in CASES:
            for a in ALPHAS:
                row.append(get_cell(df, algo, layers, placement, act, a))
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("Wrote:", OUT_PATH)


if __name__ == "__main__":
    main()
