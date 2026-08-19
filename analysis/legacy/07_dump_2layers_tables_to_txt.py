from __future__ import annotations

import os
import argparse
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd


ROOT_DEFAULT = os.path.expanduser("~/Desktop/new-Fr")
IN_DEFAULT = os.path.join(ROOT_DEFAULT, "outputs/analysis/auc_by_seed_normalized.csv")
OUT_DEFAULT = os.path.join(ROOT_DEFAULT, "outputs/paper/appendix_2layers_numbers.txt")

FRACTIONAL_ACTS = ["frelu", "flrelu", "fprelu", "fswish", "fgelu"]
ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5]


def iqm(values: np.ndarray, trim: float = 0.25) -> float:
    """Interquartile mean (IQM). For 5 seeds, this equals the mean of the middle 3."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    x.sort()
    k = int(np.floor(trim * x.size))
    if 2 * k >= x.size:
        return float(np.mean(x))
    return float(np.mean(x[k : x.size - k]))


def canon(s: str) -> str:
    return str(s).strip().lower()


def fmt(x: float, decimals: int = 1) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.{decimals}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default=IN_DEFAULT)
    ap.add_argument("--out_txt", default=OUT_DEFAULT)
    ap.add_argument("--decimals", type=int, default=1)
    ap.add_argument("--exclude_none", action="store_true", help="Exclude placement == none")
    args = ap.parse_args()

    if not os.path.exists(args.in_csv):
        raise FileNotFoundError(args.in_csv)

    df = pd.read_csv(args.in_csv)

    needed = {"algo", "layers", "placement", "activation", "alpha", "task", "seed"}
    missing = sorted(list(needed - set(df.columns)))
    if missing:
        raise ValueError(f"Missing columns in {args.in_csv}: {missing}")

    df["activation"] = df["activation"].map(canon)
    df["placement"] = df["placement"].astype(str)
    df["layers"] = df["layers"].astype(str)
    df["algo"] = df["algo"].astype(str)
    df["task"] = df["task"].astype(str)
    df["seed"] = df["seed"].astype(str)
    df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")

    # Choose metric
    metric_col = "auc_used" if "auc_used" in df.columns else "auc"
    if metric_col not in df.columns:
        raise ValueError("Neither auc_used nor auc found.")
    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")

    # Filter to 2layers
    df2 = df[df["layers"] == "2layers"].copy()
    if df2.empty:
        raise ValueError("No rows with layers == 2layers.")

    placements = sorted(df2["placement"].dropna().unique().tolist())
    if args.exclude_none:
        placements = [p for p in placements if canon(p) != "none"]

    algos = sorted(df2["algo"].dropna().unique().tolist())
    tasks = sorted(df2["task"].dropna().unique().tolist())

    lines: List[str] = []
    lines.append("# Appendix numbers dump")
    lines.append(f"# Source: {args.in_csv}")
    lines.append("# Scope: 2layers, all placements, both algos, all tasks")
    lines.append("# Values: IQM over seeds")
    lines.append(f"# Metric column: {metric_col}")
    lines.append(f"# Alphas: {ALPHAS}")
    lines.append(f"# Fractional activations: {FRACTIONAL_ACTS}")
    lines.append("")

    for algo in algos:
        lines.append("============================================================")
        lines.append(f"ALGO = {algo}")
        lines.append("============================================================")
        lines.append("")
        for placement in placements:
            sub = df2[(df2["algo"] == algo) & (df2["placement"] == placement)].copy()
            if sub.empty:
                continue

            lines.append("------------------------------------------------------------")
            lines.append(f"PLACEMENT = {placement}")
            lines.append("FORMAT:")
            lines.append("Task | ReLU | FReLU(a=0.1..0.5) | FLReLU(a=0.1..0.5) | FPReLU(a=0.1..0.5)")
            lines.append("Numbers are IQM(AUC) over seeds.")
            lines.append("------------------------------------------------------------")

            # Determine tasks present for this subset
            subset_tasks = sorted(sub["task"].unique().tolist())

            # Precompute per task
            for task in subset_tasks:
                row = sub[sub["task"] == task]

                # ReLU
                relu_vals = row[row["activation"] == "relu"][metric_col].to_numpy(dtype=float)
                relu_iqm = iqm(relu_vals)

                parts: List[str] = []
                parts.append(task)
                parts.append(fmt(relu_iqm, args.decimals))

                # fractional act blocks
                for act in FRACTIONAL_ACTS:
                    for a in ALPHAS:
                        vals = row[(row["activation"] == act) & (np.isclose(row["alpha"], a))][metric_col].to_numpy(dtype=float)
                        parts.append(fmt(iqm(vals), args.decimals))

                # Build a fixed-width line to be easy to paste back
                # Columns: Task | ReLU | 5 FReLU | 5 FLReLU | 5 FPReLU
                line = " | ".join(parts)
                lines.append(line)

            lines.append("")  # blank line after placement block

        lines.append("")  # blank line after algo block

    os.makedirs(os.path.dirname(args.out_txt), exist_ok=True)
    with open(args.out_txt, "w") as f:
        f.write("\n".join(lines))

    print(f"[OK] Wrote: {args.out_txt}")
    print(f"[INFO] algos: {algos}")
    print(f"[INFO] placements: {placements}")
    print(f"[INFO] total tasks (global): {tasks}")
    print(f"[INFO] metric_col: {metric_col}")


if __name__ == "__main__":
    main()
