#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import pandas as pd

ROOT = os.path.expanduser("~/Desktop/new-Fr")
IN_NORM = os.path.join(ROOT, "outputs", "analysis", "auc_by_seed_normalized.csv")

OUT_PAPER = os.path.join(ROOT, "outputs", "paper")
OUT_ALPHA_DIR = os.path.join(OUT_PAPER, "alpha_sensitivity")
os.makedirs(OUT_ALPHA_DIR, exist_ok=True)

OUT_ALPHA = os.path.join(OUT_ALPHA_DIR, "table_alpha_sensitivity.csv")
OUT_ROBUST = os.path.join(OUT_ALPHA_DIR, "table_alpha_robustness_index.csv")
FRACTIONAL = ["frelu", "flrelu", "fprelu", "fgelu", "fswish"]


def require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing columns: {missing}\nFound: {list(df.columns)}")


def main() -> None:
    if not os.path.exists(IN_NORM):
        raise FileNotFoundError(f"Missing input: {IN_NORM}")

    df = pd.read_csv(IN_NORM)

    require_cols(
        df,
        ["algo", "task", "layers", "placement", "activation", "alpha", "seed", "delta_pct_vs_relu"],
        "auc_by_seed_normalized.csv",
    )

    df["algo"] = df["algo"].astype(str).str.upper()
    df["activation"] = df["activation"].astype(str).str.lower()
    df["layers"] = df["layers"].astype(str).str.lower()
    df["placement"] = df["placement"].astype(str).str.lower()

    # fractional only
    df_f = df[df["activation"].isin(FRACTIONAL)].copy()

    # 1) mean across seeds per config
    cfg_cols = ["algo", "task", "layers", "placement", "activation", "alpha"]
    mean_alpha = (
        df_f.groupby(cfg_cols, as_index=False)["delta_pct_vs_relu"]
        .mean()
        .rename(columns={"delta_pct_vs_relu": "mean_delta_pct"})
    )

    mean_alpha.to_csv(OUT_ALPHA, index=False)
    print("Wrote:", OUT_ALPHA, "rows:", len(mean_alpha))

    # 2) robustness index per (algo, layers, placement, activation)
    robust = (
        mean_alpha.groupby(["algo", "layers", "placement", "activation"], as_index=False)
        .agg(
            mean_over_alpha=("mean_delta_pct", "mean"),
            std_over_alpha=("mean_delta_pct", "std"),
            n_alpha=("mean_delta_pct", "size"),
        )
    )

    robust["mean_over_alpha"] = robust["mean_over_alpha"].round(1)
    robust["std_over_alpha"] = robust["std_over_alpha"].round(2)

    robust.to_csv(OUT_ROBUST, index=False)
    print("Wrote:", OUT_ROBUST, "rows:", len(robust))

    # Tests
    print("[T11] unique alphas per activation:")
    print(mean_alpha.groupby("activation")["alpha"].nunique())

    print("[T12] any missing alpha:", mean_alpha["alpha"].isna().sum())
    print("[T13] algos present:", sorted(robust["algo"].unique().tolist()))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e), file=sys.stderr)
        raise
