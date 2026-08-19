#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import pandas as pd

ROOT = os.path.expanduser("~/Desktop/new-Fr")

ALPHA_DIR = os.path.join(ROOT, "outputs", "paper", "alpha_sensitivity")
os.makedirs(ALPHA_DIR, exist_ok=True)

IN_ALPHA = os.path.join(ALPHA_DIR, "table_alpha_sensitivity.csv")
OUT_BVM = os.path.join(ALPHA_DIR, "table_alpha_best_vs_mean.csv")

FRACTIONAL = ["frelu", "flrelu", "fprelu", "fgelu", "fswish"]
PLACEMENTS_2L = ["all-actor", "all-critic", "all-both", "first-actor", "first-both"]


def require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing columns: {missing}\nFound: {list(df.columns)}")


def main() -> None:
    if not os.path.exists(IN_ALPHA):
        raise FileNotFoundError(f"Missing input: {IN_ALPHA}")

    df = pd.read_csv(IN_ALPHA)

    require_cols(
        df,
        ["algo", "task", "layers", "placement", "activation", "alpha", "mean_delta_pct"],
        "table_alpha_sensitivity.csv",
    )

    df["algo"] = df["algo"].astype(str).str.upper().str.strip()
    df["task"] = df["task"].astype(str).str.strip()
    df["layers"] = df["layers"].astype(str).str.lower().str.strip()
    df["placement"] = df["placement"].astype(str).str.lower().str.strip()
    df["activation"] = df["activation"].astype(str).str.lower().str.strip()
    df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")
    df["mean_delta_pct"] = pd.to_numeric(df["mean_delta_pct"], errors="coerce")

    df = df[df["activation"].isin(FRACTIONAL)].copy()
    df = df.dropna(subset=["alpha", "mean_delta_pct"]).copy()

    # 1-layer has fixed placement.
    df.loc[df["layers"] == "1layer", "placement"] = "none"

    df1 = df[df["layers"] == "1layer"].copy()
    df1 = df1.assign(placement="none")

    # 2-layer uses the best placement per task, activation, and alpha.
    df2 = df[
        (df["layers"] == "2layers")
        & (df["placement"].isin(PLACEMENTS_2L))
    ].copy()

    best_pl = (
        df2.groupby(
            ["algo", "task", "activation", "alpha"],
            as_index=False,
        )["mean_delta_pct"]
        .max()
        .assign(layers="2layers", placement="best-placement")
    )

    keep_cols = ["algo", "layers", "placement", "task", "activation", "alpha", "mean_delta_pct"]

    combo = pd.concat(
        [df1[keep_cols], best_pl[keep_cols]],
        ignore_index=True,
    )

    # For each task and activation:
    # mean_over_alpha averages performance across alpha values.
    # best_over_alpha selects the strongest alpha value.
    per_task = (
        combo.groupby(
            ["algo", "layers", "placement", "task", "activation"],
            as_index=False,
        )
        .agg(
            mean_over_alpha=("mean_delta_pct", "mean"),
            best_over_alpha=("mean_delta_pct", "max"),
        )
    )

    per_task["gain_best_minus_mean"] = (
        per_task["best_over_alpha"] - per_task["mean_over_alpha"]
    )

    # Summarise across tasks.
    out = (
        per_task.groupby(
            ["algo", "layers", "placement", "activation"],
            as_index=False,
        )
        .agg(
            avg_mean_over_alpha=("mean_over_alpha", "mean"),
            avg_best_over_alpha=("best_over_alpha", "mean"),
            avg_gain=("gain_best_minus_mean", "mean"),
        )
    )

    out["avg_mean_over_alpha"] = out["avg_mean_over_alpha"].round(1)
    out["avg_best_over_alpha"] = out["avg_best_over_alpha"].round(1)
    out["avg_gain"] = out["avg_gain"].round(1)

    out["activation"] = pd.Categorical(
        out["activation"],
        categories=["frelu", "flrelu", "fprelu", "fgelu", "fswish"],
        ordered=True,
    )

    out = out.sort_values(["algo", "layers", "placement", "activation"])

    out.to_csv(OUT_BVM, index=False)

    print("Wrote:", OUT_BVM, "rows:", len(out))

    print("[T4f] expected rows:", "2 algos * 2 layer views * 5 fractional activations = 20")
    print("[T4f] actual rows:", len(out))
    print("[T4f] placements:", sorted(out["placement"].unique().tolist()))
    print("[T4f] layers:", sorted(out["layers"].unique().tolist()))
    print("[T4f] activations:", sorted(out["activation"].unique().tolist()))
    print("[T4f] any missing:", int(out.isna().sum().sum()))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e), file=sys.stderr)
        raise
