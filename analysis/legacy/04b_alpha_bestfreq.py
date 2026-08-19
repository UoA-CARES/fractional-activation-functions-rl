#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import pandas as pd

ROOT = os.path.expanduser("~/Desktop/new-Fr")
ALPHA_DIR = os.path.join(ROOT, "outputs", "paper", "alpha_sensitivity")
os.makedirs(ALPHA_DIR, exist_ok=True)

IN_ALPHA = os.path.join(ALPHA_DIR, "table_alpha_sensitivity.csv")
OUT_DIR = ALPHA_DIR

OUT_BEST_BY_TASK = os.path.join(OUT_DIR, "table_best_alpha_by_task.csv")
OUT_FREQ = os.path.join(OUT_DIR, "table_best_alpha_frequency.csv")
OUT_GAP = os.path.join(OUT_DIR, "table_alpha_gap_summary.csv")

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

    # For 1layer, placement is fixed.
    df.loc[df["layers"] == "1layer", "placement"] = "none"

    # For 2layers, use the best placement per task for each activation and alpha.
    df_2 = df[
        (df["layers"] == "2layers")
        & (df["placement"].isin(PLACEMENTS_2L))
    ].copy()

    best_pl = (
        df_2.groupby(
            ["algo", "layers", "task", "activation", "alpha"],
            as_index=False,
        )["mean_delta_pct"]
        .max()
        .assign(placement="best-placement")
    )

    # Combine 1layer rows and 2layers best-placement rows.
    df_1 = df[df["layers"] == "1layer"].copy()
    df_1 = df_1.assign(placement="none")

    keep_cols = ["algo", "layers", "placement", "task", "activation", "alpha", "mean_delta_pct"]
    combo = pd.concat([df_1[keep_cols], best_pl[keep_cols]], ignore_index=True)

    # ------------------------------------------------------------
    # 1) Best alpha per task
    # ------------------------------------------------------------
    idx = combo.groupby(
        ["algo", "layers", "placement", "task", "activation"]
    )["mean_delta_pct"].idxmax()

    best_alpha_by_task = combo.loc[
        idx,
        ["algo", "layers", "placement", "task", "activation", "alpha", "mean_delta_pct"],
    ].copy()

    best_alpha_by_task = best_alpha_by_task.rename(
        columns={"mean_delta_pct": "best_mean_delta_pct"}
    )

    best_alpha_by_task["best_mean_delta_pct"] = (
        best_alpha_by_task["best_mean_delta_pct"].round(1)
    )

    best_alpha_by_task = best_alpha_by_task.sort_values(
        ["algo", "layers", "placement", "activation", "task"]
    )

    best_alpha_by_task.to_csv(OUT_BEST_BY_TASK, index=False)
    print("Wrote:", OUT_BEST_BY_TASK, "rows:", len(best_alpha_by_task))

    # ------------------------------------------------------------
    # 2) Best alpha frequency
    # ------------------------------------------------------------
    freq = (
        best_alpha_by_task.groupby(
            ["algo", "layers", "placement", "activation", "alpha"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "win_count"})
    )

    totals = (
        best_alpha_by_task.groupby(
            ["algo", "layers", "placement", "activation"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "n_tasks"})
    )

    freq = freq.merge(
        totals,
        on=["algo", "layers", "placement", "activation"],
        how="left",
    )

    freq["win_pct"] = (100.0 * freq["win_count"] / freq["n_tasks"]).round(1)

    freq = freq.sort_values(
        ["algo", "layers", "placement", "activation", "win_count", "alpha"],
        ascending=[True, True, True, True, False, True],
    )

    freq.to_csv(OUT_FREQ, index=False)
    print("Wrote:", OUT_FREQ, "rows:", len(freq))

    # ------------------------------------------------------------
    # 3) Alpha gap summary
    # gap per task = best alpha result - worst alpha result
    # ------------------------------------------------------------
    per_task = (
        combo.groupby(
            ["algo", "layers", "placement", "task", "activation"],
            as_index=False,
        )
        .agg(
            best=("mean_delta_pct", "max"),
            worst=("mean_delta_pct", "min"),
        )
    )

    per_task["gap"] = per_task["best"] - per_task["worst"]

    gap_summary = (
        per_task.groupby(
            ["algo", "layers", "placement", "activation"],
            as_index=False,
        )
        .agg(
            avg_gap=("gap", "mean"),
            median_gap=("gap", "median"),
        )
    )

    gap_summary["avg_gap"] = gap_summary["avg_gap"].round(1)
    gap_summary["median_gap"] = gap_summary["median_gap"].round(1)

    gap_summary = gap_summary.sort_values(
        ["algo", "layers", "placement", "avg_gap"],
        ascending=[True, True, True, False],
    )

    gap_summary.to_csv(OUT_GAP, index=False)
    print("Wrote:", OUT_GAP, "rows:", len(gap_summary))

    print("[T4b] best-alpha rows expected:", "2 algos * (1layer+2layers-best) * 8 tasks * 5 acts = 160")
    print("[T4b] best-alpha rows actual:", len(best_alpha_by_task))
    print("[T4c] unique alphas:", sorted(combo["alpha"].unique().tolist()))
    print("[T4d] activations used:", sorted(combo["activation"].unique().tolist()))
    print("[T4e] layers:", sorted(combo["layers"].unique().tolist()))
    print("[T4f] placements:", sorted(combo["placement"].unique().tolist()))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e), file=sys.stderr)
        raise
