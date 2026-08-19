#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import pandas as pd

ROOT = os.environ.get("FRORL_RESULTS_ROOT", "results")

ALPHA_DIR = os.path.join(ROOT, "outputs", "paper", "alpha_sensitivity")
os.makedirs(ALPHA_DIR, exist_ok=True)

IN_BEST = os.path.join(ALPHA_DIR, "table_best_alpha_by_task.csv")
OUT_DOM = os.path.join(ALPHA_DIR, "table_alpha_dominance_summary.csv")

FRACTIONAL = ["frelu", "flrelu", "fprelu", "fgelu", "fswish"]
LOW_ALPHAS = {0.1, 0.2}


def require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing columns: {missing}\nFound: {list(df.columns)}")


def main() -> None:
    if not os.path.exists(IN_BEST):
        raise FileNotFoundError(
            f"Missing {IN_BEST}. Run scripts/analysis/04b_alpha_bestfreq.py first."
        )

    df = pd.read_csv(IN_BEST)

    require_cols(
        df,
        ["algo", "layers", "placement", "task", "activation", "alpha", "best_mean_delta_pct"],
        "table_best_alpha_by_task.csv",
    )

    df["algo"] = df["algo"].astype(str).str.upper().str.strip()
    df["layers"] = df["layers"].astype(str).str.lower().str.strip()
    df["placement"] = df["placement"].astype(str).str.lower().str.strip()
    df["activation"] = df["activation"].astype(str).str.lower().str.strip()
    df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")
    df["best_mean_delta_pct"] = pd.to_numeric(df["best_mean_delta_pct"], errors="coerce")

    df = df[df["activation"].isin(FRACTIONAL)].copy()
    df = df.dropna(subset=["alpha", "best_mean_delta_pct"]).copy()

    df["is_low_alpha"] = df["alpha"].isin(LOW_ALPHAS)

    dom = (
        df.groupby(["algo", "layers", "placement", "activation"], as_index=False)
        .agg(
            n_tasks=("task", "nunique"),
            low_alpha_wins=("is_low_alpha", "sum"),
            mean_best_delta=("best_mean_delta_pct", "mean"),
        )
    )

    dom["low_alpha_pct"] = (100.0 * dom["low_alpha_wins"] / dom["n_tasks"]).round(1)
    dom["mean_best_delta"] = dom["mean_best_delta"].round(1)

    freq = (
        df.groupby(["algo", "layers", "placement", "activation", "alpha"], as_index=False)
        .size()
        .rename(columns={"size": "win_count"})
    )

    freq = freq.sort_values(
        ["algo", "layers", "placement", "activation", "win_count", "alpha"],
        ascending=[True, True, True, True, False, True],
    )

    modal = (
        freq.groupby(["algo", "layers", "placement", "activation"], as_index=False)
        .first()[["algo", "layers", "placement", "activation", "alpha", "win_count"]]
        .rename(columns={"alpha": "modal_alpha", "win_count": "modal_alpha_wins"})
    )

    out = dom.merge(
        modal,
        on=["algo", "layers", "placement", "activation"],
        how="left",
    )

    out["activation"] = pd.Categorical(
        out["activation"],
        categories=FRACTIONAL,
        ordered=True,
    )

    out = out.sort_values(["algo", "layers", "placement", "activation"])

    out.to_csv(OUT_DOM, index=False)

    print("Wrote:", OUT_DOM, "rows:", len(out))
    print("[T4e] expected rows:", "2 algos * 2 layer views * 5 fractional activations = 20")
    print("[T4e] actual rows:", len(out))
    print("[T4e] algos:", sorted(out["algo"].unique().tolist()))
    print("[T4e] placements:", sorted(out["placement"].unique().tolist()))
    print("[T4e] activations:", [str(x) for x in out["activation"].drop_duplicates().tolist()])
    print("[T4e] any missing modal_alpha:", int(out["modal_alpha"].isna().sum()))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e), file=sys.stderr)
        raise
