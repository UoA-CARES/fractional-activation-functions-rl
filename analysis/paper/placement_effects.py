#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import pandas as pd

ROOT = os.environ.get("FRORL_RESULTS_ROOT", "results")
IN_NORM = os.path.join(ROOT, "outputs", "analysis", "auc_by_seed_normalized.csv")

OUT_PAPER = os.path.join(ROOT, "outputs", "paper")
OUT_PLACEMENT = os.path.join(OUT_PAPER, "placement_ablation")
os.makedirs(OUT_PLACEMENT, exist_ok=True)

OUT_SUMMARY = os.path.join(OUT_PLACEMENT, "table_placement_summary.csv")
OUT_BEST_BY_TASK = os.path.join(OUT_PLACEMENT, "table_best_placement_by_task.csv")

PLACEMENTS_2L = ["all-actor", "all-critic", "all-both", "first-actor", "first-both"]

ACT_MAP = {
    "r": "relu",
    "relu": "relu",

    "l": "lrelu",
    "lrelu": "lrelu",
    "leakyrelu": "lrelu",

    "p": "prelu",
    "prelu": "prelu",

    "g": "gelu",
    "gelu": "gelu",

    "s": "swish",
    "swish": "swish",
    "silu": "swish",

    "fr": "frelu",
    "frelu": "frelu",

    "fl": "flrelu",
    "flrelu": "flrelu",

    "fp": "fprelu",
    "fprelu": "fprelu",

    "fg": "fgelu",
    "fgelu": "fgelu",
    "fractionalgelu": "fgelu",
    "fractional_gelu": "fgelu",
    "fractionalgelubeta": "fgelu",
    "fractional_gelu_beta": "fgelu",

    "fs": "fswish",
    "fswish": "fswish",
    "fractionalswish": "fswish",
    "fractional_swish": "fswish",
    "fractionalswishbeta": "fswish",
    "fractional_swish_beta": "fswish",
}


def require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing columns: {missing}\nFound: {list(df.columns)}")


def norm_activation(x: object) -> str:
    s = str(x).strip().lower()
    return ACT_MAP.get(s, s)


def norm_alpha(x: object) -> str:
    if pd.isna(x):
        return "NA"

    s = str(x).strip()

    if s == "" or s.lower() in {"nan", "none", "na"}:
        return "NA"

    try:
        return f"{float(s):.1f}"
    except Exception:
        return s


def main() -> None:
    if not os.path.exists(IN_NORM):
        raise FileNotFoundError(f"Missing input: {IN_NORM}")

    df = pd.read_csv(IN_NORM)

    require_cols(
        df,
        ["algo", "task", "layers", "placement", "activation", "seed", "delta_pct_vs_relu"],
        "auc_by_seed_normalized.csv",
    )

    df["algo"] = df["algo"].astype(str).str.upper().str.strip()
    df["task"] = df["task"].astype(str).str.strip()
    df["layers"] = df["layers"].astype(str).str.lower().str.strip()
    df["placement"] = df["placement"].astype(str).str.lower().str.strip()
    df["activation"] = df["activation"].apply(norm_activation)
    df["delta_pct_vs_relu"] = pd.to_numeric(df["delta_pct_vs_relu"], errors="coerce")

    df = df.dropna(subset=["delta_pct_vs_relu"]).copy()

    if "alpha_key" not in df.columns:
        if "alpha" in df.columns:
            df["alpha_key"] = df["alpha"].apply(norm_alpha)
        else:
            df["alpha_key"] = "NA"
    else:
        df["alpha_key"] = df["alpha_key"].apply(norm_alpha)

    # Focus: 2layers placements only, excluding baseline placement r.
    df2 = df[
        (df["layers"] == "2layers")
        & (df["placement"].isin(PLACEMENTS_2L))
    ].copy()

    if df2.empty:
        raise ValueError("No rows found for 2layers placements. Check placement names in the normalized CSV.")

    # 1) Mean across seeds for each full config, including alpha.
    cfg_cols = ["algo", "task", "placement", "activation", "alpha_key"]

    mean_cfg = (
        df2.groupby(cfg_cols, as_index=False)["delta_pct_vs_relu"]
        .mean()
        .rename(columns={"delta_pct_vs_relu": "mean_delta_pct"})
    )

    # 2) Fix alpha by choosing best alpha per algo, task, placement, activation.
    best_alpha = (
        mean_cfg.groupby(
            ["algo", "task", "placement", "activation"],
            as_index=False,
        )["mean_delta_pct"]
        .max()
        .rename(columns={"mean_delta_pct": "best_mean_delta_pct_over_alpha"})
    )

    # 3A) Placement summary across tasks and activations.
    placement_summary = (
        best_alpha.groupby(["algo", "placement"], as_index=False)
        .agg(
            avg_delta_pct=("best_mean_delta_pct_over_alpha", "mean"),
            median_delta_pct=("best_mean_delta_pct_over_alpha", "median"),
            win_count=("best_mean_delta_pct_over_alpha", lambda s: int((s > 0).sum())),
            n=("best_mean_delta_pct_over_alpha", "size"),
        )
    )

    placement_summary["avg_delta_pct"] = placement_summary["avg_delta_pct"].round(1)
    placement_summary["median_delta_pct"] = placement_summary["median_delta_pct"].round(1)

    placement_summary = placement_summary.sort_values(
        ["algo", "avg_delta_pct"],
        ascending=[True, False],
    )

    placement_summary.to_csv(OUT_SUMMARY, index=False)
    print("Wrote:", OUT_SUMMARY, "rows:", len(placement_summary))

    # 3B) Best placement by task, overall best over activation and alpha.
    best_overall_cfg = (
        best_alpha.groupby(
            ["algo", "task", "placement"],
            as_index=False,
        )["best_mean_delta_pct_over_alpha"]
        .max()
        .rename(columns={"best_mean_delta_pct_over_alpha": "best_delta_pct_this_placement"})
    )

    idx = best_overall_cfg.groupby(["algo", "task"])["best_delta_pct_this_placement"].idxmax()

    best_place_by_task = best_overall_cfg.loc[idx].copy()
    best_place_by_task["best_delta_pct_this_placement"] = (
        best_place_by_task["best_delta_pct_this_placement"].round(1)
    )

    best_place_by_task = best_place_by_task.sort_values(["algo", "task"])

    best_place_by_task.to_csv(OUT_BEST_BY_TASK, index=False)
    print("Wrote:", OUT_BEST_BY_TASK, "rows:", len(best_place_by_task))

    print("[T8] placements in summary:", sorted(placement_summary["placement"].unique().tolist()))
    print("[T9] algos in summary:", sorted(placement_summary["algo"].unique().tolist()))
    print("[T10] best placement by task rows:", len(best_place_by_task))
    print("[T11] activations used:", sorted(df2["activation"].unique().tolist()))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e), file=sys.stderr)
        raise
