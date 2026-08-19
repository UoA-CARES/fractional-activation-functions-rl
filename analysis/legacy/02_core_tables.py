#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import pandas as pd


ROOT = os.path.expanduser("~/Desktop/new-Fr")
IN_NORM = os.path.join(ROOT, "outputs", "analysis", "auc_by_seed_normalized.csv")
OUT_PAPER = os.path.join(ROOT, "outputs", "paper")
os.makedirs(OUT_PAPER, exist_ok=True)

OUT_BEST_BY_TASK = os.path.join(OUT_PAPER, "table_best_by_task.csv")
OUT_FAM_SUMMARY = os.path.join(OUT_PAPER, "table_activation_family_summary.csv")
OUT_1V2 = os.path.join(OUT_PAPER, "table_1layer_vs_2layers.csv")


# Only updated to include Swish, GELU, FSwish, and FGELU.
# The rest of the script logic is unchanged.
ACT_FAMILY = {
    "relu": "relu",
    "lrelu": "lrelu",
    "prelu": "prelu",
    "swish": "swish",
    "gelu": "gelu",
    "frelu": "frelu",
    "flrelu": "flrelu",
    "fprelu": "fprelu",
    "fswish": "fswish",
    "fgelu": "fgelu",

    # allow old folder/code forms just in case
    "r": "relu",
    "l": "lrelu",
    "p": "prelu",
    "s": "swish",
    "g": "gelu",
    "fr": "frelu",
    "fl": "flrelu",
    "fp": "fprelu",
    "fs": "fswish",
    "fg": "fgelu",
}


def require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing columns: {missing}\nFound: {list(df.columns)}")


def fmt_delta(x) -> str:
    if pd.isna(x):
        return ""
    return f"{x:+.1f}"


def main() -> None:
    if not os.path.exists(IN_NORM):
        raise FileNotFoundError(f"Missing input: {IN_NORM}")

    df = pd.read_csv(IN_NORM)

    require_cols(
        df,
        [
            "algo",
            "task",
            "layers",
            "placement",
            "activation",
            "seed",
            "delta_pct_vs_relu",
        ],
        "auc_by_seed_normalized.csv",
    )

    # normalize
    df["algo"] = df["algo"].astype(str).str.upper()
    df["task"] = df["task"].astype(str)
    df["layers"] = df["layers"].astype(str).str.lower()
    df["placement"] = df["placement"].astype(str).str.lower()
    df["activation"] = df["activation"].astype(str).str.lower()
    df["act_family"] = df["activation"].map(lambda a: ACT_FAMILY.get(a, a))

    # -----------------------------------------
    # A) Best configuration per task per algo
    #    best config = max mean Delta% across seeds
    # -----------------------------------------
    cfg_cols = ["algo", "task", "layers", "placement", "activation"]

    mean_cfg = (
        df.groupby(cfg_cols, as_index=False)["delta_pct_vs_relu"]
        .mean()
        .rename(columns={"delta_pct_vs_relu": "mean_delta_pct"})
    )

    # pick best config per (algo, task) across all layers/placements/activations
    idx = mean_cfg.groupby(["algo", "task"])["mean_delta_pct"].idxmax()
    best_by_task = mean_cfg.loc[idx].copy()

    best_by_task["mean_delta_pct"] = best_by_task["mean_delta_pct"].round(1)
    best_by_task = best_by_task.sort_values(["algo", "task"])

    best_by_task.to_csv(OUT_BEST_BY_TASK, index=False)
    print("Wrote:", OUT_BEST_BY_TASK, "rows:", len(best_by_task))

    # -----------------------------------------
    # B) Activation-family summary
    #    Use best config per task within each family
    # -----------------------------------------
    mean_cfg_fam = mean_cfg.copy()
    mean_cfg_fam["act_family"] = mean_cfg_fam["activation"].map(
        lambda a: ACT_FAMILY.get(a, a)
    )

    # best within each family per (algo, task)
    idx2 = mean_cfg_fam.groupby(["algo", "task", "act_family"])["mean_delta_pct"].idxmax()
    best_task_per_family = mean_cfg_fam.loc[idx2].copy()

    fam_summary = (
        best_task_per_family.groupby(["algo", "act_family"], as_index=False)
        .agg(
            avg_delta_pct=("mean_delta_pct", "mean"),
            win_count=("mean_delta_pct", lambda s: int((s > 0).sum())),
        )
    )

    fam_summary["avg_delta_pct"] = fam_summary["avg_delta_pct"].round(1)
    fam_summary = fam_summary.sort_values(
        ["algo", "avg_delta_pct"],
        ascending=[True, False],
    )

    fam_summary.to_csv(OUT_FAM_SUMMARY, index=False)
    print("Wrote:", OUT_FAM_SUMMARY, "rows:", len(fam_summary))

    # -----------------------------------------
    # C) 1layer vs 2layers comparison
    #    For each (algo, task):
    #      best_1layer Delta%
    #      best_2layers Delta%
    #      difference
    # -----------------------------------------
    best_1 = (
        mean_cfg[mean_cfg["layers"] == "1layer"]
        .groupby(["algo", "task"], as_index=False)["mean_delta_pct"]
        .max()
        .rename(columns={"mean_delta_pct": "best_1layer_delta_pct"})
    )

    best_2 = (
        mean_cfg[mean_cfg["layers"] == "2layers"]
        .groupby(["algo", "task"], as_index=False)["mean_delta_pct"]
        .max()
        .rename(columns={"mean_delta_pct": "best_2layers_delta_pct"})
    )

    comp = best_1.merge(best_2, on=["algo", "task"], how="outer")
    comp["diff_2minus1"] = (
        comp["best_2layers_delta_pct"] - comp["best_1layer_delta_pct"]
    )

    comp["best_1layer_delta_pct"] = comp["best_1layer_delta_pct"].round(1)
    comp["best_2layers_delta_pct"] = comp["best_2layers_delta_pct"].round(1)
    comp["diff_2minus1"] = comp["diff_2minus1"].round(1)

    comp = comp.sort_values(["algo", "task"])
    comp.to_csv(OUT_1V2, index=False)
    print("Wrote:", OUT_1V2, "rows:", len(comp))

    # Tests
    print("[T5] best-by-task rows:", len(best_by_task))
    print(
        "[T6] best-by-task null activation:",
        int(best_by_task["activation"].isna().sum()),
        "null mean_delta_pct:",
        int(best_by_task["mean_delta_pct"].isna().sum()),
    )
    print("[T7] 1layer-vs-2layers rows:", len(comp))

    print("\nActivation families found:")
    print(sorted(df["act_family"].dropna().unique()))

    print("\nRows per activation:")
    print(df["activation"].value_counts().sort_index())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e), file=sys.stderr)
        raise
