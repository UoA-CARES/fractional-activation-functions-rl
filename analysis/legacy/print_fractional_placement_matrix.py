#!/usr/bin/env python3
from __future__ import annotations

import os
import pandas as pd

ROOT = os.path.expanduser("~/Desktop/new-Fr")
IN_NORM = os.path.join(ROOT, "outputs", "analysis", "auc_by_seed_normalized.csv")

OUT_DIR = os.path.join(ROOT, "outputs", "paper", "placement_matrices")
os.makedirs(OUT_DIR, exist_ok=True)

ALGOS = ["SAC", "TD3"]
PLACEMENTS_2L = ["all-actor", "all-critic", "all-both", "first-actor", "first-both"]
ROWS = ["1layer"] + PLACEMENTS_2L
FRACTIONAL = ["frelu", "flrelu", "fprelu", "fswish", "fgelu"]


def main():
    df = pd.read_csv(IN_NORM)

    df["algo"] = df["algo"].str.upper()
    df["task"] = df["task"].astype(str)
    df["layers"] = df["layers"].str.lower()
    df["placement"] = df["placement"].str.lower()
    df["activation"] = df["activation"].str.lower()

    if "alpha_key" not in df.columns:
        df["alpha_key"] = df.get("alpha", "NA")

    df.loc[df["layers"] == "1layer", "placement"] = "none"
    df = df[df["activation"].isin(FRACTIONAL)].copy()

    cfg = (
        df.groupby(
            ["algo", "task", "layers", "placement", "activation", "alpha_key"],
            as_index=False
        )["delta_pct_vs_relu"]
        .mean()
        .rename(columns={"delta_pct_vs_relu": "mean_delta_pct"})
    )

    # 1layer best
    cfg_1 = cfg[cfg["layers"] == "1layer"]
    best_1 = (
        cfg_1.groupby(["algo", "task"], as_index=False)["mean_delta_pct"]
        .max()
        .rename(columns={"mean_delta_pct": "best_delta_pct"})
    )
    best_1["row"] = "1layer"

    # 2layers placements best
    cfg_2 = cfg[(cfg["layers"] == "2layers") &
                (cfg["placement"].isin(PLACEMENTS_2L))]

    best_2 = (
        cfg_2.groupby(["algo", "task", "placement"], as_index=False)["mean_delta_pct"]
        .max()
        .rename(columns={"mean_delta_pct": "best_delta_pct",
                         "placement": "row"})
    )

    best_all = pd.concat(
        [best_1[["algo", "task", "row", "best_delta_pct"]],
         best_2[["algo", "task", "row", "best_delta_pct"]]],
        ignore_index=True
    )

    txt_output = []

    for algo in ALGOS:
        header = f"\n{'='*80}\n{algo} — Fractional Placement Matrix (Δ% vs ReLU)\n{'='*80}"
        print(header)
        txt_output.append(header)

        sub = best_all[best_all["algo"] == algo]
        tasks = sorted(sub["task"].unique())

        pivot = (
            sub.pivot(index="row", columns="task", values="best_delta_pct")
            .reindex(index=ROWS, columns=tasks)
        ).round(1)

        print(pivot.to_string())
        txt_output.append(pivot.to_string())
        print("="*80)
        txt_output.append("="*80)

        # Save CSV per algo
        csv_path = os.path.join(OUT_DIR, f"{algo.lower()}_fractional_placement_matrix.csv")
        pivot.to_csv(csv_path)
        print("Saved CSV:", csv_path)

    # Save TXT file
    txt_path = os.path.join(OUT_DIR, "fractional_placement_matrices.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(txt_output))

    print("\nSaved TXT:", txt_path)


if __name__ == "__main__":
    main()
