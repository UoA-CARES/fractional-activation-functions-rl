#!/usr/bin/env python3
from __future__ import annotations

import os
import pandas as pd

ROOT = os.path.expanduser("~/Desktop/new-Fr")
IN_ROOT = os.path.join(ROOT, "outputs", "analysis", "auc_subsets")
OUT_ROOT = os.path.join(ROOT, "outputs", "paper", "tables_best2layers")
os.makedirs(OUT_ROOT, exist_ok=True)

ALGOS = ["TD3", "SAC"]
PLACEMENTS_2L = ["all-actor", "all-critic", "all-both", "first-actor", "first-both"]

ACT_ORDER = ["lrelu", "prelu", "swish", "gelu", "frelu", "flrelu", "fprelu", "fswish", "fgelu"]


def load_one_placement(placement: str) -> pd.DataFrame:
    best_path = os.path.join(IN_ROOT, "2layers", placement, "auc_best_by_act_task.csv")
    base_path = os.path.join(IN_ROOT, "2layers", placement, "baseline_relu_by_task.csv")

    best = pd.read_csv(best_path)
    base = pd.read_csv(base_path)

    best["algo"] = best["algo"].str.upper()
    best["layers"] = best["layers"].str.lower()
    best["placement"] = best["placement"].str.lower()
    best["activation"] = best["activation"].str.lower()

    base["algo"] = base["algo"].str.upper()
    base["layers"] = base["layers"].str.lower()
    base["placement"] = base["placement"].str.lower()

    key = ["algo", "task", "layers", "placement"]
    merged = best.merge(base[key + ["mean_auc_relu"]], on=key, how="left")

    merged["delta_pct"] = 100.0 * (
        (merged["best_mean_auc"] - merged["mean_auc_relu"])
        / merged["mean_auc_relu"]
    )

    merged["placement_src"] = placement
    return merged


def build_best_across_placements(algo: str) -> pd.DataFrame:
    dfs = []
    for pl in PLACEMENTS_2L:
        dfs.append(load_one_placement(pl))

    df = pd.concat(dfs, ignore_index=True)
    df = df[(df["algo"] == algo) & (df["layers"] == "2layers")]
    df = df[df["activation"].isin(ACT_ORDER)]

    idx = df.groupby(["activation", "task"])["delta_pct"].idxmax()
    best = df.loc[idx, ["activation", "task", "delta_pct", "placement_src"]].copy()

    best["delta_pct"] = best["delta_pct"].round(1)

    best = best.sort_values(["activation", "task"])
    return best


def save_latex_table(df: pd.DataFrame, path: str, caption: str, label: str):
    with open(path, "w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{" + caption + "}\n")
        f.write("\\label{" + label + "}\n")
        f.write("\\begin{tabular}{llll}\n")
        f.write("\\toprule\n")
        f.write("Activation & Task & Best $\\Delta\\%$ & Placement \\\\\n")
        f.write("\\midrule\n")

        for _, row in df.iterrows():
            f.write(
                f"{row.activation} & "
                f"{row.task} & "
                f"{row.delta_pct:+.1f} & "
                f"{row.placement_src} \\\\\n"
            )

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def main():
    for algo in ALGOS:
        best = build_best_across_placements(algo)

        # Save CSV
        csv_path = os.path.join(OUT_ROOT, f"best_2layers_{algo.lower()}.csv")
        best.to_csv(csv_path, index=False)

        # Save TXT
        txt_path = os.path.join(OUT_ROOT, f"best_2layers_{algo.lower()}.txt")
        best.to_string(open(txt_path, "w"), index=False)

        # Save LaTeX
        tex_path = os.path.join(OUT_ROOT, f"best_2layers_{algo.lower()}.tex")
        save_latex_table(
            best,
            tex_path,
            caption=f"Best 2-layer placement per activation and task ({algo}).",
            label=f"tab:best-2layers-{algo.lower()}",
        )

        print("Saved:", csv_path)
        print("Saved:", txt_path)
        print("Saved:", tex_path)

    print("Done.")


if __name__ == "__main__":
    main()
