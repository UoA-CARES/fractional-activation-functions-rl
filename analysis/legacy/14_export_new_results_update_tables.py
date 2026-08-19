#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import pandas as pd
import numpy as np


ROOT = os.path.expanduser("~/Desktop/new-Fr")

# Use normalized seed-level AUC, because the old paired stats file only has FReLU/FLReLU/FPReLU.
IN_CSV = os.path.join(ROOT, "outputs", "analysis", "auc_by_seed_normalized.csv")

# New output folder. This will not overwrite previous paper tables.
OUT_DIR = os.path.join(ROOT, "outputs", "paper", "tables_new_results_update")
os.makedirs(OUT_DIR, exist_ok=True)


NEW_ACTIVATIONS = [
    "swish",
    "gelu",
    "fswish",
    "fgelu",
]

ACT_LABEL = {
    "relu": "ReLU",
    "lrelu": "LReLU",
    "prelu": "PReLU",
    "swish": "Swish",
    "gelu": "GELU",
    "frelu": "FReLU",
    "flrelu": "FLReLU",
    "fprelu": "FPReLU",
    "fswish": "FSwish",
    "fgelu": "FGELU",
}


def norm_activation(x) -> str:
    s = str(x).strip().lower()

    amap = {
        "s": "swish",
        "swish": "swish",
        "silu": "swish",

        "g": "gelu",
        "gelu": "gelu",

        "fs": "fswish",
        "fswish": "fswish",
        "fractionalswish": "fswish",
        "fractional_swish": "fswish",
        "fractionalswishbeta": "fswish",
        "fractional_swish_beta": "fswish",

        "fg": "fgelu",
        "fgelu": "fgelu",
        "fractionalgelu": "fgelu",
        "fractional_gelu": "fgelu",
        "fractionalgelubeta": "fgelu",
        "fractional_gelu_beta": "fgelu",
    }

    return amap.get(s, s)


def norm_layers(x) -> str:
    s = str(x).strip().lower()

    if s in {"1", "1layer", "1layers"}:
        return "1layer"

    if s in {"2", "2layer", "2layers"}:
        return "2layers"

    if "1" in s:
        return "1layer"

    if "2" in s:
        return "2layers"

    return s


def norm_alpha(x) -> str:
    if pd.isna(x):
        return "--"

    s = str(x).strip()

    if s == "" or s.lower() in {"nan", "none", "na"}:
        return "--"

    try:
        v = float(s)
        return f"{v:.1f}"
    except Exception:
        return s


def norm_placement(x) -> str:
    if pd.isna(x):
        return "--"

    s = str(x).strip()

    if s == "" or s.lower() in {"nan", "none", "na"}:
        return "--"

    if s == "r":
        return "--"

    return s


def latex_escape(x) -> str:
    return str(x).replace("_", r"\_")


def iqm(x: pd.Series) -> float:
    v = np.asarray(x.dropna().values, dtype=float)

    if v.size == 0:
        return np.nan

    q1 = np.quantile(v, 0.25)
    q3 = np.quantile(v, 0.75)

    mid = v[(v >= q1) & (v <= q3)]

    if mid.size == 0:
        return float(np.mean(v))

    return float(np.mean(mid))


def require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]

    if missing:
        raise KeyError(
            "Missing required columns: "
            + str(missing)
            + "\nFound columns: "
            + str(list(df.columns))
        )


def write_tex_table(sub: pd.DataFrame, algo_name: str, layer_name: str) -> None:
    if sub.empty:
        print(f"[WARN] No rows for {algo_name} {layer_name}")
        return

    caption_layer = "single-layer" if layer_name == "1layer" else "two-layer"

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.12}")
    lines.append(
        rf"\caption{{New activation results for the {caption_layer} architecture under {algo_name}. "
        rf"The table reports only Swish, GELU, FSwish, and FGELU. "
        rf"For each activation and task, the best configuration is selected using seed-level IQM of $\Delta\%$ AUC relative to ReLU.}}"
    )
    lines.append(rf"\label{{tab:new-results-{algo_name.lower()}-{layer_name}}}")

    if layer_name == "1layer":
        lines.append(r"\begin{tabular}{llcc}")
        lines.append(r"\toprule")
        lines.append(r"Activation & Task & $\alpha$ & Best $\Delta\%$ \\")
        lines.append(r"\midrule")

        for _, r in sub.iterrows():
            act = ACT_LABEL.get(r["activation"], r["activation"])
            task = latex_escape(r["task"])
            alpha = r["alpha_key"]
            d = float(r["delta_iqm"])
            sign = "+" if d >= 0 else ""

            lines.append(
                rf"{act} & {task} & {alpha} & {sign}{d:.1f} \\"
            )

    else:
        lines.append(r"\begin{tabular}{lllcc}")
        lines.append(r"\toprule")
        lines.append(r"Activation & Task & Placement & $\alpha$ & Best $\Delta\%$ \\")
        lines.append(r"\midrule")

        for _, r in sub.iterrows():
            act = ACT_LABEL.get(r["activation"], r["activation"])
            task = latex_escape(r["task"])
            placement = latex_escape(r["placement"])
            alpha = r["alpha_key"]
            d = float(r["delta_iqm"])
            sign = "+" if d >= 0 else ""

            lines.append(
                rf"{act} & {task} & {placement} & {alpha} & {sign}{d:.1f} \\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out_path = os.path.join(
        OUT_DIR,
        f"table_new_results_{algo_name.lower()}_{layer_name}.tex",
    )

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("Wrote:", out_path)


def main() -> None:
    if not os.path.exists(IN_CSV):
        raise FileNotFoundError(
            f"Missing input file: {IN_CSV}\n"
            f"Run this first:\n"
            f"python3 scripts/analysis/01_normalize.py"
        )

    df = pd.read_csv(IN_CSV)

    require_cols(
        df,
        [
            "algo",
            "layers",
            "task",
            "seed",
            "placement",
            "activation",
            "alpha",
            "delta_pct_vs_relu",
        ],
    )

    df["algo"] = df["algo"].astype(str).str.upper().str.strip()
    df["layers"] = df["layers"].apply(norm_layers)
    df["task"] = df["task"].astype(str).str.strip()
    df["activation"] = df["activation"].apply(norm_activation)
    df["placement"] = df["placement"].apply(norm_placement)
    df["alpha_key"] = df["alpha"].apply(norm_alpha)
    df["delta_seed"] = pd.to_numeric(df["delta_pct_vs_relu"], errors="coerce")

    df = df.dropna(subset=["delta_seed"])

    print("\nAvailable activations in normalized input:")
    print(sorted(df["activation"].dropna().unique()))

    df_new = df[df["activation"].isin(NEW_ACTIVATIONS)].copy()

    if df_new.empty:
        print("\n[ERROR] No rows found for new activations:")
        print(NEW_ACTIVATIONS)
        print("\nThis means your normalized file is still old.")
        print("Run:")
        print("python3 scripts/analysis/01_normalize.py")
        raise SystemExit(1)

    agg = (
        df_new.groupby(
            ["algo", "layers", "task", "placement", "activation", "alpha_key"],
            as_index=False,
            dropna=False,
        )
        .agg(
            delta_iqm=("delta_seed", iqm),
            delta_mean=("delta_seed", "mean"),
            delta_std=("delta_seed", "std"),
            n_seeds=("delta_seed", "count"),
        )
    )

    idx = agg.groupby(
        ["algo", "layers", "activation", "task"],
        dropna=False,
    )["delta_iqm"].idxmax()

    best = agg.loc[idx].copy()
    best = best.sort_values(["algo", "layers", "activation", "task"])

    out_all = os.path.join(OUT_DIR, "new_results_all_configs_iqm.csv")
    out_best = os.path.join(OUT_DIR, "new_results_best_by_activation_task_iqm.csv")

    agg.to_csv(out_all, index=False)
    best.to_csv(out_best, index=False)

    print("\nWrote:", out_all)
    print("Wrote:", out_best)

    for algo_name in ["SAC", "TD3"]:
        for layer_name in ["1layer", "2layers"]:
            sub = best[
                (best["algo"] == algo_name)
                & (best["layers"] == layer_name)
            ].copy()

            write_tex_table(sub, algo_name, layer_name)

    print("\nActivation counts in new-results input:")
    print(df_new["activation"].value_counts().sort_index())

    print("\nBest rows by algo and layer:")
    print(best.groupby(["algo", "layers"]).size())

    print("\nDone. Output dir:", OUT_DIR)


if __name__ == "__main__":
    main()
