#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import pandas as pd

ROOT = os.path.expanduser("~/Desktop/new-Fr")
IN_NORM = os.path.join(ROOT, "outputs", "analysis", "auc_by_seed_normalized.csv")

OUT_DIR = os.path.join(ROOT, "outputs", "paper", "activation_placement_effects_all_activations")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_ACT_SUMMARY_CSV = os.path.join(OUT_DIR, "table_activation_summary.csv")
OUT_ACT_SUMMARY_TEX = os.path.join(OUT_DIR, "table_activation_summary.tex")

OUT_PLACE_FREQ_CSV = os.path.join(OUT_DIR, "table_placement_frequency.csv")
OUT_PLACE_FREQ_TEX = os.path.join(OUT_DIR, "table_placement_frequency.tex")

OUT_PLACE_BY_TASK_CSV = os.path.join(OUT_DIR, "table_placement_by_task.csv")
OUT_PLACE_BY_TASK_TEX = os.path.join(OUT_DIR, "table_placement_by_task.tex")

PLACEMENTS_2L = ["all-actor", "first-actor", "all-critic", "all-both", "first-both"]

TASK_ORDER = [
    "Ant-v4",
    "Cartpole-Swingup",
    "Cheetah-Run",
    "Finger-Spin",
    "HalfCheetah-v4",
    "Hopper-v4",
    "Humanoid-v4",
    "Walker-Walk",
]

ACT_ORDER = [
    "lrelu",
    "prelu",
    "gelu",
    "swish",
    "frelu",
    "flrelu",
    "fprelu",
    "fgelu",
    "fswish",
]

ACT_LABEL = {
    "relu": "ReLU",
    "lrelu": "LReLU",
    "prelu": "PReLU",
    "gelu": "GELU",
    "swish": "Swish",
    "frelu": "FReLU",
    "flrelu": "FLReLU",
    "fprelu": "FPReLU",
    "fgelu": "FGELU",
    "fswish": "FSwish",
}

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


def fmt_signed(x: float, digits: int = 1) -> str:
    x = float(x)
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.{digits}f}"


def load_data() -> pd.DataFrame:
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

    if "alpha_key" not in df.columns:
        if "alpha" in df.columns:
            df["alpha_key"] = df["alpha"].apply(norm_alpha)
        else:
            df["alpha_key"] = "NA"
    else:
        df["alpha_key"] = df["alpha_key"].apply(norm_alpha)

    df = df.dropna(subset=["delta_pct_vs_relu"]).copy()

    df2 = df[
        (df["layers"] == "2layers")
        & (df["placement"].isin(PLACEMENTS_2L))
        & (df["activation"].isin(ACT_ORDER))
    ].copy()

    if df2.empty:
        raise ValueError("No rows found for 2layers placement analysis.")

    return df2


def build_mean_config(df2: pd.DataFrame) -> pd.DataFrame:
    cfg_cols = ["algo", "task", "placement", "activation", "alpha_key"]

    mean_cfg = (
        df2.groupby(cfg_cols, as_index=False)["delta_pct_vs_relu"]
        .mean()
        .rename(columns={"delta_pct_vs_relu": "mean_delta_pct"})
    )

    return mean_cfg


def build_activation_summary(mean_cfg: pd.DataFrame) -> pd.DataFrame:
    best_per_act_task = (
        mean_cfg.groupby(["algo", "task", "activation"], as_index=False)["mean_delta_pct"]
        .max()
        .rename(columns={"mean_delta_pct": "best_delta_pct"})
    )

    out = (
        best_per_act_task.groupby("activation", as_index=False)
        .agg(
            avg_delta_pct=("best_delta_pct", "mean"),
            tasks_gt_relu=("best_delta_pct", lambda s: int((s > 0).sum())),
            n_pairs=("best_delta_pct", "size"),
            max_delta_pct=("best_delta_pct", "max"),
        )
    )

    out["activation_label"] = out["activation"].map(ACT_LABEL).fillna(out["activation"])
    out["activation"] = pd.Categorical(out["activation"], categories=ACT_ORDER, ordered=True)
    out = out.sort_values("activation")

    out["avg_delta_pct"] = out["avg_delta_pct"].round(1)
    out["max_delta_pct"] = out["max_delta_pct"].round(2)

    return out


def build_best_placement_per_activation(mean_cfg: pd.DataFrame) -> pd.DataFrame:
    best_alpha_per_place = (
        mean_cfg.groupby(["algo", "task", "activation", "placement"], as_index=False)["mean_delta_pct"]
        .max()
        .rename(columns={"mean_delta_pct": "best_delta_pct"})
    )

    idx = best_alpha_per_place.groupby(["algo", "task", "activation"])["best_delta_pct"].idxmax()
    best_place = best_alpha_per_place.loc[idx].copy()

    return best_place


def build_placement_frequency(best_place: pd.DataFrame) -> pd.DataFrame:
    freq = (
        best_place.groupby(["placement", "algo"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )

    pivot = (
        freq.pivot(index="placement", columns="algo", values="count")
        .fillna(0)
        .astype(int)
        .reset_index()
    )

    for algo in ["SAC", "TD3"]:
        if algo not in pivot.columns:
            pivot[algo] = 0

    pivot["placement"] = pd.Categorical(pivot["placement"], categories=PLACEMENTS_2L, ordered=True)
    pivot = pivot.sort_values("placement")

    return pivot[["placement", "SAC", "TD3"]]


def build_placement_by_task(best_place: pd.DataFrame) -> pd.DataFrame:
    freq = (
        best_place.groupby(["task", "placement"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )

    pivot = (
        freq.pivot(index="task", columns="placement", values="count")
        .fillna(0)
        .astype(int)
        .reset_index()
    )

    for p in PLACEMENTS_2L:
        if p not in pivot.columns:
            pivot[p] = 0

    pivot["task"] = pd.Categorical(pivot["task"], categories=TASK_ORDER, ordered=True)
    pivot = pivot.sort_values("task")

    return pivot[["task"] + PLACEMENTS_2L]


def write_activation_summary_tex(df: pd.DataFrame) -> None:
    lines = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Summary statistics of activation performance in the two-layer experiments. For each activation family, the table reports the average improvement relative to the ReLU baseline across the 16 algorithm--task pairs, the number of algorithm--task pairs where the activation outperforms ReLU, and the maximum observed improvement.}")
    lines.append(r"\label{tab:activation-summary}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{6pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.1}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Activation & Avg $\Delta\%$ & Tasks $>$ ReLU & Max $\Delta\%$ \\")
    lines.append(r"\midrule")

    for _, r in df.iterrows():
        lines.append(
            f"{r['activation_label']} & "
            f"{fmt_signed(r['avg_delta_pct'], 1)} & "
            f"{int(r['tasks_gt_relu'])} / {int(r['n_pairs'])} & "
            f"{fmt_signed(r['max_delta_pct'], 2)} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    with open(OUT_ACT_SUMMARY_TEX, "w") as f:
        f.write("\n".join(lines))


def write_placement_frequency_tex(df: pd.DataFrame) -> None:
    lines = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Frequency of activation placement appearing as the best configuration in the two-layer experiments. Counts are aggregated across 9 activation families and 8 tasks for each algorithm, giving 72 comparisons per algorithm.}")
    lines.append(r"\label{tab:placement-frequency-all}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{6pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.1}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"Placement Strategy & SAC (2-layer) & TD3 (2-layer) \\")
    lines.append(r"\midrule")

    for _, r in df.iterrows():
        lines.append(f"{r['placement']} & {int(r['SAC'])} & {int(r['TD3'])} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    with open(OUT_PLACE_FREQ_TEX, "w") as f:
        f.write("\n".join(lines))


def write_placement_by_task_tex(df: pd.DataFrame) -> None:
    lines = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Most frequent best placement per task across both algorithms and all 9 activation families in the two-layer experiments. Each row sums to 18, corresponding to 9 activation families evaluated under TD3 and SAC.}")
    lines.append(r"\label{tab:placement-by-task-all}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.08}")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"Task & all-actor & first-actor & all-critic & all-both & first-both \\")
    lines.append(r"\midrule")

    for _, r in df.iterrows():
        lines.append(
            f"{r['task']} & "
            f"{int(r['all-actor'])} & "
            f"{int(r['first-actor'])} & "
            f"{int(r['all-critic'])} & "
            f"{int(r['all-both'])} & "
            f"{int(r['first-both'])} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    with open(OUT_PLACE_BY_TASK_TEX, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    df2 = load_data()
    mean_cfg = build_mean_config(df2)

    activation_summary = build_activation_summary(mean_cfg)
    best_place = build_best_placement_per_activation(mean_cfg)
    placement_frequency = build_placement_frequency(best_place)
    placement_by_task = build_placement_by_task(best_place)

    activation_summary.to_csv(OUT_ACT_SUMMARY_CSV, index=False)
    placement_frequency.to_csv(OUT_PLACE_FREQ_CSV, index=False)
    placement_by_task.to_csv(OUT_PLACE_BY_TASK_CSV, index=False)

    write_activation_summary_tex(activation_summary)
    write_placement_frequency_tex(placement_frequency)
    write_placement_by_task_tex(placement_by_task)

    print("Wrote:", OUT_ACT_SUMMARY_CSV)
    print("Wrote:", OUT_ACT_SUMMARY_TEX)
    print("Wrote:", OUT_PLACE_FREQ_CSV)
    print("Wrote:", OUT_PLACE_FREQ_TEX)
    print("Wrote:", OUT_PLACE_BY_TASK_CSV)
    print("Wrote:", OUT_PLACE_BY_TASK_TEX)

    print("[T1] activations used:", sorted(df2["activation"].unique().tolist()))
    print("[T2] placements used:", sorted(df2["placement"].unique().tolist()))
    print("[T3] activation summary rows:", len(activation_summary))
    print("[T4] placement frequency rows:", len(placement_frequency))
    print("[T5] placement by task rows:", len(placement_by_task))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e), file=sys.stderr)
        raise
