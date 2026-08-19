#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import pandas as pd

ROOT = os.path.expanduser("~/Desktop/new-Fr")

ALPHA_DIR = os.path.join(ROOT, "outputs", "paper", "alpha_sensitivity")
os.makedirs(ALPHA_DIR, exist_ok=True)

IN_BEST = os.path.join(ALPHA_DIR, "table_best_alpha_by_task.csv")

OUT_CSV = os.path.join(ALPHA_DIR, "table_alpha_mode_per_task.csv")
OUT_TEX = os.path.join(ALPHA_DIR, "table_alpha_mode_per_task.tex")

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

FRACTIONAL = ["frelu", "flrelu", "fprelu", "fgelu", "fswish"]


def require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing columns: {missing}\nFound: {list(df.columns)}")


def fmt_alpha(x: float) -> str:
    return f"${float(x):.1f}$"


def choose_mode(group: pd.DataFrame) -> dict:
    freq = (
        group.groupby("alpha", as_index=False)
        .agg(
            count=("activation", "nunique"),
            mean_delta=("best_mean_delta_pct", "mean"),
        )
    )

    # Tie break:
    # 1. higher count
    # 2. higher mean delta
    # 3. smaller alpha
    freq = freq.sort_values(
        ["count", "mean_delta", "alpha"],
        ascending=[False, False, True],
    )

    best = freq.iloc[0]

    return {
        "preferred_alpha": float(best["alpha"]),
        "count": int(best["count"]),
        "mean_delta": float(best["mean_delta"]),
    }


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
    df["task"] = df["task"].astype(str).str.strip()
    df["activation"] = df["activation"].astype(str).str.lower().str.strip()
    df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")
    df["best_mean_delta_pct"] = pd.to_numeric(df["best_mean_delta_pct"], errors="coerce")

    df = df[
        (df["layers"] == "2layers")
        & (df["placement"] == "best-placement")
        & (df["activation"].isin(FRACTIONAL))
    ].copy()

    df = df.dropna(subset=["alpha", "best_mean_delta_pct"]).copy()

    if df.empty:
        raise ValueError("No valid 2-layer best-placement alpha rows found.")

    rows = []

    for (algo, task), group in df.groupby(["algo", "task"], sort=False):
        m = choose_mode(group)
        rows.append(
            {
                "algo": algo,
                "task": task,
                "preferred_alpha": m["preferred_alpha"],
                "count": m["count"],
                "mean_delta": m["mean_delta"],
                "cell": f"{fmt_alpha(m['preferred_alpha'])} ({m['count']}/5)",
            }
        )

    modes = pd.DataFrame(rows)

    pivot = modes.pivot(index="task", columns="algo", values="cell").reset_index()

    for algo in ["SAC", "TD3"]:
        if algo not in pivot.columns:
            pivot[algo] = "--"

    pivot["task"] = pd.Categorical(
        pivot["task"],
        categories=TASK_ORDER,
        ordered=True,
    )

    pivot = pivot.sort_values("task")

    out = pivot[["task", "SAC", "TD3"]].copy()
    out = out.rename(
        columns={
            "task": "Task",
            "SAC": "SAC Preferred alpha",
            "TD3": "TD3 Preferred alpha",
        }
    )

    out.to_csv(OUT_CSV, index=False)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Preferred fractional order $\alpha$ per task in the two-layer experiments. For each task and algorithm, the value corresponds to the most frequently selected $\alpha$ across the five fractional activation families. The count in parentheses indicates how many activation families selected that value.}")
    lines.append(r"\label{tab:alpha-mode-per-task}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{8pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"Task & SAC Preferred $\alpha$ & TD3 Preferred $\alpha$ \\")
    lines.append(r"\midrule")

    for _, r in out.iterrows():
        task = r["Task"]
        sac_cell = r["SAC Preferred alpha"]
        td3_cell = r["TD3 Preferred alpha"]
        lines.append(f"{task} & {sac_cell} & {td3_cell} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    with open(OUT_TEX, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Wrote:", OUT_CSV)
    print("Wrote:", OUT_TEX)
    print("[T4g] rows:", len(out))
    print("[T4g] tasks:", out["Task"].tolist())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e), file=sys.stderr)
        raise
