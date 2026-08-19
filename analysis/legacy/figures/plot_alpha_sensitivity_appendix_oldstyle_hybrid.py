#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(os.path.expanduser("~/Desktop/new-Fr"))

# Old file that produced your previous appendix figures
OLD_CSV = Path(os.environ.get("OLD_ALPHA_CSV", str(ROOT / "outputs" / "paper" / "table_alpha_sensitivity.csv")))

# New file that contains FGELU and FSwish
NEW_CSV = ROOT / "outputs" / "paper" / "alpha_sensitivity" / "table_alpha_sensitivity.csv"

OUT_APPENDIX = ROOT / "Appendix" / "alpha_sensitivity_best_task"
OUT_BACKUP = ROOT / "outputs" / "paper" / "alpha_sensitivity" / "appendix_figures_oldstyle_hybrid"

OUT_APPENDIX.mkdir(parents=True, exist_ok=True)
OUT_BACKUP.mkdir(parents=True, exist_ok=True)

OLD_ACTS = ["frelu", "flrelu", "fprelu"]
NEW_ACTS = ["fgelu", "fswish"]

ACT_ORDER = ["frelu", "flrelu", "fprelu", "fgelu", "fswish"]

ACT_LABEL = {
    "frelu": "FReLU",
    "flrelu": "FLReLU",
    "fprelu": "FPReLU",
    "fgelu": "FGELU",
    "fswish": "FSwish",
}

ACT_COLOR = {
    "frelu": "#1f77b4",   # blue
    "flrelu": "#2ca02c",  # green
    "fprelu": "#d62728",  # red
    "fgelu": "#9467bd",   # purple
    "fswish": "#ffbf00",  # yellow
}

REAL_PLACEMENTS = [
    "all-both",
    "all-actor",
    "all-critic",
    "first-both",
    "first-actor",
]

PLOT_PLACEMENTS = [
    "all-both",
    "all-actor",
    "all-critic",
    "first-both",
    "first-actor",
    "best-placement",
]

ALGOS = ["SAC", "TD3"]
ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required = [
        "algo",
        "task",
        "layers",
        "placement",
        "activation",
        "alpha",
        "mean_delta_pct",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}\nFound: {list(df.columns)}")

    df["algo"] = df["algo"].astype(str).str.upper().str.strip()
    df["task"] = df["task"].astype(str).str.strip()
    df["layers"] = df["layers"].astype(str).str.lower().str.strip()
    df["placement"] = df["placement"].astype(str).str.lower().str.strip()
    df["activation"] = df["activation"].astype(str).str.lower().str.strip()
    df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")
    df["mean_delta_pct"] = pd.to_numeric(df["mean_delta_pct"], errors="coerce")

    df = df.dropna(subset=["alpha", "mean_delta_pct"]).copy()
    return df


def load_hybrid() -> pd.DataFrame:
    if not OLD_CSV.exists():
        raise FileNotFoundError(f"Missing old CSV: {OLD_CSV}")

    if not NEW_CSV.exists():
        raise FileNotFoundError(f"Missing new CSV: {NEW_CSV}")

    old = clean(pd.read_csv(OLD_CSV))
    new = clean(pd.read_csv(NEW_CSV))

    old_part = old[old["activation"].isin(OLD_ACTS)].copy()
    new_part = new[new["activation"].isin(NEW_ACTS)].copy()

    df = pd.concat([old_part, new_part], ignore_index=True)

    df = df[
        (df["layers"] == "2layers")
        & (df["placement"].isin(REAL_PLACEMENTS))
        & (df["activation"].isin(ACT_ORDER))
    ].copy()

    if df.empty:
        raise ValueError("No valid two-layer rows found after hybrid merge.")

    print("Hybrid source")
    print("  old activations from:", OLD_CSV)
    print("  new activations from:", NEW_CSV)
    print()
    print(df["activation"].value_counts().reindex(ACT_ORDER))

    return df


def make_oldstyle_curve(df: pd.DataFrame, algo: str, placement: str) -> pd.DataFrame:
    sub = df[df["algo"] == algo].copy()

    if placement == "best-placement":
        # Old appendix behaviour:
        # pick the best task and placement for each activation and alpha.
        curve = (
            sub.groupby(["activation", "alpha"], as_index=False)["mean_delta_pct"]
            .max()
            .rename(columns={"mean_delta_pct": "best_delta_pct"})
        )
    else:
        # Old appendix behaviour for fixed placement:
        # pick the best task for each activation and alpha.
        curve = (
            sub[sub["placement"] == placement]
            .groupby(["activation", "alpha"], as_index=False)["mean_delta_pct"]
            .max()
            .rename(columns={"mean_delta_pct": "best_delta_pct"})
        )

    rows = []
    for act, g in curve.groupby("activation", sort=False):
        g = g.copy()
        max_abs = g["best_delta_pct"].abs().max()

        if pd.isna(max_abs) or max_abs == 0:
            g["norm_y"] = 0.0
        else:
            g["norm_y"] = g["best_delta_pct"] / max_abs

        rows.append(g)

    return pd.concat(rows, ignore_index=True)


def plot_one(curve: pd.DataFrame, algo: str, placement: str) -> None:
    out_dir = OUT_APPENDIX / algo
    backup_dir = OUT_BACKUP / algo

    out_dir.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5.5))

    for act in ACT_ORDER:
        g = curve[curve["activation"] == act].sort_values("alpha")

        if g.empty:
            continue

        xs = g["alpha"].tolist()
        ys = g["norm_y"].tolist()
        raw = g["best_delta_pct"].tolist()

        plt.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.2,
            markersize=6,
            color=ACT_COLOR[act],
            label=ACT_LABEL[act],
        )

        for x, y, r in zip(xs, ys, raw):
            sign = "+" if r >= 0 else ""
            va = "bottom" if y >= 0 else "top"
            plt.text(
                x,
                y,
                f"{sign}{r:.1f}",
                fontsize=7.5,
                fontweight="bold",
                ha="center",
                va=va,
            )

    plt.axhline(0, linewidth=1.2)
    plt.title(f"Alpha Sensitivity, {algo} 2-layer ({placement})", fontsize=10)
    plt.xlabel("alpha", fontsize=9)
    plt.ylabel("Normalized best Δ%", fontsize=9)
    plt.xticks(ALPHAS, fontsize=8)
    plt.yticks(fontsize=8)
    plt.ylim(-1.08, 1.08)
    plt.grid(True, linewidth=0.4, alpha=0.3)
    plt.legend(fontsize=7, loc="best")
    plt.tight_layout()

    filename = f"{placement}.png"

    plt.savefig(out_dir / filename, dpi=300, bbox_inches="tight")
    plt.savefig(backup_dir / filename, dpi=300, bbox_inches="tight")

    # Compatibility with older LaTeX filenames.
    if placement == "best-placement":
        plt.savefig(out_dir / "best placement.png", dpi=300, bbox_inches="tight")
        plt.savefig(backup_dir / "best placement.png", dpi=300, bbox_inches="tight")

    if placement == "first-both":
        plt.savefig(out_dir / "fisrt-both.png", dpi=300, bbox_inches="tight")
        plt.savefig(backup_dir / "fisrt-both.png", dpi=300, bbox_inches="tight")

    plt.close()

    print("Wrote:", out_dir / filename)


def main() -> None:
    df = load_hybrid()

    for algo in ALGOS:
        for placement in PLOT_PLACEMENTS:
            curve = make_oldstyle_curve(df, algo, placement)
            plot_one(curve, algo, placement)

    print("Done.")


if __name__ == "__main__":
    main()
