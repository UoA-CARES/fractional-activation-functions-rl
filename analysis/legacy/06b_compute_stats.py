from __future__ import annotations

import os
import sys
import argparse
from typing import Tuple

import numpy as np
import pandas as pd


ROOT_DEFAULT = os.path.expanduser("~/Desktop/new-Fr")
IN_DEFAULT = os.path.join(ROOT_DEFAULT, "outputs/paper/table_stats_paired_seed_rows.csv")
OUT_DEFAULT = os.path.join(ROOT_DEFAULT, "outputs/paper/table_stats_summary.csv")


def die(msg: str, code: int = 1) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    raise SystemExit(code)


def ensure_cols(df: pd.DataFrame, cols: list[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        die(f"{where} missing columns: {missing}\nFound: {list(df.columns)}")


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cliff's delta: P(x > y) - P(x < y)
    Returns value in [-1, 1].
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    n = x.size * y.size
    if n == 0:
        return np.nan
    return float((gt - lt) / n)


def effect_size_category(delta: float) -> str:
    a = abs(delta)
    if np.isnan(a):
        return "unknown"
    if a < 0.147:
        return "negligible"
    if a < 0.33:
        return "small"
    if a < 0.474:
        return "medium"
    return "large"


def bootstrap_mean_ci(
    diffs: np.ndarray,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float]:
    """
    Bootstrap CI for the mean of diffs (paired differences).
    Resamples diffs with replacement.
    """
    diffs = np.asarray(diffs, dtype=float)
    n = diffs.size
    if n == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diffs[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo = np.quantile(means, alpha)
    hi = np.quantile(means, 1.0 - alpha)
    return (float(lo), float(hi))


def wilcoxon_pvalue(diffs: np.ndarray) -> float:
    """
    Paired Wilcoxon signed-rank test on diffs (two-sided).
    Uses SciPy if available.
    """
    diffs = np.asarray(diffs, dtype=float)

    # Wilcoxon cannot handle all-zero diffs
    if np.allclose(diffs, 0.0):
        return 1.0

    try:
        from scipy.stats import wilcoxon  # type: ignore
    except Exception as e:
        die(
            "SciPy is required for Wilcoxon in this script but could not be imported.\n"
            f"Import error: {e}\n"
            "Fix: pip install scipy (or run inside your conda env that has scipy)."
        )

    # zero_method="wilcox" drops zeros, standard choice
    res = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided", mode="auto")
    return float(res.pvalue)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default=IN_DEFAULT)
    ap.add_argument("--out_csv", default=OUT_DEFAULT)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--boot_seed", type=int, default=0)
    args = ap.parse_args()

    if not os.path.exists(args.in_csv):
        die(f"Input file not found: {args.in_csv}")

    df = pd.read_csv(args.in_csv)

    required = [
        "algo",
        "layers",
        "task",
        "seed",
        "auc_relu",
        "auc_frac",
        "delta_auc",
        "delta_pct_vs_relu_seed",
        "placement",
        "activation",
        "alpha_key",
    ]
    ensure_cols(df, required, where=os.path.basename(args.in_csv))

    # Ensure seed treated as str for grouping consistency
    df["seed"] = df["seed"].astype(str)

    out_rows = []
    group_cols = ["algo", "layers", "task"]

    for (algo, layers, task), g in df.groupby(group_cols):
        diffs = g["delta_auc"].to_numpy(dtype=float)

        n = int(g["seed"].nunique())
        mean_delta_auc = float(np.mean(diffs)) if diffs.size else np.nan

        # Bootstrap CI on mean delta AUC
        ci_lo, ci_hi = bootstrap_mean_ci(diffs, n_boot=args.n_boot, seed=args.boot_seed)
        ci_excludes_zero = bool((ci_lo > 0.0) or (ci_hi < 0.0)) if not (np.isnan(ci_lo) or np.isnan(ci_hi)) else False

        # Wilcoxon
        p = wilcoxon_pvalue(diffs) if n >= 3 else np.nan
        significant_005 = bool((not np.isnan(p)) and (p < 0.05))

        # Cliff's delta: compare auc_frac vs auc_relu (unpaired definition, still informative)
        cd = cliffs_delta(g["auc_frac"].to_numpy(), g["auc_relu"].to_numpy())
        cd_cat = effect_size_category(cd)

        # For traceability, keep the selected best config info (should be constant within group)
        placement = g["placement"].mode(dropna=True).iloc[0] if not g["placement"].mode(dropna=True).empty else ""
        activation = g["activation"].mode(dropna=True).iloc[0] if not g["activation"].mode(dropna=True).empty else ""
        alpha_key = g["alpha_key"].mode(dropna=True).iloc[0] if not g["alpha_key"].mode(dropna=True).empty else ""

        mean_delta_pct = float(g["delta_pct_vs_relu_seed"].mean())

        out_rows.append(
            {
                "algo": algo,
                "layers": layers,
                "task": task,
                "best_placement": placement,
                "best_activation": activation,
                "best_alpha_key": alpha_key,
                "n_paired_seeds": n,
                "mean_delta_auc": mean_delta_auc,
                "bootstrap_ci_low": ci_lo,
                "bootstrap_ci_high": ci_hi,
                "ci_excludes_zero": ci_excludes_zero,
                "wilcoxon_p": p,
                "significant_0.05": significant_005,
                "cliffs_delta": cd,
                "effect_size_category": cd_cat,
                "mean_delta_pct_vs_relu": mean_delta_pct,
            }
        )

    out = pd.DataFrame(out_rows)

    # Friendly ordering
    out = out.sort_values(["algo", "layers", "task"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"[OK] Wrote: {args.out_csv}")

    # Console summary
    total = len(out)
    sig = int(out["significant_0.05"].sum())
    ci_ok = int(out["ci_excludes_zero"].sum())
    print(f"[INFO] Groups: {total}, p<0.05: {sig}, CI excludes 0: {ci_ok}")

    # Show top positive mean_delta_pct
    show_cols = [
        "algo",
        "layers",
        "task",
        "best_placement",
        "best_activation",
        "best_alpha_key",
        "n_paired_seeds",
        "mean_delta_pct_vs_relu",
        "wilcoxon_p",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "cliffs_delta",
        "effect_size_category",
    ]
    print("\n[INFO] Top 10 by mean_delta_pct_vs_relu:")
    print(out.sort_values("mean_delta_pct_vs_relu", ascending=False)[show_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
