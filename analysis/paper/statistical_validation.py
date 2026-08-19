from __future__ import annotations

import os
import sys
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


ROOT_DEFAULT = os.environ.get("FRORL_RESULTS_ROOT", "results")
IN_DEFAULT = os.path.join(ROOT_DEFAULT, "outputs/analysis/auc_by_seed_normalized.csv")
OUT_DIR_DEFAULT = os.path.join(ROOT_DEFAULT, "outputs/paper")


FRACTIONAL_ACTS = {"frelu", "flrelu", "fprelu", "fswish", "fgelu"}
RELU_NAMES = {"relu", "ReLU"}  # be tolerant


REQUIRED_COLS = [
    "algo",
    "task",
    "layers",
    "placement",
    "activation",
    "alpha",   # numeric in your file
    "seed",
    "auc",
    "delta_pct_vs_relu",
]


def die(msg: str, code: int = 1) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    raise SystemExit(code)


def ensure_cols(df: pd.DataFrame, cols: list[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        die(f"{where} missing columns: {missing}\nFound: {list(df.columns)}")


def canonical_activation(x: str) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    return s.lower()


def load_seed_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        die(f"Input CSV not found: {path}")
    df = pd.read_csv(path)
    ensure_cols(df, REQUIRED_COLS, where=os.path.basename(path))

    # Normalize types
    df["activation"] = df["activation"].map(canonical_activation)
    df["seed"] = df["seed"].astype(str)

    # ---- ADD THIS BLOCK HERE ----
    df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")

    def _alpha_key(row) -> str:
        act = row["activation"]
        a = row["alpha"]
        if act in {"frelu", "flrelu", "fprelu", "fswish", "fgelu"}:
            if pd.isna(a):
                return ""
            return f"{float(a):.1f}"
        return "na"

    df["alpha_key"] = df.apply(_alpha_key, axis=1)

    bad_frac = df[
        (df["activation"].isin({"frelu", "flrelu", "fprelu", "fswish", "fgelu"}))
        & (df["alpha_key"] == "")
    ]
    if not bad_frac.empty:
        sample = bad_frac[
            ["algo", "layers", "placement", "activation", "task", "seed", "alpha"]
        ].head(10)
        die("Some fractional rows have missing alpha. Example rows:\n" +
            sample.to_string(index=False))
    # ---- END BLOCK ----

    return df


def split_relu_fractional(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    relu = df[df["activation"].isin({a.lower() for a in RELU_NAMES})].copy()
    frac = df[df["activation"].isin(FRACTIONAL_ACTS)].copy()

    if relu.empty:
        die("No ReLU rows found in seed table. Expected activation == relu/ReLU.")
    if frac.empty:
        die("No fractional rows found in seed table. Expected activations in {frelu, flrelu, fprelu, fswish, fgelu, fswish, fgelu}.")

    return relu, frac


def pick_best_fractional_per_group(frac: pd.DataFrame) -> pd.DataFrame:
    """
    Choose best fractional config per (algo, layers, task) using mean delta_pct_vs_relu across seeds.

    Returns a table with one row per (algo, layers, task) describing the winning config:
    placement, activation, alpha_key, and its mean delta.
    """
    group_cols = ["algo", "layers", "task", "placement", "activation", "alpha_key"]

    # Mean delta across seeds for each config
    cfg = (
        frac.groupby(group_cols, as_index=False)
        .agg(mean_delta_pct=("delta_pct_vs_relu", "mean"),
             mean_auc=("auc", "mean"),
             n_seeds=("seed", "nunique"))
    )

    # Pick best config per (algo, layers, task)
    key_cols = ["algo", "layers", "task"]
    cfg = cfg.sort_values(key_cols + ["mean_delta_pct", "mean_auc"], ascending=[True, True, True, False, False])

    best = cfg.groupby(key_cols, as_index=False).head(1).copy()

    # Basic sanity
    if best.empty:
        die("Best fractional selection produced empty table.")

    # Warn if not all seeds available for a chosen config
    if (best["n_seeds"] < 2).any():
        bad = best[best["n_seeds"] < 2][key_cols + ["placement", "activation", "alpha_key", "n_seeds"]]
        print("[WARN] Some chosen best configs have <2 seeds. Stats will be weak for these groups.")
        print(bad.to_string(index=False))

    return best


def build_paired_seed_rows(
    relu: pd.DataFrame,
    frac: pd.DataFrame,
    best_cfg: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build per-seed paired rows for each (algo, layers, task, seed):
    - auc_relu
    - auc_frac (for the chosen best config of that group)
    - delta_auc = auc_frac - auc_relu
    - delta_pct_vs_relu_seed (from frac row, for reference)
    """

    # ReLU per seed (should be unique per algo,layers,task,seed)
    relu_key = ["algo", "layers", "task", "seed"]
    relu_seed = relu[relu_key + ["auc"]].copy()
    relu_seed = relu_seed.rename(columns={"auc": "auc_relu"})

    # If duplicates exist, average them but warn
    dup = relu_seed.duplicated(relu_key).any()
    if dup:
        print("[WARN] Duplicate ReLU rows found per (algo,layers,task,seed). Averaging them.")
        relu_seed = relu_seed.groupby(relu_key, as_index=False).agg(auc_relu=("auc_relu", "mean"))

    # Join best config onto fractional rows to filter to those winners
    cfg_cols = ["algo", "layers", "task", "placement", "activation", "alpha_key"]
    winners = best_cfg[cfg_cols].copy()

    frac_w = frac.merge(winners, on=cfg_cols, how="inner")
    if frac_w.empty:
        die("No fractional rows matched the selected best configs. Check keys and naming.")

    frac_key = ["algo", "layers", "task", "seed"]
    frac_seed = frac_w[frac_key + ["placement", "activation", "alpha_key", "auc", "delta_pct_vs_relu"]].copy()
    frac_seed = frac_seed.rename(columns={"auc": "auc_frac", "delta_pct_vs_relu": "delta_pct_vs_relu_seed"})

    # If duplicates exist, average them but warn
    if frac_seed.duplicated(frac_key).any():
        print("[WARN] Duplicate fractional rows found for winners per (algo,layers,task,seed). Averaging them.")
        frac_seed = (
            frac_seed.groupby(frac_key + ["placement", "activation", "alpha_key"], as_index=False)
            .agg(
                auc_frac=("auc_frac", "mean"),
                delta_pct_vs_relu_seed=("delta_pct_vs_relu_seed", "mean"),
            )
        )

    paired = frac_seed.merge(relu_seed, on=relu_key, how="inner")
    if paired.empty:
        die("Paired dataset is empty after inner-join with ReLU. Seeds or keys may not align.")

    paired["delta_auc"] = paired["auc_frac"] - paired["auc_relu"]

    # Check seed counts per group
    gcols = ["algo", "layers", "task"]
    counts = paired.groupby(gcols)["seed"].nunique().reset_index(name="n_paired_seeds")
    low = counts[counts["n_paired_seeds"] < 3]
    if not low.empty:
        print("[WARN] Some (algo,layers,task) groups have <3 paired seeds. Wilcoxon may be unreliable.")
        print(low.to_string(index=False))

    return paired


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT_DEFAULT)
    ap.add_argument("--in_csv", default=IN_DEFAULT)
    ap.add_argument("--out_dir", default=OUT_DIR_DEFAULT)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_seed_table(args.in_csv)
    relu, frac = split_relu_fractional(df)

    best_cfg = pick_best_fractional_per_group(frac)

    # Save best configs for transparency
    best_cfg_path = os.path.join(args.out_dir, "table_stats_best_fractional_configs.csv")
    best_cfg.to_csv(best_cfg_path, index=False)
    print(f"[OK] Wrote: {best_cfg_path}")

    paired = build_paired_seed_rows(relu, frac, best_cfg)

    paired_path = os.path.join(args.out_dir, "table_stats_paired_seed_rows.csv")
    paired.to_csv(paired_path, index=False)
    print(f"[OK] Wrote: {paired_path}")

    # Small console summary
    print("\n[INFO] Paired rows preview:")
    cols = ["algo", "layers", "task", "seed", "placement", "activation", "alpha_key", "auc_relu", "auc_frac", "delta_auc"]
    print(paired[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
