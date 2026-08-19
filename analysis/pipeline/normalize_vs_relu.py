#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

IN_SEED = str(ROOT / "analysis_outputs" / "auc" / "auc_by_seed.csv")
IN_RUN = str(ROOT / "analysis_outputs" / "auc" / "auc_by_run.csv")

OUT_DIR = str(ROOT / "analysis_outputs" / "normalized")
OUT_SEED = str(Path(OUT_DIR) / "auc_by_seed_normalized.csv")
OUT_RUN = str(Path(OUT_DIR) / "auc_by_run_normalized.csv")
OUT_BASE = str(Path(OUT_DIR) / "relu_baseline_by_task.csv")


ALL_ACTIVATIONS = [
    "relu", "lrelu", "prelu", "swish", "gelu",
    "frelu", "flrelu", "fprelu", "fswish", "fgelu",
]

FRACTIONAL_ACTIVATIONS = [
    "frelu", "flrelu", "fprelu", "fswish", "fgelu",
]


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing columns: {missing}\nFound: {list(df.columns)}")


def norm_layers(x) -> str:
    s = str(x).strip().lower()
    if s in {"1", "1layer", "1layers"}:
        return "1layer"
    if s in {"2", "2layer", "2layers"}:
        return "2layers"
    return s


def norm_algo(x) -> str:
    return str(x).strip().upper()


def norm_str(x) -> str:
    return str(x).strip()


def norm_activation(x) -> str:
    s = str(x).strip().lower()

    # Accept both canonical names and folder-code forms.
    amap = {
        "r": "relu",
        "relu": "relu",

        "l": "lrelu",
        "lrelu": "lrelu",
        "leakyrelu": "lrelu",
        "leaky_relu": "lrelu",

        "p": "prelu",
        "prelu": "prelu",
        "parametricrelu": "prelu",
        "parametric_relu": "prelu",

        "s": "swish",
        "swish": "swish",
        "silu": "swish",

        "g": "gelu",
        "gelu": "gelu",

        "fr": "frelu",
        "frelu": "frelu",
        "fractionalrelu": "frelu",
        "fractional_relu": "frelu",

        "fl": "flrelu",
        "flrelu": "flrelu",
        "fractionalleakyrelu": "flrelu",
        "fractional_leaky_relu": "flrelu",

        "fp": "fprelu",
        "fprelu": "fprelu",
        "fractionalprelu": "fprelu",
        "fractional_prelu": "fprelu",

        "fs": "fswish",
        "fswish": "fswish",
        "fractionalswish": "fswish",
        "fractional_swish": "fswish",
        "fractionalswishbeta": "fswish",
        "fractional_swish_beta": "fswish",
        "residualfractionalswish": "fswish",
        "residual_fractional_swish": "fswish",

        "fg": "fgelu",
        "fgelu": "fgelu",
        "fractionalgelu": "fgelu",
        "fractional_gelu": "fgelu",
        "fractionalgelubeta": "fgelu",
        "fractional_gelu_beta": "fgelu",
        "residualfractionalgelu": "fgelu",
        "residual_fractional_gelu": "fgelu",
    }

    return amap.get(s, s)


def norm_placement(x) -> str:
    s = str(x).strip().lower()

    if s in {"none", "nan", "na", ""}:
        return "none"

    return s


def resolve_auc_col(df: pd.DataFrame) -> str:
    candidates = [
        "auc_used",
        "auc_norm",
        "auc",
        "mean_auc",
        "auc_mean",
        "best_mean_auc",
        "auc_value",
        "auc_total",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    raise KeyError(
        "No AUC column found. Tried: "
        + ", ".join(candidates)
        + "\nFound columns: "
        + ", ".join(list(df.columns))
    )


def standardize(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]

    if "layers" not in out.columns and "layer" in out.columns:
        out = out.rename(columns={"layer": "layers"})

    require_cols(out, ["algo", "task", "layers", "placement", "activation"], name)

    out["algo"] = out["algo"].apply(norm_algo)
    out["task"] = out["task"].apply(norm_str)
    out["layers"] = out["layers"].apply(norm_layers)
    out["placement"] = out["placement"].apply(norm_placement)
    out["activation"] = out["activation"].apply(norm_activation)

    # For 1-layer, force placement to none so it never splits baselines.
    out.loc[out["layers"] == "1layer", "placement"] = "none"

    return out


def build_relu_baseline(df_seed: pd.DataFrame, auc_col: str) -> pd.DataFrame:
    """
    Baseline is from the dedicated ReLU baseline runs.

    In raw logs this can appear as placement == 'r' and/or activation == 'r'/'relu'.
    We treat any row with placement == 'r' OR activation == 'relu' as eligible for baseline,
    then compute mean across seeds per (algo, task, layers).

    For 2-layer networks, we do not split baselines by placement. The ReLU baseline is
    shared across placements.
    """
    cand = df_seed[
        (df_seed["placement"] == "r")
        | (df_seed["activation"] == "relu")
    ].copy()

    if cand.empty:
        raise ValueError(
            "No baseline candidates found. Expected placement == 'r' or activation == 'relu' rows."
        )

    base = (
        cand.groupby(["algo", "task", "layers"], as_index=False)[auc_col]
        .mean()
        .rename(columns={auc_col: "auc_relu_base"})
    )

    return base


def attach_baseline(df: pd.DataFrame, base: pd.DataFrame, auc_col: str) -> pd.DataFrame:
    out = df.merge(base, on=["algo", "task", "layers"], how="left")

    if out["auc_relu_base"].isna().any():
        miss = (
            out[out["auc_relu_base"].isna()][["algo", "task", "layers"]]
            .drop_duplicates()
            .head(30)
        )
        raise ValueError(f"Missing baseline for some groups. Example:\n{miss}")

    out["auc_used"] = out[auc_col]

    # Keep the previous formula unchanged.
    out["delta_pct_vs_relu"] = (
        100.0 * (out["auc_used"] - out["auc_relu_base"]) / out["auc_relu_base"]
    )

    return out


def main() -> None:
    if not os.path.exists(IN_SEED):
        raise FileNotFoundError(f"Missing input: {IN_SEED}")

    if not os.path.exists(IN_RUN):
        raise FileNotFoundError(f"Missing input: {IN_RUN}")

    ensure_dir(OUT_DIR)

    print("Reading seed AUC from:", IN_SEED)
    print("Reading run AUC from:", IN_RUN)

    df_seed_raw = pd.read_csv(IN_SEED)
    df_run_raw = pd.read_csv(IN_RUN)

    df_seed = standardize(df_seed_raw, "auc_by_seed.csv")
    df_run = standardize(df_run_raw, "auc_by_run.csv")

    auc_col_seed = resolve_auc_col(df_seed)
    auc_col_run = resolve_auc_col(df_run)

    base = build_relu_baseline(df_seed, auc_col_seed)
    base.to_csv(OUT_BASE, index=False)

    seed_norm = attach_baseline(df_seed, base, auc_col_seed)
    run_norm = attach_baseline(df_run, base, auc_col_run)

    seed_norm.to_csv(OUT_SEED, index=False)
    run_norm.to_csv(OUT_RUN, index=False)

    print("[T1] any NaN in auc_relu_base:", int(seed_norm["auc_relu_base"].isna().sum()))
    print("[T2] baseline groups:", len(base), "expected:", 2 * 2 * 8)
    print("[T3] any NaN in delta_pct_vs_relu:", int(seed_norm["delta_pct_vs_relu"].isna().sum()))
    print("[T4] rows normalized == rows input:", len(seed_norm), "==", len(df_seed_raw))

    print("\nSeed normalized activation counts:")
    print(seed_norm["activation"].value_counts().sort_index())

    print("\nRun normalized activation counts:")
    print(run_norm["activation"].value_counts().sort_index())

    expected = set(ALL_ACTIVATIONS)
    found = set(seed_norm["activation"].dropna().unique())
    print("\nMissing expected activations:", sorted(expected - found))

    print("\nUNKNOWN tasks:", int((seed_norm["task"] == "UNKNOWN").sum()))

    if "alpha" in seed_norm.columns:
        frac = seed_norm[seed_norm["activation"].isin(FRACTIONAL_ACTIVATIONS)]
        print("Fractional missing alpha:", int(frac["alpha"].isna().sum()))

    print("\nWrote:")
    print(" ", OUT_SEED)
    print(" ", OUT_RUN)
    print(" ", OUT_BASE)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e), file=sys.stderr)
        raise
