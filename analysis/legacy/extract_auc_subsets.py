from __future__ import annotations

import os
import sys
import pandas as pd

ROOT = os.path.expanduser("~/Desktop/new-Fr")
IN_CSV = os.path.join(ROOT, "outputs", "analysis", "auc_by_seed_normalized.csv")
OUT_ROOT = os.path.join(ROOT, "outputs", "analysis", "auc_subsets")

PLACEMENTS_2L = ["all-actor", "all-critic", "all-both", "first-actor", "first-both"]

# Map folder codes to canonical activation names used everywhere
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

    "fs": "fswish",
    "fswish": "fswish",
    "fractionalswish": "fswish",
    "fractional_swish": "fswish",
    "fractionalswishbeta": "fswish",
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing columns: {missing}\nFound: {list(df.columns)}")


def normalize_layers_value(x: str) -> str:
    s = str(x).strip().lower()
    if s in {"1", "1layer", "1layers"}:
        return "1layer"
    if s in {"2", "2layer", "2layers"}:
        return "2layers"
    return s


def resolve_auc_col(df: pd.DataFrame) -> str:
    for c in ["auc_used", "auc_norm", "auc"]:
        if c in df.columns:
            return c
    raise KeyError("No AUC column found. Expected one of: auc_used, auc_norm, auc")


def normalize_activation_value(x: str) -> str:
    s = str(x).strip().lower()
    return ACT_MAP.get(s, s)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]

    if "layers" not in out.columns and "layer" in out.columns:
        out = out.rename(columns={"layer": "layers"})

    require_columns(out, ["algo", "task", "layers", "placement", "activation", "seed"], "normalized CSV")

    out["algo"] = out["algo"].astype(str).str.strip().str.upper()
    out["task"] = out["task"].astype(str).str.strip()
    out["layers"] = out["layers"].apply(normalize_layers_value)

    out["placement"] = out["placement"].astype(str).str.strip().str.lower()
    out["activation"] = out["activation"].apply(normalize_activation_value)

    # For 1layer, always unify placement
    out.loc[out["layers"] == "1layer", "placement"] = "none"

    # alpha_key must exist for reliable grouping
    if "alpha_key" not in out.columns:
        if "alpha" in out.columns:
            out["alpha_key"] = out["alpha"].astype(object).where(out["alpha"].notna(), "NA")
        else:
            out["alpha"] = pd.NA
            out["alpha_key"] = "NA"

    if "alpha" not in out.columns:
        out["alpha"] = pd.NA

    return out


def compute_mean_by_config(df_sub: pd.DataFrame, auc_col: str) -> pd.DataFrame:
    cfg_cols = ["algo", "task", "layers", "placement", "activation", "alpha_key"]
    return (
        df_sub.groupby(cfg_cols, as_index=False)[auc_col]
        .mean()
        .rename(columns={auc_col: "mean_auc"})
    )


def compute_baseline_relu_by_task_from_mean(mean_by_cfg: pd.DataFrame) -> pd.DataFrame:
    relu = mean_by_cfg[mean_by_cfg["activation"] == "relu"].copy()
    group_cols = ["algo", "task", "layers", "placement"]
    return (
        relu.groupby(group_cols, as_index=False)["mean_auc"]
        .max()
        .rename(columns={"mean_auc": "mean_auc_relu"})
    )


def compute_baseline_2layers_from_global_relu(df_all: pd.DataFrame, auc_col: str) -> pd.DataFrame:
    """
    For 2layers, baseline is stored in the dedicated ReLU folder (placement like 'r' or maybe 'none').
    We compute one baseline per (algo, task) from all 2layers rows with activation==relu, then
    later replicate across placements.
    """
    df2 = df_all[df_all["layers"] == "2layers"].copy()
    df2 = df2[df2["activation"] == "relu"].copy()
    if df2.empty:
        return pd.DataFrame(columns=["algo", "task", "layers", "mean_auc_relu"])

    # Mean across seeds, then baseline per task
    mean_relu = (
        df2.groupby(["algo", "task", "layers"], as_index=False)[auc_col]
        .mean()
        .rename(columns={auc_col: "mean_auc_relu"})
    )
    return mean_relu


def write_subset_tables(
    df_sub: pd.DataFrame,
    outdir: str,
    auc_col: str,
    subset_tag: str,
    baseline_override: pd.DataFrame | None = None,
) -> None:
    ensure_dir(outdir)

    # 1) by-seed table
    keep_cols = [
        "algo", "task", "layers", "placement",
        "activation", "alpha", "alpha_key",
        "seed", auc_col,
        "auc_relu_base", "delta_pct_vs_relu",
        "run_name", "run_dir", "eval_csv", "train_csv",
        "final_return", "n_points", "first_step", "last_step", "auc_cutoff",
        "task_alias",
    ]
    by_seed = df_sub.loc[:, [c for c in keep_cols if c in df_sub.columns]].copy()
    by_seed.to_csv(os.path.join(outdir, "auc_by_seed.csv"), index=False)

    # 2) mean-by-config
    mean_by_cfg = compute_mean_by_config(df_sub, auc_col)
    mean_by_cfg.to_csv(os.path.join(outdir, "auc_mean_by_config.csv"), index=False)

    # 3) baseline relu by task
    if baseline_override is None:
        baseline = compute_baseline_relu_by_task_from_mean(mean_by_cfg)
    else:
        baseline = baseline_override.copy()
    baseline.to_csv(os.path.join(outdir, "baseline_relu_by_task.csv"), index=False)

    # 4) best mean AUC per activation per task (max over alpha_key)
    best = (
        mean_by_cfg.groupby(["algo", "task", "layers", "placement", "activation"], as_index=False)["mean_auc"]
        .max()
        .rename(columns={"mean_auc": "best_mean_auc"})
    )
    best.to_csv(os.path.join(outdir, "auc_best_by_act_task.csv"), index=False)

    # 5) summary across tasks (mean of best_mean_auc across tasks)
    summary = (
        best.groupby(["algo", "layers", "placement", "activation"], as_index=False)["best_mean_auc"]
        .mean()
        .rename(columns={"best_mean_auc": "mean_best_auc_across_tasks"})
        .sort_values(["algo", "layers", "placement", "mean_best_auc_across_tasks"], ascending=[True, True, True, False])
    )
    summary.to_csv(os.path.join(outdir, "auc_summary_mean_across_tasks.csv"), index=False)

    # 6) info
    with open(os.path.join(outdir, "_subset_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"subset: {subset_tag}\n")
        f.write(f"rows: {len(df_sub)}\n")
        f.write(f"auc_col: {auc_col}\n")
        f.write(f"algos: {sorted(df_sub['algo'].unique().tolist())}\n")
        f.write(f"layers: {sorted(df_sub['layers'].unique().tolist())}\n")
        f.write(f"placements: {sorted(df_sub['placement'].unique().tolist())}\n")
        f.write(f"activations: {sorted(df_sub['activation'].unique().tolist())}\n")


def main() -> None:
    if not os.path.exists(IN_CSV):
        raise FileNotFoundError(f"Missing input: {IN_CSV}")

    df = pd.read_csv(IN_CSV)
    df = normalize_columns(df)

    require_columns(df, ["auc_relu_base", "delta_pct_vs_relu"], "auc_by_seed_normalized.csv")
    auc_col = resolve_auc_col(df)

    ensure_dir(OUT_ROOT)

    print("[DEBUG] using auc column:", auc_col)
    print("[DEBUG] layers counts:", df["layers"].value_counts(dropna=False).to_dict())
    print("[DEBUG] activation counts:", df["activation"].value_counts(dropna=False).head(20).to_dict())

    # 1layer subset
    df_1 = df[df["layers"] == "1layer"].copy()
    write_subset_tables(
        df_sub=df_1,
        outdir=os.path.join(OUT_ROOT, "1layer"),
        auc_col=auc_col,
        subset_tag="1layer",
    )
    print("Wrote 1layer:", os.path.join(OUT_ROOT, "1layer"), "rows:", len(df_1))

    # Compute global 2layers baseline from the dedicated relu runs (placement folder 'r')
    base2_global = compute_baseline_2layers_from_global_relu(df, auc_col)
    if base2_global.empty:
        print("[WARN] No 2layers relu rows found. 2layers placement baselines may be empty.")

    # 2layers per placement
    for pl in PLACEMENTS_2L:
        df_pl = df[(df["layers"] == "2layers") & (df["placement"] == pl)].copy()

        # Replicate the global baseline across this placement
        if not base2_global.empty:
            baseline_override = base2_global.copy()
            baseline_override["placement"] = pl
            baseline_override = baseline_override[["algo", "task", "layers", "placement", "mean_auc_relu"]]
        else:
            baseline_override = None

        write_subset_tables(
            df_sub=df_pl,
            outdir=os.path.join(OUT_ROOT, "2layers", pl),
            auc_col=auc_col,
            subset_tag=f"2layers/{pl}",
            baseline_override=baseline_override,
        )
        print("Wrote 2layers:", os.path.join(OUT_ROOT, "2layers", pl), "rows:", len(df_pl))

    print("Done. Wrote subsets to:", OUT_ROOT)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e), file=sys.stderr)
        raise
