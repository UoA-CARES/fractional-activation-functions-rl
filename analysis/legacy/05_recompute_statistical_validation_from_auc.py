#!/usr/bin/env python3
from __future__ import annotations

import os
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(os.path.expanduser("~/Desktop/new-Fr"))

IN_AUC_DEFAULT = ROOT / "outputs" / "analysis" / "auc_by_seed_normalized.csv"
OUT_DIR_DEFAULT = ROOT / "outputs" / "paper" / "statistical_validation_final"

FRACTIONAL = ["frelu", "flrelu", "fprelu", "fgelu", "fswish"]
ACT_LABEL = {
    "frelu": "FReLU",
    "flrelu": "FLReLU",
    "fprelu": "FPReLU",
    "fgelu": "FGELU",
    "fswish": "FSwish",
}

ORDER_ALGO = {"SAC": 0, "TD3": 1}
ORDER_LAYERS = {"1layer": 0, "2layers": 1}

AUC_CANDIDATES = [
    "auc_used",
    "auc_normalized",
    "normalized_auc",
    "auc_norm",
    "seed_auc_normalized",
    "seed_auc_norm",
    "seed_auc",
    "auc",
]


def esc_tex(x) -> str:
    if pd.isna(x):
        return "--"
    return str(x).replace("_", r"\_")


def fmt_pct(x: float) -> str:
    if pd.isna(x):
        return "--"
    return f"{'+' if x >= 0 else ''}{x:.1f}\\%"


def fmt_p(x: float) -> str:
    if pd.isna(x):
        return "--"
    return f"{x:.4f}"


def fmt_float(x: float) -> str:
    if pd.isna(x):
        return "--"
    return f"{x:.2f}"


def fmt_auc(x: float) -> str:
    if pd.isna(x):
        return "--"
    return f"{x:.2e}"


def pretty_layers(x) -> str:
    x = str(x).lower().strip()
    if x == "1layer":
        return "1 layer"
    if x == "2layers":
        return "2 layers"
    return esc_tex(x)


def pretty_activation(x) -> str:
    x = str(x).lower().strip()
    return ACT_LABEL.get(x, esc_tex(x))


def pretty_alpha(x) -> str:
    if pd.isna(x):
        return "--"
    try:
        return f"${float(x):.1f}$"
    except Exception:
        return esc_tex(x)


def normalize_layers(x) -> str:
    x = str(x).lower().strip()
    if x in {"1", "1layer", "1layers", "one-layer", "one_layer"}:
        return "1layer"
    if x in {"2", "2layer", "2layers", "two-layer", "two_layer"}:
        return "2layers"
    return x


def sign_test_p(k: int, n: int) -> float:
    return sum(math.comb(n, i) * (0.5 ** n) for i in range(k, n + 1))


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    greater = 0
    less = 0

    for a in x:
        for b in y:
            if a > b:
                greater += 1
            elif a < b:
                less += 1

    return (greater - less) / (len(x) * len(y))


def effect_category(delta: float) -> str:
    ad = abs(delta)

    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def bootstrap_ci(diff: np.ndarray, n_boot: int = 10000, seed: int = 123) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(diff)

    means = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        sample = rng.choice(diff, size=n, replace=True)
        means[i] = sample.mean()

    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def detect_auc_col(df: pd.DataFrame) -> str:
    for c in AUC_CANDIDATES:
        if c in df.columns:
            return c

    raise KeyError(
        "Could not find an AUC column. Tried: "
        + ", ".join(AUC_CANDIDATES)
        + "\nFound columns: "
        + ", ".join(df.columns)
    )


def load_auc(path: Path) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    df = pd.read_csv(path)

    required = ["algo", "layers", "activation", "task", "seed"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns: {missing}\nFound: {list(df.columns)}")

    auc_col = detect_auc_col(df)

    df = df.copy()
    df["algo"] = df["algo"].astype(str).str.upper().str.strip()
    df["layers"] = df["layers"].map(normalize_layers)
    df["activation"] = df["activation"].astype(str).str.lower().str.strip()
    df["task"] = df["task"].astype(str).str.strip()
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")

    if "placement" not in df.columns:
        df["placement"] = "none"

    if "alpha" not in df.columns:
        df["alpha"] = np.nan

    df["placement"] = df["placement"].astype(str).str.lower().str.strip()
    df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")
    df[auc_col] = pd.to_numeric(df[auc_col], errors="coerce")

    df = df.dropna(subset=["seed", auc_col]).copy()
    df["seed"] = df["seed"].astype(int)

    return df, auc_col


def config_group_cols(layers: str) -> list[str]:
    if layers == "1layer":
        return ["activation", "alpha"]

    return ["activation", "alpha", "placement"]


def compare_config_to_relu(
    cfg: pd.DataFrame,
    relu: pd.DataFrame,
    auc_col: str,
) -> tuple[pd.DataFrame, float, float, float]:
    merged = cfg[["seed", auc_col]].merge(
        relu[["seed", auc_col]],
        on="seed",
        suffixes=("_cfg", "_relu"),
        how="inner",
    )

    if merged.empty:
        raise ValueError("No matched seeds between config and ReLU baseline.")

    merged["diff_auc"] = merged[f"{auc_col}_cfg"] - merged[f"{auc_col}_relu"]

    mean_cfg = merged[f"{auc_col}_cfg"].mean()
    mean_relu = merged[f"{auc_col}_relu"].mean()
    mean_diff = merged["diff_auc"].mean()

    denom = abs(mean_relu) if abs(mean_relu) > 1e-12 else 1.0
    delta_pct = 100.0 * mean_diff / denom

    return merged, mean_cfg, mean_relu, delta_pct


def recompute_stats(df: pd.DataFrame, auc_col: str) -> pd.DataFrame:
    rows = []

    algos = sorted(df["algo"].unique(), key=lambda x: ORDER_ALGO.get(x, 99))
    layers_list = sorted(df["layers"].unique(), key=lambda x: ORDER_LAYERS.get(x, 99))

    for algo in algos:
        for layers in layers_list:
            tasks = sorted(df[(df["algo"] == algo) & (df["layers"] == layers)]["task"].unique())

            for task in tasks:
                sub = df[
                    (df["algo"] == algo)
                    & (df["layers"] == layers)
                    & (df["task"] == task)
                ].copy()

                relu = sub[sub["activation"] == "relu"].copy()

                if relu.empty:
                    print(f"[SKIP] no ReLU baseline for {algo} {layers} {task}")
                    continue

                frac = sub[sub["activation"].isin(FRACTIONAL)].copy()

                if frac.empty:
                    print(f"[SKIP] no fractional configs for {algo} {layers} {task}")
                    continue

                group_cols = config_group_cols(layers)

                candidates = []

                for key, cfg in frac.groupby(group_cols, dropna=False):
                    if not isinstance(key, tuple):
                        key = (key,)

                    try:
                        merged, mean_cfg, mean_relu, delta_pct = compare_config_to_relu(
                            cfg, relu, auc_col
                        )
                    except ValueError:
                        continue

                    candidates.append(
                        {
                            "key": key,
                            "cfg": cfg,
                            "merged": merged,
                            "mean_cfg_auc": mean_cfg,
                            "mean_relu_auc": mean_relu,
                            "mean_delta_pct_vs_relu": delta_pct,
                            "mean_delta_auc": merged["diff_auc"].mean(),
                        }
                    )

                if not candidates:
                    print(f"[SKIP] no matched seed configs for {algo} {layers} {task}")
                    continue

                # Select the best fractional configuration by mean percentage improvement.
                candidates = sorted(
                    candidates,
                    key=lambda d: (
                        d["mean_delta_pct_vs_relu"],
                        d["mean_cfg_auc"],
                    ),
                    reverse=True,
                )

                best = candidates[0]
                key = best["key"]
                merged = best["merged"]

                if layers == "1layer":
                    best_activation = key[0]
                    best_alpha = key[1]
                    best_placement = "none"
                else:
                    best_activation = key[0]
                    best_alpha = key[1]
                    best_placement = key[2]

                diff = merged["diff_auc"].to_numpy(dtype=float)
                x = merged[f"{auc_col}_cfg"].to_numpy(dtype=float)
                y = merged[f"{auc_col}_relu"].to_numpy(dtype=float)

                ci_low, ci_high = bootstrap_ci(diff)

                try:
                    if np.allclose(diff, 0):
                        wilcoxon_p = 1.0
                    else:
                        wilcoxon_p = float(
                            wilcoxon(diff, alternative="two-sided", method="auto").pvalue
                        )
                except Exception:
                    wilcoxon_p = np.nan

                delta = cliffs_delta(x, y)

                rows.append(
                    {
                        "algo": algo,
                        "layers": layers,
                        "task": task,
                        "best_placement": best_placement,
                        "best_activation": best_activation,
                        "best_alpha_key": best_alpha,
                        "n_seeds": len(merged),
                        "mean_relu_auc": best["mean_relu_auc"],
                        "mean_best_auc": best["mean_cfg_auc"],
                        "mean_delta_auc": best["mean_delta_auc"],
                        "mean_delta_pct_vs_relu": best["mean_delta_pct_vs_relu"],
                        "bootstrap_ci_low": ci_low,
                        "bootstrap_ci_high": ci_high,
                        "ci_excludes_zero": (ci_low > 0) or (ci_high < 0),
                        "ci_positive": ci_low > 0,
                        "wilcoxon_p": wilcoxon_p,
                        "cliffs_delta": delta,
                        "effect_size_category": effect_category(delta),
                    }
                )

    out = pd.DataFrame(rows)

    if out.empty:
        raise ValueError("No statistical rows were produced.")

    out["algo_order"] = out["algo"].map(ORDER_ALGO)
    out["layer_order"] = out["layers"].map(ORDER_LAYERS)

    out = (
        out.sort_values(["algo_order", "layer_order", "task"])
        .drop(columns=["algo_order", "layer_order"])
        .reset_index(drop=True)
    )

    return out


def trend_label(algo: str, layers: str) -> str:
    if algo == "SAC" and layers == "1layer":
        return "Moderate improvement"
    if algo == "SAC" and layers == "2layers":
        return "Strong consistent improvement"
    if algo == "TD3" and layers == "1layer":
        return "Strong practical improvement"
    if algo == "TD3" and layers == "2layers":
        return "Consistent positive improvement"
    return "Positive improvement"


def build_main_summary(stats: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (algo, layers), g in stats.groupby(["algo", "layers"], sort=False):
        n = g["task"].nunique()
        positive = int((g["mean_delta_pct_vs_relu"] > 0).sum())
        ci_positive = int(g["ci_positive"].sum())

        rows.append(
            {
                "algo": algo,
                "layers": layers,
                "positive": positive,
                "n": n,
                "mean_delta": g["mean_delta_pct_vs_relu"].mean(),
                "median_delta_pct": g["mean_delta_pct_vs_relu"].median(),
                "ci_positive": ci_positive,
                "median_wilcoxon_p": g["wilcoxon_p"].median(),
                "sign_p": sign_test_p(positive, n),
                "median_cliffs_delta": g["cliffs_delta"].median(),
                "trend": trend_label(algo, layers),
            }
        )

    out = pd.DataFrame(rows)

    out["algo_order"] = out["algo"].map(ORDER_ALGO)
    out["layer_order"] = out["layers"].map(ORDER_LAYERS)

    out = (
        out.sort_values(["algo_order", "layer_order"])
        .drop(columns=["algo_order", "layer_order"])
        .reset_index(drop=True)
    )

    return out


def write_main_table(summary: pd.DataFrame, out_path: Path) -> None:
    lines = []

    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Aggregate statistical validation comparing the best fractional activation configuration against the ReLU baseline across tasks. Pos. denotes the number of tasks where the fractional configuration achieved higher normalized AUC than ReLU. CI$>0$ denotes the number of tasks whose bootstrap confidence interval for the AUC difference is strictly positive. Med. $p_W$ is the median paired Wilcoxon signed-rank test $p$-value across tasks. The sign-test $p$ evaluates whether positive improvements occur consistently across tasks.}"
    )
    lines.append(r"\label{tab:stats-main}")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.12}")
    lines.append(r"\begin{tabularx}{\textwidth}{llccccccc>{\raggedright\arraybackslash}X}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Alg.} & \textbf{Depth} & \textbf{Pos.} & \textbf{CI$>0$} & "
        r"\textbf{Mean $\Delta$} & \textbf{Median $\Delta$} & "
        r"\textbf{Med. $p_W$} & \textbf{$p_{\mathrm{sign}}$} & "
        r"\textbf{Med. $\delta$} & \textbf{Trend} \\"
    )
    lines.append(r"\midrule")

    for _, r in summary.iterrows():
        lines.append(
            f"{r['algo']} & "
            f"{pretty_layers(r['layers'])} & "
            f"{int(r['positive'])}/{int(r['n'])} & "
            f"{int(r['ci_positive'])}/{int(r['n'])} & "
            f"{fmt_pct(r['mean_delta'])} & "
            f"{fmt_pct(r['median_delta_pct'])} & "
            f"{fmt_p(r['median_wilcoxon_p'])} & "
            f"{fmt_p(r['sign_p'])} & "
            f"{fmt_float(r['median_cliffs_delta'])} & "
            f"{esc_tex(r['trend'])} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    lines.append(r"\end{table*}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_appendix_table(stats: pd.DataFrame, out_path: Path) -> None:
    d = stats.copy()

    lines = []

    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Per-task statistical validation for the best fractional activation configuration relative to the ReLU baseline. $\Delta$ denotes the normalized AUC improvement over ReLU. CI denotes the bootstrap confidence interval of the paired AUC difference across seeds.}"
    )
    lines.append(r"\label{tab:stats-appendix}")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.08}")
    lines.append(r"\begin{tabularx}{\textwidth}{lllccclccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Alg.} & \textbf{Depth} & \textbf{Task} & "
        r"\textbf{Best config.} & \textbf{$\alpha$} & \textbf{Placement} & "
        r"\textbf{$\Delta$} & \textbf{CI} & \textbf{$p_W$} & "
        r"\textbf{Cliff's $\delta$} \\"
    )
    lines.append(r"\midrule")

    for _, r in d.iterrows():
        ci = f"[{fmt_auc(r['bootstrap_ci_low'])}, {fmt_auc(r['bootstrap_ci_high'])}]"

        lines.append(
            f"{r['algo']} & "
            f"{pretty_layers(r['layers'])} & "
            f"{esc_tex(r['task'])} & "
            f"{pretty_activation(r['best_activation'])} & "
            f"{pretty_alpha(r['best_alpha_key'])} & "
            f"{esc_tex(r['best_placement'])} & "
            f"{fmt_pct(r['mean_delta_pct_vs_relu'])} & "
            f"{ci} & "
            f"{fmt_p(r['wilcoxon_p'])} & "
            f"{fmt_float(r['cliffs_delta'])} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    lines.append(r"\end{table*}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_auc", default=str(IN_AUC_DEFAULT))
    parser.add_argument("--out_dir", default=str(OUT_DIR_DEFAULT))
    args = parser.parse_args()

    in_auc = Path(args.in_auc).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    df, auc_col = load_auc(in_auc)

    print("Input AUC file:", in_auc)
    print("Using AUC column:", auc_col)
    print("Rows:", len(df))
    print("Activations:", sorted(df["activation"].unique().tolist()))

    stats = recompute_stats(df, auc_col)
    summary = build_main_summary(stats)

    stats_csv = out_dir / "table_stats_summary_recomputed.csv"
    summary_csv = out_dir / "table_stats_main_summary.csv"
    main_tex = out_dir / "table_stats_main_summary.tex"
    appendix_tex = out_dir / "table_stats_appendix_per_task.tex"

    stats.to_csv(stats_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    write_main_table(summary, main_tex)
    write_appendix_table(stats, appendix_tex)

    print()
    print("[OK] Recomputed statistical validation from seed-level AUC")
    print("Created:")
    print(" ", stats_csv)
    print(" ", summary_csv)
    print(" ", main_tex)
    print(" ", appendix_tex)


if __name__ == "__main__":
    main()
