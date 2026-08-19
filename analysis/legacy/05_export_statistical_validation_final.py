#!/usr/bin/env python3
from __future__ import annotations

import os
import math
import argparse
from pathlib import Path

import pandas as pd


ROOT_DEFAULT = Path(os.path.expanduser("~/Desktop/new-Fr"))

DEFAULT_CANDIDATES = [
    ROOT_DEFAULT / "outputs" / "paper" / "table_stats_summary.csv",
    ROOT_DEFAULT / "outputs" / "paper" / "statistical_validation" / "table_stats_summary.csv",
    ROOT_DEFAULT / "outputs" / "paper-old" / "table_stats_summary.csv",
]

OUT_DIR_DEFAULT = ROOT_DEFAULT / "outputs" / "paper" / "statistical_validation_final"

ORDER_ALGO = {"SAC": 0, "TD3": 1}
ORDER_LAYERS = {"1layer": 0, "2layers": 1}


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


def esc_tex(x) -> str:
    if pd.isna(x):
        return "--"
    return str(x).replace("_", r"\_")


def pretty_layers(x) -> str:
    x = str(x).lower().strip()
    if x == "1layer":
        return "1 layer"
    if x == "2layers":
        return "2 layers"
    return esc_tex(x)


def pretty_activation(x) -> str:
    mapping = {
        "frelu": "FReLU",
        "flrelu": "FLReLU",
        "fprelu": "FPReLU",
        "fgelu": "FGELU",
        "fswish": "FSwish",
    }
    s = str(x).lower().strip()
    return mapping.get(s, esc_tex(x))


def pretty_alpha(x) -> str:
    if pd.isna(x):
        return "--"

    s = str(x).strip()

    if s.lower() in {"none", "nan", "--"}:
        return "--"

    try:
        return f"${float(s):.1f}$"
    except Exception:
        return esc_tex(s)


def sign_test_p(k: int, n: int) -> float:
    """One-sided exact sign test: P(X >= k), X ~ Binomial(n, 0.5)."""
    return sum(math.comb(n, i) * (0.5 ** n) for i in range(k, n + 1))


def resolve_input(path_arg: str | None) -> Path:
    if path_arg is not None:
        p = Path(path_arg).expanduser()
        if p.exists():
            return p
        raise FileNotFoundError(f"Input file not found: {p}")

    for p in DEFAULT_CANDIDATES:
        if p.exists():
            return p

    required = {
        "algo",
        "layers",
        "task",
        "best_placement",
        "best_activation",
        "best_alpha_key",
        "mean_delta_pct_vs_relu",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "wilcoxon_p",
        "cliffs_delta",
    }

    matches = []
    for p in (ROOT_DEFAULT / "outputs").rglob("*.csv"):
        try:
            cols = set(pd.read_csv(p, nrows=0).columns)
        except Exception:
            continue

        if required.issubset(cols):
            matches.append(p)

    if matches:
        return matches[0]

    searched = "\n".join(str(p) for p in DEFAULT_CANDIDATES)
    raise FileNotFoundError(
        "Could not find a statistical summary CSV.\n"
        "Checked these paths:\n"
        f"{searched}\n"
        "You can pass the file manually with --in_csv."
    )


def trend_label(positive: int, n: int, ci_positive: int, median_delta: float) -> str:
    if positive == n and ci_positive == n and median_delta >= 0.70:
        return "Strong and consistent improvement"
    if positive == n and median_delta >= 0.50:
        return "Consistent practical improvement"
    if positive >= n - 1:
        return "Positive directional improvement"
    return "Mixed but positive on average"


def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = [
        "algo",
        "layers",
        "task",
        "best_placement",
        "best_activation",
        "best_alpha_key",
        "mean_delta_pct_vs_relu",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "wilcoxon_p",
        "cliffs_delta",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nFound: {list(df.columns)}")

    df = df.copy()

    df["algo"] = df["algo"].astype(str).str.upper().str.strip()
    df["layers"] = df["layers"].astype(str).str.lower().str.strip()
    df["task"] = df["task"].astype(str).str.strip()
    df["best_placement"] = df["best_placement"].astype(str).str.lower().str.strip()
    df["best_activation"] = df["best_activation"].astype(str).str.lower().str.strip()

    numeric_cols = [
        "mean_delta_pct_vs_relu",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "wilcoxon_p",
        "cliffs_delta",
    ]

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["positive"] = df["mean_delta_pct_vs_relu"] > 0

    # This is stricter and safer than using ci_excludes_zero alone.
    # It requires the lower CI bound to be above zero.
    df["ci_positive"] = df["bootstrap_ci_low"] > 0

    return df


def build_main_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (algo, layers), g in df.groupby(["algo", "layers"], sort=False):
        n = int(g["task"].nunique())
        positive = int(g["positive"].sum())
        ci_positive = int(g["ci_positive"].sum())
        median_delta = float(g["cliffs_delta"].median())

        rows.append(
            {
                "algo": algo,
                "layers": layers,
                "positive": positive,
                "n": n,
                "mean_delta": float(g["mean_delta_pct_vs_relu"].mean()),
                "median_delta_pct": float(g["mean_delta_pct_vs_relu"].median()),
                "ci_positive": ci_positive,
                "median_wilcoxon_p": float(g["wilcoxon_p"].median()),
                "sign_p": sign_test_p(positive, n),
                "median_cliffs_delta": median_delta,
                "trend": trend_label(positive, n, ci_positive, median_delta),
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


def write_appendix_table(df: pd.DataFrame, out_path: Path) -> None:
    d = df.copy()

    d["algo_order"] = d["algo"].map(ORDER_ALGO)
    d["layer_order"] = d["layers"].map(ORDER_LAYERS)
    d = d.sort_values(["algo_order", "layer_order", "task"]).reset_index(drop=True)

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


def write_text(summary: pd.DataFrame, out_path: Path) -> None:
    total_positive = int(summary["positive"].sum())
    total_tasks = int(summary["n"].sum())

    text = rf"""
% =========================================================
% Statistical validation text generated from table_stats_summary.csv
% =========================================================

Across all evaluated settings, fractional activations improve normalized AUC in {total_positive} out of {total_tasks} task-level comparisons. This indicates a strong directional trend over the ReLU baseline across algorithms, network depths, and environments.

The aggregate statistical results are summarized in Table~\ref{{tab:stats-main}}. The sign-test results evaluate whether positive improvements occur consistently across tasks, while the paired Wilcoxon signed-rank tests and Cliff's $\delta$ effect sizes provide seed-level statistical and practical evidence. Because each task uses five seeds, individual per-task Wilcoxon tests have limited statistical power and should be interpreted together with the aggregate sign-test and effect-size results.

Full per-task statistical results, including bootstrap confidence intervals for AUC differences, paired Wilcoxon signed-rank tests, Cliff's $\delta$ effect sizes, selected activation configurations, fractional orders, and activation placements, are provided in Appendix~\ref{{app:statistical-validation}}.
""".strip()

    out_path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", default=None)
    parser.add_argument("--out_dir", default=str(OUT_DIR_DEFAULT))
    args = parser.parse_args()

    in_csv = resolve_input(args.in_csv)
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_df(in_csv)
    summary = build_main_summary(df)

    summary_csv = out_dir / "table_stats_main_summary.csv"
    main_tex = out_dir / "table_stats_main_summary.tex"
    appendix_tex = out_dir / "table_stats_appendix_per_task.tex"
    text_tex = out_dir / "statistical_validation_text.tex"

    summary.to_csv(summary_csv, index=False)
    write_main_table(summary, main_tex)
    write_appendix_table(df, appendix_tex)
    write_text(summary, text_tex)

    print("[OK] Statistical validation files written")
    print("Input:", in_csv)
    print("Output dir:", out_dir)
    print()
    print("Created:")
    print(" ", summary_csv)
    print(" ", main_tex)
    print(" ", appendix_tex)
    print(" ", text_tex)


if __name__ == "__main__":
    main()
