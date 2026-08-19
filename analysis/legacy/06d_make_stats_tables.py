from __future__ import annotations

import os
import argparse
import pandas as pd
import numpy as np


ROOT_DEFAULT = os.path.expanduser("~/Desktop/new-Fr")
IN_DEFAULT = os.path.join(ROOT_DEFAULT, "outputs/paper/table_stats_summary.csv")
OUT_DIR_DEFAULT = os.path.join(ROOT_DEFAULT, "outputs/paper")


def esc_tex(s: str) -> str:
    """Minimal LaTeX escaping for table cells."""
    if s is None:
        return ""
    s = str(s)
    s = s.replace("_", r"\_")
    return s


def fmt_pct(x: float) -> str:
    if pd.isna(x):
        return ""
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.1f}"


def fmt_float(x: float, nd: int = 2) -> str:
    if pd.isna(x):
        return ""
    return f"{x:.{nd}f}"


def build_main_summary(df: pd.DataFrame) -> pd.DataFrame:
    # Large effects count
    df = df.copy()
    df["is_large"] = df["effect_size_category"].astype(str).str.lower().eq("large")

    rows = []
    for (algo, layers), g in df.groupby(["algo", "layers"], sort=True):
        ci_pos = int(g["ci_excludes_zero"].sum())
        n_tasks = int(g["task"].nunique())
        median_delta = float(g["cliffs_delta"].median())
        mean_delta_pct = float(g["mean_delta_pct_vs_relu"].mean())
        pct_large = 100.0 * float(g["is_large"].mean()) if len(g) else np.nan

        rows.append(
            {
                "Algo": algo,
                "Layers": layers,
                "CI>0": f"{ci_pos} / {n_tasks}",
                "Median_Cliffs_delta": median_delta,
                "Large_Effects_%": pct_large,
                "Mean_Delta_%": mean_delta_pct,
            }
        )

    out = pd.DataFrame(rows).sort_values(["Algo", "Layers"]).reset_index(drop=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default=IN_DEFAULT)
    ap.add_argument("--out_dir", default=OUT_DIR_DEFAULT)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.in_csv)

    required = [
        "algo",
        "layers",
        "task",
        "best_placement",
        "best_activation",
        "best_alpha_key",
        "n_paired_seeds",
        "mean_delta_auc",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "ci_excludes_zero",
        "wilcoxon_p",
        "cliffs_delta",
        "effect_size_category",
        "mean_delta_pct_vs_relu",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {args.in_csv}: {missing}")

    # -------------------------
    # 1) MAIN SUMMARY (4 rows)
    # -------------------------
    main_sum = build_main_summary(df)

    # Save CSV for sanity checking
    main_csv = os.path.join(args.out_dir, "table_stats_summary_main.csv")
    main_sum.to_csv(main_csv, index=False)

    # Build LaTeX for main table
    main_tex = os.path.join(args.out_dir, "table_stats_summary_main.tex")

    # Format display columns
    main_disp = main_sum.copy()
    main_disp["Median $\\delta$"] = main_disp["Median_Cliffs_delta"].map(lambda x: fmt_float(x, 2))
    main_disp["Large effects (\\%)"] = main_disp["Large_Effects_%"].map(lambda x: fmt_float(x, 0))
    main_disp["Mean $\\Delta\\%$"] = main_disp["Mean_Delta_%"].map(fmt_pct)
    main_disp = main_disp[["Algo", "Layers", "CI>0", "Median $\\delta$", "Large effects (\\%)", "Mean $\\Delta\\%$"]]

    latex_main = main_disp.to_latex(
        index=False,
        escape=False,
        caption=(
            "Statistical validation of fractional activations versus ReLU aggregated across 8 tasks. "
            "CI>0 indicates the number of tasks whose 95\\% bootstrap confidence interval for $\\Delta$AUC excludes zero. "
            "Effect size is measured using Cliff's $\\delta$."
        ),
        label="tab:stats_summary",
        column_format="lccccc",
    )

    # Add booktabs-friendly sizing lines around the tabular environment
    latex_main = latex_main.replace(r"\begin{table}", r"\begin{table}[t]" + "\n" + r"\centering" + "\n" + r"\small")
    latex_main = latex_main.replace(r"\end{table}", r"\end{table}")

    with open(main_tex, "w") as f:
        f.write(latex_main)

    print(f"[OK] Wrote: {main_csv}")
    print(f"[OK] Wrote: {main_tex}")

    # -------------------------
    # 2) FULL APPENDIX TABLE
    # -------------------------
    full_tex = os.path.join(args.out_dir, "table_stats_summary_full.tex")

    full = df.copy()

    # Friendly formatting
    full["task"] = full["task"].map(esc_tex)
    full["best_placement"] = full["best_placement"].map(esc_tex)
    full["best_activation"] = full["best_activation"].map(esc_tex)
    full["best_alpha_key"] = full["best_alpha_key"].map(esc_tex)
    full["mean_delta_pct_vs_relu"] = full["mean_delta_pct_vs_relu"].map(fmt_pct)
    full["wilcoxon_p"] = full["wilcoxon_p"].map(lambda x: fmt_float(x, 4))
    full["cliffs_delta"] = full["cliffs_delta"].map(lambda x: fmt_float(x, 2))

    # CI string (ΔAUC CI, raw scale)
    def ci_str(r) -> str:
        lo = r["bootstrap_ci_low"]
        hi = r["bootstrap_ci_high"]
        if pd.isna(lo) or pd.isna(hi):
            return ""
        # scientific notation, compact
        return f"[{lo:.2e}, {hi:.2e}]"

    full["CI$_{\\Delta\\mathrm{AUC}}$"] = full.apply(ci_str, axis=1)

    full["CI>0"] = full["ci_excludes_zero"].map(lambda b: "True" if bool(b) else "False")

    full_disp = full[
        [
            "algo",
            "layers",
            "task",
            "best_placement",
            "best_activation",
            "best_alpha_key",
            "n_paired_seeds",
            "mean_delta_pct_vs_relu",
            "CI$_{\\Delta\\mathrm{AUC}}$",
            "CI>0",
            "wilcoxon_p",
            "cliffs_delta",
            "effect_size_category",
        ]
    ].rename(
        columns={
            "algo": "Algo",
            "layers": "Layers",
            "task": "Task",
            "best_placement": "Placement",
            "best_activation": "Activation",
            "best_alpha_key": "$\\alpha$",
            "n_paired_seeds": "Seeds",
            "mean_delta_pct_vs_relu": "Mean $\\Delta\\%$",
            "wilcoxon_p": "$p$",
            "cliffs_delta": "Cliff's $\\delta$",
            "effect_size_category": "Effect",
        }
    )

    # Sort for readability
    full_disp = full_disp.sort_values(["Algo", "Layers", "Task"]).reset_index(drop=True)

    latex_full = full_disp.to_latex(
        index=False,
        escape=False,
        longtable=False,
        caption="Per-task statistical validation results comparing the best fractional configuration against ReLU.",
        label="tab:stats_full",
        column_format="llp{3.0cm}llllrlllrll",
    )

    latex_full = latex_full.replace(r"\begin{table}", r"\begin{table*}[t]" + "\n" + r"\centering" + "\n" + r"\scriptsize")
    latex_full = latex_full.replace(r"\end{table}", r"\end{table*}")

    with open(full_tex, "w") as f:
        f.write(latex_full)

    print(f"[OK] Wrote: {full_tex}")


if __name__ == "__main__":
    main()
