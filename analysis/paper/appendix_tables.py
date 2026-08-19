from __future__ import annotations

import os
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5]
FRACTIONAL_ACTS = ["frelu", "flrelu", "fprelu", "fswish", "fgelu"]


def iqm(values: np.ndarray, trim: float = 0.25) -> float:
    """Interquartile mean. For 5 seeds, equals mean of middle 3."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    x.sort()
    k = int(np.floor(trim * x.size))
    if 2 * k >= x.size:
        return float(np.mean(x))
    return float(np.mean(x[k : x.size - k]))


def canon(s: str) -> str:
    return str(s).strip().lower()


def tex_escape(s: str) -> str:
    return str(s).replace("_", r"\_")


def fmt(x: float, decimals: int, scale: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return f"{x/scale:.{decimals}f}"


def best_alpha_idx(vals5: List[float]) -> int:
    arr = np.array(vals5, dtype=float)
    if np.all(np.isnan(arr)):
        return -1
    return int(np.nanargmax(arr))


def overall_best_key(relu_val: float, nums15: List[float]) -> Tuple[str, int]:
    """Return ('relu', -1) or (act, alpha_index)."""
    best_v = relu_val
    best = ("relu", -1)

    k = 0
    for act in FRACTIONAL_ACTS:
        for j in range(5):
            v = nums15[k]
            if not np.isnan(v) and v > best_v:
                best_v = v
                best = (act, j)
            k += 1
    return best


def build_matrix(df: pd.DataFrame, algo: str, metric_col: str) -> Tuple[List[str], Dict[str, float], Dict[str, List[float]]]:
    """
    Returns:
      tasks
      relu_map[task] = IQM(AUC) for ReLU
      frac_map[task] = 15 values for (frelu 5, flrelu 5, fprelu 5)
    """
    sub = df[(df["algo"] == algo) & (df["layers"] == "1layer")].copy()
    tasks = sorted(sub["task"].unique().tolist())

    relu_map: Dict[str, float] = {}
    frac_map: Dict[str, List[float]] = {}

    for t in tasks:
        row = sub[sub["task"] == t]

        relu_vals = row[row["activation"] == "relu"][metric_col].to_numpy(dtype=float)
        relu_map[t] = iqm(relu_vals)

        nums15: List[float] = []
        for act in FRACTIONAL_ACTS:
            for a in ALPHAS:
                vals = row[(row["activation"] == act) & (np.isclose(row["alpha"], a))][metric_col].to_numpy(dtype=float)
                nums15.append(iqm(vals))
        frac_map[t] = nums15

    return tasks, relu_map, frac_map


def make_subtable_tex(
    algo: str,
    tasks: List[str],
    relu_map: Dict[str, float],
    frac_map: Dict[str, List[float]],
    decimals: int,
    scale: float,
) -> str:
    out: List[str] = []
    out.append(r"\begin{subtable}{\textwidth}")
    out.append(r"\centering")
    out.append(rf"\caption{{{tex_escape(algo)}}}")
    out.append(r"\resizebox{\textwidth}{!}{%")
    out.append(r"\begin{tabular}{l || >{\columncolor{gray!20}}c || *{5}{c} || *{5}{c} || *{5}{c} || *{5}{c} || *{5}{c}}")
    out.append(r"\toprule")
    out.append(r"\multirow{2}{*}{\textbf{Task}} & \multirow{2}{*}{\textbf{ReLU}} &")
    out.append(r"\multicolumn{5}{c||}{\textbf{FReLU}} &")
    out.append(r"\multicolumn{5}{c||}{\textbf{FLReLU}} &")
    out.append(r"\multicolumn{5}{c||}{\textbf{FPReLU}} &")
    out.append(r"\multicolumn{5}{c||}{\textbf{FSwish}} &")
    out.append(r"\multicolumn{5}{c}{\textbf{FGELU}} \\")
    out.append(r"\cmidrule(lr){3-7}\cmidrule(lr){8-12}\cmidrule(lr){13-17}\cmidrule(lr){18-22}\cmidrule(lr){23-27}")
    out.append(
        r"& & $\alpha{=}$0.1 & 0.2 & 0.3 & 0.4 & 0.5 &"
        r" 0.1 & 0.2 & 0.3 & 0.4 & 0.5 &"
        r" 0.1 & 0.2 & 0.3 & 0.4 & 0.5 &"
        r" 0.1 & 0.2 & 0.3 & 0.4 & 0.5 &"
        r" 0.1 & 0.2 & 0.3 & 0.4 & 0.5 \\"
    )
    out.append(r"\toprule")

    for t in tasks:
        relu_val = relu_map.get(t, float("nan"))
        nums = frac_map.get(t, [float("nan")] * 15)

        frelu_best = best_alpha_idx(nums[0:5])
        flrelu_best = best_alpha_idx(nums[5:10])
        fprelu_best = best_alpha_idx(nums[10:15])
        best = overall_best_key(relu_val, nums)

        relu_cell = fmt(relu_val, decimals, scale)
        if best == ("relu", -1) and relu_cell != "NA":
            relu_cell = rf"\textcolor{{custompurple}}{{\textbf{{{relu_cell}}}}}"
        else:
            if relu_cell != "NA":
                relu_cell = rf"\textbf{{{relu_cell}}}"

        cells: List[str] = []
        k = 0
        for act in FRACTIONAL_ACTS:
            start = k
            best_idx = best_alpha_idx(nums[start:start + 5])
            for j in range(5):
                v = nums[k]
                s = fmt(v, decimals, scale)
                if s != "NA" and j == best_idx:
                    s = rf"\textbf{{{s}}}"
                if best == (act, j) and s != "NA":
                    raw = fmt(v, decimals, scale)
                    s = rf"\textcolor{{custompurple}}{{\textbf{{{raw}}}}}"
                cells.append(s)
                k += 1

        out.append(rf"\textbf{{{tex_escape(t)}}} & {relu_cell} & " + " & ".join(cells) + r" \\")
        out.append(r"\addlinespace[2pt]")

    out.append(r"\bottomrule")
    out.append(r"\end{tabular}%")
    out.append(r"}")
    out.append(r"\end{subtable}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default=os.path.join(os.environ.get("FRORL_RESULTS_ROOT", "results"), "outputs", "analysis", "auc_by_seed_normalized.csv"))
    ap.add_argument("--out_dir", default=os.path.join(os.environ.get("FRORL_RESULTS_ROOT", "results"), "outputs", "paper", "appendix_tables_1layer_tex"))
    ap.add_argument("--decimals", type=int, default=2)
    ap.add_argument("--scale", type=float, default=1e9, help="Use 1e9 to report AUC in billions")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.in_csv)

    needed = {"algo", "layers", "placement", "activation", "alpha", "task", "seed"}
    missing = sorted(list(needed - set(df.columns)))
    if missing:
        raise ValueError(f"Missing columns in {args.in_csv}: {missing}")

    df["activation"] = df["activation"].map(canon)
    df["layers"] = df["layers"].astype(str)
    df["task"] = df["task"].astype(str)
    df["algo"] = df["algo"].astype(str)
    df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")

    metric_col = "auc_used" if "auc_used" in df.columns else "auc"
    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")

    # For 1layer baseline, placement should be "none". We do not need placement filter for 1layer.
    df1 = df[df["layers"] == "1layer"].copy()
    if df1.empty:
        raise ValueError("No rows found for layers == 1layer")

    # Build TD3 and SAC
    out_path = os.path.join(args.out_dir, "appendix_1layer_full_TD3_SAC.tex")

    caption = (
        r"\textbf{Full results for the 1-layer architecture.} "
        r"Normalised area under the learning curve (AUC, $\times 10^9$) up to total training steps "
        r"(interquartile mean over five seeds). "
        r"Best $\alpha$ per activation type (FReLU, FLReLU, FPReLU, FSwish, FGELU) is shown in \textbf{bold}; "
        r"the overall best configuration for each task is highlighted in "
        r"\textbf{\textcolor{custompurple}{purple}}."
    )

    td3_tasks, td3_relu, td3_frac = build_matrix(df1, algo="TD3", metric_col=metric_col)
    sac_tasks, sac_relu, sac_frac = build_matrix(df1, algo="SAC", metric_col=metric_col)

    tex: List[str] = []
    tex.append(r"\begin{table*}[t]")
    tex.append(r"\centering")
    tex.append(r"\small")
    tex.append(r"\setlength{\tabcolsep}{4pt}")
    tex.append(r"\renewcommand{\arraystretch}{1.3}")
    tex.append(rf"\caption{{{caption}}}")
    tex.append(r"\label{tab:appendix_full_results_1layer}")
    tex.append("")
    tex.append(make_subtable_tex("TD3", td3_tasks, td3_relu, td3_frac, args.decimals, args.scale))
    tex.append(r"\vspace{1em}")
    tex.append(make_subtable_tex("SAC", sac_tasks, sac_relu, sac_frac, args.decimals, args.scale))
    tex.append(r"\end{table*}")
    tex.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(tex))

    print(f"[OK] wrote {out_path}")
    print(f"[INFO] metric_col used: {metric_col}")
    print(f"[INFO] scale: {args.scale}  decimals: {args.decimals}")


if __name__ == "__main__":
    main()
