from __future__ import annotations

import os
import pandas as pd

ROOT = os.path.expanduser("~/Desktop/new-Fr")
IN = os.path.join(ROOT, "outputs", "paper", "table_stats_paired_seed_rows_clean.csv")
OUTDIR = os.path.join(ROOT, "outputs", "paper", "tables_best_per_activation_by_task")
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(IN)

# Normalize key columns for robust matching
for c in ["algo", "layers", "activation", "placement", "alpha_key", "task"]:
    if c in df.columns:
        df[c] = df[c].astype(str).str.strip()

if "algo" in df.columns:
    df["algo"] = df["algo"].str.upper()

if "activation" in df.columns:
    df["activation"] = df["activation"].str.lower()

if "placement" in df.columns:
    df["placement"] = df["placement"].str.lower()
    df["placement"] = df["placement"].replace({"nan": "none", "": "none"})

if "alpha_key" in df.columns:
    df["alpha_key"] = df["alpha_key"].str.lower()
    df["alpha_key"] = df["alpha_key"].replace({"nan": "base", "": "base"})

ACT_ORDER = [
    "relu",
    "lrelu",
    "prelu",
    "swish",
    "gelu",
    "frelu",
    "flrelu",
    "fprelu",
    "fswish",
    "fgelu",
]

ACT_DISPLAY = {
    "relu": "ReLU",
    "lrelu": "LReLU",
    "prelu": "PReLU",
    "swish": "Swish",
    "gelu": "GELU",
    "frelu": "FReLU",
    "flrelu": "FLReLU",
    "fprelu": "FPReLU",
    "fswish": "FSwish",
    "fgelu": "FGELU",
}


def _filter_layers(sub: pd.DataFrame, layers_tag: str) -> pd.DataFrame:
    """
    Robust filtering for layers.

    If the layers column is numeric, match 1 or 2 directly.
    Otherwise, match strings such as 1layer or 2layers.
    """
    if "layers" not in sub.columns:
        raise ValueError("Missing 'layers' column in input CSV.")

    s = sub["layers"]

    try:
        s_num = pd.to_numeric(s, errors="coerce")
        if s_num.notna().any():
            want = int(layers_tag)
            return sub[s_num == want].copy()
    except Exception:
        pass

    return sub[s.astype(str).str.contains(str(layers_tag), na=False)].copy()


def best_per_activation_table(algo: str, layers_tag: str) -> pd.DataFrame:
    sub = df[df["algo"] == algo].copy()
    sub = _filter_layers(sub, layers_tag)

    if sub.empty:
        raise ValueError(f"No rows for algo={algo}, layers_tag={layers_tag}")

    relu_rows = sub[sub["activation"] == "relu"].copy()
    if relu_rows.empty:
        raise ValueError(f"No ReLU rows found for algo={algo}, layers_tag={layers_tag}")

    if "placement" in relu_rows.columns and (relu_rows["placement"] == "r").any():
        relu_rows = relu_rows[relu_rows["placement"] == "r"]

    relu_mean = (
        relu_rows.groupby(["task"], as_index=False)["auc_frac"]
        .mean()
        .rename(columns={"auc_frac": "mean_auc_relu"})
    )

    cfg_cols = ["task", "activation"]

    if "alpha_key" in sub.columns:
        cfg_cols.append("alpha_key")
    else:
        sub["alpha_key"] = "base"
        cfg_cols.append("alpha_key")

    if "placement" in sub.columns:
        cfg_cols.append("placement")
    else:
        sub["placement"] = "none"
        cfg_cols.append("placement")

    g = (
        sub.groupby(cfg_cols, as_index=False)["auc_frac"]
        .mean()
        .rename(columns={"auc_frac": "mean_auc"})
    )

    g = g.merge(relu_mean, on="task", how="left")

    if g["mean_auc_relu"].isna().any():
        missing = g[g["mean_auc_relu"].isna()]["task"].unique().tolist()
        raise ValueError(f"Missing ReLU baseline for tasks: {missing}")

    g["delta"] = 100.0 * (g["mean_auc"] - g["mean_auc_relu"]) / g["mean_auc_relu"]

    idx = g.groupby(["task", "activation"])["delta"].idxmax()
    best = g.loc[
        idx,
        ["task", "activation", "alpha_key", "placement", "mean_auc", "mean_auc_relu", "delta"],
    ].copy()

    best["activation"] = best["activation"].astype(str).str.lower()
    best["task"] = best["task"].astype(str)

    best["act_rank"] = best["activation"].apply(
        lambda a: ACT_ORDER.index(a) if a in ACT_ORDER else 999
    )
    best = best.sort_values(["task", "act_rank", "activation"]).drop(columns=["act_rank"])

    relu_mask = best["activation"] == "relu"
    if relu_mask.any():
        best.loc[relu_mask, "delta"] = 0.0
        best.loc[relu_mask, "alpha_key"] = "base"

        if "placement" in sub.columns and (sub["placement"] == "r").any():
            best.loc[relu_mask, "placement"] = "r"
        else:
            best.loc[relu_mask, "placement"] = "none"

    return best


def export_tex(
    best: pd.DataFrame,
    out_path: str,
    caption: str,
    label: str,
    show_placement: bool,
) -> None:
    with open(out_path, "w") as f:
        f.write("{\\scriptsize\n")
        f.write("\\setlength{\\tabcolsep}{4pt}\n")
        f.write("\\renewcommand{\\arraystretch}{1.08}\n")

        if show_placement:
            f.write("\\begin{longtable}{llllr}\n")
            f.write(f"\\caption{{{caption}}}\\label{{{label}}}\\\\\n")
            f.write("\\toprule\n")
            f.write("Task & Activation & Best $\\alpha$ & Best placement & $\\Delta\\%$ \\\\\n")
            f.write("\\midrule\n")
            f.write("\\endfirsthead\n")

            f.write("\\caption[]{Continued from previous page}\\\\\n")
            f.write("\\toprule\n")
            f.write("Task & Activation & Best $\\alpha$ & Best placement & $\\Delta\\%$ \\\\\n")
            f.write("\\midrule\n")
            f.write("\\endhead\n")

            f.write("\\midrule\n")
            f.write("\\multicolumn{5}{r}{\\emph{Continued on next page}}\\\\\n")
            f.write("\\endfoot\n")

            f.write("\\bottomrule\n")
            f.write("\\endlastfoot\n")

            for _, r in best.iterrows():
                d = float(r["delta"])
                sign = "+" if d >= 0 else ""

                act_key = str(r["activation"]).lower()
                act = ACT_DISPLAY.get(act_key, str(r["activation"]))

                alpha = str(r["alpha_key"])
                placement = str(r["placement"])

                f.write(
                    f"{r['task']} & {act} & {alpha} & {placement} & {sign}{d:.2f} \\\\\n"
                )

            f.write("\\end{longtable}\n")

        else:
            f.write("\\begin{longtable}{lllr}\n")
            f.write(f"\\caption{{{caption}}}\\label{{{label}}}\\\\\n")
            f.write("\\toprule\n")
            f.write("Task & Activation & Best $\\alpha$ & $\\Delta\\%$ \\\\\n")
            f.write("\\midrule\n")
            f.write("\\endfirsthead\n")

            f.write("\\caption[]{Continued from previous page}\\\\\n")
            f.write("\\toprule\n")
            f.write("Task & Activation & Best $\\alpha$ & $\\Delta\\%$ \\\\\n")
            f.write("\\midrule\n")
            f.write("\\endhead\n")

            f.write("\\midrule\n")
            f.write("\\multicolumn{4}{r}{\\emph{Continued on next page}}\\\\\n")
            f.write("\\endfoot\n")

            f.write("\\bottomrule\n")
            f.write("\\endlastfoot\n")

            for _, r in best.iterrows():
                d = float(r["delta"])
                sign = "+" if d >= 0 else ""

                act_key = str(r["activation"]).lower()
                act = ACT_DISPLAY.get(act_key, str(r["activation"]))

                alpha = str(r["alpha_key"])

                f.write(
                    f"{r['task']} & {act} & {alpha} & {sign}{d:.2f} \\\\\n"
                )

            f.write("\\end{longtable}\n")

        f.write("}\n")


def export_csv(best: pd.DataFrame, out_path: str) -> None:
    best.to_csv(out_path, index=False)


for algo in ["SAC", "TD3"]:
    for layer in ["1", "2"]:
        best = best_per_activation_table(algo, layer)

        tex_path = os.path.join(
            OUTDIR,
            f"table_best_per_activation_{algo.lower()}_{layer}layer_meanauc_relur.tex",
        )
        csv_path = os.path.join(
            OUTDIR,
            f"table_best_per_activation_{algo.lower()}_{layer}layer_meanauc_relur.csv",
        )

        if layer == "1":
            caption = (
                f"Best configuration per activation and task for {algo} with a 1-layer network. "
                "Values report mean normalized AUC improvement relative to the ReLU baseline."
            )
        else:
            caption = (
                f"Best configuration per activation and task for {algo} with a 2-layer network. "
                "Values report mean normalized AUC improvement relative to the ReLU baseline. "
                "The placement column indicates where the activation is applied."
            )

        label = f"tab:best-per-activation-{algo.lower()}-{layer}-meanauc-relur"
        show_placement = layer == "2"

        export_tex(
            best,
            tex_path,
            caption,
            label,
            show_placement,
        )

        export_csv(best, csv_path)

        print("Wrote:", tex_path)
        print("Wrote:", csv_path)

print("Done.")
