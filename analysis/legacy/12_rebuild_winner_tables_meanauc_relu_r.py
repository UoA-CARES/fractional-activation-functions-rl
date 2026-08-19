
# scripts/analysis/12_rebuild_winner_tables_meanauc_relu_r.py
# Old logic preserved:
# mean AUC per config, ReLU baseline, then best config per task.
# Updated to include GELU, Swish, FGELU, FSwish.

import os
import pandas as pd

ROOT = os.path.expanduser("~/Desktop/new-Fr")

# Current complete file
IN = os.path.join(ROOT, "outputs", "analysis", "auc_by_seed_normalized.csv")

OUTDIR = os.path.join(ROOT, "outputs", "paper", "tables_best2layers")
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(IN)

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

ACT_LABEL = {
    "relu": "ReLU",
    "lrelu": "LReLU",
    "prelu": "PReLU",
    "gelu": "GELU",
    "swish": "Swish",
    "frelu": "FReLU",
    "flrelu": "FLReLU",
    "fprelu": "FPReLU",
    "fgelu": "FGELU",
    "fswish": "FSwish",
}

FRACTIONAL = {"frelu", "flrelu", "fprelu", "fgelu", "fswish"}


def norm_layers(x):
    s = str(x).lower().strip()
    if s in {"1", "1layer", "1layers"}:
        return "1layer"
    if s in {"2", "2layer", "2layers"}:
        return "2layers"
    return s


def norm_alpha(x):
    if pd.isna(x):
        return "--"
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "na"}:
        return "--"
    try:
        return f"{float(s):.1f}"
    except Exception:
        return s


def clean_alpha(act, alpha):
    if act in FRACTIONAL:
        return alpha
    return "--"


def clean_place(place):
    s = str(place).strip()
    if s.lower() in {"", "nan", "none", "na", "r"}:
        return "--"
    return s


df["algo"] = df["algo"].astype(str).str.upper().str.strip()
df["layers"] = df["layers"].apply(norm_layers)
df["task"] = df["task"].astype(str).str.strip()
df["activation"] = (
    df["activation"]
    .astype(str)
    .str.lower()
    .str.strip()
    .map(lambda x: ACT_MAP.get(x, x))
)

if "alpha_key" not in df.columns:
    df["alpha_key"] = df["alpha"].apply(norm_alpha)
else:
    df["alpha_key"] = df["alpha_key"].apply(norm_alpha)

df["placement"] = df["placement"].astype(str).str.lower().str.strip()

# Match old script column name
if "auc_frac" not in df.columns:
    df["auc_frac"] = pd.to_numeric(df["auc_used"], errors="coerce")
else:
    df["auc_frac"] = pd.to_numeric(df["auc_frac"], errors="coerce")

df = df.dropna(subset=["auc_frac"]).copy()


def mean_auc_table(algo: str, layers_tag: str) -> pd.DataFrame:
    layer_name = "1layer" if layers_tag == "1" else "2layers"

    sub = df[
        (df["algo"] == algo)
        & (df["layers"] == layer_name)
    ].copy()

    # Baseline: ReLU from placement r if it exists, otherwise any ReLU rows.
    relu_rows = sub[sub["activation"] == "relu"].copy()

    if "placement" in relu_rows.columns and (relu_rows["placement"] == "r").any():
        relu_rows = relu_rows[relu_rows["placement"] == "r"]

    relu_mean = (
        relu_rows.groupby(["task"], as_index=False)["auc_frac"]
        .mean()
        .rename(columns={"auc_frac": "mean_auc_relu"})
    )

    # Mean AUC per config: activation + alpha + placement
    g = (
        sub.groupby(["task", "activation", "alpha_key", "placement"], as_index=False)["auc_frac"]
        .mean()
        .rename(columns={"auc_frac": "mean_auc"})
    )

    g = g.merge(relu_mean, on="task", how="left")

    g["delta"] = 100.0 * (
        g["mean_auc"] - g["mean_auc_relu"]
    ) / g["mean_auc_relu"]

    g = g.dropna(subset=["delta"]).copy()

    # Best config per task across all activations, alphas, placements
    idx = g.groupby("task")["delta"].idxmax()

    win = g.loc[
        idx,
        ["task", "activation", "alpha_key", "placement", "mean_auc", "mean_auc_relu", "delta"],
    ].copy()

    win["activation_label"] = win["activation"].map(ACT_LABEL).fillna(win["activation"])
    win["alpha_print"] = win.apply(
        lambda r: clean_alpha(r["activation"], r["alpha_key"]),
        axis=1,
    )
    win["placement_print"] = win["placement"].apply(clean_place)

    win = win.sort_values("task")

    return win


def export_tex(win: pd.DataFrame, out_path: str, caption: str, label: str):
    with open(out_path, "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write("\\begin{tabular}{llllr}\n\\toprule\n")
        f.write("Task & Best Activation & Best $\\alpha$ & Placement & $\\Delta\\%$ \\\\\n")
        f.write("\\midrule\n")

        for _, r in win.iterrows():
            d = float(r["delta"])
            sign = "+" if d >= 0 else ""
            f.write(
                f"{r['task']} & "
                f"{r['activation_label']} & "
                f"{r['alpha_print']} & "
                f"{r['placement_print']} & "
                f"{sign}{d:.2f} \\\\\n"
            )

        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def export_csv(win: pd.DataFrame, out_path: str):
    win.to_csv(out_path, index=False)


print("Activations found:")
print(sorted(df["activation"].unique()))

print("\nRows per activation:")
print(df["activation"].value_counts().sort_index())

for algo in ["SAC", "TD3"]:
    for layer in ["1", "2"]:
        win = mean_auc_table(algo, layer)

        tex_path = os.path.join(
            OUTDIR,
            f"table_winner_{algo.lower()}_{layer}_meanauc_relur.tex",
        )
        csv_path = os.path.join(
            OUTDIR,
            f"table_winner_{algo.lower()}_{layer}_meanauc_relur.csv",
        )

        export_tex(
            win,
            tex_path,
            f"Best configuration per task ({algo}, {layer}-layer), mean AUC vs ReLU baseline.",
            f"tab:winner-{algo.lower()}-{layer}-meanauc-relur",
        )
        export_csv(win, csv_path)

        print("Wrote:", tex_path)
        print("Wrote:", csv_path)

print("Done.")

