from __future__ import annotations
import os
import pandas as pd
import numpy as np

ROOT = os.environ.get("FRORL_RESULTS_ROOT", "results")
IN = os.path.join(
    ROOT,
    "outputs",
    "paper",
    "table_stats_paired_seed_rows_allacts_correctbaseline.csv",
)
OUT_DIR = os.path.join(ROOT, "outputs", "paper", "tables_best2layers")
os.makedirs(OUT_DIR, exist_ok=True)

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
    "fractional_gelu_beta": "fgelu",

    "fs": "fswish",
    "fswish": "fswish",
    "fractionalswish": "fswish",
    "fractional_swish": "fswish",
    "fractionalswishbeta": "fswish",
    "fractional_swish_beta": "fswish",
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

ACT_ORDER = [
    "relu",
    "lrelu",
    "prelu",
    "gelu",
    "swish",
    "frelu",
    "flrelu",
    "fprelu",
    "fgelu",
    "fswish",
]

df["algo"] = df["algo"].astype(str).str.upper().str.strip()
df["layers"] = df["layers"].astype(str).str.lower().str.strip()
df["task"] = df["task"].astype(str)
df["activation"] = (
    df["activation"]
    .astype(str)
    .str.lower()
    .str.strip()
    .map(lambda x: ACT_MAP.get(x, x))
)

# auc_by_seed_normalized.csv uses delta_pct_vs_relu
if "delta_pct_vs_relu_seed" in df.columns:
    delta_col = "delta_pct_vs_relu_seed"
else:
    delta_col = "delta_pct_vs_relu"

df["delta_seed"] = pd.to_numeric(df[delta_col], errors="coerce")
df = df.dropna(subset=["delta_seed"])

# Keep only expected activations
df = df[df["activation"].isin(ACT_ORDER)].copy()


def iqm(x):
    v = np.asarray(x.dropna().values, dtype=float)
    if len(v) == 0:
        return np.nan

    q1 = np.quantile(v, 0.25)
    q3 = np.quantile(v, 0.75)

    mid = v[(v >= q1) & (v <= q3)]

    return float(np.mean(mid)) if len(mid) else float(np.mean(v))


# Aggregate seeds → activation-level
agg = (
    df.groupby(["algo", "layers", "task", "activation"], as_index=False)
      .agg(delta_iqm=("delta_seed", iqm))
)

# Winner per task
idx = agg.groupby(["algo", "layers", "task"])["delta_iqm"].idxmax()
winners = agg.loc[idx].copy()
winners["activation_label"] = winners["activation"].map(ACT_LABEL).fillna(winners["activation"])
winners = winners.sort_values(["algo", "layers", "task"])


def write_tex(algo_name, layer_name):
    sub = winners[
        (winners["algo"] == algo_name)
        & (winners["layers"].str.contains(layer_name))
    ].copy()

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{Best activation per task ({algo_name}, {layer_name}).}}")
    lines.append(rf"\label{{tab:winner-{algo_name.lower()}-{layer_name}}}")
    lines.append(r"\begin{tabular}{lll}")
    lines.append(r"\toprule")
    lines.append(r"Task & Best Activation & $\Delta\%$ \\")
    lines.append(r"\midrule")

    for _, r in sub.iterrows():
        d = float(r["delta_iqm"])
        sign = "+" if d >= 0 else ""
        lines.append(
            f"{r['task']} & {r['activation_label']} & {sign}{d:.1f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out_path = os.path.join(
        OUT_DIR,
        f"table_winner_{algo_name.lower()}_{layer_name}.tex",
    )

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("Wrote:", out_path)


print("Activations found after mapping:")
print(sorted(df["activation"].unique()))

print("\nRows per activation:")
print(df["activation"].value_counts().sort_index())

for algo in ["SAC", "TD3"]:
    for layer in ["1", "2"]:
        write_tex(algo, layer)

print("Done.")
