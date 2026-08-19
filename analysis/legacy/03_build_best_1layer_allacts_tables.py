from __future__ import annotations
import os
import pandas as pd
import numpy as np

ROOT = os.path.expanduser("~/Desktop/new-Fr")
IN_CSV = os.path.join(ROOT, "outputs", "paper", "table_stats_paired_seed_rows.csv")
OUT_DIR = os.path.join(ROOT, "outputs", "paper", "tables_best2layers")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(IN_CSV)

# your columns (confirmed)
# ['algo','layers','task','seed','placement','activation','alpha_key',
#  'auc_frac','delta_pct_vs_relu_seed','auc_relu','delta_auc']

df["algo"] = df["algo"].astype(str).str.upper().str.strip()
df["layers"] = df["layers"].astype(str).str.lower().str.strip()
df["task"] = df["task"].astype(str)
df["activation"] = df["activation"].astype(str)
df["placement"] = df["placement"].astype(str)
df["alpha_key"] = df["alpha_key"].astype(str)
df["delta_seed"] = pd.to_numeric(df["delta_pct_vs_relu_seed"], errors="coerce")
df = df.dropna(subset=["delta_seed"])

# 1-layer only
df = df[df["layers"].str.contains("1")].copy()

def iqm(x: pd.Series) -> float:
    v = np.asarray(x.dropna().values, dtype=float)
    if v.size == 0:
        return np.nan
    q1 = np.quantile(v, 0.25)
    q3 = np.quantile(v, 0.75)
    mid = v[(v >= q1) & (v <= q3)]
    return float(np.mean(mid)) if mid.size else float(np.mean(v))

# Aggregate seeds -> config level
agg = (
    df.groupby(["algo","task","activation","placement","alpha_key"], as_index=False)
      .agg(delta_iqm=("delta_seed", iqm))
)

# IMPORTANT: best over placement + alpha, but KEEP activation
idx = agg.groupby(["algo","task","activation"])["delta_iqm"].idxmax()
best = agg.loc[idx, ["algo","activation","task","delta_iqm","placement"]].copy()
best = best.sort_values(["algo","activation","task"])

def write_tex(algo_name: str):
    sub = best[best["algo"] == algo_name].copy()
    out_path = os.path.join(OUT_DIR, f"table_best_1layer_allacts_{algo_name.lower()}.tex")

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{Best 1-layer placement per activation and task ({algo_name}).}}")
    lines.append(rf"\label{{tab:best-1layer-allacts-{algo_name.lower()}}}")
    lines.append(r"\begin{tabular}{llll}")
    lines.append(r"\toprule")
    lines.append(r"Activation & Task & Best $\Delta\%$ & Placement \\")
    lines.append(r"\midrule")

    for _, r in sub.iterrows():
        d = float(r["delta_iqm"])
        sign = "+" if d >= 0 else ""
        lines.append(f"{r['activation']} & {r['task']} & {sign}{d:.1f} & {r['placement']} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("Wrote:", out_path)

write_tex("SAC")
write_tex("TD3")
print("Done. Output dir:", OUT_DIR)
