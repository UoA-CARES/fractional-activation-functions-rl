from __future__ import annotations
import os
import pandas as pd

ROOT = os.path.expanduser("~/Desktop/new-Fr")
IN = os.path.join(ROOT, "outputs", "analysis", "auc_by_seed_normalized.csv")
OUT = os.path.join(ROOT, "outputs", "paper",
                   "table_stats_paired_seed_rows_clean.csv")

os.makedirs(os.path.dirname(OUT), exist_ok=True)

df = pd.read_csv(IN)

# Required columns
need = ["algo","layers","placement","activation","task","seed","auc"]
missing = [c for c in need if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns {missing}. Found: {list(df.columns)}")

# Clean
df["algo"] = df["algo"].astype(str).str.upper().str.strip()
df["layers"] = df["layers"].astype(str).str.lower().str.strip()
df["placement"] = df["placement"].astype(str)
df["activation"] = df["activation"].astype(str).str.lower().str.strip()
df["task"] = df["task"].astype(str)
df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
df["auc"] = pd.to_numeric(df["auc"], errors="coerce")

df = df.dropna(subset=["seed","auc"]).copy()
df["seed"] = df["seed"].astype(int)

# Build baseline: ReLU AUC per exact key
key = ["algo","layers","placement","task","seed"]

relu = (
    df[df["activation"] == "relu"][key + ["auc"]]
    .rename(columns={"auc":"auc_relu"})
)

m = df.merge(relu, on=key, how="left")

# alpha handling
if "alpha" in m.columns:
    m["alpha_key"] = m["alpha"].where(m["alpha"].notna(), "base")
    m["alpha_key"] = m["alpha_key"].astype(str)
else:
    m["alpha_key"] = "base"

out = m.rename(columns={"auc":"auc_frac"})[
    ["algo","layers","task","seed","placement",
     "activation","alpha_key",
     "auc_frac","auc_relu"]
].copy()

out.to_csv(OUT, index=False)
print("Wrote:", OUT)
