from __future__ import annotations

import os
import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/Desktop/new-Fr")
IN_SEED = os.path.join(ROOT, "outputs/analysis/auc_by_seed_normalized.csv")

OUT_PAPER = os.path.join(ROOT, "outputs/paper")
OUT_TABLES = os.path.join(OUT_PAPER, "tables")
os.makedirs(OUT_PAPER, exist_ok=True)
os.makedirs(OUT_TABLES, exist_ok=True)

OUT_CSV = os.path.join(OUT_PAPER, "table_best_by_task_fractional.csv")
OUT_TEX = os.path.join(OUT_TABLES, "table_best_by_task_fractional.tex")

N_BOOT = 5000
CI_LO, CI_HI = 2.5, 97.5
RNG_SEED = 123

FRACTIONAL_ACTS = ["frelu", "flrelu", "fprelu", "fswish", "fgelu"]


def bootstrap_ci_mean(x: np.ndarray, n_boot: int = N_BOOT, seed: int = RNG_SEED):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n = x.size
    boots = rng.choice(x, size=(n_boot, n), replace=True).mean(axis=1)
    return (float(np.percentile(boots, CI_LO)), float(np.percentile(boots, CI_HI)))


def to_booktabs(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = list(df.columns)
    colspec = " ".join(["l"] * len(cols))
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\begin{tabular}{" + colspec + r"}")
    lines.append(r"\toprule")
    lines.append(" & ".join(cols) + r" \\")
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        lines.append(" & ".join(str(row[c]) for c in cols) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main():
    df = pd.read_csv(IN_SEED)

    required = ["algo", "task", "layers", "placement", "activation", "alpha", "seed", "delta_pct_vs_relu"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["activation"] = df["activation"].astype(str).str.lower()

    dff = df[df["activation"].isin(FRACTIONAL_ACTS)].copy()
    if dff.empty:
        raise ValueError(f"No rows found for fractional activations: {FRACTIONAL_ACTS}")

    config_cols = ["algo", "task", "layers", "placement", "activation", "alpha"]

    cfg = (
        dff.groupby(config_cols, as_index=False)["delta_pct_vs_relu"]
        .mean()
        .rename(columns={"delta_pct_vs_relu": "delta_mean"})
    )

    idx = cfg.groupby(["algo", "task"])["delta_mean"].idxmax()
    best_cfg = cfg.loc[idx].reset_index(drop=True)

    ci_lo, ci_hi = [], []
    for _, r in best_cfg.iterrows():
        mask = np.ones(len(dff), dtype=bool)
        for c in config_cols:
            if pd.isna(r[c]):
                mask &= dff[c].isna()
            else:
                mask &= (dff[c] == r[c])
        vals = dff.loc[mask, "delta_pct_vs_relu"].to_numpy(dtype=float)
        lo, hi = bootstrap_ci_mean(vals)
        ci_lo.append(lo)
        ci_hi.append(hi)

    best_cfg["ci_lo"] = ci_lo
    best_cfg["ci_hi"] = ci_hi

    out = pd.DataFrame({
        "Task": best_cfg["task"],
        "Algo": best_cfg["algo"],
        "Placement": best_cfg["placement"],
        "Activation": best_cfg["activation"],
        r"$\alpha$": best_cfg["alpha"].astype(str),
        r"$\Delta\%$": best_cfg["delta_mean"].map(lambda x: f"{x:+.1f}"),
        "95\\% CI": [f"[{lo:.1f}, {hi:.1f}]" for lo, hi in zip(best_cfg["ci_lo"], best_cfg["ci_hi"])],
    }).sort_values(["Algo", "Task"]).reset_index(drop=True)

    out.to_csv(OUT_CSV, index=False)

    tex = to_booktabs(
        out,
        caption="Best fractional configuration per task and algorithm (selected from fractional variants only).",
        label="tab:best_fractional_by_task",
    )

    with open(OUT_TEX, "w", encoding="utf-8") as f:
        f.write(tex)

    if len(out) != 16:
        raise ValueError(f"Expected 16 rows (2 algos × 8 tasks), got {len(out)}")

    print("Saved:")
    print(" -", OUT_CSV)
    print(" -", OUT_TEX)
    print("Rows:", len(out))


if __name__ == "__main__":
    main()
