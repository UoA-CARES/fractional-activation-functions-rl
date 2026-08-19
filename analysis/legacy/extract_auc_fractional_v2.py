import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_activation(x: str) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "unknown"
    s = str(x).strip().lower()
    s = s.replace(" ", "").replace("_", "").replace("-", "")

    # Baselines
    if s in {"relu"}:
        return "relu"
    if s in {"lrelu", "leakyrelu", "leaky"}:
        return "lrelu"
    if s in {"prelu", "parametricrelu"}:
        return "prelu"

    # Fractional family
    if s in {"frelu"}:
        return "frelu"
    if s in {"flrelu", "fractionalleakyrelu"}:
        return "flrelu"
    if s in {"fprelu", "fractionalprelu"}:
        return "fprelu"

    return s


def trapezoid_auc(x: np.ndarray, y: np.ndarray) -> float:
    # assumes x sorted
    if len(x) < 2:
        return float("nan")
    return float(np.trapz(y, x))


def normalized_auc(steps: np.ndarray, values: np.ndarray, budget: float) -> float:
    """
    Normalized AUC on [0, budget], divided by budget.
    If steps do not reach budget, we clip to last observed step.
    """
    if len(steps) == 0:
        return float("nan")

    # sort and deduplicate by step keeping last
    order = np.argsort(steps)
    steps = steps[order]
    values = values[order]

    # dedup steps
    df = pd.DataFrame({"s": steps, "v": values}).groupby("s", as_index=False).last()
    steps = df["s"].to_numpy()
    values = df["v"].to_numpy()

    # clip to [0, budget]
    steps = np.clip(steps, 0, budget)

    # ensure starts at 0 for proper area
    if steps[0] > 0:
        steps = np.insert(steps, 0, 0.0)
        values = np.insert(values, 0, values[0])

    # ensure ends at budget by extending last value
    if steps[-1] < budget:
        steps = np.append(steps, budget)
        values = np.append(values, values[-1])

    auc = trapezoid_auc(steps, values)
    return auc / float(budget)


def find_eval_files(root: Path, pattern: str) -> list[Path]:
    return list(root.rglob(pattern))


def parse_metadata_from_path(p: Path) -> dict:
    """
    Adjust this to your directory naming.
    This tries to infer task, algo, placement, activation, alpha, seed from the run path.
    """
    s = str(p)

    # Task: look for ".../Ant-v4/..." or ".../Walker-Walk/..."
    task = None
    m = re.search(r"/(Ant-v4|HalfCheetah-v4|Humanoid-v4|Walker2d-v4|Cartpole-Swingup|Cheetah-Run|Finger-Spin|Ball-in-Cup|Walker-Walk)/", s)
    if m:
        task = m.group(1)

    # Algo
    algo = None
    if re.search(r"/SAC/", s, re.IGNORECASE):
        algo = "SAC"
    if re.search(r"/TD3/", s, re.IGNORECASE):
        algo = "TD3"

    # Placement
    placement = None
    m = re.search(r"(all-actor|all-critic|first-actor|first-critic|actor|critic)", s)
    if m:
        placement = m.group(1)

    # Activation and alpha
    activation = None
    alpha = np.nan
    m = re.search(r"(fprelu|flrelu|frelu|prelu|lrelu|relu)", s, re.IGNORECASE)
    if m:
        activation = normalize_activation(m.group(1))
    m = re.search(r"alpha([0-9]*\.?[0-9]+)", s, re.IGNORECASE)
    if m:
        alpha = float(m.group(1))

    # Seed
    seed = None
    m = re.search(r"seed([0-9]+)", s, re.IGNORECASE)
    if m:
        seed = int(m.group(1))

    return {
        "task": task,
        "algo": algo,
        "placement": placement,
        "activation": activation,
        "alpha": alpha,
        "seed": seed,
        "run_dir": str(p.parent),
    }


def load_eval_csv(csv_path: Path, step_col: str, value_col: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    if step_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"{csv_path} missing columns. Have {list(df.columns)} expected {step_col}, {value_col}")
    steps = pd.to_numeric(df[step_col], errors="coerce").dropna().to_numpy()
    vals = pd.to_numeric(df[value_col], errors="coerce").dropna().to_numpy()
    n = min(len(steps), len(vals))
    return steps[:n], vals[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Root directory containing run folders")
    ap.add_argument("--eval_pattern", type=str, default="*eval*.csv", help="Glob pattern for eval CSV files")
    ap.add_argument("--step_col", type=str, default="step", help="Step column name in eval CSV")
    ap.add_argument("--value_col", type=str, default="return", help="Eval metric column name in eval CSV")
    ap.add_argument("--budget", type=float, required=True, help="Training budget steps for normalized AUC")
    ap.add_argument("--out", type=str, default="outputs/auc/auc_long.csv")
    args = ap.parse_args()

    root = Path(args.root)
    files = find_eval_files(root, args.eval_pattern)
    if not files:
        raise SystemExit(f"No eval files found under {root} with pattern {args.eval_pattern}")

    rows = []
    for f in files:
        try:
            meta = parse_metadata_from_path(f)
            steps, vals = load_eval_csv(f, args.step_col, args.value_col)
            aucn = normalized_auc(steps, vals, args.budget)
            rows.append({
                **meta,
                "eval_file": str(f),
                "auc_norm": aucn,
                "n_points": int(len(steps)),
                "metric_name": args.value_col,
            })
        except Exception as e:
            rows.append({
                "eval_file": str(f),
                "error": str(e),
            })

    out_df = pd.DataFrame(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    # Diagnostics that catch your exact issue
    ok = out_df.dropna(subset=["auc_norm"])
    print(f"Wrote: {out_path}")
    print(f"Total files: {len(out_df)} | OK rows: {len(ok)} | Error rows: {len(out_df) - len(ok)}")

    if "activation" in ok.columns:
        print("Activation counts:")
        print(ok["activation"].value_counts(dropna=False).to_string())

        missing_baselines = [a for a in ["lrelu", "prelu"] if a not in set(ok["activation"].dropna().unique())]
        if missing_baselines:
            print("WARNING: Missing baseline activations in extracted output:", missing_baselines)
            print("This usually means your run paths or eval files do not contain the activation name, or parsing needs adjustment.")

    # Show a few rows with missing metadata so you can fix parsing fast
    bad_meta = ok[ok["task"].isna() | ok["algo"].isna() | ok["placement"].isna() | ok["activation"].isna()]
    if len(bad_meta) > 0:
        print("\nRows with missing metadata (first 10):")
        cols = [c for c in ["eval_file", "task", "algo", "placement", "activation", "alpha", "run_dir"] if c in bad_meta.columns]
        print(bad_meta[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
