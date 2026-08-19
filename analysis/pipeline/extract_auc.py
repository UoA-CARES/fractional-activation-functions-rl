#!/usr/bin/env python3
"""
Extract per-seed and aggregated AUC metrics from the Fractional-RL run manifest.

Inputs:
  - scripts/manifests/run_manifest.csv
  - <seed_dir>/data/eval.csv or eval_csv path from the manifest

Outputs:
  - analysis_outputs/auc/auc_by_seed.csv
  - analysis_outputs/auc/auc_by_run.csv

This version supports all activations:
  ReLU, LReLU, PReLU, Swish, GELU,
  FReLU, FLReLU, FPReLU, FSwish, FGELU

Usage:
  cd /path/to/experiment/root

  python3 scripts/extract_auc_from_manifest.py

  or explicitly:

  python3 scripts/extract_auc_from_manifest.py \
    --manifest scripts/manifests/run_manifest.csv \
    --outdir outputs/analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd


STEP_CANDIDATES = [
    "env_step",
    "environment_step",
    "total_steps",
    "total_step",
    "timestep",
    "timesteps",
    "step",
    "steps",
    "global_step",
]

RETURN_CANDIDATES = [
    "eval/return",
    "eval_return",
    "evaluation_return",
    "return",
    "returns",
    "episode_return",
    "episodic_return",
    "episode_reward",
    "episode_rewards",
    "mean_return",
    "avg_return",
    "reward",
    "rewards",
    "eval_reward",
    "eval/episode_reward",
    "score",
    "scores",
]


ALL_ACTIVATIONS = [
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

FRACTIONAL_ACTIVATIONS = [
    "frelu",
    "flrelu",
    "fprelu",
    "fswish",
    "fgelu",
]


def _find_first_existing_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
) -> Optional[str]:
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}

    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    return None


def detect_step_and_return_columns(df: pd.DataFrame) -> Tuple[str, str]:
    step_col = _find_first_existing_column(df, STEP_CANDIDATES)
    ret_col = _find_first_existing_column(df, RETURN_CANDIDATES)

    if step_col is None:
        raise ValueError(f"Could not detect step column. Columns={list(df.columns)}")

    if ret_col is None:
        raise ValueError(f"Could not detect return column. Columns={list(df.columns)}")

    return step_col, ret_col


def trapezoid_auc(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    return float(np.trapz(y, x))


def auc_up_to_cutoff(x: np.ndarray, y: np.ndarray, cutoff: float) -> float:
    if len(x) < 2:
        return float("nan")

    if cutoff <= x[0]:
        return 0.0

    if x[-1] <= cutoff:
        return trapezoid_auc(x, y)

    idx = np.searchsorted(x, cutoff, side="right") - 1
    idx = int(np.clip(idx, 0, len(x) - 2))

    x0, x1 = x[idx], x[idx + 1]
    y0, y1 = y[idx], y[idx + 1]

    if x1 == x0:
        y_cut = y0
    else:
        t = (cutoff - x0) / (x1 - x0)
        y_cut = y0 + t * (y1 - y0)

    x_new = np.concatenate([x[: idx + 1], np.array([cutoff], dtype=float)])
    y_new = np.concatenate([y[: idx + 1], np.array([y_cut], dtype=float)])

    return trapezoid_auc(x_new, y_new)


def compute_metrics(eval_csv: Path, step_cutoff: Optional[float]) -> dict:
    df = pd.read_csv(eval_csv)

    step_col, ret_col = detect_step_and_return_columns(df)

    sub = df[[step_col, ret_col]].copy()
    sub = sub.rename(columns={step_col: "step", ret_col: "ret"})
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
    sub["step"] = pd.to_numeric(sub["step"], errors="coerce")
    sub["ret"] = pd.to_numeric(sub["ret"], errors="coerce")
    sub = sub.dropna()
    sub = sub.sort_values("step")

    x = sub["step"].to_numpy(dtype=float)
    y = sub["ret"].to_numpy(dtype=float)

    # Drop duplicate steps and keep the last value.
    if len(x) > 1:
        dd = pd.DataFrame({"step": x, "ret": y})
        dd = dd.drop_duplicates("step", keep="last")
        x = dd["step"].to_numpy(dtype=float)
        y = dd["ret"].to_numpy(dtype=float)

    out = {
        "auc": trapezoid_auc(x, y),
        "final_return": float(y[-1]) if len(y) else float("nan"),
        "n_points": int(len(x)),
        "first_step": float(x[0]) if len(x) else float("nan"),
        "last_step": float(x[-1]) if len(x) else float("nan"),
    }

    if step_cutoff is not None:
        out["auc_cutoff"] = auc_up_to_cutoff(x, y, float(step_cutoff))
    else:
        out["auc_cutoff"] = np.nan

    return out


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--manifest",
        type=str,
        default="analysis_outputs/run_manifest.csv",
        help="Path to run manifest CSV.",
    )

    ap.add_argument(
        "--outdir",
        type=str,
        default="analysis_outputs/auc",
        help="Output directory.",
    )

    ap.add_argument(
        "--eval_relpath",
        type=str,
        default="data/eval.csv",
        help="Relative eval path under seed_dir if eval_csv is not present in manifest.",
    )

    ap.add_argument(
        "--step_cutoff",
        type=float,
        default=None,
        help="Optional cutoff step for truncated AUC.",
    )

    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail fast instead of skipping files with errors.",
    )

    args = ap.parse_args()

    mf_path = Path(args.manifest).expanduser()
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    if not mf_path.exists():
        raise FileNotFoundError(f"Manifest not found: {mf_path}")

    mf = pd.read_csv(mf_path)

    if "seed_dir" not in mf.columns:
        raise ValueError("Manifest must contain column: seed_dir")

    if "activation" in mf.columns:
        mf["activation"] = mf["activation"].astype(str).str.lower()

    if "placement" in mf.columns:
        mf["placement"] = mf["placement"].fillna("NA")

    unknown_activations = sorted(
        set(mf["activation"].dropna().unique()) - set(ALL_ACTIVATIONS)
    )

    if unknown_activations:
        print("[WARN] Unknown activations found in manifest:")
        print(unknown_activations)
        print("[WARN] They will still be kept in the output.")

    rows = []
    errors = []

    for i, r in mf.iterrows():
        if i % 500 == 0:
            print(f"Processing {i}/{len(mf)}", flush=True)
        seed_dir = Path(str(r["seed_dir"])).expanduser()

        # Prefer eval_csv from manifest because it is the most reliable.
        if "eval_csv" in mf.columns and pd.notna(r.get("eval_csv", None)):
            eval_csv = Path(str(r["eval_csv"])).expanduser()
        else:
            eval_csv = seed_dir / args.eval_relpath

        if not eval_csv.exists():
            msg = f"Missing eval.csv: {eval_csv}"
            if args.strict:
                raise FileNotFoundError(msg)
            errors.append(msg)
            continue

        try:
            metrics = compute_metrics(eval_csv, args.step_cutoff)
            out = r.to_dict()
            out["eval_csv"] = str(eval_csv)
            out.update(metrics)
            rows.append(out)
        except Exception as e:
            msg = f"Failed {eval_csv}: {type(e).__name__}: {e}"
            if args.strict:
                raise
            errors.append(msg)

    auc_by_seed = pd.DataFrame(rows)

    if len(auc_by_seed) == 0:
        raise RuntimeError("No AUC rows were produced. Check manifest and eval.csv paths.")

    # Keep placement clean.
    if "placement" in auc_by_seed.columns:
        auc_by_seed["placement"] = auc_by_seed["placement"].fillna("NA")

    # Important:
    # Non-fractional activations have alpha = NaN.
    # Pandas groupby drops NaN keys by default, so we replace NaN alpha with "NA".
    if "alpha" in auc_by_seed.columns:
        auc_by_seed["alpha"] = auc_by_seed["alpha"].where(
            auc_by_seed["alpha"].notna(), "NA"
        )

    auc_by_seed.to_csv(outdir / "auc_by_seed.csv", index=False)

    key_cols = ["algo", "layers", "placement", "activation", "alpha", "task"]
    key_cols = [c for c in key_cols if c in auc_by_seed.columns]

    auc_by_run = (
        auc_by_seed.groupby(key_cols, as_index=False, dropna=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            auc_n=("auc", "count"),
            final_return_mean=("final_return", "mean"),
            final_return_std=("final_return", "std"),
            final_return_n=("final_return", "count"),
            n_points_mean=("n_points", "mean"),
            first_step_min=("first_step", "min"),
            last_step_max=("last_step", "max"),
        )
    )

    auc_by_run.to_csv(outdir / "auc_by_run.csv", index=False)

    if errors:
        error_path = outdir / "auc_errors.txt"
        error_path.write_text("\n".join(errors) + "\n")
        print("Warnings:", len(errors), "See:", error_path)

    print("Wrote:", outdir / "auc_by_seed.csv")
    print("Wrote:", outdir / "auc_by_run.csv")

    print("\nSeed-level activation counts:")
    print(auc_by_seed["activation"].value_counts().sort_index())

    print("\nRun-level activation counts:")
    print(auc_by_run["activation"].value_counts().sort_index())

    if "task" in auc_by_seed.columns:
        print("\nUNKNOWN tasks:", int((auc_by_seed["task"] == "UNKNOWN").sum()))

    if "alpha" in auc_by_seed.columns:
        frac = auc_by_seed[auc_by_seed["activation"].isin(FRACTIONAL_ACTIVATIONS)]
        print(
            "Fractional missing alpha:",
            int((frac["alpha"] == "NA").sum()),
        )


if __name__ == "__main__":
    main()


