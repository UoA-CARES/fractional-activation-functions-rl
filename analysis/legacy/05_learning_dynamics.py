#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/Desktop/new-Fr")
IN_NORM = os.path.join(ROOT, "outputs", "analysis", "auc_by_seed_normalized.csv")

OUT_DIR = os.path.join(ROOT, "outputs", "paper")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_DYNAMICS = os.path.join(OUT_DIR, "table_learning_dynamics.csv")
OUT_STABILITY = os.path.join(OUT_DIR, "table_stability_summary.csv")


# Heuristics for column detection in eval CSVs
STEP_CANDIDATES = [
    "env_step", "env_steps", "total_steps", "step", "steps", "timestep", "timesteps",
    "global_step", "frame", "frames",
]
RETURN_CANDIDATES = [
    "eval_return", "return", "episode_return", "avg_return", "average_return",
    "mean_return", "reward", "eval_reward",
]


@dataclass(frozen=True)
class CurveMetrics:
    final_return: float
    steps_to_80pct: float  # NaN if not reached or not computable
    n_points: int
    first_step: float
    last_step: float


def require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing columns: {missing}\nFound: {list(df.columns)}")


def _pick_first_present(columns: list[str], candidates: list[str]) -> Optional[str]:
    colset = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in colset:
            return colset[cand.lower()]
    return None


def detect_step_and_return_cols(df: pd.DataFrame, path: str) -> Tuple[str, str]:
    step_col = _pick_first_present(list(df.columns), STEP_CANDIDATES)
    ret_col = _pick_first_present(list(df.columns), RETURN_CANDIDATES)

    # If not found, try numeric fallback: choose first monotonic-like column for step, and a non-step numeric for return
    if step_col is None or ret_col is None:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if step_col is None:
            # pick a column with many unique values and non-decreasing tendency
            best = None
            best_score = -1
            for c in num_cols:
                s = df[c].dropna().values
                if len(s) < 3:
                    continue
                score = float(len(np.unique(s)))
                if score > best_score:
                    best = c
                    best_score = score
            step_col = best

        if ret_col is None:
            # pick a numeric column different from step_col with reasonable variance
            best = None
            best_var = -1.0
            for c in num_cols:
                if c == step_col:
                    continue
                v = float(np.nanvar(df[c].values.astype(float))) if len(df[c]) else 0.0
                if v > best_var:
                    best = c
                    best_var = v
            ret_col = best

    if step_col is None or ret_col is None:
        raise ValueError(
            f"Could not detect step/return columns in eval csv: {path}\n"
            f"Columns: {list(df.columns)}"
        )

    return step_col, ret_col


def compute_steps_to_target(steps: np.ndarray, returns: np.ndarray, frac: float = 0.8) -> float:
    """
    Define target as: start + frac * (final - start).
    Then return first step where curve crosses target in the direction of improvement.
    """
    if len(steps) == 0 or len(returns) == 0:
        return float("nan")

    start = float(returns[0])
    final = float(returns[-1])
    target = start + frac * (final - start)

    improving_up = final >= start
    if improving_up:
        idx = np.where(returns >= target)[0]
    else:
        idx = np.where(returns <= target)[0]

    if idx.size == 0:
        return float("nan")

    return float(steps[int(idx[0])])


def load_curve_metrics(eval_csv_path: str) -> CurveMetrics:
    df = pd.read_csv(eval_csv_path)
    if df.empty:
        return CurveMetrics(final_return=float("nan"), steps_to_80pct=float("nan"),
                            n_points=0, first_step=float("nan"), last_step=float("nan"))

    step_col, ret_col = detect_step_and_return_cols(df, eval_csv_path)

    d = df[[step_col, ret_col]].dropna().copy()
    if d.empty:
        return CurveMetrics(final_return=float("nan"), steps_to_80pct=float("nan"),
                            n_points=0, first_step=float("nan"), last_step=float("nan"))

    d = d.sort_values(step_col)
    steps = d[step_col].to_numpy(dtype=float)
    rets = d[ret_col].to_numpy(dtype=float)

    final_return = float(rets[-1])
    steps_to_80 = compute_steps_to_target(steps, rets, frac=0.8)

    return CurveMetrics(
        final_return=final_return,
        steps_to_80pct=steps_to_80,
        n_points=int(len(d)),
        first_step=float(steps[0]),
        last_step=float(steps[-1]),
    )


def safe_cv(mean: float, std: float) -> float:
    if np.isnan(mean) or np.isnan(std):
        return float("nan")
    denom = abs(mean)
    if denom < 1e-12:
        return float("nan")
    return float(std / denom)


def main() -> None:
    if not os.path.exists(IN_NORM):
        raise FileNotFoundError(f"Missing input: {IN_NORM}")

    df = pd.read_csv(IN_NORM)

    # We need eval_csv and config identifiers
    require_cols(
        df,
        ["algo", "task", "layers", "placement", "activation", "seed", "eval_csv"],
        "auc_by_seed_normalized.csv",
    )
    require_cols(df, ["delta_pct_vs_relu"], "auc_by_seed_normalized.csv")

    # Normalize strings
    df["algo"] = df["algo"].astype(str).str.upper()
    df["task"] = df["task"].astype(str)
    df["layers"] = df["layers"].astype(str).str.lower()
    df["placement"] = df["placement"].astype(str).str.lower()
    df["activation"] = df["activation"].astype(str).str.lower()

    # Ensure alpha_key exists
    if "alpha_key" not in df.columns:
        if "alpha" in df.columns:
            df["alpha_key"] = df["alpha"].astype(object).where(df["alpha"].notna(), "NA")
        else:
            df["alpha_key"] = "NA"
            df["alpha"] = pd.NA

    # Force 1layer placement to none for stability
    df.loc[df["layers"] == "1layer", "placement"] = "none"

    # Compute per-row curve metrics using caching
    cache: Dict[str, CurveMetrics] = {}

    finals = []
    steps80 = []
    npts = []
    fstep = []
    lstep = []

    for p in df["eval_csv"].astype(str).tolist():
        if p not in cache:
            if not os.path.exists(p):
                cache[p] = CurveMetrics(final_return=float("nan"), steps_to_80pct=float("nan"),
                                        n_points=0, first_step=float("nan"), last_step=float("nan"))
            else:
                cache[p] = load_curve_metrics(p)
        m = cache[p]
        finals.append(m.final_return)
        steps80.append(m.steps_to_80pct)
        npts.append(m.n_points)
        fstep.append(m.first_step)
        lstep.append(m.last_step)

    df["final_return_from_eval"] = finals
    df["steps_to_80pct"] = steps80
    df["eval_n_points"] = npts
    df["eval_first_step"] = fstep
    df["eval_last_step"] = lstep

    # Table 1: per config aggregated across seeds
    cfg_cols = ["algo", "task", "layers", "placement", "activation", "alpha_key"]
    if "alpha" in df.columns:
        # keep alpha numeric for readability where present
        pass

    agg = (
        df.groupby(cfg_cols, as_index=False)
        .agg(
            mean_delta_pct=("delta_pct_vs_relu", "mean"),
            mean_final_return=("final_return_from_eval", "mean"),
            std_final_return=("final_return_from_eval", "std"),
            mean_steps_to_80pct=("steps_to_80pct", "mean"),
            std_steps_to_80pct=("steps_to_80pct", "std"),
            n_seeds=("seed", "nunique"),
        )
    )

    agg["mean_delta_pct"] = agg["mean_delta_pct"].round(1)
    agg["mean_final_return"] = agg["mean_final_return"].round(2)
    agg["std_final_return"] = agg["std_final_return"].round(2)
    agg["mean_steps_to_80pct"] = agg["mean_steps_to_80pct"].round(0)
    agg["std_steps_to_80pct"] = agg["std_steps_to_80pct"].round(0)

    # CV for final return
    agg["cv_final_return"] = [
        round(safe_cv(m, s), 3) if not np.isnan(safe_cv(m, s)) else float("nan")
        for m, s in zip(agg["mean_final_return"].astype(float), agg["std_final_return"].astype(float))
    ]

    agg.to_csv(OUT_DYNAMICS, index=False)
    print("Wrote:", OUT_DYNAMICS, "rows:", len(agg))

    # Table 2: stability summary across tasks for each activation/config family
    # Keep it simple: average across tasks for each (algo, layers, placement, activation)
    stab = (
        agg.groupby(["algo", "layers", "placement", "activation"], as_index=False)
        .agg(
            avg_mean_delta_pct=("mean_delta_pct", "mean"),
            avg_steps_to_80pct=("mean_steps_to_80pct", "mean"),
            avg_cv_final_return=("cv_final_return", "mean"),
            n_tasks=("task", "nunique"),
        )
    )
    stab["avg_mean_delta_pct"] = stab["avg_mean_delta_pct"].round(1)
    stab["avg_steps_to_80pct"] = stab["avg_steps_to_80pct"].round(0)
    stab["avg_cv_final_return"] = stab["avg_cv_final_return"].round(3)

    stab = stab.sort_values(["algo", "layers", "avg_mean_delta_pct"], ascending=[True, True, False])
    stab.to_csv(OUT_STABILITY, index=False)
    print("Wrote:", OUT_STABILITY, "rows:", len(stab))

    # Basic tests
    print("[T14] dynamics NaNs in final_return_from_eval:", int(df["final_return_from_eval"].isna().sum()))
    print("[T15] steps_to_80pct NaNs:", int(df["steps_to_80pct"].isna().sum()))
    print("[T15] steps_to_80pct min/max (finite):",
          float(np.nanmin(df["steps_to_80pct"].values)) if np.isfinite(df["steps_to_80pct"]).any() else float("nan"),
          float(np.nanmax(df["steps_to_80pct"].values)) if np.isfinite(df["steps_to_80pct"]).any() else float("nan"),
          )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e), file=sys.stderr)
        raise
