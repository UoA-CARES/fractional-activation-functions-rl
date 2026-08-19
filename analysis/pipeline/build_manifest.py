#!/usr/bin/env python3
"""
build_run_manifest.py

Builds a per-seed manifest for the Fractional Activation Study.

Folder root (default):
  /path/to/experiment/root

Expected structure:
  new-Fr/
    TD3/
      1layer/<act>/<task_alias>/<run_name>/<seed>/data/eval.csv
      2layers/<placement>/<act>/<task_alias>/<run_name>/<seed>/data/eval.csv
      2layers/r/<task_group>/<seed>/data/eval.csv                       # TD3 baseline layout A
      2layers/r/<task_group>/<run_name>/<seed>/data/eval.csv            # TD3 baseline layout B
    SAC/
      1layer/<act>/<task_alias>/<run_name>/<seed>/data/eval.csv
      2layers/<placement>/<act>/<task_alias>/<run_name>/<seed>/data/eval.csv

Output:
  scripts/manifests/run_manifest.csv  (or any path you pass with --out)

Usage:
  python3 scripts/build_run_manifest.py --out scripts/manifests/run_manifest.csv --strict_eval
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# -----------------------------
# Canonical mappings
# -----------------------------

ACT_DIR_TO_ACTIVATION: Dict[str, str] = {
    "r": "relu",
    "l": "lrelu",
    "p": "prelu",
    "s": "swish",
    "g": "gelu",
    "fr": "frelu",
    "fl": "flrelu",
    "fp": "fprelu",
    "fs": "fswish",
    "fg": "fgelu",
}

FRACTIONAL_ACTIVATIONS = {"frelu", "flrelu", "fprelu", "fswish", "fgelu"}

# 2layers placements
PLACEMENT_DIRS = {
    "all-actor",
    "all-critic",
    "all-both",
    "first-actor",
    "first-both",
    "r",  # baseline relu all layers (special TD3 layout(s))
}

# Task alias mapping
TASK_ALIAS_TO_CANON: Dict[str, str] = {
    # MuJoCo
    "ant": "Ant-v4",
    "ant-v4": "Ant-v4",
    "hopper": "Hopper-v4",
    "hopper-v4": "Hopper-v4",
    "halfcheetah": "HalfCheetah-v4",
    "halfcheetah-v4": "HalfCheetah-v4",
    "humanoid": "Humanoid-v4",
    "humanoid-v4": "Humanoid-v4",
    # DM Control
    "walker": "Walker-Walk",
    "walker-walk": "Walker-Walk",
    "walker_walk": "Walker-Walk",
    "cheetah-run": "Cheetah-Run",
    "cheetah_run": "Cheetah-Run",
    "cheetah-runner": "Cheetah-Run",
    "cheetah": "Cheetah-Run",
    "finger-spin": "Finger-Spin",
    "finger_spin": "Finger-Spin",
    "finger-s": "Finger-Spin",
    "finger": "Finger-Spin",
    # Cartpole is swingup in your project
    "cartpole": "Cartpole-Swingup",
    "cartpole-s": "Cartpole-Swingup",
    "cartpole-swingup": "Cartpole-Swingup",
    "cartpole_swingup": "Cartpole-Swingup",
    "cartpole-balance": "Cartpole-Swingup",
    "cartpole_balance": "Cartpole-Swingup",
}


# -----------------------------
# Alpha parsing
# -----------------------------

ALPHA_ANYWHERE_RE = re.compile(r"(?<!\d)(?P<alpha>0\.[1-5])(?!\d)")


def parse_alpha_from_run_name(run_name: str) -> Optional[float]:
    m = ALPHA_ANYWHERE_RE.search(run_name)
    if m:
        return float(m.group("alpha"))
    return None


# -----------------------------
# Helpers
# -----------------------------

def norm_task_alias(s: str) -> str:
    return s.strip().lower().replace(" ", "_")


def canonical_task_from_folder(task_folder_name: str) -> str:
    key = norm_task_alias(task_folder_name)
    return TASK_ALIAS_TO_CANON.get(key, "UNKNOWN")

def infer_task_from_run_name(run_name: str) -> str:
    """Infer canonical task from a run folder name by substring match."""
    s = run_name.strip().lower().replace("td3-", "").replace("sac-", "")
    s = s.replace(" ", "_")
    # Try exact canonical_task_from_folder on a few cleaned tokens
    # Then try substring match against known aliases (longest first).
    t = canonical_task_from_folder(s)
    if t != "UNKNOWN":
        return t

    keys = sorted(TASK_ALIAS_TO_CANON.keys(), key=len, reverse=True)
    for k in keys:
        if k in s:
            return TASK_ALIAS_TO_CANON[k]
    return "UNKNOWN"



def is_seed_dir(p: Path) -> bool:
    return p.is_dir() and p.name.isdigit()


def safe_exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:
        return False


@dataclass
class Row:
    algo: str
    layers: str
    placement: str
    activation: str
    alpha: Optional[float]
    task: str
    task_alias: str
    run_name: str
    run_dir: str
    seed: int
    seed_dir: str
    eval_csv: str
    train_csv: str


def build_rows(root: Path, strict_eval: bool) -> List[Row]:
    rows: List[Row] = []

    for algo_dir in ["TD3", "SAC"]:
        base_algo = root / algo_dir
        if not base_algo.exists():
            continue

        for layers_dir in ["1layer", "2layers"]:
            base_layers = base_algo / layers_dir
            if not base_layers.exists():
                continue

            # -----------------------------
            # 1layer
            # -----------------------------
            if layers_dir == "1layer":
                for act_dir in base_layers.iterdir():
                    if not act_dir.is_dir():
                        continue
                    act_key = act_dir.name.lower()
                    if act_key not in ACT_DIR_TO_ACTIVATION:
                        continue
                    activation = ACT_DIR_TO_ACTIVATION[act_key]
                    # Support two layouts under act_dir:
                    #   (A) act/task_alias/run_name/seed/...
                    #   (B) act/run_name/seed/...    (task inferred from run_name)
                    for child in act_dir.iterdir():
                        if not child.is_dir():
                            continue

                        # Layout B: act/run_name/seed/...
                        # Detect by checking if child contains digit seed dirs directly.
                        has_seed = any(p.is_dir() and p.name.isdigit() for p in child.iterdir())
                        if has_seed:
                            task_alias = child.name
                            task = infer_task_from_run_name(child.name)
                            run_dir = child
                            run_name = child.name

                            alpha = None
                            if activation in FRACTIONAL_ACTIVATIONS:
                                alpha = parse_alpha_from_run_name(run_name)

                            for seed_dir in run_dir.iterdir():
                                if not is_seed_dir(seed_dir):
                                    continue
                                seed = int(seed_dir.name)

                                eval_csv = seed_dir / "data" / "eval.csv"
                                train_csv = seed_dir / "data" / "train.csv"

                                if strict_eval and not safe_exists(eval_csv):
                                    continue

                                rows.append(
                                    Row(
                                        algo=algo_dir,
                                        layers=layers_dir,
                                        placement="NA",
                                        activation=activation,
                                        alpha=alpha,
                                        task=task,
                                        task_alias=task_alias,
                                        run_name=run_name,
                                        run_dir=str(run_dir),
                                        seed=seed,
                                        seed_dir=str(seed_dir),
                                        eval_csv=str(eval_csv),
                                        train_csv=str(train_csv),
                                    )
                                )
                            continue

                        # Layout A: act/task_alias/run_name/seed/...
                        task_dir = child
                        task_alias = task_dir.name
                        task = canonical_task_from_folder(task_alias)

                        for run_dir in task_dir.iterdir():
                            if not run_dir.is_dir():
                                continue
                            run_name = run_dir.name

                            alpha = None
                            if activation in FRACTIONAL_ACTIVATIONS:
                                alpha = parse_alpha_from_run_name(run_name)

                            for seed_dir in run_dir.iterdir():
                                if not is_seed_dir(seed_dir):
                                    continue
                                seed = int(seed_dir.name)

                                eval_csv = seed_dir / "data" / "eval.csv"
                                train_csv = seed_dir / "data" / "train.csv"

                                if strict_eval and not safe_exists(eval_csv):
                                    continue

                                rows.append(
                                    Row(
                                        algo=algo_dir,
                                        layers=layers_dir,
                                        placement="NA",
                                        activation=activation,
                                        alpha=alpha,
                                        task=task,
                                        task_alias=task_alias,
                                        run_name=run_name,
                                        run_dir=str(run_dir),
                                        seed=seed,
                                        seed_dir=str(seed_dir),
                                        eval_csv=str(eval_csv),
                                        train_csv=str(train_csv),
                                    )
                                )

            # -----------------------------
            # 2layers
            # -----------------------------
            else:
                for placement_dir in base_layers.iterdir():
                    if not placement_dir.is_dir():
                        continue

                    placement = placement_dir.name
                    if placement not in PLACEMENT_DIRS:
                        continue

                    # Special case: baseline folder "r" (TD3 has layout A or B)
                    if placement == "r":
                        activation = "relu"
                        alpha = None

                        for task_group_dir in placement_dir.iterdir():
                            if not task_group_dir.is_dir():
                                continue

                            # Examples:
                            #   TD3-cartpole-swingup
                            #   TD3-Ant-v4
                            name = task_group_dir.name.lower()
                            name = name.replace("td3-", "").replace("sac-", "")
                            task_alias = name
                            task = canonical_task_from_folder(task_alias)

                            # Detect whether seeds are directly inside task_group_dir
                            children = [p for p in task_group_dir.iterdir() if p.is_dir()]
                            direct_seeds = [p for p in children if p.name.isdigit()]

                            if direct_seeds:
                                # Layout A: 2layers/r/<task_group>/<seed>/data/eval.csv
                                run_dir = task_group_dir
                                run_name = task_group_dir.name

                                for seed_dir in direct_seeds:
                                    seed = int(seed_dir.name)

                                    eval_csv = seed_dir / "data" / "eval.csv"
                                    train_csv = seed_dir / "data" / "train.csv"

                                    if strict_eval and not safe_exists(eval_csv):
                                        continue

                                    rows.append(
                                        Row(
                                            algo=algo_dir,
                                            layers=layers_dir,
                                            placement=placement,
                                            activation=activation,
                                            alpha=alpha,
                                            task=task,
                                            task_alias=task_alias,
                                            run_name=run_name,
                                            run_dir=str(run_dir),
                                            seed=seed,
                                            seed_dir=str(seed_dir),
                                            eval_csv=str(eval_csv),
                                            train_csv=str(train_csv),
                                        )
                                    )
                            else:
                                # Layout B: 2layers/r/<task_group>/<run_name>/<seed>/data/eval.csv
                                for run_dir in task_group_dir.iterdir():
                                    if not run_dir.is_dir():
                                        continue
                                    run_name = run_dir.name

                                    for seed_dir in run_dir.iterdir():
                                        if not is_seed_dir(seed_dir):
                                            continue
                                        seed = int(seed_dir.name)

                                        eval_csv = seed_dir / "data" / "eval.csv"
                                        train_csv = seed_dir / "data" / "train.csv"

                                        if strict_eval and not safe_exists(eval_csv):
                                            continue

                                        rows.append(
                                            Row(
                                                algo=algo_dir,
                                                layers=layers_dir,
                                                placement=placement,
                                                activation=activation,
                                                alpha=alpha,
                                                task=task,
                                                task_alias=task_alias,
                                                run_name=run_name,
                                                run_dir=str(run_dir),
                                                seed=seed,
                                                seed_dir=str(seed_dir),
                                                eval_csv=str(eval_csv),
                                                train_csv=str(train_csv),
                                            )
                                        )

                        continue  # done with placement "r"

                    # Normal case: 2layers/<placement>/<act>/<task_alias>/<run>/<seed>/data/eval.csv
                    for act_dir in placement_dir.iterdir():
                        if not act_dir.is_dir():
                            continue
                        act_key = act_dir.name.lower()
                        if act_key not in ACT_DIR_TO_ACTIVATION:
                            continue
                        activation = ACT_DIR_TO_ACTIVATION[act_key]
                        # Support two layouts under act_dir:
                        #   (A) act/task_alias/run_name/seed/...
                        #   (B) act/run_name/seed/...    (task inferred from run_name)
                        for child in act_dir.iterdir():
                            if not child.is_dir():
                                continue

                            has_seed = any(p.is_dir() and p.name.isdigit() for p in child.iterdir())
                            if has_seed:
                                task_alias = child.name
                                task = infer_task_from_run_name(child.name)
                                run_dir = child
                                run_name = child.name

                                alpha = None
                                if activation in FRACTIONAL_ACTIVATIONS:
                                    alpha = parse_alpha_from_run_name(run_name)

                                for seed_dir in run_dir.iterdir():
                                    if not is_seed_dir(seed_dir):
                                        continue
                                    seed = int(seed_dir.name)

                                    eval_csv = seed_dir / "data" / "eval.csv"
                                    train_csv = seed_dir / "data" / "train.csv"

                                    if strict_eval and not safe_exists(eval_csv):
                                        continue

                                    rows.append(
                                        Row(
                                            algo=algo_dir,
                                            layers=layers_dir,
                                            placement=placement,
                                            activation=activation,
                                            alpha=alpha,
                                            task=task,
                                            task_alias=task_alias,
                                            run_name=run_name,
                                            run_dir=str(run_dir),
                                            seed=seed,
                                            seed_dir=str(seed_dir),
                                            eval_csv=str(eval_csv),
                                            train_csv=str(train_csv),
                                        )
                                    )
                                continue

                            task_dir = child
                            task_alias = task_dir.name
                            task = canonical_task_from_folder(task_alias)

                            for run_dir in task_dir.iterdir():
                                if not run_dir.is_dir():
                                    continue
                                run_name = run_dir.name

                                alpha = None
                                if activation in FRACTIONAL_ACTIVATIONS:
                                    alpha = parse_alpha_from_run_name(run_name)

                                for seed_dir in run_dir.iterdir():
                                    if not is_seed_dir(seed_dir):
                                        continue
                                    seed = int(seed_dir.name)

                                    eval_csv = seed_dir / "data" / "eval.csv"
                                    train_csv = seed_dir / "data" / "train.csv"

                                    if strict_eval and not safe_exists(eval_csv):
                                        continue

                                    rows.append(
                                        Row(
                                            algo=algo_dir,
                                            layers=layers_dir,
                                            placement=placement,
                                            activation=activation,
                                            alpha=alpha,
                                            task=task,
                                            task_alias=task_alias,
                                            run_name=run_name,
                                            run_dir=str(run_dir),
                                            seed=seed,
                                            seed_dir=str(seed_dir),
                                            eval_csv=str(eval_csv),
                                            train_csv=str(train_csv),
                                        )
                                    )

    return rows



# ---------------------------------------------------------------------
# Standalone Fractional-RL result layout
# ---------------------------------------------------------------------

STANDALONE_ACTIVATIONS = {
    "relu": "relu",
    "lrelu": "lrelu",
    "prelu": "prelu",
    "gelu": "gelu",
    "swish": "swish",
    "frelu": "frelu",
    "flrelu": "flrelu",
    "fprelu": "fprelu",
    "fswish": "fswish",
    "fgelu": "fgelu",
}


def build_standalone_rows(root: Path) -> List[Row]:
    """Scan results produced by the standalone scripts/train.py.

    Expected layout:

      results/
        TD3/
          HalfCheetah-v4/
            FReLU/
              alpha_0.2/
                2layer/
                  all_both/
                    seed_10/
                      config.json
                      train.csv
                      eval.csv
    """

    rows: List[Row] = []

    for algo in ("TD3", "SAC"):
        algo_dir = root / algo

        if not algo_dir.exists():
            continue

        for task_dir in algo_dir.iterdir():
            if not task_dir.is_dir():
                continue

            task_alias = task_dir.name
            task = canonical_task_from_folder(task_alias)

            if task == "UNKNOWN":
                continue

            for act_dir in task_dir.iterdir():
                if not act_dir.is_dir():
                    continue

                activation = STANDALONE_ACTIVATIONS.get(
                    act_dir.name.lower()
                )

                if activation is None:
                    continue

                for alpha_dir in act_dir.iterdir():
                    if not alpha_dir.is_dir():
                        continue

                    alpha = None

                    if activation in FRACTIONAL_ACTIVATIONS:
                        if not alpha_dir.name.startswith("alpha_"):
                            continue

                        value = alpha_dir.name.removeprefix("alpha_")

                        try:
                            alpha = float(value)
                        except ValueError:
                            continue

                    for layer_dir in alpha_dir.iterdir():
                        if not layer_dir.is_dir():
                            continue

                        if layer_dir.name == "1layer":
                            layers = "1layer"
                        elif layer_dir.name in {"2layer", "2layers"}:
                            layers = "2layers"
                        else:
                            continue

                        for placement_dir in layer_dir.iterdir():
                            if not placement_dir.is_dir():
                                continue

                            placement_name = (
                                placement_dir.name.replace("_", "-")
                            )

                            if layers == "1layer":
                                placement = "NA"
                            elif activation == "relu":
                                # ReLU is the shared two-layer reference.
                                placement = "r"
                            else:
                                placement = placement_name

                            for seed_dir in placement_dir.iterdir():
                                if not seed_dir.is_dir():
                                    continue

                                if not seed_dir.name.startswith("seed_"):
                                    continue

                                try:
                                    seed = int(
                                        seed_dir.name.removeprefix("seed_")
                                    )
                                except ValueError:
                                    continue

                                eval_csv = seed_dir / "eval.csv"
                                train_csv = seed_dir / "train.csv"

                                run_dir = placement_dir

                                run_name = (
                                    f"{algo}-{task_alias}-"
                                    f"{act_dir.name}-{alpha_dir.name}-"
                                    f"{layer_dir.name}-{placement_dir.name}"
                                )

                                rows.append(
                                    Row(
                                        algo=algo,
                                        layers=layers,
                                        placement=placement,
                                        activation=activation,
                                        alpha=alpha,
                                        task=task,
                                        task_alias=task_alias,
                                        run_name=run_name,
                                        run_dir=str(run_dir),
                                        seed=seed,
                                        seed_dir=str(seed_dir),
                                        eval_csv=str(eval_csv),
                                        train_csv=str(train_csv),
                                    )
                                )

    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build the authoritative per-seed Fractional-RL run manifest."
        )
    )

    ap.add_argument(
        "--root",
        type=str,
        default="results",
        help=(
            "Experiment-result root. Supports both the historical paper "
            "layout and the standalone repository layout."
        ),
    )

    ap.add_argument(
        "--out",
        type=str,
        default="analysis_outputs/run_manifest.csv",
    )

    ap.add_argument(
        "--strict-eval",
        "--strict_eval",
        dest="strict_eval",
        action="store_true",
        help="Write only runs with an existing eval.csv.",
    )

    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_path = Path(args.out).expanduser()

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Historical paper layout.
    paper_layout_rows = build_rows(
        root,
        strict_eval=False,
    )

    # New standalone-repository layout.
    standalone_rows = build_standalone_rows(root)

    records = []

    for row in paper_layout_rows:
        record = row.__dict__.copy()
        record["layout"] = "paper_layout"
        records.append(record)

    for row in standalone_rows:
        record = row.__dict__.copy()
        record["layout"] = "standalone"
        records.append(record)

    if not records:
        raise SystemExit(
            f"No experiment runs were found under: {root}"
        )

    df_all = pd.DataFrame(records)

    # Avoid accidental double discovery.
    identity = [
        "algo",
        "layers",
        "placement",
        "activation",
        "alpha",
        "task",
        "seed",
        "eval_csv",
    ]

    df_all = (
        df_all
        .drop_duplicates(subset=identity)
        .reset_index(drop=True)
    )

    df_all["eval_exists"] = df_all["eval_csv"].map(
        lambda value: Path(value).exists()
    )

    df_all["train_exists"] = df_all["train_csv"].map(
        lambda value: Path(value).exists()
    )

    df_all["complete"] = (
        df_all["eval_exists"]
        & df_all["train_exists"]
    )

    # Always export missing-evaluation diagnostics.
    missing_eval = df_all[~df_all["eval_exists"]].copy()

    missing_path = out_path.with_name(
        out_path.stem + "_missing_eval.csv"
    )

    missing_eval.to_csv(
        missing_path,
        index=False,
    )

    if args.strict_eval:
        df = df_all[df_all["eval_exists"]].copy()
    else:
        df = df_all.copy()

    sort_cols = [
        "algo",
        "layers",
        "task",
        "activation",
        "alpha",
        "placement",
        "seed",
    ]

    df = df.sort_values(
        sort_cols,
        na_position="first",
    ).reset_index(drop=True)

    df.to_csv(
        out_path,
        index=False,
    )

    print("=" * 68)
    print("Fractional-RL run manifest")
    print("=" * 68)
    print(f"Root             : {root}")
    print(f"Discovered rows  : {len(df_all):,}")
    print(f"Written rows     : {len(df):,}")
    print(f"Usable eval.csv  : {int(df_all['eval_exists'].sum()):,}")
    print(f"Missing eval.csv : {int((~df_all['eval_exists']).sum()):,}")
    print(f"Missing train.csv: {int((~df_all['train_exists']).sum()):,}")
    print(f"Manifest         : {out_path}")
    print(f"Missing report   : {missing_path}")

    print()
    print("Algorithms :", df_all["algo"].value_counts().to_dict())
    print("Layers     :", df_all["layers"].value_counts().to_dict())
    print("Activations:", df_all["activation"].value_counts().to_dict())
    print("Tasks      :", df_all["task"].value_counts().to_dict())

    frac = df_all[
        df_all["activation"].isin(FRACTIONAL_ACTIVATIONS)
    ]

    print(
        "Alphas     :",
        sorted(
            frac["alpha"]
            .dropna()
            .unique()
            .tolist()
        ),
    )

    unknown = int((df_all["task"] == "UNKNOWN").sum())

    if unknown:
        print(f"WARNING: {unknown} run(s) have UNKNOWN task labels.")


if __name__ == "__main__":
    main()
