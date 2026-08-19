"""Build exact seed-paired activation-vs-ReLU AUC comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="analysis_outputs/auc/auc_by_seed.csv",
    )

    parser.add_argument(
        "--output",
        default="analysis_outputs/normalized/paired_seed_rows.csv",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input)

    required = [
        "algo",
        "task",
        "layers",
        "activation",
        "placement",
        "seed",
        "auc",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["algo"] = df["algo"].astype(str).str.upper().str.strip()
    df["task"] = df["task"].astype(str).str.strip()
    df["layers"] = df["layers"].astype(str).str.lower().str.strip()
    df["activation"] = (
        df["activation"].astype(str).str.lower().str.strip()
    )
    df["placement"] = (
        df["placement"].fillna("NA").astype(str).str.lower().str.strip()
    )

    df["seed"] = pd.to_numeric(df["seed"], errors="raise").astype(int)
    df["auc"] = pd.to_numeric(df["auc"], errors="raise")

    # ---------------------------------------------------------
    # Exact ReLU seed baseline.
    #
    # ReLU is shared across placements, therefore placement is
    # deliberately NOT part of the pairing key.
    # ---------------------------------------------------------

    pair_key = [
        "algo",
        "task",
        "layers",
        "seed",
    ]

    relu = (
        df[df["activation"] == "relu"]
        [pair_key + ["auc"]]
        .rename(columns={"auc": "auc_relu"})
    )

    duplicate_relu = relu.duplicated(pair_key).sum()

    if duplicate_relu:
        raise ValueError(
            f"Found {duplicate_relu} duplicate ReLU seed baselines."
        )

    candidates = df[df["activation"] != "relu"].copy()

    paired = candidates.merge(
        relu,
        on=pair_key,
        how="left",
        validate="many_to_one",
    )

    missing_baseline = int(paired["auc_relu"].isna().sum())

    if missing_baseline:
        raise ValueError(
            f"{missing_baseline} candidate rows have no paired ReLU seed."
        )

    paired = paired.rename(
        columns={"auc": "auc_candidate"}
    )

    paired["delta_auc"] = (
        paired["auc_candidate"] - paired["auc_relu"]
    )

    paired["delta_pct_vs_relu_seed"] = (
        100.0
        * paired["delta_auc"]
        / paired["auc_relu"]
    )

    if "alpha" in paired.columns:
        paired["alpha_key"] = paired["alpha"].where(
            paired["alpha"].notna(),
            "base",
        )
    else:
        paired["alpha_key"] = "base"

    columns = [
        "algo",
        "task",
        "layers",
        "placement",
        "activation",
        "alpha_key",
        "seed",
        "auc_candidate",
        "auc_relu",
        "delta_auc",
        "delta_pct_vs_relu_seed",
        "eval_csv",
    ]

    columns = [
        c for c in columns
        if c in paired.columns
    ]

    paired = paired[columns].sort_values(
        [
            "algo",
            "task",
            "layers",
            "activation",
            "placement",
            "alpha_key",
            "seed",
        ]
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    paired.to_csv(output, index=False)

    print("=" * 65)
    print("Exact seed-paired comparisons")
    print("=" * 65)
    print(f"Candidate rows : {len(paired):,}")
    print(f"ReLU baselines : {len(relu):,}")
    print(f"Missing pairs  : {missing_baseline}")
    print(f"Output         : {output}")


if __name__ == "__main__":
    main()
