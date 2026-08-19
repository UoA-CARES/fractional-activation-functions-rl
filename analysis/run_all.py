#!/usr/bin/env python3
"""Run the authoritative portable Fractional-RL analysis pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def execute(args: list[str]) -> None:
    print()
    print("=" * 72)
    print("$", sys.executable, *args)
    print("=" * 72)

    subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Fractional-RL analysis pipeline."
    )

    parser.add_argument(
        "--results",
        required=True,
        help="Experiment root to analyse.",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing or invalid evaluation files.",
    )

    args = parser.parse_args()

    out = ROOT / "analysis_outputs"
    auc = out / "auc"

    out.mkdir(parents=True, exist_ok=True)
    auc.mkdir(parents=True, exist_ok=True)

    manifest = out / "run_manifest.csv"

    execute([
        "-m",
        "analysis.pipeline.build_manifest",
        "--root",
        args.results,
        "--out",
        str(manifest),
    ])

    auc_command = [
        "-m",
        "analysis.pipeline.extract_auc",
        "--manifest",
        str(manifest),
        "--outdir",
        str(auc),
    ]

    if args.strict:
        auc_command.append("--strict")

    execute(auc_command)

    execute([
        "-m",
        "analysis.pipeline.normalize_vs_relu",
    ])

    execute([
        "-m",
        "analysis.pipeline.pair_seeds_vs_relu",
    ])

    print()
    print("=" * 72)
    print("Fractional-RL core analysis complete")
    print("=" * 72)
    print(f"Manifest   : {manifest}")
    print(f"AUC        : {auc}")
    print(f"Normalized : {out / 'normalized'}")


if __name__ == "__main__":
    main()
