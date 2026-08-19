#!/usr/bin/env python3
"""Run or preview one activation sweep across alpha values and seeds.

Examples
--------
Preview the paper FReLU sweep:

    python scripts/run_sweep.py \
        --algo TD3 \
        --env HalfCheetah-v4 \
        --activation FReLU \
        --layers 2 \
        --placement all_both

Execute it by adding:

    --run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "scripts" / "train.py"

OFFICIAL_FRACTIONAL = {
    "frelu",
    "flrelu",
    "fprelu",
    "fswish",
    "fgelu",
}

DEFAULT_ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5]
DEFAULT_SEEDS = [10, 20, 30, 40, 50]

PLACEMENTS = {
    "all_both",
    "all_actor",
    "all_critic",
    "first_both",
    "first_actor",
    "first_critic",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview or execute a Fractional-RL alpha/seed sweep."
    )

    parser.add_argument("--algo", choices=["TD3", "SAC"], required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--activation", required=True)

    parser.add_argument(
        "--layers",
        type=int,
        choices=[1, 2],
        default=2,
    )
    parser.add_argument(
        "--placement",
        default="all_both",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=DEFAULT_ALPHAS,
    )

    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--output", default="results")

    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute jobs. Without this flag commands are only printed.",
    )

    args = parser.parse_args()

    placement = args.placement.replace("-", "_")

    if args.layers == 1:
        placement = "all_both"

    if placement not in PLACEMENTS:
        raise ValueError(
            f"Unknown placement '{placement}'. "
            f"Choose from {sorted(PLACEMENTS)}."
        )

    is_fractional = args.activation.lower() in OFFICIAL_FRACTIONAL

    alpha_values = args.alphas if is_fractional else [None]

    jobs: list[list[str]] = []

    for alpha in alpha_values:
        for seed in args.seeds:
            command = [
                sys.executable,
                str(TRAIN),
                "--algo",
                args.algo,
                "--env",
                args.env,
                "--activation",
                args.activation,
                "--layers",
                str(args.layers),
                "--placement",
                placement,
                "--seed",
                str(seed),
                "--steps",
                str(args.steps),
                "--output",
                args.output,
            ]

            if alpha is not None:
                command += ["--alpha", str(alpha)]

            jobs.append(command)

    print(
        f"# {len(jobs)} jobs | "
        f"{args.algo} | {args.env} | {args.activation} | "
        f"{args.layers} layer(s) | {placement}"
    )

    for index, command in enumerate(jobs, start=1):
        print(
            f"[{index:03d}/{len(jobs):03d}] "
            + " ".join(command),
            flush=True,
        )

        if args.run:
            subprocess.run(
                command,
                cwd=ROOT,
                check=True,
            )


if __name__ == "__main__":
    main()
