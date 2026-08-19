#!/usr/bin/env python3
"""Generate or execute the Fractional-RL experimental matrix.

By default this represents the reported experimental design:

Algorithms
    TD3, SAC

Environment families
    MuJoCo:
        HalfCheetah-v4
        Humanoid-v4
        Ant-v4
        Hopper-v4

    DeepMind Control Suite:
        cheetah-run
        cartpole-swingup
        finger-spin
        walker-walk

Architectures
    1 hidden layer
    2 hidden layers

Fractional orders
    0.1, 0.2, 0.3, 0.4, 0.5

Seeds
    10, 20, 30, 40, 50

Reported two-layer placements
    all_both
    all_actor
    all_critic
    first_both
    first_actor

The implementation also supports first_critic through
--include-first-critic, but this placement is not included in the
default reported-paper matrix.

The script is dry-run by default. Add --run to execute jobs.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "scripts" / "train.py"

DEFAULT_ALGORITHMS = ["TD3", "SAC"]

DEFAULT_MUJOCO = [
    "HalfCheetah-v4",
    "Humanoid-v4",
    "Ant-v4",
    "Hopper-v4",
]

DEFAULT_DMC = [
    "cheetah-run",
    "cartpole-swingup",
    "finger-spin",
    "walker-walk",
]

DEFAULT_BASELINES = [
    "ReLU",
    "GELU",
    "SiLU",
    "Tanh",
]

DEFAULT_FRACTIONAL = [
    "FReLU",
    "FLReLU",
    "FPReLU",
    "FSwish",
    "FGELU",
]

DEFAULT_ALPHAS = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
]

DEFAULT_SEEDS = [
    10,
    20,
    30,
    40,
    50,
]

PAPER_PLACEMENTS = [
    "all_both",
    "all_actor",
    "all_critic",
    "first_both",
    "first_actor",
]


def build_command(
    *,
    algorithm: str,
    environment: str,
    activation: str,
    alpha: float | None,
    layers: int,
    placement: str,
    seed: int,
    steps: int,
    output: str,
) -> list[str]:
    command = [
        sys.executable,
        str(TRAIN),
        "--algo",
        algorithm,
        "--env",
        environment,
        "--activation",
        activation,
        "--layers",
        str(layers),
        "--placement",
        placement,
        "--seed",
        str(seed),
        "--steps",
        str(steps),
        "--output",
        output,
    ]

    if alpha is not None:
        command += [
            "--alpha",
            str(alpha),
        ]

    return command


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate or execute the Fractional-RL paper matrix."
    )

    parser.add_argument(
        "--algos",
        nargs="+",
        choices=["TD3", "SAC"],
        default=DEFAULT_ALGORITHMS,
    )

    parser.add_argument(
        "--envs",
        nargs="+",
        default=DEFAULT_MUJOCO + DEFAULT_DMC,
    )

    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        choices=[1, 2],
        default=[1, 2],
    )

    parser.add_argument(
        "--baselines",
        nargs="+",
        default=DEFAULT_BASELINES,
    )

    parser.add_argument(
        "--fractional",
        nargs="+",
        default=DEFAULT_FRACTIONAL,
    )

    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=DEFAULT_ALPHAS,
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
    )

    parser.add_argument(
        "--placements",
        nargs="+",
        default=PAPER_PLACEMENTS,
    )

    parser.add_argument(
        "--include-first-critic",
        action="store_true",
        help=(
            "Also generate the supported first_critic placement. "
            "It is not part of the default reported-paper matrix."
        ),
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=1_000_000,
    )

    parser.add_argument(
        "--output",
        default="results",
    )

    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional file to save every generated command.",
    )

    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute jobs. Otherwise only preview commands.",
    )

    args = parser.parse_args()

    placements = [
        value.replace("-", "_")
        for value in args.placements
    ]

    if args.include_first_critic and "first_critic" not in placements:
        placements.append("first_critic")

    jobs: list[list[str]] = []

    for algorithm in args.algos:
        for environment in args.envs:
            for layer_count in args.layers:

                active_placements = (
                    ["all_both"]
                    if layer_count == 1
                    else placements
                )

                # Standard activation baselines.
                for activation in args.baselines:
                    for placement in active_placements:
                        for seed in args.seeds:
                            jobs.append(
                                build_command(
                                    algorithm=algorithm,
                                    environment=environment,
                                    activation=activation,
                                    alpha=None,
                                    layers=layer_count,
                                    placement=placement,
                                    seed=seed,
                                    steps=args.steps,
                                    output=args.output,
                                )
                            )

                # Fractional activation family.
                for activation in args.fractional:
                    for alpha in args.alphas:
                        for placement in active_placements:
                            for seed in args.seeds:
                                jobs.append(
                                    build_command(
                                        algorithm=algorithm,
                                        environment=environment,
                                        activation=activation,
                                        alpha=alpha,
                                        layers=layer_count,
                                        placement=placement,
                                        seed=seed,
                                        steps=args.steps,
                                        output=args.output,
                                    )
                                )

    print("=" * 72)
    print("Fractional-RL experiment matrix")
    print("=" * 72)
    print(f"Algorithms      : {', '.join(args.algos)}")
    print(f"Environments    : {len(args.envs)}")
    print(f"Layers          : {args.layers}")
    print(f"Baselines       : {', '.join(args.baselines)}")
    print(f"Fractional      : {', '.join(args.fractional)}")
    print(f"Alphas          : {args.alphas}")
    print(f"Seeds           : {args.seeds}")
    print(f"2-layer places  : {placements}")
    print(f"Total jobs      : {len(jobs):,}")
    print("=" * 72)

    command_lines = [
        shlex.join(command)
        for command in jobs
    ]

    if args.manifest:
        manifest = Path(args.manifest)

        if not manifest.is_absolute():
            manifest = ROOT / manifest

        manifest.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        manifest.write_text(
            "\n".join(command_lines) + "\n"
        )

        print(f"Manifest written: {manifest}")

    for index, command in enumerate(jobs, start=1):
        print(
            f"[{index:05d}/{len(jobs):05d}] "
            f"{shlex.join(command)}",
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
