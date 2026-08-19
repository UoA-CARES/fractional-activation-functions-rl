#!/usr/bin/env python3
"""Train TD3 or SAC with baseline or fractional activations.

Examples
--------
TD3 + FReLU:
    python scripts/train.py \
        --algo TD3 \
        --env HalfCheetah-v4 \
        --activation FReLU \
        --alpha 0.2 \
        --layers 2 \
        --placement all_both \
        --seed 10

SAC + FGELU:
    python scripts/train.py \
        --algo SAC \
        --env walker-walk \
        --activation FGELU \
        --alpha 0.2 \
        --layers 2 \
        --placement first_actor \
        --seed 10
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from frorl.algorithms.sac import SAC
from frorl.algorithms.td3 import TD3
from frorl.configurations import (
    FunctionLayer,
    MLPConfig,
    SACConfig,
    TD3Config,
    TrainableLayer,
)
from frorl.memory import MemoryBuffer
from frorl.models.sac_actor import Actor as SACActor
from frorl.models.sac_critic import Critic as SACCritic
from frorl.models.td3_actor import Actor as TD3Actor
from frorl.models.td3_critic import Critic as TD3Critic
from frorl.training.train_loop import train


STANDARD_ACTIVATIONS = {
    "relu": "ReLU",
    "lrelu": "LReLU",
    "leakyrelu": "LReLU",
    "prelu": "PReLU",
    "gelu": "GELU",
    "swish": "Swish",
    "silu": "Swish",
    "tanh": "Tanh",
}

OFFICIAL_FRACTIONAL_ACTIVATIONS = {
    "frelu": "FReLU",
    "flrelu": "FLReLU",
    "fprelu": "FPReLU",
    "fswish": "FSwish",
    "fgelu": "FGELU",
}

PLACEMENTS = {
    "all_both",
    "all_actor",
    "all_critic",
    "first_both",
    "first_actor",
    "first_critic",
}

DMC_TASKS = {
    "cheetah-run",
    "cartpole-swingup",
    "finger-spin",
    "walker-walk",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def canonical_activation(name: str) -> str:
    key = name.lower()

    if key in STANDARD_ACTIVATIONS:
        return STANDARD_ACTIVATIONS[key]

    if key in OFFICIAL_FRACTIONAL_ACTIVATIONS:
        return OFFICIAL_FRACTIONAL_ACTIVATIONS[key]

    # Preserve access to additional activation classes by their exact
    # class names, e.g. FractionalSwish, FALU, ResidualFractionalGELU.
    return name


def make_activation_layer(
    activation_name: str,
    alpha: float | None,
) -> FunctionLayer:
    activation_name = canonical_activation(activation_name)

    # Paper baselines use the same parameterisation as the experiments.
    if activation_name == "LReLU":
        return FunctionLayer(
            layer_type="LeakyReLU",
            params={"negative_slope": 0.1},
        )

    if activation_name == "PReLU":
        return FunctionLayer(
            layer_type="PReLU",
            params={"init": 0.25},
        )

    # PyTorch implements Swish as SiLU.
    if activation_name == "Swish":
        return FunctionLayer(layer_type="SiLU")

    if activation_name in {"ReLU", "GELU", "Tanh"}:
        return FunctionLayer(layer_type=activation_name)

    if alpha is None:
        raise ValueError(
            f"{activation_name} requires --alpha for fractional experiments."
        )

    return FunctionLayer(
        layer_type=activation_name,
        params={"a": alpha},
    )


def hidden_activation_layout(
    activation_name: str,
    layers: int,
    placement: str,
) -> tuple[list[str], list[str]]:
    """Return actor and critic hidden-layer activation layouts."""

    activation_name = canonical_activation(activation_name)

    if layers == 1:
        if placement != "all_both":
            raise ValueError(
                "For one hidden layer, placement must be 'all_both'."
            )

        return [activation_name], [activation_name]

    relu = "ReLU"

    layouts = {
        "all_both": (
            [activation_name, activation_name],
            [activation_name, activation_name],
        ),
        "all_actor": (
            [activation_name, activation_name],
            [relu, relu],
        ),
        "all_critic": (
            [relu, relu],
            [activation_name, activation_name],
        ),
        "first_both": (
            [activation_name, relu],
            [activation_name, relu],
        ),
        "first_actor": (
            [activation_name, relu],
            [relu, relu],
        ),
        "first_critic": (
            [relu, relu],
            [activation_name, relu],
        ),
    }

    return layouts[placement]


def build_actor_config(
    algorithm: str,
    hidden_activations: list[str],
    activation_name: str,
    alpha: float | None,
) -> MLPConfig:
    layers = []

    layers.append(
        TrainableLayer(
            layer_type="Linear",
            out_features=256,
        )
    )
    layers.append(
        make_activation_layer(
            hidden_activations[0],
            alpha if hidden_activations[0] != "ReLU" else None,
        )
    )

    if len(hidden_activations) == 2:
        layers.append(
            TrainableLayer(
                layer_type="Linear",
                in_features=256,
                out_features=256,
            )
        )
        layers.append(
            make_activation_layer(
                hidden_activations[1],
                alpha if hidden_activations[1] != "ReLU" else None,
            )
        )

    if algorithm == "TD3":
        layers.append(
            TrainableLayer(
                layer_type="Linear",
                in_features=256,
            )
        )
        layers.append(FunctionLayer(layer_type="Tanh"))

    return MLPConfig(layers=layers)


def build_critic_config(
    hidden_activations: list[str],
    alpha: float | None,
) -> MLPConfig:
    layers = []

    layers.append(
        TrainableLayer(
            layer_type="Linear",
            out_features=256,
        )
    )
    layers.append(
        make_activation_layer(
            hidden_activations[0],
            alpha if hidden_activations[0] != "ReLU" else None,
        )
    )

    if len(hidden_activations) == 2:
        layers.append(
            TrainableLayer(
                layer_type="Linear",
                in_features=256,
                out_features=256,
            )
        )
        layers.append(
            make_activation_layer(
                hidden_activations[1],
                alpha if hidden_activations[1] != "ReLU" else None,
            )
        )

    layers.append(
        TrainableLayer(
            layer_type="Linear",
            in_features=256,
            out_features=1,
        )
    )

    return MLPConfig(layers=layers)


def make_gym_env(name: str, seed: int):
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise SystemExit(
            "Gymnasium is required. Install the Gymnasium dependencies first."
        ) from exc

    env = gym.make(name)
    env.reset(seed=seed)
    env.action_space.seed(seed)

    return env


def make_dmc_env(label: str, seed: int):
    from frorl.environments import DMCEnv

    domain, task = label.split("-", 1)
    return DMCEnv(domain, task, seed)


def make_env(name: str, seed: int):
    if name.lower() in DMC_TASKS:
        return make_dmc_env(name.lower(), seed)

    return make_gym_env(name, seed)


def environment_dimensions(env) -> tuple[int, int]:
    if hasattr(env, "observation_space"):
        observation_size = int(np.prod(env.observation_space.shape))
        num_actions = int(np.prod(env.action_space.shape))
    else:
        state = env.reset()

        if isinstance(state, tuple):
            state = state[0]

        observation_size = int(np.asarray(state).size)
        num_actions = int(env.action_num)

    return observation_size, num_actions


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fractional activation experiments for TD3 and SAC."
    )

    parser.add_argument(
        "--algo",
        choices=["TD3", "SAC"],
        required=True,
    )
    parser.add_argument("--env", required=True)

    parser.add_argument(
        "--activation",
        default="ReLU",
        help="ReLU, FReLU, FLReLU, FPReLU, FSwish, FGELU, or another supported class.",
    )
    parser.add_argument("--alpha", type=float, default=None)

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

    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--eval-seed", type=int, default=None)

    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--start-steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)

    parser.add_argument("--train-every", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)

    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)

    parser.add_argument("--output", default="results")

    return parser.parse_args()


def main():
    args = parse_args()

    # Accept either all_both or all-both spelling.
    placement = args.placement.replace("-", "_")

    if placement not in PLACEMENTS:
        raise ValueError(
            f"Unknown placement '{args.placement}'. "
            f"Choose from {sorted(PLACEMENTS)}."
        )

    if args.layers == 1:
        placement = "all_both"

    activation_name = canonical_activation(args.activation)

    set_seed(args.seed)

    # Original runner defaults eval_seed to train_seed.
    eval_seed = (
        args.seed
        if args.eval_seed is None
        else args.eval_seed
    )

    env = make_env(args.env, args.seed)
    eval_env = make_env(args.env, eval_seed)

    observation_size, num_actions = environment_dimensions(env)

    actor_hidden, critic_hidden = hidden_activation_layout(
        activation_name,
        args.layers,
        placement,
    )

    if args.algo == "TD3":
        config = TD3Config()
    else:
        config = SACConfig()

    config.max_steps_training = args.steps
    config.max_steps_exploration = args.start_steps
    config.batch_size = args.batch_size
    config.buffer_size = args.buffer_size
    config.number_steps_per_train_policy = args.train_every
    config.G = args.gradient_steps

    config.actor_config = build_actor_config(
        args.algo,
        actor_hidden,
        activation_name,
        args.alpha,
    )

    config.critic_config = build_critic_config(
        critic_hidden,
        args.alpha,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if args.algo == "TD3":
        actor = TD3Actor(
            observation_size=observation_size,
            num_actions=num_actions,
            config=config,
        )
        critic = TD3Critic(
            observation_size=observation_size,
            num_actions=num_actions,
            config=config,
        )

        agent = TD3(
            actor_network=actor,
            critic_network=critic,
            config=config,
            device=device,
        )

    else:
        actor = SACActor(
            observation_size=observation_size,
            num_actions=num_actions,
            config=config,
        )
        critic = SACCritic(
            observation_size=observation_size,
            num_actions=num_actions,
            config=config,
        )

        agent = SAC(
            actor_network=actor,
            critic_network=critic,
            config=config,
            device=device,
        )

    alpha_label = (
        "base"
        if args.alpha is None
        else f"{args.alpha:g}"
    )

    output_dir = (
        Path(args.output)
        / args.algo
        / args.env
        / activation_name
        / f"alpha_{alpha_label}"
        / f"{args.layers}layer"
        / placement
        / f"seed_{args.seed}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = vars(args).copy()
    run_config["activation"] = activation_name
    run_config["placement"] = placement
    run_config["eval_seed"] = eval_seed
    run_config["device"] = str(device)

    (output_dir / "config.json").write_text(
        json.dumps(run_config, indent=2)
    )

    memory = MemoryBuffer(
        max_capacity=args.buffer_size,
    )

    print(
        f"{args.algo} | {args.env} | {activation_name} | "
        f"alpha={args.alpha} | {args.layers} layer(s) | "
        f"{placement} | seed={args.seed} | device={device}"
    )

    train(
        env=env,
        eval_env=eval_env,
        agent=agent,
        memory=memory,
        total_steps=args.steps,
        max_steps_exploration=args.start_steps,
        batch_size=args.batch_size,
        number_steps_per_train_policy=args.train_every,
        gradient_steps=args.gradient_steps,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        out_dir=output_dir,
    )

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
