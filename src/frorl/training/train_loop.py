"""Standalone training and evaluation loops for Fractional-RL.

The execution order follows the training protocol used for the reported
experiments:

1. Random exploration for ``max_steps_exploration`` environment steps.
2. Store every transition in replay memory.
3. Train according to ``number_steps_per_train_policy``.
4. Perform ``G`` gradient updates at every policy-training event.
5. Evaluate periodically using deterministic/evaluation policy actions.
6. Reset algorithm episode state whenever an episode terminates or truncates.

The RL update equations themselves remain implemented in the TD3 and SAC
algorithm classes.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from frorl import helpers as hlp
from frorl.training_context import ActionContext, TrainingContext


def _action_bounds(env) -> tuple[np.ndarray, np.ndarray]:
    """Return the environment action-space lower and upper bounds."""

    if hasattr(env, "action_space"):
        low = np.asarray(env.action_space.low, dtype=np.float32)
        high = np.asarray(env.action_space.high, dtype=np.float32)
        return low, high

    if hasattr(env, "_action_spec"):
        low = np.asarray(env._action_spec.minimum, dtype=np.float32)
        high = np.asarray(env._action_spec.maximum, dtype=np.float32)
        return low, high

    raise AttributeError(
        "Environment must expose action_space or a dm_control action specification."
    )


def _reset_env(env):
    """Reset Gymnasium/DMC-style environments and return only the observation."""

    result = env.reset()

    if isinstance(result, tuple):
        return result[0]

    return result


def _step_env(env, action):
    """Normalize the environment step API."""

    result = env.step(action)

    if len(result) == 5:
        next_state, reward, terminated, truncated, info = result
    elif len(result) == 4:
        next_state, reward, done, info = result
        terminated = done
        truncated = False
    else:
        raise RuntimeError(
            f"Unexpected environment step result with {len(result)} elements."
        )

    return next_state, float(reward), bool(terminated), bool(truncated), info


def _sample_exploration_action(env) -> np.ndarray:
    """Sample an action directly from the environment action space."""

    if hasattr(env, "action_space"):
        return np.asarray(env.action_space.sample(), dtype=np.float32)

    if hasattr(env, "sample_action"):
        return np.asarray(env.sample_action(), dtype=np.float32)

    raise AttributeError("Environment does not provide random action sampling.")


def _normalize_action(
    action: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    """Map an environment action to the policy's normalized [-1, 1] space."""

    # The original training framework used the CARES normalize helper.
    return np.asarray(
        hlp.normalize(action, high, low),
        dtype=np.float32,
    )


def _denormalize_action(
    action: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    """Map a normalized policy action back to environment action bounds."""

    return np.asarray(
        hlp.denormalize(action, high, low),
        dtype=np.float32,
    )


def evaluate(
    env,
    agent,
    episodes: int = 10,
) -> tuple[float, float]:
    """Evaluate an agent over complete episodes."""

    low, high = _action_bounds(env)

    returns: list[float] = []

    for _ in range(episodes):
        state = _reset_env(env)

        episode_reward = 0.0
        episode_done = False

        while not episode_done:
            action_context = ActionContext(
                state=np.asarray(state, dtype=np.float32),
                evaluation=True,
                available_actions=np.array([], dtype=np.float32),
            )

            normalized_action = agent.select_action_from_policy(action_context)

            environment_action = _denormalize_action(
                normalized_action,
                low,
                high,
            )

            (
                state,
                reward,
                terminated,
                truncated,
                _,
            ) = _step_env(env, environment_action)

            episode_reward += reward
            episode_done = terminated or truncated

        returns.append(episode_reward)

        # Matches the original runner's per-episode algorithm reset.
        agent.episode_done()

    return float(np.mean(returns)), float(np.std(returns))


def train(
    env,
    eval_env,
    agent,
    memory,
    total_steps: int,
    max_steps_exploration: int,
    batch_size: int,
    number_steps_per_train_policy: int,
    gradient_steps: int,
    eval_every: int,
    eval_episodes: int,
    out_dir,
):
    """Train a TD3/SAC agent using the Fractional-RL experiment protocol."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_csv = out / "train.csv"
    eval_csv = out / "eval.csv"

    action_low, action_high = _action_bounds(env)

    state = _reset_env(env)

    episode_num = 0
    episode_steps = 0
    episode_reward = 0.0

    with (
        train_csv.open("w", newline="") as train_file,
        eval_csv.open("w", newline="") as eval_file,
    ):
        train_writer = csv.writer(train_file)
        eval_writer = csv.writer(eval_file)

        train_writer.writerow(
            [
                "total_steps",
                "episode",
                "episode_reward",
                "episode_length",
            ]
        )

        eval_writer.writerow(
            [
                "total_steps",
                "mean_reward",
                "std_reward",
            ]
        )

        # Keep zero-based training-step semantics internally, matching the
        # original TrainingRunner.
        for train_step_counter in range(int(total_steps)):
            episode_steps += 1

            # -------------------------------------------------------------
            # Action selection
            # -------------------------------------------------------------
            if train_step_counter < max_steps_exploration:
                environment_action = _sample_exploration_action(env)

                normalized_action = _normalize_action(
                    environment_action,
                    action_low,
                    action_high,
                )

            else:
                action_context = ActionContext(
                    state=np.asarray(state, dtype=np.float32),
                    evaluation=False,
                    available_actions=np.array([], dtype=np.float32),
                )

                normalized_action = agent.select_action_from_policy(
                    action_context
                )

                environment_action = _denormalize_action(
                    normalized_action,
                    action_low,
                    action_high,
                )

            # -------------------------------------------------------------
            # Environment step
            # -------------------------------------------------------------
            (
                next_state,
                reward,
                terminated,
                truncated,
                _,
            ) = _step_env(env, environment_action)

            episode_done = terminated or truncated

            # -------------------------------------------------------------
            # Replay storage
            #
            # Important: the original experiments stored normalized actions
            # in replay memory while sending denormalized actions to the env.
            # -------------------------------------------------------------
            memory.add(
                state,
                normalized_action,
                reward,
                next_state,
                terminated,
            )

            state = next_state
            episode_reward += reward

            # -------------------------------------------------------------
            # Policy update
            # -------------------------------------------------------------
            if (
                train_step_counter >= max_steps_exploration
                and (train_step_counter + 1)
                % number_steps_per_train_policy
                == 0
            ):
                training_context = TrainingContext(
                    memory=memory,
                    batch_size=batch_size,
                    training_step=train_step_counter,
                    episode=episode_num + 1,
                    episode_steps=episode_steps,
                    episode_reward=episode_reward,
                    episode_done=episode_done,
                )

                for _ in range(gradient_steps):
                    agent.train_policy(training_context)

            # -------------------------------------------------------------
            # Evaluation
            #
            # In the original runner evaluation occurred before episode
            # finalisation when both happened on the same training step.
            # -------------------------------------------------------------
            if (train_step_counter + 1) % eval_every == 0:
                mean_reward, std_reward = evaluate(
                    eval_env,
                    agent,
                    eval_episodes,
                )

                eval_writer.writerow(
                    [
                        train_step_counter + 1,
                        mean_reward,
                        std_reward,
                    ]
                )
                eval_file.flush()

                print(
                    f"step={train_step_counter + 1} "
                    f"eval_mean={mean_reward:.3f} "
                    f"eval_std={std_reward:.3f}"
                )

            # -------------------------------------------------------------
            # Episode finalisation
            # -------------------------------------------------------------
            if episode_done:
                train_writer.writerow(
                    [
                        train_step_counter + 1,
                        episode_num + 1,
                        episode_reward,
                        episode_steps,
                    ]
                )
                train_file.flush()

                state = _reset_env(env)

                episode_num += 1
                episode_steps = 0
                episode_reward = 0.0

                agent.episode_done()

    agent.save_models(str(out), "model")
