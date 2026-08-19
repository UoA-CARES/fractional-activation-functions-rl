import numpy as np
import torch

from frorl.algorithms.td3 import TD3
from frorl.algorithms.sac import SAC

from frorl.configurations import (
    TD3Config,
    SACConfig,
    MLPConfig,
    TrainableLayer,
    FunctionLayer,
)

from frorl.memory import MemoryBuffer
from frorl.training_context import TrainingContext

from frorl.models.td3_actor import Actor as TD3Actor
from frorl.models.td3_critic import Critic as TD3Critic
from frorl.models.sac_actor import Actor as SACActor
from frorl.models.sac_critic import Critic as SACCritic


def make_memory():
    memory = MemoryBuffer(max_capacity=1000)

    rng = np.random.default_rng(7)

    for _ in range(300):
        state = rng.normal(size=3).astype(np.float32)
        action = rng.uniform(-1, 1, size=2).astype(np.float32)
        reward = float(rng.normal())
        next_state = rng.normal(size=3).astype(np.float32)
        done = False

        memory.add(
            state,
            action,
            reward,
            next_state,
            done,
        )

    return memory


def make_context():
    return TrainingContext(
        memory=make_memory(),
        batch_size=64,
        training_step=1000,
        episode=1,
        episode_steps=10,
        episode_reward=1.0,
        episode_done=False,
    )


def test_td3_training_step_with_frelu():

    config = TD3Config()

    config.actor_config = MLPConfig(
        layers=[
            TrainableLayer(
                layer_type="Linear",
                out_features=32,
            ),
            FunctionLayer(
                layer_type="FReLU",
                params={"a": 0.2},
            ),
            TrainableLayer(
                layer_type="Linear",
                in_features=32,
            ),
            FunctionLayer(
                layer_type="Tanh",
            ),
        ]
    )

    config.critic_config = MLPConfig(
        layers=[
            TrainableLayer(
                layer_type="Linear",
                out_features=32,
            ),
            FunctionLayer(
                layer_type="FReLU",
                params={"a": 0.2},
            ),
            TrainableLayer(
                layer_type="Linear",
                in_features=32,
                out_features=1,
            ),
        ]
    )

    actor = TD3Actor(
        observation_size=3,
        num_actions=2,
        config=config,
    )

    critic = TD3Critic(
        observation_size=3,
        num_actions=2,
        config=config,
    )

    agent = TD3(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=torch.device("cpu"),
    )

    info = agent.train_policy(make_context())

    assert "critic_loss_total" in info
    assert np.isfinite(info["critic_loss_total"])


def test_sac_training_step_with_fgelu():

    config = SACConfig()

    config.actor_config = MLPConfig(
        layers=[
            TrainableLayer(
                layer_type="Linear",
                out_features=32,
            ),
            FunctionLayer(
                layer_type="FGELU",
                params={"a": 0.2},
            ),
        ]
    )

    config.critic_config = MLPConfig(
        layers=[
            TrainableLayer(
                layer_type="Linear",
                out_features=32,
            ),
            FunctionLayer(
                layer_type="FGELU",
                params={"a": 0.2},
            ),
            TrainableLayer(
                layer_type="Linear",
                in_features=32,
                out_features=1,
            ),
        ]
    )

    actor = SACActor(
        observation_size=3,
        num_actions=2,
        config=config,
    )

    critic = SACCritic(
        observation_size=3,
        num_actions=2,
        config=config,
    )

    agent = SAC(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=torch.device("cpu"),
    )

    info = agent.train_policy(make_context())

    assert "critic_loss_total" in info
    assert "alpha" in info

    assert np.isfinite(info["critic_loss_total"])
    assert np.isfinite(info["alpha"])
