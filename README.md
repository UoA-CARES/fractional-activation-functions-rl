# Fractional Activation Functions for Off-Policy Reinforcement Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![RL Algorithms](https://img.shields.io/badge/Algorithms-TD3%20%7C%20SAC-6f42c1.svg)](#experiments)

<p align="center">
  <img src="assets/fractional_activation_overview.png"
       alt="Fractional activation functions across different fractional orders"
       width="100%">
</p>

<p align="center">
  <em>Fractional activation families across different fractional orders.</em>
</p>

A research repository for fractional-order activation functions in off-policy deep reinforcement learning.

This project evaluates fractional activations as lightweight architectural modifications for actor and critic networks. The activations are integrated into Twin Delayed Deep Deterministic Policy Gradient (TD3) and Soft Actor-Critic (SAC), without changing their learning rules, replay mechanisms, target updates, optimizers, or training budgets.

## Overview

Activation functions determine the nonlinear transformations used by actor and critic networks. Standard rectifier functions such as ReLU are computationally efficient, but their piecewise-linear form may limit representational flexibility. Fractional-order formulations introduce a fixed order parameter, \(\alpha\), that continuously controls activation shape.

This project evaluates five conventional activation functions and five fractional counterparts:

| Family | Conventional activation | Fractional activation | Main characteristic |
|---|---|---|---|
| Rectifier | ReLU | FReLU | Fractional curvature in the positive branch |
| Rectifier | Leaky ReLU | FLReLU | Fractional branches with fixed negative slope |
| Rectifier | Parametric ReLU | FPReLU | Fractional branches with a learnable negative coefficient |
| Smooth | GELU | FGELU | Fractional modulation of an adaptive GELU core |
| Smooth | Swish | FSwish | Fractional modulation of an adaptive Swish core |

For the rectifier-based variants, the central transformation is based on

$$
x^{1-\alpha}, \qquad \alpha \in \{0.1, 0.2, 0.3, 0.4, 0.5\}.
$$

The fractional order is fixed within each experiment and treated as an activation-specific hyperparameter. It should not be interpreted as a quantity that must be maximized.

## Research findings

- Fractional activations frequently outperform ReLU and their corresponding non-fractional baselines.
- The best fractional configurations improve normalized area under the learning curve by approximately 26% on average relative to ReLU across the evaluated algorithm and architecture settings.
- Fractional extensions of GELU and Swish provide gains beyond already strong smooth baselines in several settings.
- The preferred fractional order depends on the activation family, task, algorithm, network depth, and placement.
- SAC shows the most consistent evidence, while TD3 exhibits greater seed-level variability.

These results support fractional activation functions as simple, computationally lightweight design choices for off-policy actor-critic learning.

## Experiments

### Algorithms

- Twin Delayed Deep Deterministic Policy Gradient (TD3)
- Soft Actor-Critic (SAC)

### Benchmarks

| Benchmark suite | Tasks |
|---|---|
| MuJoCo | Ant-v4, HalfCheetah-v4, Hopper-v4, Humanoid-v4 |
| DeepMind Control Suite | Walker-Walk, Cheetah-Run, Finger-Spin, Cartpole-Swingup |

### Evaluation protocol

- One-hidden-layer and two-hidden-layer actor-critic networks
- Five random seeds per configuration
- Five fractional orders: \(\alpha \in \{0.1, 0.2, 0.3, 0.4, 0.5\}\)
- Multiple actor and critic activation-placement strategies
- Matched training budgets, optimizers, architectures, seeds, and evaluation settings
- Normalized area under the learning curve and complete learning trajectories
- Bootstrap confidence intervals
- Paired Wilcoxon signed-rank tests
- Cliff's delta effect sizes
- Aggregate sign-test summaries

## Installation

Clone the repository and create an isolated Python environment:

```bash
git clone https://github.com/UoA-CARES/fractional-activation-functions-rl.git
cd fractional-activation-functions-rl
python -m venv .venv
```

Activate the environment:

```bash
# Linux or macOS
source .venv/bin/activate

# Windows Command Prompt
.venv\Scripts\activate
```

Install the package and Gymnasium dependencies:

```bash
python -m pip install --upgrade pip
pip install -e ".[gym]"
```

The experiments build on the [CARES Reinforcement Learning](https://github.com/UoA-CARES/cares_reinforcement_learning) framework and its standardized environment interfaces.

## Running experiments

Experiments are launched through the CARES RL configuration interface:

```bash
cares-rl train config --data_path <CONFIGURATION_DIRECTORY>
```

Example baseline runs:

```bash
cares-rl train cli --gym openai --task HalfCheetah-v4 TD3
cares-rl train cli --gym openai --task HalfCheetah-v4 SAC
```

Use the supplied experiment configurations to select the activation family, fractional order, network depth, placement strategy, task, and random seeds. Exact configuration paths and reproduction commands will be documented alongside the finalized configuration directory.

## Reproducibility

To support controlled comparisons, each fractional configuration should be evaluated against its matched non-fractional baseline using identical:

- environment and task settings;
- algorithm hyperparameters;
- network width and depth;
- optimizer settings;
- training and evaluation budgets; and
- random seeds.

Training outputs include the algorithm, environment, and training configurations together with seed-level learning logs. These files support reconstruction of learning curves, normalized AUC values, and statistical comparisons.

## Repository structure

```text
fractional-activation-functions-rl/
|-- cares_reinforcement_learning/   # TD3, SAC, networks, and activations
|-- configs/                        # Reproducible experiment configurations
|-- analysis/                       # AUC and statistical analysis scripts
|-- results/                        # Processed tables and summary results
|-- tests/                          # Activation and configuration tests
|-- assets/                         # README figures
|-- README.md
|-- LICENSE
`-- pyproject.toml
```

Large raw training logs and model checkpoints are not stored directly in Git. Where released, they will be linked through an archival data record.

## Acknowledgements

This work was developed within the Centre for Automation and Robotic Engineering Science at The University of Auckland. The implementation builds on the open-source CARES Reinforcement Learning framework.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details. Third-party components remain subject to their respective licenses and attribution requirements.
