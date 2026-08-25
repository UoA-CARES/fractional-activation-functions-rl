# Fractional Activation Functions for Off-Policy Reinforcement Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TD3 & SAC](https://img.shields.io/badge/Algorithms-TD3%20%7C%20SAC-6F42C1)](#experiments)
[![MuJoCo](https://img.shields.io/badge/Benchmark-MuJoCo-00599C)](https://mujoco.org/)
[![DeepMind Control Suite](https://img.shields.io/badge/Benchmark-DeepMind%20Control%20Suite-blue)](https://github.com/google-deepmind/dm_control)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A PyTorch research implementation for studying **fractional-order activation functions in off-policy deep reinforcement learning**.


Fractional activation functions introduce a controllable fractional-order parameter into familiar nonlinearities, allowing the shape of these nonlinearities to be modified while keeping the underlying TD3 and SAC learning rules unchanged.

This repository evaluates five fractional activation families across continuous-control tasks from **MuJoCo** and the **DeepMind Control Suite**, using controlled actor-critic architectures, matched training settings, and five random seeds.

---

## At a Glance

| Component | Study setting |
|---|---|
| RL algorithms | TD3, SAC |
| Conventional activations | ReLU, LReLU, PReLU, GELU, Swish |
| Fractional activations | FReLU, FLReLU, FPReLU, FGELU, FSwish |
| Fractional orders tested | `0.1, 0.2, 0.3, 0.4, 0.5` |
| Benchmarks | 4 MuJoCo + 4 DeepMind Control Suite tasks |
| Network depths | 1 and 2 hidden layers |
| Hidden width | 256 |
| Random seeds | `10, 20, 30, 40, 50` |
| Training budget | 1,000,000 environment steps |
| Total seed-level runs | **14,080** |

---

## What Does "Fractional" Mean?

Fractional calculus generalizes ordinary integer-order differentiation and integration to **non-integer orders**.

A useful property of fractional calculus is that applying a fractional-order operation to a power function changes its exponent continuously. This motivates the use of activation functions whose nonlinear shape can be controlled by a fractional-order parameter `alpha`.

For the rectifier-based activations used in this work, the central fractional term is:

```text
x^(1-alpha)
```

For example, the positive branch of conventional ReLU is:

```text
ReLU(x) = x
```

while FReLU uses:

```text
FReLU_alpha(x) = x^(1-alpha),   x > 0
```

When `alpha` is close to zero, the exponent `1 - alpha` remains close to `1`, so the activation remains close to the conventional linear ReLU branch.

As `alpha` increases, the exponent decreases and the fractional branch becomes progressively more curved and sublinear.

| `alpha` | Exponent `1 - alpha` | Interpretation |
|---:|---:|---|
| 0.0 | 1.0 | conventional ReLU limit |
| 0.1 | 0.9 | mild fractional modification |
| 0.2 | 0.8 | modest fractional curvature |
| 0.3 | 0.7 | intermediate fractional curvature |
| 0.4 | 0.6 | stronger fractional curvature |
| 0.5 | 0.5 | strongest order evaluated in this study |

The basic intuition is:

```text
alpha close to 0
       |
       v
closer to the conventional activation


larger alpha
       |
       v
stronger fractional modification
```

### Why Does This Study Use `alpha = 0.1` to `0.5`?

Fractional orders are **not inherently limited to 0.5**.

For the power-based rectifier interpretation, fractional orders can extend beyond the range evaluated here. For example:

```text
alpha = 0.6  ->  exponent = 0.4
alpha = 0.7  ->  exponent = 0.3
alpha = 0.8  ->  exponent = 0.2
alpha = 0.9  ->  exponent = 0.1
```

This study deliberately evaluates:

```text
alpha ∈ {0.1, 0.2, 0.3, 0.4, 0.5}
```

to investigate **mild-to-moderate fractional transformations** while keeping the resulting nonlinearities reasonably close to their corresponding conventional activation families.

For the power-based fractional rectifiers, values above `alpha = 0.5` imply:

```text
1 - alpha < 0.5
```

which introduces substantially stronger compression of the activation response and increasingly sharp behaviour near zero.

This can also produce larger changes in activation magnitudes and gradients. Such effects are particularly relevant in off-policy actor-critic learning, where actor and critic networks are repeatedly optimized using replayed transitions and bootstrapped value targets.

The selected range therefore provides meaningful control over fractional curvature without deliberately moving into more extreme fractional regimes.

Importantly:

> **A larger fractional order is not necessarily better.**

`alpha` is treated as an **activation-specific hyperparameter**, not as a quantity that should be maximized.

Its preferred value can depend on:

- activation family;
- environment;
- RL algorithm;
- network depth;
- actor/critic placement.

For the smooth fractional activations, FGELU and FSwish, `alpha` controls the **strength of fractional modulation** rather than directly acting as the exponent of a power-law branch.

The study therefore asks whether a controlled amount of fractional modification can improve learning, rather than searching for the largest possible fractional order.

---

<p align="center">
  <img
    src="assets/fractional_activation_overview.png"
    alt="Conventional and fractional activation functions across fractional orders"
    width="100%"
  >
</p>

<p align="center">
  <em>
    Effect of fractional order on the evaluated activation families.
    Increasing alpha changes the nonlinear response while the underlying TD3 and SAC algorithms remain unchanged.
  </em>
</p>

---

## Activation Families

The study evaluates five conventional activation families and their fractional counterparts.

| Family | Conventional | Fractional | Main fractional modification |
|---|---|---|---|
| Rectifier | ReLU | **FReLU** | fractional positive-branch curvature |
| Rectifier | LReLU | **FLReLU** | fractional positive and negative branches |
| Parametric rectifier | PReLU | **FPReLU** | fractional branches with adaptive negative response |
| Smooth | GELU | **FGELU** | finite fractional transformation of a smooth GELU core |
| Smooth | Swish | **FSwish** | fractional modulation of a smooth Swish core |

### Conventional Activations

```text
ReLU
LReLU
PReLU
GELU
Swish
```

### Fractional Activations

```text
FReLU
FLReLU
FPReLU
FGELU
FSwish
```

This comparison allows the study to test whether fractionalization is useful only for ReLU-style activations or whether it can also improve already strong smooth nonlinearities such as GELU and Swish.

---

## Fractional Activation Definitions

### FReLU

Fractional ReLU modifies only the positive branch of ReLU:

```text
FReLU_alpha(x) =
    x^(1-alpha),   if x > 0
    0,             otherwise
```

The fractional order controls the curvature of the positive branch.

At the non-fractional limit:

```text
alpha = 0
```

the exponent becomes `1`, recovering the conventional positive ReLU branch.

For numerical stability, the implementation safeguards positive inputs near zero before applying the fractional power.

**The FReLU implementation used in this repository does not apply Gamma normalization.**

---

### FLReLU

Fractional Leaky ReLU applies fractional-power transformations to both positive and negative branches:

```text
FLReLU_alpha(x) =

     x^(1-alpha) / Gamma(2-alpha),       if x > 0

    -k*(-x)^(1-alpha) / Gamma(2-alpha),  if x < 0

     0,                                  if x = 0
```

where:

- `alpha` controls the curvature of both branches;
- `k` controls the magnitude of the negative response;
- `k` is initialized at `0.1`;
- the implementation constrains `k` during the forward computation.

The Gamma term provides scale control across fractional orders.

---

### FPReLU

Fractional Parametric ReLU uses the same Gamma-normalized fractional branches while allowing the negative-side coefficient to adapt during training:

```text
FPReLU_alpha(x) =

     x^(1-alpha) / Gamma(2-alpha),       if x > 0

    -p*(-x)^(1-alpha) / Gamma(2-alpha),  if x < 0

     0,                                  if x = 0
```

where:

- `alpha` controls fractional curvature;
- `p` controls the negative-side response;
- `p` is learned jointly with the neural-network parameters;
- a positive parameterization is used for numerical stability.

FPReLU therefore combines **fractional curvature control** with an **adaptive negative branch**.

---

### FSwish

FSwish extends the smooth Swish family.

The adaptive Swish core is:

```text
sigmoid_term = sigmoid(beta * x)

swish(x) = x * sigmoid_term
```

The fractional formulation used in this repository is:

```text
FSwish(x) =
    swish(x)
    + alpha * sigmoid_term * (1 - swish(x))
```

where:

- `alpha` controls the strength of fractional modulation;
- `beta` is learned during training;
- `beta` controls the smooth gating response.

Unlike the rectifier-based fractional activations, `alpha` does not directly appear as the exponent of a fractional-power branch.

---

### FGELU

FGELU extends GELU using a finite fractional approximation applied to an adaptive smooth GELU core.

The implementation uses:

```text
fractional order alpha
learnable beta
step size h = 0.5
finite expansion length N = 3
```

The finite Grünwald-Letnikov-style transformation provides fractional modulation while keeping the activation computationally practical for repeated actor and critic updates.

The implementation also includes numerical safeguards for finite outputs.

FGELU therefore combines:

- smooth GELU-style gating;
- adaptive scaling;
- finite fractional modulation.

---

## Implementation

The public activation interface is located in:

```text
src/frorl/models/activations.py
```

The exact fractional activation implementations used by the experiments are located in:

```text
src/frorl/models/fractional_activations.py
```

The SHA-256 fingerprint of the validated fractional activation source is:

```text
46b35edb9d4a87e75907284c00e73f6d4f7ad83dad910b8aafe1dd6e8b4e15a8
```

This fingerprint can be used to verify that the validated experimental implementation has not been modified.

---

## Research Findings

The study investigates whether changing only hidden-layer activation design can affect off-policy actor-critic learning.

Across the evaluated tasks, algorithms, architectures, and configurations, the experiments show that:

- fractional activations frequently outperform their corresponding conventional activation families;
- fractional rectifier variants can improve over ReLU, LReLU, and PReLU;
- FGELU and FSwish can improve over already competitive smooth GELU and Swish baselines;
- no single fractional order is universally optimal;
- the preferred `alpha` depends on the activation family, task, algorithm, and architecture;
- activation placement affects performance in two-hidden-layer actor-critic networks;
- actor-side and early-layer placements are frequently competitive;
- SAC and TD3 can respond differently to the same fractional activation design.

These gains are obtained without modifying the underlying reinforcement-learning objectives, replay mechanisms, target updates, or optimization procedures.

The results therefore support fractional activation design as a lightweight architectural mechanism for modifying nonlinear representation in continuous-control reinforcement learning.

---

## Experiments

### Algorithms

The repository contains standalone implementations of:

- **Twin Delayed Deep Deterministic Policy Gradient (TD3)**
- **Soft Actor-Critic (SAC)**

Fractional-activation experiments do **not** change the underlying TD3 or SAC learning rules.

---

### Benchmarks

#### MuJoCo

```text
Ant-v4
HalfCheetah-v4
Hopper-v4
Humanoid-v4
```

The `v4` environments are retained to match the reported experimental protocol.

#### DeepMind Control Suite

```text
walker-walk
cheetah-run
finger-spin
cartpole-swingup
```

Together, these tasks include:

- high-dimensional locomotion;
- coordinated whole-body control;
- balance;
- manipulation;
- dynamic continuous control.

---

### Network Architectures

All hidden layers use width:

```text
256
```

The study evaluates:

```text
1 hidden layer
2 hidden layers
```

#### One-Hidden-Layer Network

```text
Input
  |
Linear(256)
  |
Activation
  |
Output
```

#### Two-Hidden-Layer Network

```text
Input
  |
Linear(256)
  |
Activation 1
  |
Linear(256)
  |
Activation 2
  |
Output
```

Only hidden-layer activation configuration changes between activation conditions.

---

### Activation Placement

For two-hidden-layer architectures, five placement strategies are included in the reported study.

| Placement | Description |
|---|---|
| `all_actor` | selected activation in all actor hidden layers |
| `all_critic` | selected activation in all critic hidden layers |
| `all_both` | selected activation in all actor and critic hidden layers |
| `first_actor` | selected activation only in the first actor hidden layer |
| `first_both` | selected activation only in the first hidden layer of both actor and critic |

Hidden layers not assigned the selected activation use ReLU.

The implementation also supports:

```text
first_critic
```

for additional experiments, but this placement is not part of the main reported five-placement comparison.

For the single-hidden-layer study, the reported configuration is:

```text
all_both
```

---

### Experimental Protocol

| Component | Setting |
|---|---|
| Algorithms | TD3, SAC |
| Benchmark tasks | 8 |
| Training budget | 1,000,000 environment steps |
| Initial random exploration | 1,000 steps |
| Replay capacity | 1,000,000 |
| Batch size | 256 |
| Hidden width | 256 |
| Hidden depth | 1 or 2 |
| Fractional orders | 0.1, 0.2, 0.3, 0.4, 0.5 |
| Seeds | 10, 20, 30, 40, 50 |
| Evaluation interval | 10,000 environment steps |
| Evaluation episodes | 10 |

Controlled comparisons use matched:

- algorithms;
- environments;
- actor and critic widths;
- network depth;
- replay-buffer settings;
- optimization settings;
- training budgets;
- evaluation schedules;
- random seeds.

The primary experimental variables are:

```text
activation family
fractional order
network depth
activation placement
```

No activation-specific RL algorithm modification is introduced.

---

## Quick Start

### Clone the Repository

```bash
git clone https://github.com/UoA-CARES/fractional-activation-functions-rl.git
cd fractional-activation-functions-rl
```

### Create a Virtual Environment

```bash
python -m venv .venv
```

#### Linux / macOS

```bash
source .venv/bin/activate
```

#### Windows Command Prompt

```bat
.venv\Scripts\activate
```

#### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### Install

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

---

## Verification

Run the unit tests:

```bash
pytest -q
```

The validated repository passes:

```text
19/19 tests
```

For broader end-to-end verification:

```bash
./scripts/smoke_test_all.sh
```

The validated smoke-test suite passes:

```text
22/22 end-to-end smoke tests
```

The smoke tests cover:

- TD3;
- SAC;
- all five primary fractional activations;
- conventional activations;
- one- and two-hidden-layer networks;
- activation placement;
- MuJoCo;
- DeepMind Control Suite;
- replay-buffer interaction;
- optimization;
- evaluation;
- result generation;
- checkpoint generation.

---

## Running Experiments

The main training entry point is:

```text
scripts/train.py
```

### TD3 with ReLU

```bash
python scripts/train.py \
  --algo TD3 \
  --env HalfCheetah-v4 \
  --activation relu \
  --layers 1 \
  --placement all_both \
  --seed 10
```

### TD3 with FReLU

```bash
python scripts/train.py \
  --algo TD3 \
  --env HalfCheetah-v4 \
  --activation frelu \
  --alpha 0.3 \
  --layers 2 \
  --placement all_both \
  --seed 10
```

### SAC with FGELU

```bash
python scripts/train.py \
  --algo SAC \
  --env walker-walk \
  --activation fgelu \
  --alpha 0.2 \
  --layers 2 \
  --placement all_both \
  --seed 10
```

### Two-Layer Placement Experiment

```bash
python scripts/train.py \
  --algo TD3 \
  --env HalfCheetah-v4 \
  --activation flrelu \
  --alpha 0.1 \
  --layers 2 \
  --placement first_actor \
  --seed 10
```

---

## Running an Alpha and Seed Sweep

Use:

```text
scripts/run_sweep.py
```

Example:

```bash
python scripts/run_sweep.py \
  --algo TD3 \
  --env HalfCheetah-v4 \
  --activation frelu \
  --layers 2 \
  --placement all_both
```

The standard fractional search uses:

```text
alpha = 0.1, 0.2, 0.3, 0.4, 0.5
seeds = 10, 20, 30, 40, 50
```

The launcher previews generated experiments by default.

Use:

```text
--run
```

to execute them.

---

## Complete Experiment Matrix

The complete paper experiment launcher is:

```text
scripts/run_matrix.py
```

Preview the experiment matrix with:

```bash
python scripts/run_matrix.py \
  --manifest experiment_commands.txt
```

The complete study contains:

```text
14,080 seed-level experiments
```

covering:

```text
2 algorithms
8 tasks
2 network depths
5 conventional activations
5 fractional activations
5 fractional orders
5 random seeds
5 reported two-layer placement strategies
```

### Shared ReLU Baseline

In the two-layer placement study, ReLU is evaluated as a **shared baseline**.

Changing a nominal placement label does not alter the architecture when all relevant layers already use ReLU. Repeating identical ReLU networks under every placement label would therefore create redundant runs.

The experiment matrix stores only the required shared ReLU baseline rather than duplicating architecturally identical configurations.

---

## Results

Compact paper-level numerical results are included directly in the repository.

```text
results/
├── README.md
└── paper/
    ├── auc_by_seed.csv
    └── auc_by_run.csv
```

### Seed-Level Results

[`results/paper/auc_by_seed.csv`](results/paper/auc_by_seed.csv)

contains:

```text
14,080 seed-level experiment results
```

This is the primary source for:

- seed-level analysis;
- exact paired comparisons;
- statistical validation.

### Configuration-Level Results

[`results/paper/auc_by_run.csv`](results/paper/auc_by_run.csv)

contains:

```text
2,816 configuration-level results
```

aggregated over the five seeds.

### Raw Training Outputs

The complete raw training collection is not committed because the full study contains thousands of experiment directories and model checkpoints.

A typical raw run contains:

```text
config.json
train.csv
eval.csv
model_checkpoint.pth
```

The compact paper-level AUC tables are included so that the reported analyses can be reproduced without downloading the complete raw experiment archive.

---

## Area Under the Learning Curve

The core repository analysis computes the **area under the learning curve (AUC)** from evaluation returns over recorded environment steps using trapezoidal integration.

AUC captures performance across the complete observed learning trajectory rather than using only the final evaluation return.

The implementation is provided in:

```text
analysis/pipeline/extract_auc.py
```

The raw AUC values are subsequently used to construct matched ReLU-relative comparisons and other derived statistics.

---

## Analysis

The analysis code is contained in:

```text
analysis/
```

The core portable pipeline is:

```text
analysis/
├── README.md
├── auc.py
├── run_all.py
│
├── pipeline/
│   ├── __init__.py
│   ├── build_manifest.py
│   ├── extract_auc.py
│   ├── normalize_vs_relu.py
│   └── pair_seeds_vs_relu.py
│
└── paper/
    ├── activation_winners.py
    ├── alpha_best_frequency.py
    ├── alpha_dominance.py
    ├── alpha_sensitivity.py
    ├── alpha_tables.py
    ├── alpha_tuning_gain.py
    ├── appendix_tables.py
    ├── best_configurations.py
    ├── best_one_layer.py
    ├── best_two_layer.py
    ├── effect_statistics.py
    ├── fractional_performance.py
    ├── fractional_placement_effects.py
    ├── learning_dynamics.py
    ├── overall_performance.py
    ├── placement_effects.py
    ├── results_tables.py
    ├── statistical_validation.py
    └── statistical_validation_from_auc.py
```

A detailed description of every analysis module is provided in:

[`analysis/README.md`](analysis/README.md)

---

### Core Analysis Pipeline

Run:

```bash
python -m analysis.run_all \
  --results /path/to/experiment/root \
  --strict
```

The pipeline performs:

```text
raw experiment runs
        |
        v
run manifest
        |
        v
seed-level AUC
        |
        v
configuration-level AUC
        |
        v
matched ReLU normalization
        |
        v
exact same-seed pairing
```

For the complete experiment collection, the validated pipeline finds:

```text
14,080 total seed-level runs
14,080 usable evaluation files
0 missing evaluation files
0 missing training files
```

Algorithm counts:

```text
TD3 = 7,040
SAC = 7,040
```

Network-depth counts:

```text
1 layer  = 2,400
2 layers = 11,680
```

Exact matched comparison validation gives:

```text
160 ReLU seed baselines
13,920 non-ReLU candidate rows
0 missing ReLU pairs
```

---

## Analysis Outputs

Derived paper outputs are stored separately from primary numerical results:

```text
analysis_outputs/
```

The repository therefore follows the structure:

```text
raw experiment runs
        |
        v
results/paper/
primary compact numerical results
        |
        v
analysis/
reproducible analysis code
        |
        v
analysis_outputs/
derived tables, statistics, and figures
```

The curated analysis outputs include:

```text
analysis_outputs/
├── README.md
│
├── normalized/
│   ├── auc_by_run_normalized.csv
│   ├── auc_by_seed_normalized.csv
│   ├── paired_seed_rows.csv
│   └── relu_baseline_by_task.csv
│
└── paper/
    ├── overall/
    ├── alpha/
    │   └── figures/
    ├── appendix/
    ├── best_configurations/
    ├── figures/
    │   ├── one_layer/
    │   └── two_layer/
    ├── new_results/
    ├── placement/
    │   ├── ablation/
    │   ├── all_activations/
    │   └── fractional/
    └── statistics/
```

These outputs include:

- ReLU-relative AUC comparisons;
- exact seed-paired data;
- activation-family summaries;
- best configurations;
- fractional-order sensitivity;
- best-alpha frequency;
- alpha dominance;
- alpha tuning gain;
- activation-placement analysis;
- statistical summaries;
- appendix tables;
- publication figures and heatmaps.

---

## Statistical Interpretation

The repository reports several complementary forms of statistical evidence.

These include:

- exact same-seed comparisons;
- bootstrap confidence intervals;
- Wilcoxon signed-rank statistics;
- effect-size summaries;
- sign-test summaries;
- task-level consistency;
- alpha-sensitivity analysis;
- placement analysis.

### Best-Configuration Results

Best-configuration results should be interpreted as a **tuned-performance view**.

The best activation, `alpha`, and placement are selected from the evaluated experimental grid. When the same finite set of seeds is used both to select and summarize the strongest configuration, the resulting comparison is not an independent confirmatory significance test.

This is why the repository provides the full seed-level data and complementary robustness analyses rather than relying on a single best-configuration statistic.

### Five-Seed Statistical Resolution

Each configuration contains five seeds:

```text
10
20
30
40
50
```

With only five paired observations per task, exact non-parametric tests have relatively coarse p-value resolution.

For this reason, statistical interpretation should consider:

```text
direction of improvement
effect magnitude
bootstrap uncertainty
cross-task consistency
alpha robustness
placement robustness
```

alongside individual p-values.

---

## Reproducibility

The experiments are designed to isolate activation-function effects as far as practical within empirical deep reinforcement learning.

Across activation comparisons, the following components are held fixed:

- RL algorithm;
- benchmark environment;
- actor architecture;
- critic architecture;
- hidden width;
- network depth within each comparison;
- replay-buffer settings;
- learning rates;
- optimizer settings;
- target-network updates;
- policy-update rules;
- training budget;
- evaluation frequency;
- random seeds.

The controlled experimental variables are:

```text
activation family
fractional order
activation placement
network depth
```

The validated experiment inventory contains:

```text
14,080 / 14,080
```

planned seed-level runs.

---

## Repository Structure

```text
fractional-activation-functions-rl/
├── src/
│   └── frorl/
│       ├── algorithms/
│       │   ├── base.py
│       │   ├── td3.py
│       │   └── sac.py
│       │
│       ├── models/
│       │   ├── activations.py
│       │   ├── fractional_activations.py
│       │   ├── common.py
│       │   ├── td3_actor.py
│       │   ├── td3_critic.py
│       │   ├── sac_actor.py
│       │   ├── sac_critic.py
│       │   └── batchrenorm.py
│       │
│       ├── memory/
│       │   ├── memory_buffer.py
│       │   └── sum_tree.py
│       │
│       ├── encoders/
│       ├── environments/
│       │
│       ├── training/
│       │   └── train_loop.py
│       │
│       ├── configurations.py
│       ├── helpers.py
│       ├── training_context.py
│       └── training_utils.py
│
├── scripts/
│   ├── train.py
│   ├── run_sweep.py
│   ├── run_matrix.py
│   └── smoke_test_all.sh
│
├── analysis/
│   ├── README.md
│   ├── auc.py
│   ├── run_all.py
│   ├── pipeline/
│   └── paper/
│
├── results/
│   ├── README.md
│   └── paper/
│       ├── auc_by_run.csv
│       └── auc_by_seed.csv
│
├── analysis_outputs/
│   ├── README.md
│   ├── normalized/
│   └── paper/
│
├── tests/
├── configs/
├── assets/
│   └── fractional_activation_overview.png
│
├── README.md
├── LICENSE
├── CITATION.cff
├── pyproject.toml
└── .gitignore
```

---

## Design Principle

The experimental principle behind the repository is deliberately simple:

> **Change the activation function, not the reinforcement-learning algorithm.**

TD3 and SAC retain their standard:

- replay mechanisms;
- Bellman targets;
- target networks;
- policy updates;
- delayed updates where applicable;
- entropy optimization where applicable;
- optimization procedures.

Fractional activation design is introduced as a **lightweight architectural modification**, allowing its empirical effect to be studied without simultaneously introducing a new reinforcement-learning algorithm.

---

## Acknowledgements

This work was developed within the **Centre for Automation and Robotic Engineering Science (CARES)** at **The University of Auckland**.



## License

This repository is released under the **MIT License**.

See [LICENSE](LICENSE) for details.

Third-party software components remain subject to their respective licenses and attribution requirements.
