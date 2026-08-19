# Analysis and Reproducibility

This directory contains the complete analysis workflow used to evaluate the Fractional-RL experiments.

The analysis is intentionally separated into two levels:

1. **Core pipeline** — portable scripts that reconstruct the experiment inventory, compute AUC, normalize against ReLU, and build exact seed-paired comparisons.
2. **Paper analyses** — focused analyses used to study activation performance, fractional order, placement, learning dynamics, statistical evidence, and publication tables.

The validated experiment set contains:

```text
14,080 seed-level runs
```

with:

```text
7,040 TD3 runs
7,040 SAC runs
```

across eight continuous-control tasks.

---

# Directory Structure

```text
analysis/
├── README.md
├── __init__.py
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
    ├── __init__.py
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

---

# Quick Start

Run the complete portable core analysis with:

```bash
python -m analysis.run_all \
    --results /path/to/experiment/root \
    --strict
```

The core workflow performs:

```text
experiment directories
        ↓
run manifest
        ↓
seed-level AUC
        ↓
configuration-level AUC
        ↓
ReLU normalization
        ↓
exact same-seed pairing
```

Generated outputs are written under:

```text
analysis_outputs/
```

---

# Top-Level Analysis Files

## `auc.py`

Small AUC utility used by the test suite and lightweight analysis tasks.

Its main purpose is to provide a simple normalized-AUC calculation for small inputs and unit testing.

This is separate from the full experiment extractor in:

```text
pipeline/extract_auc.py
```

For complete experiment analysis, use the pipeline version.

---

## `run_all.py`

Main entry point for the portable analysis workflow.

It runs the core stages in sequence:

1. `build_manifest.py`
2. `extract_auc.py`
3. `normalize_vs_relu.py`
4. `pair_seeds_vs_relu.py`

Example:

```bash
python -m analysis.run_all \
    --results /path/to/experiment/root \
    --strict
```

Use this when reproducing the complete analysis from raw experiment result directories.

---

# Core Pipeline

The files under `analysis/pipeline/` form the reproducible foundation of the analysis.

These scripts should normally be run before any paper-specific analysis.

---

## `pipeline/build_manifest.py`

### Purpose

Discovers experiment runs and creates a structured inventory of the complete experiment set.

### What it identifies

For every run, the manifest records information such as:

- algorithm;
- task;
- activation;
- fractional order;
- network depth;
- activation placement;
- random seed;
- training file;
- evaluation file;
- run completeness.

### Why it matters

The manifest provides one consistent representation of the experiment directory tree.

Instead of every analysis script independently searching folders, downstream analysis can operate from one validated table.

### Validated result

For the complete study:

```text
Discovered rows  : 14,080
Usable eval.csv  : 14,080
Missing eval.csv : 0
Missing train.csv: 0
```

### Output

Typical output:

```text
analysis_outputs/run_manifest.csv
analysis_outputs/run_manifest_missing_eval.csv
```

---

## `pipeline/extract_auc.py`

### Purpose

Computes learning-curve area under the curve for every seed-level experiment.

### Input

The run manifest produced by:

```text
build_manifest.py
```

### Processing

For each evaluation trajectory, the script:

- reads training steps and evaluation returns;
- converts values to numeric form;
- sorts observations by environment step;
- handles duplicate evaluation steps consistently;
- computes trapezoidal AUC;
- records final return;
- records the number of evaluation observations;
- records first and last evaluation step;
- supports an optional AUC cutoff.

The reported AUC is the raw trapezoidal integral over evaluation steps.

### Output

Seed-level output:

```text
analysis_outputs/auc/auc_by_seed.csv
```

Configuration-level aggregation:

```text
analysis_outputs/auc/auc_by_run.csv
```

The configuration-level table aggregates seeds for each combination of:

```text
algorithm
task
network depth
placement
activation
alpha
```

### Validation

This implementation was numerically compared against the full historical paper AUC outputs and produced matching results.

---

## `pipeline/normalize_vs_relu.py`

### Purpose

Expresses activation performance relative to the corresponding ReLU reference.

### Why ReLU is used

ReLU provides the common reference architecture against which both conventional and fractional activations are compared.

### Matching variables

The baseline is matched using:

```text
algorithm
task
network depth
```

For the two-hidden-layer placement study, ReLU is treated as a shared reference because assigning ReLU to different nominal placement positions does not change the resulting ReLU network.

### Main quantities

The normalized outputs include quantities such as:

```text
candidate AUC
ReLU reference AUC
absolute difference
percentage difference relative to ReLU
```

### Validated result

The complete experiment set contains:

```text
32 ReLU baseline groups
```

### Output

```text
analysis_outputs/normalized/auc_by_seed_normalized.csv
analysis_outputs/normalized/auc_by_run_normalized.csv
analysis_outputs/normalized/relu_baseline_by_task.csv
```

---

## `pipeline/pair_seeds_vs_relu.py`

### Purpose

Builds exact candidate-versus-ReLU pairs using the same random seed.

### Pairing key

Each candidate run is paired with ReLU using:

```text
algorithm
task
network depth
seed
```

Placement is intentionally not part of the ReLU pairing key because ReLU is a shared reference in the placement study.

### Why this is important

Seed-paired comparison reduces unnecessary variation caused by comparing unrelated random seeds.

The paired table is the preferred source for analyses that examine:

- candidate-minus-ReLU difference;
- percentage improvement;
- sign consistency;
- paired statistical tests;
- paired effect summaries.

### Validated result

```text
Candidate rows : 13,920
ReLU baselines : 160
Missing pairs  : 0
```

### Output

```text
analysis_outputs/normalized/paired_seed_rows.csv
```

---

# Paper Analysis

The scripts under `analysis/paper/` operate on processed results and answer specific research questions used in the manuscript.

They are deliberately separated by purpose so that each analysis can be reproduced independently.

---

# Overall Performance

## `paper/overall_performance.py`

Evaluates broad performance across the complete activation set.

This analysis is useful for questions such as:

- Which activation families perform best overall?
- How often do activations improve on ReLU?
- How do results differ between TD3 and SAC?
- How do results differ across tasks and network depths?

It includes both conventional and fractional activations.

---

## `paper/fractional_performance.py`

Restricts the overall analysis to the fractional activation family:

```text
FReLU
FLReLU
FPReLU
FGELU
FSwish
```

Use this when the research question concerns differences **within the fractional family** rather than comparisons with conventional activations.

---

# Best-Configuration Analysis

## `paper/best_one_layer.py`

Identifies the strongest configurations for the one-hidden-layer architecture.

The one-layer experiment does not use the full two-layer placement matrix, so it is analyzed separately.

Typical questions include:

- Which activation performs best in one-layer networks?
- Which fractional order is preferred?
- How does one-layer performance compare across tasks and algorithms?

---

## `paper/best_two_layer.py`

Identifies the strongest configurations for two-hidden-layer networks.

Because the two-layer experiments include multiple activation placements, this analysis jointly considers:

- activation;
- fractional order;
- placement.

---

## `paper/best_configurations.py`

Produces a broader best-configuration view by activation and task.

It is useful for reporting the strongest tuned configuration for each activation family.

These results should be interpreted as a **tuned-performance view**, not as independent confirmatory evidence.

---

## `paper/activation_winners.py`

Summarizes which activation wins across tasks or experimental conditions.

Typical outputs can answer:

- How frequently does each activation achieve the best result?
- Which activation families dominate particular algorithms?
- Is one activation consistently strongest, or are winners task dependent?

This provides a complementary view to mean performance.

---

# Fractional-Order Analysis

## `paper/alpha_sensitivity.py`

Studies how performance changes with fractional order:

```text
0.1
0.2
0.3
0.4
0.5
```

This analysis addresses whether performance is sensitive to \(\alpha\) and whether the preferred fractional order depends on:

- activation family;
- task;
- algorithm;
- architecture;
- placement.

This is important because \(\alpha\) is treated as an activation-specific hyperparameter rather than a monotonic control variable.

---

## `paper/alpha_best_frequency.py`

Counts how frequently each fractional order becomes the best-performing choice.

For example, it can answer:

```text
How often is alpha = 0.1 selected as best?
How often is alpha = 0.3 selected as best?
```

This provides a simple view of whether one fractional order dominates globally or whether the preferred order varies across conditions.

---

## `paper/alpha_dominance.py`

Studies whether particular fractional orders systematically outperform others.

This is broader than simply counting the single best alpha.

It helps distinguish:

- a genuinely dominant fractional order;
- several similarly competitive orders;
- strongly task-dependent alpha preferences.

---

## `paper/alpha_tuning_gain.py`

Quantifies the benefit of tuning the fractional order.

Conceptually, it compares performance obtained from a selected best alpha with performance averaged or aggregated across alpha values.

This analysis helps answer:

> How much performance comes from the fractional activation itself, and how much additional benefit comes from tuning its fractional order?

---

## `paper/alpha_tables.py`

Exports fractional-order results into publication-oriented tables.

This script is used for presenting alpha sensitivity and best-alpha results in a compact manuscript or supplementary format.

---

# Activation Placement Analysis

## `paper/placement_effects.py`

Analyzes activation placement in the two-hidden-layer architecture.

The reported placements are:

```text
all_actor
all_critic
all_both
first_actor
first_both
```

This analysis can answer questions such as:

- Is actor-side placement more effective than critic-side placement?
- Is applying the activation everywhere better than only in the first layer?
- Are placement preferences consistent across algorithms or tasks?

---

## `paper/fractional_placement_effects.py`

Restricts placement analysis to the five fractional activations.

This is useful for understanding whether fractional activations behave differently depending on where they are introduced into actor and critic networks.

---

# Learning-Dynamics Analysis

## `paper/learning_dynamics.py`

Examines the shape and progression of learning rather than only aggregate AUC.

Possible quantities include:

- learning speed;
- trajectory behaviour;
- milestone-based learning measures;
- early versus late training performance.

This analysis provides additional context for understanding *how* an activation changes learning.

Learning-dynamics metrics should be interpreted carefully when their target level depends on each individual curve.

---

# Statistical Analysis

## `paper/effect_statistics.py`

Computes statistical summaries and effect-oriented quantities used to quantify the magnitude and consistency of activation improvements.

This is intended to complement simple averages and p-values.

---

## `paper/statistical_validation.py`

Performs the principal statistical validation used for activation comparisons.

It is designed to summarize evidence such as:

- paired differences;
- consistency of improvement;
- statistical tests;
- effect measures.

Because experiments use five paired seeds, exact two-sided tests have relatively coarse p-value resolution.

Statistical interpretation should therefore consider:

- effect magnitude;
- direction consistency;
- uncertainty;
- cross-task replication;

alongside p-values.

---

## `paper/statistical_validation_from_auc.py`

Reconstructs statistical validation directly from AUC-level results.

This provides an alternative route for reproducing statistical summaries from processed AUC tables rather than from intermediate paper-specific files.

---

# Publication Tables

## `paper/results_tables.py`

Produces manuscript-ready tables for the main performance results.

This script is intended for the core results presented in the paper.

---

## `paper/appendix_tables.py`

Produces more detailed supplementary or appendix tables.

These tables can contain results that are useful for full reproducibility but too large for the main manuscript.

---

# Result-Root Configuration

Paper-analysis scripts do not depend on a user-specific filesystem path.

Set the experiment root with:

```bash
export FRORL_RESULTS_ROOT=/path/to/experiment/root
```

If the variable is not set, paper-analysis scripts use:

```text
results/
```

as the default experiment root.

---

# Generated Output Structure

Core analysis output is organized under:

```text
analysis_outputs/
├── run_manifest.csv
├── run_manifest_missing_eval.csv
├── auc/
│   ├── auc_by_seed.csv
│   └── auc_by_run.csv
└── normalized/
    ├── auc_by_seed_normalized.csv
    ├── auc_by_run_normalized.csv
    ├── relu_baseline_by_task.csv
    └── paired_seed_rows.csv
```

Additional paper analyses may create further tables or derived summaries beneath `analysis_outputs/`.

Generated analysis outputs are intentionally excluded from Git version control.

---

# Recommended Analysis Order

For a new complete experiment set, the recommended workflow is:

```text
1. build_manifest.py
        ↓
2. extract_auc.py
        ↓
3. normalize_vs_relu.py
        ↓
4. pair_seeds_vs_relu.py
        ↓
5. overall_performance.py
        ↓
6. fractional_performance.py
        ↓
7. alpha_sensitivity.py
        ↓
8. placement_effects.py
        ↓
9. learning_dynamics.py
        ↓
10. effect_statistics.py / statistical_validation.py
        ↓
11. results_tables.py / alpha_tables.py / appendix_tables.py
```

The first four stages can be run automatically with:

```bash
python -m analysis.run_all \
    --results /path/to/experiment/root \
    --strict
```

---

# Statistical Interpretation of Tuned Results

Some analyses identify the strongest activation, fractional order, or placement using the same finite set of seeds later used to summarize its performance.

These results should be interpreted as a:

> **tuned-performance view**

rather than as an independent confirmatory test.

The repository therefore keeps several complementary forms of evidence:

- full seed-level results;
- exact seed-paired comparisons;
- activation-specific effect sizes;
- alpha sensitivity;
- placement sensitivity;
- task-level consistency;
- statistical validation.

This distinction is particularly important when reporting a single best configuration selected from many candidate hyperparameter combinations.

---

# Reproducibility Checks

The core analysis has been validated on the complete experiment set with:

```text
14,080 discovered runs
14,080 usable evaluation files
0 missing evaluation files
0 missing training files
```

AUC extraction produces:

```text
14,080 seed-level rows
2,816 configuration-level rows
```

Seed pairing produces:

```text
13,920 candidate rows
160 ReLU seed baselines
0 missing pairs
```

The portable AUC results were also numerically checked against the full paper-analysis outputs.

---

# Design Principle

The analysis follows the same principle as the experimental code:

> **Keep the reinforcement-learning algorithm fixed and measure the effect of activation design.**

The analysis therefore emphasizes controlled comparisons across:

```text
activation family
fractional order
network depth
activation placement
```

while matching algorithms, environments, training budgets, evaluation protocols, and random seeds.
