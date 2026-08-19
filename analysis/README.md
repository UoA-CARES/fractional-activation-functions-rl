# Analysis and Reproducibility

This directory contains the reproducible analysis workflow for the Fractional-RL experiments.

## Structure

```text
analysis/
├── run_all.py
├── pipeline/
│   ├── build_manifest.py
│   ├── extract_auc.py
│   ├── normalize_vs_relu.py
│   └── pair_seeds_vs_relu.py
├── paper/
│   ├── activation_winners.py
│   ├── alpha_best_frequency.py
│   ├── alpha_dominance.py
│   ├── alpha_sensitivity.py
│   ├── alpha_tables.py
│   ├── alpha_tuning_gain.py
│   ├── appendix_tables.py
│   ├── best_configurations.py
│   ├── best_one_layer.py
│   ├── best_two_layer.py
│   ├── effect_statistics.py
│   ├── fractional_performance.py
│   ├── fractional_placement_effects.py
│   ├── learning_dynamics.py
│   ├── overall_performance.py
│   ├── placement_effects.py
│   ├── results_tables.py
│   ├── statistical_validation.py
│   └── statistical_validation_from_auc.py
```

## Core portable pipeline

Run all core stages with:

```bash
python -m analysis.run_all \
    --results /path/to/experiment/root \
    --strict
```

For the historical paper experiment directory:

```bash
python -m analysis.run_all \
    --results /path/to/experiment/root \
    --strict
```

The validated historical dataset contains **14,080 seed-level runs**.

## 1. Run manifest

`pipeline/build_manifest.py` inventories each seed-level experiment and records:

- algorithm
- task
- network depth
- activation
- fractional order
- activation placement
- random seed
- training and evaluation result paths

The complete dataset contains:

```text
TD3: 7,040 runs
SAC: 7,040 runs
Total: 14,080 runs
```

across eight continuous-control tasks.

## 2. Learning-curve AUC

`pipeline/extract_auc.py` computes per-seed area under the learning curve using trapezoidal integration of evaluation return over environment steps.

Before integration, evaluation points are:

- converted to numeric values;
- ordered by environment step;
- cleaned of invalid values;
- deduplicated by step, retaining the final recorded value.

The extractor also records:

- final evaluation return;
- number of evaluation points;
- first evaluation step;
- last evaluation step;
- optional cutoff AUC.

Outputs:

```text
analysis_outputs/auc/auc_by_seed.csv
analysis_outputs/auc/auc_by_run.csv
```

The portable implementation has been numerically verified against the original full paper-analysis AUC outputs.

## 3. ReLU normalization

`pipeline/normalize_vs_relu.py` compares activation performance with the matched ReLU reference.

The normalization matches on:

- algorithm;
- task;
- network depth.

For the two-layer placement study, ReLU is treated as a shared reference because changing the nominal placement label does not alter a ReLU network.

The validated analysis contains **32 ReLU baseline groups**.

## 4. Exact same-seed pairing

`pipeline/pair_seeds_vs_relu.py` pairs every non-ReLU candidate with the ReLU result from the same:

- algorithm;
- task;
- network depth;
- random seed.

The validated full dataset produces:

```text
Total seed-level runs : 14,080
ReLU seed baselines   : 160
Candidate seed rows   : 13,920
Missing ReLU pairs    : 0
```

The paired output includes candidate AUC, ReLU AUC, absolute AUC difference, and percentage change relative to the paired ReLU seed.

## Paper analyses

The clean modules under `analysis/paper/` organize the main manuscript analyses.

### Performance

- `overall_performance.py`
- `fractional_performance.py`
- `best_one_layer.py`
- `best_two_layer.py`
- `best_configurations.py`
- `activation_winners.py`

### Fractional-order analysis

- `alpha_sensitivity.py`
- `alpha_best_frequency.py`
- `alpha_dominance.py`
- `alpha_tuning_gain.py`
- `alpha_tables.py`

### Placement analysis

- `placement_effects.py`
- `fractional_placement_effects.py`

### Learning dynamics

- `learning_dynamics.py`

### Statistical validation

- `effect_statistics.py`
- `statistical_validation.py`
- `statistical_validation_from_auc.py`

### Paper and appendix exports

- `results_tables.py`
- `appendix_tables.py`

## Statistical interpretation

Best-configuration reporting is a **tuned-performance analysis**.

When the best activation/order/placement configuration is selected and evaluated using the same finite seed set, that result should not be interpreted as an independent confirmatory significance test.

For this reason, the repository retains complementary evidence including:

- exact seed-paired effects;
- effect magnitudes;
- confidence-oriented summaries;
- cross-task consistency;
- fractional-order sensitivity;
- placement robustness;
- statistical tests.

With five paired seeds, exact two-sided tests have coarse p-value resolution. Effect size and consistency are therefore important alongside p-values.

