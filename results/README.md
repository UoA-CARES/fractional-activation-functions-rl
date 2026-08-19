# Paper Results

This directory contains compact result files from the reported Fractional-RL experiments.

The complete study contains **14,080 seed-level runs** across TD3 and SAC, eight continuous-control tasks, one- and two-layer networks, conventional and fractional activations, fractional-order settings, activation placements, and five random seeds.

## Included files

### `paper/auc_by_seed.csv`

Seed-level learning-curve results for all **14,080 experiments**.

This file is the primary source for seed-level comparisons, normalization against ReLU, paired analyses, and statistical validation.

### `paper/auc_by_run.csv`

Configuration-level aggregation containing **2,816 configurations**.

Configurations are grouped by algorithm, task, network depth, placement, activation, and fractional order, with performance aggregated across seeds.

## AUC computation

Area under the learning curve is computed using trapezoidal integration of evaluation return over environment steps.

The reproducible implementation is provided in:

`analysis/pipeline/extract_auc.py`

The committed result files were numerically verified against the reproducible analysis pipeline.

## Raw experiment outputs

Raw training trajectories and model checkpoints are not committed because the complete experiment set contains thousands of run directories.

A complete raw run typically contains:

- `config.json`
- `train.csv`
- `eval.csv`
- `model_checkpoint.pth`

The repository instead provides the training code, complete experiment-matrix generator, reproducible analysis pipeline, and compact paper-level results required for the reported analyses.
