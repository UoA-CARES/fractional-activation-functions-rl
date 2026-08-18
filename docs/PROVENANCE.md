# Provenance

The study originally lived across two development codebases:

- CARES reinforcement-learning framework branch `feature/fractional-swish-gelu`
- environment/experiment framework branch `feature/fractional-activations`

Those projects contained many algorithms, environment adapters and experimental activation prototypes unrelated to the final paper. This repository is a standalone extraction intended to make the reported activation study easier to understand and reproduce.

The TD3 and SAC implementations in `src/frorl/algorithms/` are cleaned versions of the original code. The generic MLP/policy/critic code preserves per-hidden-layer activation factories so placement can be changed without altering the learning algorithm.

The final public registry includes only ReLU, LReLU, PReLU, GELU, Swish and the five reported fractional variants FReLU, FLReLU, FPReLU, FGELU and FSwish.

`src/frorl/models/legacy_activations.py` contains compatibility implementations recovered from earlier development files. It is provided for traceability and is not used by default experiments.
