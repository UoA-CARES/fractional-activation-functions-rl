# Implementation notes

## Controlled experimental variable

Only hidden-layer activations are changed. TD3 and SAC learning rules are standard and shared across activation conditions.

## Rectifier fractional family

For fractional order `alpha`, the paper-facing rectifiers use Gamma normalization `Gamma(2-alpha)` and numerically safe positive magnitudes.

- FReLU: positive branch only.
- FLReLU: fixed leaky coefficient `k=0.1` by default.
- FPReLU: learnable positive negative-side coefficient, represented through `softplus` and clipped to `[0.01, 0.3]` for stability.

All three return exactly zero at `x=0`.

## FSwish

The implementation follows the experimental smooth fractional form

`swish + alpha * sigmoid(x) * (1 - swish)`

where `swish = x * sigmoid(x)`.

## FGELU

FGELU uses the finite Grünwald-Letnikov-style approximation used in the development code, with defaults `h=0.5` and `n_iter=3`. The inner GELU uses the tanh approximation.

## Placement

The activation factory is selected independently for every hidden layer. This makes actor/critic and first/all placement explicit without modifying TD3 or SAC.
