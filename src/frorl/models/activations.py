"""
Fractional activation functions for off-policy reinforcement learning.

This module defines the public activation interface used throughout the
Fractional-RL repository.

Five fractional activation functions form the main activation family evaluated
in the associated study:

    FReLU
    FLReLU
    FPReLU
    FSwish
    FGELU

Additional fractional, residual, adaptive, beta-scaled, and numerically
protected variants are also available for further experimentation.

The implementation supports conventional activation baselines together with
fractional-order alternatives for TD3 and SAC.
"""

from __future__ import annotations

from typing import Callable

from torch import nn

# All fractional formulations available in the project.
from .fractional_activations import *

# Internal references used to provide concise public names.
from .fractional_activations import (
    FractionalReLUPositive as _FReLUImplementation,
    FLReLU2 as _FLReLUImplementation,
    FPReLU2 as _FPReLUImplementation,
    FractionalSwishBeta as _FSwishImplementation,
    FractionalGELUBeta as _FGELUImplementation,
)


# ======================================================================
# Main activation family
# ======================================================================

class FReLU(_FReLUImplementation):
    """
    Fractional Rectified Linear Unit (FReLU).

    FReLU extends the positive branch of ReLU using a fractional power:

        FReLU_alpha(x) = x^(1-alpha),  x > 0
                         0,            x <= 0

    The fractional order alpha controls the curvature of the positive
    response. This formulation does not apply Gamma normalization.
    """


class FLReLU(_FLReLUImplementation):
    """
    Fractional Leaky Rectified Linear Unit (FLReLU).

    FLReLU applies fractional-power transformations to both branches:

        x^(1-alpha) / Gamma(2-alpha),                    x > 0

        -k (-x)^(1-alpha) / Gamma(2-alpha),              x < 0

    The fractional order alpha is fixed for an experiment. The negative
    coefficient k is learnable and constrained to the interval [0.01, 0.3].
    """


class FPReLU(_FPReLUImplementation):
    """
    Fractional Parametric Rectified Linear Unit (FPReLU).

    FPReLU combines fractional-power scaling with a learnable negative
    coefficient:

        x^(1-alpha) / Gamma(2-alpha),                    x > 0

        -p (-x)^(1-alpha) / Gamma(2-alpha),              x < 0

    The fractional order alpha is fixed for an experiment, while p is
    learned through a positive softplus parameterization.
    """


class FSwish(_FSwishImplementation):
    """
    Fractional Swish (FSwish).

    FSwish extends the smooth Swish/SiLU family using fractional modulation
    and a learnable beta-controlled sigmoid gate.

    Define

        s_beta(x) = x * sigmoid(beta*x)

    then

        FSwish_alpha(x)
            = s_beta(x)
            + alpha * sigmoid(beta*x) * (1 - s_beta(x))

    Alpha is fixed for each experimental configuration and beta is learned
    during optimization.
    """


class FGELU(_FGELUImplementation):
    """
    Fractional Gaussian Error Linear Unit (FGELU).

    FGELU extends GELU using a finite Grünwald-Letnikov-style fractional
    approximation over shifted beta-scaled GELU terms.

    Parameters
    ----------
    a:
        Fractional order.
    beta_init:
        Initial value associated with the learnable beta parameter.
    h:
        Finite-difference step size.
    n_iter:
        Number of terms in the finite fractional approximation.
    """


# ======================================================================
# Main paper activation sets
# ======================================================================

BASELINES = (
    "relu",
    "lrelu",
    "prelu",
    "gelu",
    "swish",
)

FRACTIONAL = (
    "frelu",
    "flrelu",
    "fprelu",
    "fgelu",
    "fswish",
)

ACTIVATIONS = BASELINES + FRACTIONAL


# ======================================================================
# Public activation factory
# ======================================================================

def activation_factory(
    name: str,
    alpha: float | None = None,
) -> Callable[[], nn.Module]:
    """
    Return a fresh activation module for a hidden layer.

    Parameters
    ----------
    name:
        Activation name.
    alpha:
        Fractional order. Required for fractional activations.

    Supported conventional activations
    ----------------------------------
    ReLU, Leaky ReLU, PReLU, GELU, Swish/SiLU.

    Supported main fractional activations
    -------------------------------------
    FReLU, FLReLU, FPReLU, FGELU, FSwish.
    """

    key = name.strip().lower().replace("_", "-")

    aliases = {
        # Conventional names
        "leakyrelu": "lrelu",
        "leaky-relu": "lrelu",
        "silu": "swish",

        # FReLU
        "fractionalrelu": "frelu",
        "fractional-relu": "frelu",
        "fractionalrelupositive": "frelu",

        # FLReLU
        "fractionalleakyrelu": "flrelu",
        "fractional-leaky-relu": "flrelu",
        "flrelu2": "flrelu",

        # FPReLU
        "fractionalprelu": "fprelu",
        "fractional-prelu": "fprelu",
        "fprelu2": "fprelu",

        # FGELU
        "fractionalgelubeta": "fgelu",
        "fractional-gelu-beta": "fgelu",

        # FSwish
        "fractionalswishbeta": "fswish",
        "fractional-swish-beta": "fswish",
    }

    key = aliases.get(key, key)

    if key not in ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{name}'. "
            f"Choose from: {', '.join(ACTIVATIONS)}"
        )

    # Conventional baselines
    if key == "relu":
        return nn.ReLU

    if key == "lrelu":
        return lambda: nn.LeakyReLU(negative_slope=0.1)

    if key == "prelu":
        return lambda: nn.PReLU(init=0.25)

    if key == "gelu":
        return nn.GELU

    if key == "swish":
        return nn.SiLU

    # Fractional activations require alpha.
    if alpha is None:
        raise ValueError(f"{key} requires alpha")

    if key == "frelu":
        return lambda: FReLU(a=alpha)

    if key == "flrelu":
        return lambda: FLReLU(a=alpha)

    if key == "fprelu":
        return lambda: FPReLU(a=alpha)

    if key == "fgelu":
        return lambda: FGELU(a=alpha)

    if key == "fswish":
        return lambda: FSwish(a=alpha)

    raise AssertionError(key)
