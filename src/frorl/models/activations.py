"""Paper-facing activation functions for the fractional RL study."""
from __future__ import annotations

import math
from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F


def _validate_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1] for this study; got {alpha}")
    return alpha


def _safe_power_positive(x: torch.Tensor, exponent: float, eps: float) -> torch.Tensor:
    return torch.clamp(x, min=eps).pow(exponent)


class FReLU(nn.Module):
    """Gamma-normalized fractional ReLU."""
    def __init__(self, alpha: float = 0.1, epsilon: float = 1e-8):
        super().__init__()
        self.alpha = _validate_alpha(alpha)
        self.epsilon = epsilon
        self.gamma = math.gamma(2.0 - self.alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positive = _safe_power_positive(x, 1.0 - self.alpha, self.epsilon) / self.gamma
        return torch.where(x > 0, positive, torch.zeros_like(x))


class FLReLU(nn.Module):
    """Gamma-normalized fractional Leaky ReLU with fixed negative slope."""
    def __init__(self, alpha: float = 0.1, k: float = 0.1, epsilon: float = 1e-8):
        super().__init__()
        self.alpha = _validate_alpha(alpha)
        self.k = float(k)
        self.epsilon = epsilon
        self.gamma = math.gamma(2.0 - self.alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = _safe_power_positive(x, 1.0 - self.alpha, self.epsilon) / self.gamma
        neg = -self.k * _safe_power_positive(-x, 1.0 - self.alpha, self.epsilon) / self.gamma
        return torch.where(x > 0, pos, torch.where(x < 0, neg, torch.zeros_like(x)))


class FPReLU(nn.Module):
    """Fractional PReLU with a learnable, stability-bounded negative coefficient."""
    def __init__(
        self,
        alpha: float = 0.1,
        parameter_init: float = 0.1,
        min_slope: float = 0.01,
        max_slope: float = 0.3,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.alpha = _validate_alpha(alpha)
        self.epsilon = epsilon
        self.min_slope = min_slope
        self.max_slope = max_slope
        # Inverse-softplus-ish initialization is unnecessary for exact continuity;
        # keep a direct raw value to preserve the original development convention.
        self.p_raw = nn.Parameter(torch.tensor(float(parameter_init)))
        self.gamma = math.gamma(2.0 - self.alpha)

    @property
    def negative_slope(self) -> torch.Tensor:
        return torch.clamp(F.softplus(self.p_raw), self.min_slope, self.max_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = _safe_power_positive(x, 1.0 - self.alpha, self.epsilon) / self.gamma
        neg = -self.negative_slope * _safe_power_positive(-x, 1.0 - self.alpha, self.epsilon) / self.gamma
        return torch.where(x > 0, pos, torch.where(x < 0, neg, torch.zeros_like(x)))


class FSwish(nn.Module):
    """Fractional Swish used in the experimental code."""
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = _validate_alpha(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sig = torch.sigmoid(x)
        swish = x * sig
        out = swish + self.alpha * sig * (1.0 - swish)
        return torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)


class FGELU(nn.Module):
    """Finite Grünwald-Letnikov-style fractional GELU."""
    def __init__(self, alpha: float = 0.1, h: float = 0.5, n_iter: int = 3):
        super().__init__()
        self.alpha = _validate_alpha(alpha)
        if h <= 0:
            raise ValueError("h must be positive")
        if n_iter < 1:
            raise ValueError("n_iter must be >= 1")
        self.h = float(h)
        self.n_iter = int(n_iter)
        self.register_buffer(
            "gamma_i_plus_1",
            torch.tensor([math.gamma(i + 1) for i in range(n_iter)], dtype=torch.float32),
        )

    @staticmethod
    def _gelu_tanh_twice(z: torch.Tensor) -> torch.Tensor:
        c = 0.7978845608028654  # sqrt(2/pi)
        return z * (1.0 + torch.tanh(c * (z + 0.044715 * z.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.tensor(self.alpha, dtype=x.dtype, device=x.device)
        h = torch.tensor(self.h, dtype=x.dtype, device=x.device)
        gamma_a1 = torch.exp(torch.lgamma(1.0 + alpha))
        gamma_i = self.gamma_i_plus_1.to(dtype=x.dtype, device=x.device)
        out = torch.zeros_like(x)
        for i in range(self.n_iter):
            z = x - i * h
            denom = gamma_i[i] * torch.exp(torch.lgamma(1.0 - i + alpha))
            coeff = gamma_a1 / (denom + 1e-8)
            sign = -1.0 if i % 2 else 1.0
            out = out + sign * coeff * self._gelu_tanh_twice(z)
        out = out / (2.0 * h.pow(alpha))
        return torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)


BASELINES = ("relu", "lrelu", "prelu", "gelu", "swish")
FRACTIONAL = ("frelu", "flrelu", "fprelu", "fgelu", "fswish")
ACTIVATIONS = BASELINES + FRACTIONAL


def activation_factory(name: str, alpha: float | None = None) -> Callable[[], nn.Module]:
    """Return a zero-argument module factory, suitable for one module per hidden layer."""
    key = name.strip().lower().replace("_", "-")
    aliases = {
        "leakyrelu": "lrelu", "leaky-relu": "lrelu",
        "silu": "swish",
        "fractionalrelu": "frelu", "fractional-relu": "frelu",
        "fractionalleakyrelu": "flrelu", "fractional-leaky-relu": "flrelu",
        "fractionalprelu": "fprelu", "fractional-prelu": "fprelu",
        "fractionalgelu": "fgelu", "fractional-gelu": "fgelu",
        "fractionalswish": "fswish", "fractional-swish": "fswish",
    }
    key = aliases.get(key, key)
    if key not in ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Choose from: {', '.join(ACTIVATIONS)}")
    if key == "relu": return nn.ReLU
    if key == "lrelu": return lambda: nn.LeakyReLU(negative_slope=0.1)
    if key == "prelu": return lambda: nn.PReLU(init=0.25)
    if key == "gelu": return nn.GELU
    if key == "swish": return nn.SiLU
    if alpha is None:
        raise ValueError(f"{key} requires alpha")
    if key == "frelu": return lambda: FReLU(alpha)
    if key == "flrelu": return lambda: FLReLU(alpha)
    if key == "fprelu": return lambda: FPReLU(alpha)
    if key == "fgelu": return lambda: FGELU(alpha)
    if key == "fswish": return lambda: FSwish(alpha)
    raise AssertionError(key)
