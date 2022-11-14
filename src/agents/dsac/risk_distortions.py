"""
This module defines 4 risk-distortion functions: [neutral, cvar, cpw, wang]
and a dictionary that maps these keys to DistortionFn callables.
All distortion functions take a tau tensor and a xi scalar distortion parameter.
"""

import numpy as np
import torch
from typing import Dict, Callable

DistortionFn = Callable[[torch.Tensor, float], torch.Tensor]


def normal_cdf(tau: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    """C.D.F. of the normal distribution."""
    return 0.5 * (1 + torch.erf((tau - mean) / std / np.sqrt(2)))


def normal_inverse_cdf(
    tau: torch.Tensor, mean: float = 0.0, std: float = 1.0
) -> torch.Tensor:
    """Inverse C.D.F. of the normal distribution."""
    return mean + std * torch.erfinv(2 * tau - 1) * np.sqrt(2)


def neutral(tau: torch.Tensor, _) -> torch.Tensor:
    """Neutral distortion returns the original quantiles."""
    return tau


def cvar(tau: torch.Tensor, xi: float) -> torch.Tensor:
    """Conditional value at risk distorts the slopes. Clamps between 0 and 1."""
    return torch.clamp(tau * xi, min=0.0, max=1.0)


def cpw(tau: torch.Tensor, xi: float):
    """Cumulative Prob. weighting"""
    tau_pow_xi = torch.pow(tau, xi)
    return tau_pow_xi / torch.pow((tau_pow_xi + torch.pow(1.0 - tau, xi)), (1.0 / xi))


def wang(tau: torch.Tensor, xi: float):
    return normal_cdf(normal_inverse_cdf(tau) + xi)


distortion_functions: Dict[str, DistortionFn] = {
    "neutral": neutral,
    "cvar": cvar,
    "cpw": cpw,
    "wang": wang,
}
