"""
This module defines a multi-layer perceptron (MLP) builder.
"""
import typing
from typing import Optional
from torch import nn as nn


def make_mlp(
    num_in: int,
    num_out: int,
    n_hidden_layers: int,
    n_hidden_units: int,
    final_activation: Optional[typing.Any] = None,
) -> nn.Sequential:
    """Build a multi-layer-perceptron ANN as a torch.nn.Sequential object."""
    layers = []

    layers.extend(
        [
            nn.Linear(num_in, n_hidden_units),
            nn.ReLU(),
        ]
    )

    for _ in range(n_hidden_layers):
        layers.extend(
            [
                nn.Linear(n_hidden_units, n_hidden_units),
                nn.LayerNorm(n_hidden_units),
                nn.ReLU(),
            ]
        )

    layers.append(nn.Linear(n_hidden_units, num_out))

    if final_activation is not None:
        layers.append(final_activation)

    return nn.Sequential(*layers)
