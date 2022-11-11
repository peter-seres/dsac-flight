import torch
from torch import nn as nn
from agents.mlp import make_mlp


class QNet(nn.Module):
    """Feed-forward MLP Q-function of the form: Q(s, a)"""

    def __init__(self, obs_dim, action_dim, n_hidden_layers: int, n_hidden_units: int):
        super(QNet, self).__init__()
        self.net = make_mlp(
            num_in=obs_dim + action_dim,
            num_out=1,
            n_hidden_layers=n_hidden_layers,
            n_hidden_units=n_hidden_units,
            final_activation=None,
        )

    def forward(self, states: torch.tensor, actions: torch.tensor):
        return self.net(torch.cat([states, actions], dim=1))
