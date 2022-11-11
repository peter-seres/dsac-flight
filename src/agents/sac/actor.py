import torch
import torch.nn as nn
from torch.distributions import Independent, Normal
from agents.mlp import make_mlp

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class NormalPolicyNet(nn.Module):
    """Outputs a distribution with parameters learnable by gradient descent."""

    def __init__(
        self, obs_dim: int, action_dim: int, n_hidden_layers: int, n_hidden_units: int
    ):
        super().__init__()
        self.shared_net = make_mlp(
            num_in=obs_dim,
            num_out=n_hidden_units,
            n_hidden_layers=n_hidden_layers,
            n_hidden_units=n_hidden_units,
            final_activation=nn.ReLU(),
        )
        self.mu_layer = nn.Linear(n_hidden_units, action_dim)
        self.log_std_layer = nn.Linear(n_hidden_units, action_dim)

    def forward(self, states: torch.tensor):
        out = self.shared_net(states)
        means, log_stds = self.mu_layer(out), self.log_std_layer(out)
        stds = torch.exp(torch.clamp(log_stds, LOG_STD_MIN, LOG_STD_MAX))

        return Independent(Normal(loc=means, scale=stds), reinterpreted_batch_ndims=1)

    def get_mean(self, states: torch.Tensor):
        out = self.shared_net(states)
        mean = self.mu_layer(out)
        return mean

    def get_std(self, states: torch.Tensor):
        out = self.shared_net(states)
        log_std = self.log_std_layer(out)
        return torch.exp(torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX))

    def get_bayesian(self, states: torch.Tensor):
        out = self.shared_net(states)
        means, log_stds = self.mu_layer(out), self.log_std_layer(out)
        stds = torch.exp(torch.clamp(log_stds, LOG_STD_MIN, LOG_STD_MAX))
        return means, stds
